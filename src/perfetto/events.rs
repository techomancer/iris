//! The catalogue of instrumentation points: every `EventDef` the emulator
//! emits, grouped by subsystem, so the trace's track layout can be read (and
//! kept consistent) in one place.
//!
//! Conventions:
//! - Work a thread does (a GO, a compile, a SCSI command it runs) goes on the
//!   thread's own track (`Track::Thread`, the default).
//! - State that is not one thread's work (an interrupt line, a DMA channel
//!   being busy, the guest stalled on a full FIFO) goes on a named track
//!   (`.on_track("...")`) in a group (`.in_group("...")`). Begin and end may
//!   then come from different threads.
//! - Counters are tracks of their own, named after the event.
//! - Track names are global: two events naming the same track share it.

use super::{cat, EventDef};

const IRQ_GROUP: &str = "Interrupts";
const PIN_GROUP: &str = "CPU interrupt pins";
const DMA_GROUP: &str = "DMA";
const GFX_GROUP: &str = "Graphics";
const AUDIO_GROUP: &str = "Audio";

// ── Interrupts ───────────────────────────────────────────────────────────────

const fn line(name: &'static str) -> EventDef {
    EventDef::new(name, cat::IRQ).on_track(name).in_group(IRQ_GROUP)
}
const fn tick(name: &'static str) -> EventDef {
    EventDef::new(name, cat::IRQ | cat::TIMER).on_track(name).in_group(IRQ_GROUP)
}
const fn pin(name: &'static str) -> EventDef {
    EventDef::new(name, cat::IRQ).on_track(name).in_group(PIN_GROUP)
}

/// IOC/INT2 lines, one slice per assertion (raw status, before masking),
/// indexed by `l0 | l1 << 8 | map << 16 | (IP2..IP6) << 24`; see
/// `IocState::update_interrupts`. Timer latches stay high until the guest
/// clears them, so their slices measure tick-to-acknowledge latency.
pub static IOC_LINES: [EventDef; 32] = [
    // Local 0
    line("L0 fifo full"), line("L0 scsi0"), line("L0 scsi1"), line("L0 ethernet"),
    line("L0 mc dma"), line("L0 parallel"), line("L0 graphics"), line("L0 map int0"),
    // Local 1
    line("L1 gp0"), line("L1 panel"), line("L1 gp2"), line("L1 map int1"),
    line("L1 hpc dma"), line("L1 ac fail"), line("L1 video vsync"), line("L1 vertical retrace"),
    // Mappable
    tick("slow tick (8254 timer 0)"), tick("fast tick (8254 timer 1)"), line("map2"), line("map3"),
    line("map kbd/mouse"), line("map serial"), line("map6 gio exp0 / gfx drain0"), line("map7 gio exp1 / gfx drain1"),
    // CPU pins driven by the IOC
    pin("IP2 (local 0)"), pin("IP3 (local 1)"), pin("IP4 (timer 0)"), pin("IP5 (timer 1)"),
    pin("IP6 (bus error)"), pin("unused 29"), pin("unused 30"), pin("unused 31"),
];

/// CP0 Count/Compare match, asserted by the timer thread, ended by the
/// guest's Compare write.
pub static IP7_COMPARE: EventDef =
    EventDef::new("IP7 (count/compare)", cat::IRQ | cat::TIMER).on_track("IP7 (count/compare)").in_group(PIN_GROUP);
pub static IP7_LEVEL: crate::PfLevel = crate::PfLevel::new();

/// The CPU took an interrupt exception (arg: Cause.IP & Status.IM >> 8).
pub static INT_TAKEN: EventDef = EventDef::new("interrupt taken", cat::CPU | cat::IRQ).with_arg("ip");

// ── HPC3 peripheral DMA ──────────────────────────────────────────────────────

macro_rules! pdma {
    ($name:literal, $arg:expr, $($track:literal),*) => {
        [$(EventDef::new($name, cat::DMA).on_track($track).in_group(DMA_GROUP).with_arg($arg)),*]
    };
}
macro_rules! pdma_noarg {
    ($name:literal, $($track:literal),*) => {
        [$(EventDef::new($name, cat::DMA).on_track($track).in_group(DMA_GROUP)),*]
    };
}

/// Channel active (arg: first descriptor address). Indexed by channel:
/// 0-7 PBUS (HAL2 audio and friends), 8/9 SCSI, 10 enet RX, 11 enet TX.
pub static PDMA_ACTIVE: [EventDef; 12] = pdma!("active", "nbdp",
    "PDMA pbus0", "PDMA pbus1", "PDMA pbus2", "PDMA pbus3", "PDMA pbus4", "PDMA pbus5", "PDMA pbus6", "PDMA pbus7",
    "PDMA scsi0", "PDMA scsi1", "PDMA enet rx", "PDMA enet tx");
/// A data descriptor was loaded (arg: its byte count and flags word).
pub static PDMA_DESC: [EventDef; 12] = pdma!("desc", "bc",
    "PDMA pbus0", "PDMA pbus1", "PDMA pbus2", "PDMA pbus3", "PDMA pbus4", "PDMA pbus5", "PDMA pbus6", "PDMA pbus7",
    "PDMA scsi0", "PDMA scsi1", "PDMA enet rx", "PDMA enet tx");
/// The channel raised its completion interrupt.
pub static PDMA_INT: [EventDef; 12] = pdma_noarg!("int",
    "PDMA pbus0", "PDMA pbus1", "PDMA pbus2", "PDMA pbus3", "PDMA pbus4", "PDMA pbus5", "PDMA pbus6", "PDMA pbus7",
    "PDMA scsi0", "PDMA scsi1", "PDMA enet rx", "PDMA enet tx");

// ── MC graphics DMA ──────────────────────────────────────────────────────────

const fn vdma(name: &'static str) -> EventDef {
    EventDef::new(name, cat::DMA | cat::GFX).on_track("VDMA").in_group(DMA_GROUP).with_arg("bytes")
}
pub static VDMA_TO_GIO: EventDef = vdma("mem -> gio");
pub static VDMA_TO_HOST: EventDef = vdma("gio -> mem");
pub static VDMA_FILL: EventDef = vdma("fill");

// ── SCSI (WD33C93A) ──────────────────────────────────────────────────────────

/// A WD33C93A command (arg: command byte).
pub static SCSI_WD_CMD: EventDef = EventDef::new("wd cmd", cat::SCSI).with_arg("cmd");
/// A SCSI CDB being executed (arg: target << 8 | opcode).
pub static SCSI_CDB: EventDef = EventDef::new("cdb", cat::SCSI).with_arg("target<<8|op");
/// The chip asserted INT (arg: SCSI status register).
pub static SCSI_INT: EventDef = EventDef::new("chip int", cat::SCSI).with_arg("status");

// ── Ethernet (SEEQ 8003) ─────────────────────────────────────────────────────

pub static ENET_TX: EventDef = EventDef::new("tx frame", cat::NET).with_arg("len");
pub static ENET_RX: EventDef = EventDef::new("rx frame", cat::NET).with_arg("len");
/// A frame waits: the guest has no receive buffer for it.
pub static ENET_RX_REFUSED: EventDef = EventDef::new("rx no buffer", cat::NET);
pub static ENET_RX_FILTERED: EventDef = EventDef::new("rx filtered", cat::NET);

// ── Audio (HAL2) ─────────────────────────────────────────────────────────────

/// One codec A pacing wake on the timer thread (arg: frames due).
pub static HAL2_PACE: EventDef = EventDef::new("codec A pace", cat::AUDIO).with_arg("frames");
/// A codec was armed (arg: codec << 8 | PBUS DMA channel) or disarmed (arg: codec).
/// Codecs: 0 = A (DAC), 1 = B (ADC), 2 = AES TX, 3 = AES RX.
pub static HAL2_ARM: EventDef = EventDef::new("arm", cat::AUDIO).on_track("HAL2").in_group(AUDIO_GROUP).with_arg("codec<<8|ch");
pub static HAL2_DISARM: EventDef = EventDef::new("disarm", cat::AUDIO).on_track("HAL2").in_group(AUDIO_GROUP).with_arg("codec");
/// The host audio callback found the ring empty (arg: underruns so far).
pub static HAL2_UNDERRUN: EventDef = EventDef::new("host underrun", cat::AUDIO).on_track("HAL2").in_group(AUDIO_GROUP).with_arg("total");

// ── Newport (REX3) ───────────────────────────────────────────────────────────

/// One GO on the REX3 processor thread (arg: DRAWMODE0 opcode).
pub static REX3_GO: EventDef = EventDef::new("GO", cat::PRIM).with_arg("opcode");
/// The guest reads STATUS and sees GFXBUSY: it is (most likely) waiting
/// for the engine.
pub static REX3_WAIT_IDLE: EventDef = EventDef::new("guest sees busy", cat::GFX).on_track("REX3 busy wait").in_group(GFX_GROUP);
/// The command FIFO is full and a CPU store is being retried.
pub static REX3_FIFO_FULL: EventDef = EventDef::new("gfifo full", cat::GFX).on_track("REX3 gfifo full").in_group(GFX_GROUP);
/// GFIFO level as STATUS reports it, sampled on each STATUS read.
pub static REX3_GFIFO_LEVEL: EventDef = EventDef::new("REX3 gfifo level", cat::GFX).in_group(GFX_GROUP);
/// Vertical blank begins (the refresh thread raised VRINT).
pub static REX3_VBLANK: EventDef = EventDef::new("vblank", cat::GFX).on_track("REX3 display").in_group(GFX_GROUP);
/// An XMAP9 mode table entry was written: how IRIX swaps buffers on
/// Newport (arg: entry << 24 | value).
pub static REX3_XMAP_MODE: EventDef = EventDef::new("xmap mode write", cat::GFX).on_track("REX3 display").in_group(GFX_GROUP).with_arg("entry<<24|val");
/// Composing and presenting a frame on the refresh thread.
pub static REX3_PRESENT: EventDef = EventDef::new("present", cat::GFX);

// ── XZ / Extreme (GR2) ───────────────────────────────────────────────────────

pub static GR2_WAIT: EventDef = EventDef::new("guest read stalled", cat::GFX).on_track("GR2 busy wait").in_group(GFX_GROUP);
pub static GR2_FIFO_FULL: EventDef = EventDef::new("fifo full", cat::GFX).on_track("GR2 fifo full").in_group(GFX_GROUP);
pub static GR2_VBLANK: EventDef = EventDef::new("vblank", cat::GFX).on_track("GR2 display").in_group(GFX_GROUP);

// ── IMPACT (MGRAS) ───────────────────────────────────────────────────────────

/// A read that needs the board idle is being retried.
pub static MGRAS_WAIT: EventDef = EventDef::new("guest read waits idle", cat::GFX).on_track("IMPACT busy wait").in_group(GFX_GROUP);
pub static MGRAS_FIFO_FULL: EventDef = EventDef::new("fifo full", cat::GFX).on_track("IMPACT fifo full").in_group(GFX_GROUP);
/// Command FIFO level as FIFOSTATUS reports it (0 empty, 1 until it is
/// nearly full for real).
pub static MGRAS_CFIFO_LEVEL: EventDef = EventDef::new("IMPACT cfifo level", cat::GFX).in_group(GFX_GROUP);
/// The FIFO interrupt line (GIO interrupt 0) is asserted.
pub static MGRAS_FIFO_INTR: EventDef = EventDef::new("fifo interrupt", cat::GFX).on_track("IMPACT fifo interrupt").in_group(GFX_GROUP);
/// HQ3 / raster engine thread has work (from first entry to empty FIFO).
pub static MGRAS_HQ_BUSY: EventDef = EventDef::new("HQ3 busy", cat::GFX);
pub static MGRAS_RSS_BUSY: EventDef = EventDef::new("RSS busy", cat::GFX);
/// Host DMA on the HQ3 thread (arg: DMA start word).
pub static MGRAS_DMA_TO_BOARD: EventDef = EventDef::new("host dma -> board", cat::GFX | cat::DMA).with_arg("word");
pub static MGRAS_DMA_TO_HOST: EventDef = EventDef::new("host dma -> host", cat::GFX | cat::DMA).with_arg("word");
/// GL's SCHEDULE_SWAP reached the GE (the flip request).
pub static MGRAS_SWAP: EventDef = EventDef::new("schedule swap", cat::GFX).on_track("IMPACT display").in_group(GFX_GROUP);
/// The kernel's BUF_SELECT write: the flip itself (arg: value).
pub static MGRAS_BUF_SELECT: EventDef = EventDef::new("buf select", cat::GFX).on_track("IMPACT display").in_group(GFX_GROUP).with_arg("val");
/// SCHEDULE_SWAP waiting for the raster engines to finish the frame
/// before the swap is reported (arg: raster entries queued).
pub static MGRAS_SWAP_WAIT: EventDef = EventDef::new("swap waits raster", cat::GFX).on_track("IMPACT swap wait").in_group(GFX_GROUP).with_arg("rss_queued");
pub static MGRAS_RETRACE: EventDef = EventDef::new("retrace", cat::GFX).on_track("IMPACT display").in_group(GFX_GROUP);
/// The refresh thread copying the shown frame out of the framebuffer
/// (snapshot and compose), and handing it to the window.
pub static MGRAS_CAPTURE: EventDef = EventDef::new("frame capture", cat::GFX).on_track("IMPACT capture").in_group(GFX_GROUP);
pub static MGRAS_PRESENT: EventDef = EventDef::new("present", cat::GFX).on_track("IMPACT present").in_group(GFX_GROUP);
/// An XMAP display mode written: the mode of one window ID, main or
/// overlay (arg: register << 40 | index << 32 | value).
pub static MGRAS_XMAP_MODE: EventDef = EventDef::new("xmap mode write", cat::GFX).on_track("IMPACT display").in_group(GFX_GROUP).with_arg("reg<<40|index<<32|val");
/// A VC3 register write (arg: index << 16 | value) and a VC3 SRAM write,
/// the window-ID tables among others (arg: address << 16 | value).
pub static MGRAS_VC3_WRITE: EventDef = EventDef::new("vc3 write", cat::GFX).on_track("IMPACT vc3").in_group(GFX_GROUP).with_arg("index<<16|val");
pub static MGRAS_VC3_SRAM: EventDef = EventDef::new("vc3 sram write", cat::GFX).on_track("IMPACT vc3").in_group(GFX_GROUP).with_arg("addr<<16|val");
/// The VC3 current-line register read (arg: the line answered).
pub static MGRAS_VC3_LINE: EventDef = EventDef::new("vc3 line read", cat::GFX).on_track("IMPACT vc3").in_group(GFX_GROUP).with_arg("line");
/// A STATUS read (arg: the value; bit 2 vertical blank).
pub static MGRAS_STATUS_READ: EventDef = EventDef::new("status read", cat::GFX).on_track("IMPACT status").in_group(GFX_GROUP).with_arg("val");
/// Flags raised by the engines or a host SET_FLAGS write (arg: raster
/// entries still queued << 32 | bits), and a host CLEAR_FLAGS write (arg:
/// bits).
pub static MGRAS_FLAG_SET: EventDef = EventDef::new("flags set", cat::GFX).on_track("IMPACT flags").in_group(GFX_GROUP).with_arg("rss_queued<<32|bits");
pub static MGRAS_FLAG_CLEAR: EventDef = EventDef::new("flags cleared", cat::GFX).on_track("IMPACT flags").in_group(GFX_GROUP).with_arg("bits");
/// The bank GL draws into, from VALIDATE_BANKS or a context image (arg:
/// bank, 1 = B).
pub static MGRAS_DRAW_BANK: EventDef = EventDef::new("draw bank", cat::GFX).on_track("IMPACT display").in_group(GFX_GROUP).with_arg("bank");
/// One of the board's interrupt lines changed (arg: line << 1 | level; 0
/// FIFO, 1 general, 2 retrace).
pub static MGRAS_IRQ_LINE: EventDef = EventDef::new("irq line", cat::GFX).on_track("IMPACT irq").in_group(GFX_GROUP).with_arg("line<<1|level");
/// Octane: an interrupt packet to HEART (arg: vector << 1 | set), and the
/// kernel's acknowledgements (arg: offset << 32 | bits written).
pub static MGRAS_HEART_PACKET: EventDef = EventDef::new("heart packet", cat::GFX).on_track("IMPACT irq").in_group(GFX_GROUP).with_arg("vector<<1|set");
pub static MGRAS_HQ4_ACK: EventDef = EventDef::new("hq4 ack", cat::GFX).on_track("IMPACT irq").in_group(GFX_GROUP).with_arg("off<<32|val");

// ── VPro (Odyssey) ───────────────────────────────────────────────────────────

/// The command FIFO level STATUS0 is built from: stores pushed and not yet
/// run. Sampled where the board looks (slice start and end, STATUS0 reads,
/// the low-water edge) rather than at every store, which a textport sends
/// by the hundred thousand.
pub static ODSY_CFIFO_LEVEL: EventDef = EventDef::new("VPro cfifo level", cat::GFX).in_group(GFX_GROUP);
/// The FIFO fell below the low-water mark as a store retired (arg: level).
pub static ODSY_CFIFO_LOW_WATER: EventDef = EventDef::new("cfifo low water", cat::GFX).on_track("VPro irq").in_group(GFX_GROUP).with_arg("level");
/// One slice of the board thread holding the board lock (arg: stores
/// waiting when it began), and an access draining the queue on its own
/// thread (arg: stores waiting).
pub static ODSY_SLICE: EventDef = EventDef::new("slice", cat::GFX).with_arg("waiting");
pub static ODSY_DRAIN: EventDef = EventDef::new("drain", cat::GFX).with_arg("waiting");
/// A page-table DMA descriptor run (arg: table address).
pub static ODSY_DMA: EventDef = EventDef::new("dma descriptor", cat::GFX | cat::DMA).with_arg("table");
/// The DMA-done interrupt raised after a transfer that asked for it.
pub static ODSY_DMA_DONE: EventDef = EventDef::new("dma done", cat::GFX | cat::DMA).on_track("VPro irq").in_group(GFX_GROUP);
/// An interrupt packet sent to HEART (arg: source shift << 9 | vector << 1
/// | set). Sources: flag level, softsync, retrace and DMA-done.
pub static ODSY_IRQ: EventDef = EventDef::new("interrupt packet", cat::GFX | cat::IRQ).on_track("VPro irq").in_group(GFX_GROUP).with_arg("shift<<9|vector<<1|set");
/// The driver armed window flips for the next retrace (arg: toggle mask
/// written).
pub static ODSY_SWAP_ARM: EventDef = EventDef::new("swap arm", cat::GFX).on_track("VPro display").in_group(GFX_GROUP).with_arg("mask");
/// A retrace: pending flips latched (arg: displayed-buffer bits after).
pub static ODSY_RETRACE: EventDef = EventDef::new("retrace", cat::GFX).on_track("VPro display").in_group(GFX_GROUP).with_arg("displayed");
/// The refresh thread resolving dirty lines, and handing the frame on.
pub static ODSY_CAPTURE: EventDef = EventDef::new("frame capture", cat::GFX).on_track("VPro capture").in_group(GFX_GROUP);
pub static ODSY_PRESENT: EventDef = EventDef::new("present", cat::GFX).on_track("VPro present").in_group(GFX_GROUP);

// ── jitv2 ────────────────────────────────────────────────────────────────────

/// A dispatch queued a page compile (arg: guest PC).
pub static JIT_REQUEST: EventDef = EventDef::new("compile request", cat::JIT).with_arg("pc");
/// The compile queue is full: from the first dropped request (arg: its
/// guest PC) to the next one that fits. While it lasts every dispatch of an
/// uncompiled page retries the push, so this is one slice, not millions of
/// instants.
pub static JIT_QUEUE_FULL: EventDef = EventDef::new("compile queue full", cat::JIT).on_track("jitv2 compile queue").with_arg("pc");
pub static JIT_QUEUE_FULL_LEVEL: crate::PfLevel = crate::PfLevel::new();
/// An already-queued page is hot: request jumped the queue (arg: guest PC).
pub static JIT_HOT_REQUEST: EventDef = EventDef::new("hot request", cat::JIT).with_arg("pc");
/// A page compile: on a jitv2-compile thread, or on the CPU thread in
/// inline-compile mode.
pub static JIT_COMPILE: EventDef = EventDef::new("compile", cat::JIT);
