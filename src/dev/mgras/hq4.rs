//! IMPACT (MGRAS) graphics on the Octane: the board's HQ4 host interface on
//! an XIO widget, in front of the raster pipeline the parent module models
//! for the Indigo2.
//!
//! The Octane boards carry the Indigo2 boards' chips behind the host
//! interface: the same GE11, RE4, PP1s, display control bus devices,
//! command grammar and DMA register file. What differs is the front: an
//! HQ4 on an XIO widget instead of an HQ3 on GIO. Its registers sit at
//! other offsets in a 2 MB window, the kernel switches graphics contexts
//! through a handshake of its own, DMA page tables are loaded through a
//! separate mapping port, and interrupts are vectors sent to HEART. This
//! file is that front; everything behind it is `Mgras`, reached at the
//! Indigo2 offsets.
//!
//! | HQ4 offset          | what                              | goes to                      |
//! |---------------------|-----------------------------------|------------------------------|
//! | `0x000000-0x00003F` | widget identity, control, error interrupt destination | here    |
//! | `0x011000-0x01109F` | context word, interrupt pending, ack, enables, vectors, NIC | here |
//! | `0x020000-0x020200` | status, HQ flags, bus and DMA status | HQ3 `0x70000-0x70104` |
//! | `0x020008`          | FIFO status                       | here (reads empty)           |
//! | `0x020400`          | normal command FIFO               | HQ3 user command FIFO        |
//! | `0x028000/4`        | GE readback                       | HQ3 `0x70010/4`              |
//! | `0x02C000-0x02FFFF` | raster registers, execute alias   | HQ3 `0x7C000-0x7FFFF`        |
//! | `0x040000-0x040030` | FIFO thresholds                   | here; the marks to HQ3 `0x50020/4` |
//! | `0x050000-0x054FFF` | HQ microcode RAM                  | HQ3 `0x40000`                |
//! | `0x058040-0x05804F` | GE11 program download             | here                         |
//! | `0x05822C-0x058233` | GE11 program readback             | here                         |
//! | `0x070000-0x077FFF` | display control bus               | HQ3 `0x60000` (I2C here)     |
//! | `0x120400`          | context image, then command words | HQ3 user command FIFO        |
//! | `0x130400`          | DMA mapping port (class `0x48`)   | HQ3 privileged command FIFO  |
//! | `0x130500`          | context-switch selection          | HQ3 context switch           |
//!
//! The register map, the handshake, the interrupt sources and the page-list
//! layout are what IRIX 6.5.30 drives on the Octane. Where the board's own
//! behaviour behind them is unknown, the comments below say which choice
//! is an approximation.
//!
//! Approximation: no widget errors. The widget-error interrupt destination
//! is stored but never used. Our QEMU model raises a widget error, and moves
//! the handshake to an error state, for a context-switch selection out of
//! sequence, a context word written outside an image phase, a normal-FIFO
//! write that is not 8 bytes wide, an upload past the end of a slot and an
//! unsupported width. This port does none of that: it logs a note, drops the
//! access and returns a normal bus response. It also accepts 32-bit words at
//! the command FIFO port. A guest that provokes those paths sees no error
//! interrupt where the model would send one.
//!
//! Approximation: interrupt packets go to HEART whatever the programmed
//! destination address is; only the destination widget is checked. The
//! Bridge, by contrast, requires its own address.
//!
//! Ported from hw/misc/sgi-mgras.c and hw/misc/sgi-mgras-display.c in jamesbraid/qemu-sgi-origami.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use super::{Gl, Hq3Board, Mgras, MGRAS_SLOT_GFX_BASE};
use crate::config::GraphicsBoard;
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::ip30::widget_small_base;
use crate::irq::{InterruptController, IrqLine};
use crate::traits::{
    BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Device, Resettable, BUS_ERR, BUS_OK,
};

/// The widget number of HEART, the only interrupt target on the Octane.
const WIDGET_HEART: u8 = 8;

/// The board's register window within its widget space; the rest of the
/// 16 MB small window is a hole.
pub const IMPACT_WINDOW: u32 = 0x20_0000;

/// What one XIO IMPACT variant answers to the reads IRIX tells the
/// variants apart by.
#[derive(Clone, Copy, Debug)]
pub struct ImpactBoard {
    pub name: &'static str,
    /// The top bytes of 32-bit reads of `0x072488` and `0x072408`.
    pub version1: u8,
    pub version0: u8,
    /// Widget identification word bits 31:28 at `0x000000`.
    pub hq_revision: u8,
    /// `0x028004` bits 7:0 before any state query.
    pub ge_revision: u8,
    /// `0x02C578` bits 7:4.
    pub re4_revision: u8,
    pub cmap_revision: u8,
    pub ra_revision: u8,
    /// The serial-number chip's board record: name, part and revision, as
    /// published Octane inventories name the board.
    pub record_name: &'static str,
    pub record_part: &'static str,
    pub record_revision: &'static str,
}

/// How IRIX reads the two version bytes on the Octane's boards (the IP30
/// kernel's board probe, and the PROM's names for what it finds):
///
/// - `version1`: GE11 count in bits 1:0, board revision in 4:2, product in
///   6:5. The Indigo2's products are 0 (High and Maximum) and 1 (Solid,
///   MG10); the Octane's are its own pair, GM10 and GM20. The SI's 0x45
///   makes GM10 product 2; GM20 as 3 is an inference (the kernel treats
///   the two alike).
/// - `version0`: TRAMs on raster board A in bits 1:0 and on B in 3:2, each
///   3 for none fitted and anything else for four. On these products the
///   kernel takes as many REs as GEs.
///
/// The PROM names 1 GE without TRAMs "SI", with them "SI with texture
/// option", 2 GEs without "SSI", with them "MXI", each prefixed "E" when
/// the GE11 revision is above 2. The serial-number record names the base
/// board: GM10 for one RE (SI, plus a texture mezzanine for SI with
/// texture), GM20 for two (SSI, plus two mezzanines for MXI).
///
/// Solid IMPACT. Its values are inferred from an SI with texture, with the
/// TRAM taken away: no textureless board's values are on record.
pub const SOLID_IMPACT: ImpactBoard = ImpactBoard {
    name: "SI",
    version1: 0x45,
    version0: 0x0F,
    hq_revision: 2,
    ge_revision: 2,
    re4_revision: 2,
    cmap_revision: 3,
    ra_revision: 0,
    record_name: "GM10",
    record_part: "030-0938-003",
    record_revision: "U",
};

/// SI with texture option: the GM10 base board with a texture mezzanine,
/// four TRAMs on raster board A (`version0` 0x0C). The raster model is the
/// Indigo2's High IMPACT (one GE, one RE, a TE with TRAM).
pub const SI_TEXTURE: ImpactBoard = ImpactBoard {
    name: "SI with texture option",
    version0: 0x0C,
    ..SOLID_IMPACT
};

/// MXI: the GM20 base board (two GEs, two REs) with two texture
/// mezzanines, four TRAMs behind each RE (`version0` 0x00). The raster
/// model is the Indigo2's Maximum IMPACT. The record's part number is not
/// on record: this one is made up and marked so (all zeros).
pub const MXI: ImpactBoard = ImpactBoard {
    name: "MXI",
    version1: 0x3 << 5 | 1 << 2 | 2,
    version0: 0x00,
    record_name: "GM20",
    record_part: "030-0000-000",
    ..SOLID_IMPACT
};

impl ImpactBoard {
    /// The Octane board for a configured board kind, if there is one.
    pub fn for_kind(kind: GraphicsBoard) -> Option<ImpactBoard> {
        match kind {
            GraphicsBoard::SolidImpact => Some(SOLID_IMPACT),
            GraphicsBoard::HighImpact => Some(SI_TEXTURE),
            GraphicsBoard::MaxImpact => Some(MXI),
            _ => None,
        }
    }
}

/// The serial-number chip's ROM serial and the record's barcode: made up
/// and marked so (letter G, instance 0), so nothing mistakes them for a
/// real board's.
const RECORD_SERIAL: u64 = 0x5145_4D55_4700;
const RECORD_BARCODE: &str = "EMUG00";

/// The MicroLAN master at `0x011098`: low-drive time in bits 19:10, sample
/// offset in 9:2, done in bit 1 and the sampled wire level in bit 0.
const NIC_DONE: u32 = 1 << 1;
const NIC_WRITABLE: u32 = 0x000F_FFFC;

/// The board's MicroLAN with its serial-number chip, a DS2505 holding the
/// board record. The stock IP30 PROM names the board from it ("Name:GM"
/// is first-generation, "Name:MOT" E-series); with no readable record it
/// prints an SI as ESI.
fn nic_bus(board: &ImpactBoard) -> w1::Bus {
    let mut bus = w1::Bus::new();
    let record = BoardRecord {
        barcode: RECORD_BARCODE,
        part: board.record_part,
        revision: board.record_revision,
        name: board.record_name,
    };
    bus.add(Box::new(board_record_part(&record, RECORD_SERIAL)));
    bus
}

/// The widget identification word at `0x000004`: revision 0, part `0xC003`,
/// manufacturer `0x2AA`.
const WIDGET_PART: u32 = 0xC003;
const WIDGET_MFG: u32 = 0x2AA;
const WIDGET_REV: u32 = 0;

/// The attached monitor's ID, in `0x071308` bits 7:4. Every captured board
/// read 0xF with its monitor, which selects IRIX's stock 1280x1024 60 Hz
/// timing.
const MONITOR_ID: u8 = 0xF;

/// Interrupt pending bits (`0x011020`) the model raises.
mod pending {
    pub const RETRACE: u32 = 1 << 0;
    pub const LOWATER: u32 = 1 << 2;
    /// The swap command has been reached.
    pub const SWAP: u32 = 1 << 10;
    /// The context-switch selection has been taken.
    pub const FLAGS: u32 = 1 << 19;
    /// The 63rd context image word has arrived.
    pub const CONTEXT: u32 = 1 << 28;
    /// What the general interrupt carries.
    pub const GENERAL: u32 = FLAGS | CONTEXT | SWAP;
}

/// `0x011050` bit 4: the FIFO low-water interrupt's second-level enable.
const HQ_ENABLE_LOWATER: u32 = 0x10;

/// The kernel's only context-switch selection value.
const SELECT_ENTRY: u32 = 0x4FC;
/// Words in a context image.
const CONTEXT_WORDS: u32 = 63;

/// Whether a 64-bit command-port word is a whole packet: its high word a
/// command header carrying exactly one payload word (byte count 4 in bits
/// 7:0, not pixel data), its low word that payload. Direct raster register
/// writes are of this form.
fn is_whole_packet(word: u64) -> bool {
    let header = (word >> 32) as u32;
    header & 0x8000_0000 == 0 && header & 0xFF == 4
}

/// HQ3 offsets in `Mgras`'s window.
mod hq3 {
    pub const UCODE: u32 = 0x40000;
    pub const CONTEXT_SWITCH: u32 = 0x50050;
    pub const CFIFO_PRIVILEGED: u32 = 0x50080;
    pub const INTERRUPT_ENABLE_SET: u32 = 0x50018;
    pub const INTERRUPT_ENABLE_CLEAR: u32 = 0x5001C;
    pub const CFIFO_HIGH_MARK: u32 = 0x50020;
    pub const CFIFO_LOW_MARK: u32 = 0x50024;
    pub const DCB: u32 = 0x60000;
    pub const STATUS: u32 = 0x70000;
    pub const SET_FLAGS: u32 = 0x70008;
    pub const CLEAR_FLAGS: u32 = 0x7000C;
    pub const GE_READBACK_HI: u32 = 0x70010;
    pub const GE_READBACK_LO: u32 = 0x70014;
    pub const CFIFO: u32 = 0x70080;
    pub const GIOSTATUS: u32 = 0x70100;
    pub const DMABUSY: u32 = 0x70104;
    pub const RASTER: u32 = 0x7C000;
    /// HQ3 flags the model uses for its own bookkeeping: the swap reached
    /// (10) and the two context-switch phases (19 and 6). The HQ4 reports
    /// the swap as a pending source instead and has its own switch
    /// handshake, so the guest sees these bits only as it wrote them.
    pub const FLAG_CP0: u32 = 1 << 10;
    pub const FLAGS_INTERNAL: u32 = 1 << 6 | FLAG_CP0 | 1 << 19;
}

/// HQ4 offsets.
mod hq4 {
    pub const WIDGET_ID_HI: u32 = 0x000000;
    pub const WIDGET_ID: u32 = 0x000004;
    pub const WIDGET_CONTROL: u32 = 0x000024;
    pub const BUS_STATE: u32 = 0x008000;
    pub const CONTEXT_WORD: u32 = 0x011000;
    pub const PENDING: u32 = 0x011020;
    pub const ACK: u32 = 0x011028;
    pub const ENABLE_SET: u32 = 0x011040;
    pub const ENABLE_CLEAR: u32 = 0x011048;
    pub const HQ_ENABLE: u32 = 0x011050;
    pub const HQ_ENABLE_CLEAR: u32 = 0x011058;
    pub const VECTOR_DEST_UPPER: u32 = 0x011060;
    pub const VECTOR_DEST_LOWER: u32 = 0x011068;
    pub const VECTORS: u32 = 0x011070;
    pub const NIC: u32 = 0x011098;
    pub const STATUS: u32 = 0x020000;
    pub const FIFOSTATUS: u32 = 0x020008;
    pub const FLAGS_SET: u32 = 0x020010;
    pub const FLAGS_CLEAR: u32 = 0x020018;
    pub const RETRACE_ACK: u32 = 0x020028;
    pub const RETRACE_STATUS_1: u32 = 0x020038;
    pub const RETRACE_STATUS_2: u32 = 0x020044;
    pub const GIOSTATUS: u32 = 0x020100;
    pub const DMABUSY: u32 = 0x020200;
    pub const FIFO: u32 = 0x020400;
    pub const GE_READBACK_HI: u32 = 0x028000;
    pub const GE_READBACK_LO: u32 = 0x028004;
    pub const RASTER: u32 = 0x02C000;
    pub const RASTER_END: u32 = 0x030000;
    pub const RE_STATUS: u32 = 0x02C578;
    pub const CFIFO_HIGH: u32 = 0x040000;
    pub const CFIFO_LOW: u32 = 0x040008;
    pub const UCODE: u32 = 0x050000;
    pub const UCODE_END: u32 = 0x055000;
    pub const DCB: u32 = 0x070000;
    pub const DCB_END: u32 = 0x078000;
    /// The display control bus's I2C device (number 15 here, 11 on the
    /// Indigo2's bus), whose registers hold what is written.
    pub const I2C: u32 = 0x073C00;
    pub const I2C_END: u32 = 0x074000;
    pub const COMMAND: u32 = 0x120400;
    pub const MAPPING: u32 = 0x130400;
    pub const SELECT: u32 = 0x130500;
}

/// The HQ4's 32-bit registers that hold what is written and nothing the
/// model knows more of: per-context words, FIFO thresholds, scheduling
/// words, bring-up words and the board-configuration table.
fn is_plain(off: u32) -> bool {
    off & 3 == 0
        && matches!(off,
            0x011008 | 0x011030 | 0x011038 | 0x011088
            | 0x040000..=0x040033 | 0x040068 | 0x040070 | 0x040078 | 0x040088
            | 0x040918 | 0x040920 | 0x040928
            | 0x060008 | 0x061000..=0x0610FF)
}

/// Where an access goes.
#[derive(Debug, PartialEq, Eq)]
enum Route {
    /// `Mgras`, at this HQ3 offset.
    Board(u32),
    /// The HQ4 state here.
    Local,
    /// Beyond the board's window: a bus error.
    Hole,
}

fn route(off: u32) -> Route {
    use Route::*;
    if off >= IMPACT_WINDOW {
        return Hole;
    }
    match off {
        hq4::STATUS => Board(hq3::STATUS),
        hq4::FLAGS_SET => Board(hq3::SET_FLAGS),
        hq4::FLAGS_CLEAR => Board(hq3::CLEAR_FLAGS),
        hq4::GIOSTATUS => Board(hq3::GIOSTATUS),
        hq4::DMABUSY => Board(hq3::DMABUSY),
        hq4::FIFO => Board(hq3::CFIFO),
        hq4::GE_READBACK_HI => Board(hq3::GE_READBACK_HI),
        hq4::GE_READBACK_LO => Board(hq3::GE_READBACK_LO),
        // The raster engine's status is answered here (its revision).
        hq4::RE_STATUS => Local,
        hq4::RASTER..hq4::RASTER_END => Board(hq3::RASTER + (off - hq4::RASTER)),
        hq4::UCODE..hq4::UCODE_END => Board(hq3::UCODE + (off - hq4::UCODE)),
        hq4::I2C..hq4::I2C_END => Local,
        hq4::DCB..hq4::DCB_END => Board(hq3::DCB + (off - hq4::DCB)),
        _ => Local,
    }
}

/// The HQ4 microcode's differences the engine has to know about.
struct Hq4Quirks;

impl Hq3Board for Hq4Quirks {
    /// The HQ4 microcode's GE state sits 0x8C words above the Indigo2
    /// microcode's from (at the latest) the polygon-stipple enables on:
    /// every address known in both layouts from there up (enables, stencil,
    /// depth, raster position, clear values, viewport, scissor, fog, blend,
    /// lighting, the matrix-stack indices and their bases) differs by exactly
    /// that, while the matrix-mode index, the matrices and the modelview
    /// stack index, all below 0x84, do not move. Where between 0x84 and
    /// 0x1FA the shift starts is not known; this shifts every Indigo2
    /// address from 0x84 up. An approximation, inferred from the two
    /// layouts' known addresses. Pixel pipeline words answer where they
    /// were loaded. The colour matrix (`0x1B9-0x1C8`) does not move
    /// either: the 6.5.22 Octane libGLcore reads it where the Indigo2's
    /// does (glstate `pixel/COLOR_MATRIX_SGI`), and no Indigo2 state sits
    /// 0x8C below it.
    fn ge_state(&self, gl: &mut Gl, addr: u32) -> Option<u32> {
        const SHIFT: u32 = 0x8C;
        const COLOR_MATRIX: std::ops::RangeInclusive<u32> = 0x1B9..=0x1C8;
        let hq3 = if addr >= 0x84 + SHIFT && !COLOR_MATRIX.contains(&addr) { addr - SHIFT } else { addr };
        // Pixel pipeline words are kept in the Indigo2 layout (see
        // `pixel_state_address`).
        let v = gl.state_word_in(hq3, self.pixel_state_address(addr))?;
        // The matrix-stack depths answer with an address in the GE's stack,
        // which moved with the rest.
        Some(if matches!(hq3, 0x83 | 0x896 | 0x897) { v + SHIFT } else { v })
    }

    /// That the register field is nine bits is an inference, not a
    /// published width: every raster register name known runs to 0x1FF, and
    /// the direct packets' decode used for the Octane's port takes the bit
    /// above as part of the packet class. The IP30 PROM sends one such word
    /// (0x001A6804) before its first context switch; what the board does
    /// with it is unknown, so it is dropped and reported.
    fn drop_raster_write(&self, cmd: u32) -> Option<String> {
        (cmd & 0x200 != 0).then(|| format!("raster command {cmd:#x} past register 0x1FF dropped"))
    }

    /// The pixel pipeline's ERAM blocks moved up by the same 0x8C: libGLcore
    /// on the 6.5.22 Octane writes the expand state, the transfer size and
    /// the texture-load blocks at 0xD91, 0xE34 and 0xE43 where the Indigo2's
    /// uses 0xD05, 0xDA8 and 0xDB7 (traced, pixtest). GE memory below the
    /// ERAM (the program at 0x13B, the module states at 0x187-0x1EF, the
    /// zoom at 0x191) did not move. Without this the HLE missed the host
    /// pixel shape, and glDrawPixels' rows reached the raster engine as
    /// RGBA when the host sent RGB.
    fn pixel_state_address(&self, addr: u32) -> u32 {
        const SHIFT: u32 = 0x8C;
        if addr >= super::pixpipe::ERAM_BASE + SHIFT { addr - SHIFT } else { addr }
    }
}

/// The HQ4 state this front keeps. Plain values under one lock, which is
/// never held while calling into `Mgras` (its interrupt lines come back
/// here) or into HEART. Packets are decided under it and sent after it is
/// dropped, under `Shared::order` (see there).
#[derive(Default)]
struct Hq4 {
    control: u32,
    /// The widget-error interrupt destination. Stored only: this port
    /// raises no widget errors (see the header).
    int_dest: [u32; 2],
    context_word: u32,
    context_word_written: bool,
    entry: u32,
    pending: u32,
    /// Bits written to `0x011020`, read back beside the pending sources.
    status_extra: u32,
    enable: u32,
    hq_enable: u32,
    vector_dest: [u32; 2],
    /// Low-water, general and retrace vectors in bytes 0, 1 and 2.
    vectors: u32,
    plain: HashMap<u32, u32>,
    i2c: HashMap<u32, u8>,
    mapping: u64,
    /// HQ3-internal flag bits as the guest set them (`FLAGS_INTERNAL`).
    flag_shadow: u32,
    ge: GeUpload,
    nic_control: u32,
    /// The context-switch handshake: the selection taken, the image being
    /// written (1) or done (3), the first acknowledgement, image words so
    /// far, and whether a switch has completed since reset.
    entry_selected: bool,
    protocol: u8,
    first_ack: bool,
    context_words: u32,
    ready: bool,
    lowater_asserted: bool,
    general_asserted: bool,
    no_dest_logged: bool,
}

impl Hq4 {
    fn vector(&self, shift: u32) -> u8 {
        (self.vectors >> shift) as u8
    }

    /// An HQ interrupt to the vector destination, if one is programmed and
    /// it names HEART, the only interrupt target on the Octane.
    fn send(&mut self, vector: u8, set: bool, out: &mut Vec<(u8, bool)>) {
        let widget = (self.vector_dest[0] >> 24 & 0xF) as u8;
        let why = if self.vector_dest == [0, 0] {
            "no vector destination is programmed"
        } else if widget != WIDGET_HEART {
            "its destination is not HEART"
        } else {
            crate::pf_instant!(crate::perfetto::events::MGRAS_HEART_PACKET, (vector as u64) << 1 | set as u64);
            out.push((vector, set));
            return;
        };
        if !self.no_dest_logged {
            self.no_dest_logged = true;
            eprintln!("impact: HQ interrupt not sent: {why}");
        }
    }

    /// Send a packet for each interrupt that changed, one a transition. The
    /// general interrupt stays raised while any enabled source is pending,
    /// since its handler serves one at a time.
    fn update(&mut self, out: &mut Vec<(u8, bool)>) {
        let lowater = self.pending & pending::LOWATER != 0;
        let general = self.pending & self.enable & pending::GENERAL != 0;
        if lowater != self.lowater_asserted {
            self.lowater_asserted = lowater;
            self.send(self.vector(0), lowater, out);
        }
        if general != self.general_asserted {
            self.general_asserted = general;
            self.send(self.vector(8), general, out);
        }
    }
}

/// GE11 program download ports (`0x058040-0x05804C`) and readback ports
/// (`0x05822C`, `0x058230`). The kernel loads
/// GE 0 through slot B's ports and GE 1 through slot A's. The words are
/// kept without interpretation, and the readback order below is an
/// approximation fitted to the Octane kernel's verifier. The HQ3 model's
/// GE11 diagnostic port reads a download back in the order the IP28 kernel
/// expects, so this path keeps its own copy.
mod ge {
    pub const DATA_B: u32 = 0x058040;
    pub const CTRL_B: u32 = 0x058044;
    pub const DATA_A: u32 = 0x058048;
    pub const CTRL_A: u32 = 0x05804C;
    pub const READ_1: u32 = 0x05822C;
    pub const READ_2: u32 = 0x058230;
    /// Words a slot holds: the largest program, 59392 lines of three.
    pub const WORDS: usize = 59392 * 3;
    /// A control value with this, less bit 0, starts a download; bit 0 set
    /// makes the readback start one word early.
    pub const LOAD: u32 = 0x20_0000;
    /// A control value with this rewinds the readback.
    pub const REWIND: u32 = 0x8000_0000;
}

#[derive(Default)]
struct GeUpload {
    words: [Vec<u32>; 2],
    write_ptr: [usize; 2],
    /// In pairs of words.
    read_ptr: [usize; 2],
    read_odd: [bool; 2],
    load_mode: [bool; 2],
    active: usize,
    /// Each slot's download fingerprint, kept as it is written.
    hash: [super::microcode::Running; 2],
}

impl GeUpload {
    /// A control write. When it ends a download, the fingerprint of the
    /// words loaded.
    fn control(&mut self, slot: usize, v: u32) -> Option<u64> {
        self.active = slot;
        let finished = self.load_mode[slot].then(|| self.hash[slot].get());
        if v & !1 == ge::LOAD {
            self.write_ptr[slot] = 0;
            self.read_ptr[slot] = 0;
            self.read_odd[slot] = v & 1 != 0;
            self.load_mode[slot] = true;
        } else {
            if v & ge::REWIND != 0 {
                self.read_ptr[slot] = 0;
            }
            self.load_mode[slot] = false;
        }
        finished
    }

    /// False when the slot is full.
    fn data(&mut self, slot: usize, v: u32) -> bool {
        if !self.load_mode[slot] {
            return true;
        }
        let p = self.write_ptr[slot];
        if p >= ge::WORDS {
            return false;
        }
        let w = &mut self.words[slot];
        if p < w.len() { w[p] = v } else { w.push(v) }
        self.hash[slot].word(p, v);
        self.write_ptr[slot] = p + 1;
        true
    }

    /// `READ_1` gives word 2p - odd of the active slot and moves p on;
    /// `READ_2` the word after it at the current p. Words outside the
    /// download read 0.
    fn read(&mut self, second: bool) -> u32 {
        let slot = self.active;
        let idx = 2 * self.read_ptr[slot] as i64 - self.read_odd[slot] as i64 + second as i64;
        if !second {
            self.read_ptr[slot] += 1;
        }
        usize::try_from(idx).ok().and_then(|i| self.words[slot].get(i).copied()).unwrap_or(0)
    }
}

/// What the front and the board's interrupt lines share.
struct Shared {
    hq4: Mutex<Hq4>,
    /// The command port is in its command phase: a context switch has
    /// completed and no selection has started another (`Hq4::ready` and
    /// `protocol != 1`, kept with them under `hq4`). Command words, nearly
    /// every write the port takes, check this instead of taking the lock.
    commands: AtomicBool,
    /// The board's MicroLAN, built on the first access.
    nic: Mutex<Option<w1::Bus>>,
    irq: Arc<dyn InterruptController>,
    /// Held from deciding a packet to HEART having it, so two threads
    /// (the CPU and an engine thread) cannot send one line's packets out of
    /// the order they were decided in. Taken before `hq4`, never while
    /// calling into `Mgras`; HEART must not call back into the board.
    order: Mutex<()>,
    noted: Mutex<HashSet<String>>,
}

impl Shared {
    /// Run `f` on the HQ4 state and send the packets it decides, outside
    /// the state lock and in decision order.
    fn with_state<R>(&self, f: impl FnOnce(&mut Hq4, &mut Vec<(u8, bool)>) -> R) -> R {
        let _order = self.order.lock();
        let mut out = Vec::new();
        let r = f(&mut self.hq4.lock(), &mut out);
        for (vector, set) in out {
            self.irq.set_line(vector as u32, set);
        }
        r
    }

    fn note(&self, what: String) {
        let mut n = self.noted.lock();
        if n.len() < 256 && n.insert(what.clone()) {
            eprintln!("impact: not modelled yet: {what}");
        }
    }
}

/// `Mgras`'s three GIO interrupt lines (FIFO, general, retrace), turned
/// into HQ4 sources.
struct Lines(Arc<Shared>);

impl InterruptController for Lines {
    fn set_line(&self, line: u32, level: bool) {
        self.0.with_state(|s, out| {
            match line {
                // The FIFO line is not used: the low-water source follows
                // its enable (see `HQ_ENABLE`).
                // The general line carries only the swap: `ImpactXio::new`
                // enables nothing else there. The HQ4 latches it until the
                // handler acknowledges it.
                1 if level => s.pending |= pending::SWAP,
                // A retrace is a message each frame the video timing chip
                // enables it, not a level: the kernel acknowledges nothing
                // on the board for it.
                2 if level => {
                    s.pending |= pending::RETRACE;
                    let v = s.vector(16);
                    s.send(v, true, out);
                }
                _ => {}
            }
            s.update(out);
        });
    }
}

/// IMPACT graphics in one of the Octane's graphics ports: a `BusDevice` for
/// the widget's small window.
pub struct ImpactXio {
    base: u64,
    board: ImpactBoard,
    shared: Arc<Shared>,
    mgras: Arc<Mgras>,
}

/// One phase of the command port (`0x120400`).
enum Phase {
    Command,
    Image,
    Ignore,
    Stray,
}

impl ImpactXio {
    /// A Solid IMPACT board (see `with_board`).
    pub fn new(
        widget: u8,
        irq: Arc<dyn InterruptController>,
        mem: Arc<dyn BusDevice>,
        heartbeat: Arc<AtomicU64>,
        fasttick: Arc<AtomicU64>,
    ) -> Arc<Self> {
        Self::with_board(widget, GraphicsBoard::SolidImpact, irq, mem, heartbeat, fasttick)
    }

    /// An IMPACT board of kind `kind` (Solid, High for SI with texture,
    /// Maximum for MXI) at `widget` (one of the four graphics ports):
    /// its interrupts are vectors on `irq` (HEART), its DMA reaches `mem`
    /// (the bus masters' view of memory). `heartbeat` and `fasttick` feed
    /// the status bar, as for the Indigo2 board. `Device::start` starts it.
    pub fn with_board(
        widget: u8,
        kind: GraphicsBoard,
        irq: Arc<dyn InterruptController>,
        mem: Arc<dyn BusDevice>,
        heartbeat: Arc<AtomicU64>,
        fasttick: Arc<AtomicU64>,
    ) -> Arc<Self> {
        let board = ImpactBoard::for_kind(kind).expect("an IMPACT board kind");
        let shared = Arc::new(Shared {
            hq4: Mutex::new(Hq4::default()),
            commands: AtomicBool::new(false),
            nic: Mutex::new(None),
            irq,
            order: Mutex::new(()),
            noted: Mutex::new(HashSet::new()),
        });
        let lines: Arc<dyn InterruptController> = Arc::new(Lines(shared.clone()));
        let mgras = Mgras::new(
            kind,
            [IrqLine::new(lines.clone(), 0), IrqLine::new(lines.clone(), 1), IrqLine::new(lines, 2)],
            heartbeat,
            fasttick,
        );
        // CMAP 0's revision register carries the monitor ID in bits 7:4;
        // CMAP 1's RA revision in 6:4 and, in bit 7, that there is no video
        // option.
        let cmap = (board.cmap_revision & 7) as u32;
        mgras.set_identity(
            [board.version0 as u32, board.version1 as u32],
            [(MONITOR_ID as u32) << 4 | cmap, 0x80 | ((board.ra_revision & 7) as u32) << 4 | cmap],
            board.ge_revision as u32,
        );
        mgras.set_phys(mem);
        // The Octane X server arms a pixel read with XFRCONTROL 8 and kicks
        // the DMA last, which starts the read.
        mgras.start_reads_at_kick();
        // The HQ4 microcode keeps its GE state at other addresses, and the
        // IP30 kernel's idle waits are shorter than the model's long lists.
        mgras.set_board(Box::new(Hq4Quirks));
        mgras.bound_busy_status();
        let me = Arc::new(Self { base: widget_small_base(widget), board, shared, mgras });
        // The swap reaching the HQ is the one HQ3 flag the general line
        // carries.
        me.board_write(32, hq3::INTERRUPT_ENABLE_SET, hq3::FLAG_CP0 as u64);
        me
    }

    /// The raster pipeline and its display: what the machine hands the
    /// window, the monitor (`Device`), `set_cpu_cycles` and `register_locks`.
    pub fn mgras(&self) -> &Arc<Mgras> {
        &self.mgras
    }

    pub fn display(&self) -> Arc<dyn crate::gfx_display::GfxDisplay> {
        self.mgras.clone()
    }

    /// Put the board's locks in the `locks` monitor listing.
    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_mutex_fn;
        self.mgras.register_locks();
        register_mutex_fn("hq4::state", self, |m| m.shared.hq4.is_locked());
        register_mutex_fn("hq4::order", self, |m| m.shared.order.is_locked());
        register_mutex_fn("hq4::nic", self, |m| m.shared.nic.is_locked());
        register_mutex_fn("hq4::noted", self, |m| m.shared.noted.is_locked());
    }

    /// Cold reset of the HQ4's own state: control, thresholds, interrupt
    /// routing and the context-switch handshake. The raster pipeline keeps
    /// its state, as `Mgras` has no reset of its own.
    pub fn reset(&self) {
        // The HQ3 model's side of the interrupt lines first: its flags and
        // enables back to what `new` set, and the FIFO marks cleared. The
        // lines' callbacks here only lower levels, which change nothing.
        self.board_write(32, hq3::CLEAR_FLAGS, hq3::FLAGS_INTERNAL as u64);
        self.board_write(32, hq3::INTERRUPT_ENABLE_CLEAR, u32::MAX as u64);
        self.board_write(32, hq3::INTERRUPT_ENABLE_SET, hq3::FLAG_CP0 as u64);
        self.board_write(32, hq3::CFIFO_HIGH_MARK, 0);
        self.board_write(32, hq3::CFIFO_LOW_MARK, 0);
        let mut s = self.shared.hq4.lock();
        *s = Hq4::default();
        self.shared.commands.store(false, Ordering::Release);
        drop(s);
        *self.shared.nic.lock() = None;
    }

    fn board_read(&self, bits: u32, off: u32) -> (u32, u64) {
        let a = (MGRAS_SLOT_GFX_BASE + off) as u64;
        let m = &*self.mgras;
        match bits {
            8 => { let r = m.read8(a); (r.status, r.data as u64) }
            16 => { let r = m.read16(a); (r.status, r.data as u64) }
            32 => { let r = m.read32(a); (r.status, r.data as u64) }
            _ => { let r = m.read64(a); (r.status, r.data) }
        }
    }

    fn board_write(&self, bits: u32, off: u32, val: u64) -> u32 {
        let a = (MGRAS_SLOT_GFX_BASE + off) as u64;
        let m = &*self.mgras;
        match bits {
            8 => m.write8(a, val as u8),
            16 => m.write16(a, val as u16),
            32 => m.write32(a, val as u32),
            _ => m.write64(a, val),
        }
    }

    /// The offset in the widget's space of `addr`, clamped past the
    /// board's window to a hole.
    fn offset(&self, addr: u64) -> u32 {
        addr.wrapping_sub(self.base).min(IMPACT_WINDOW as u64) as u32
    }

    fn read(&self, addr: u64, bits: u32) -> (u32, u64) {
        let off = self.offset(addr);
        match route(off) {
            Route::Hole => (BUS_ERR, 0),
            Route::Board(o) => {
                let (st, v) = self.board_read(bits, o);
                if st != BUS_OK {
                    return (st, 0);
                }
                if matches!(off, hq4::FLAGS_SET | hq4::FLAGS_CLEAR) {
                    let shadow = self.shared.hq4.lock().flag_shadow;
                    let internal = hq3::FLAGS_INTERNAL as u64;
                    return (BUS_OK, v & !internal | shadow as u64 & internal);
                }
                (BUS_OK, v)
            }
            Route::Local => self.read_local(off, bits),
        }
    }

    fn read_local(&self, off: u32, bits: u32) -> (u32, u64) {
        let b = &self.board;
        let s = self.shared.hq4.lock();
        let v: u32 = match off {
            hq4::WIDGET_ID_HI => {
                // The kernel takes the HQ revision from bits 31:28; a 64-bit
                // read has it in the high word. Why it differs from
                // `0x000004`'s revision is unknown.
                let v = (b.hq_revision as u32 & 0xF) << 28 | WIDGET_PART << 12 | WIDGET_MFG << 1 | 1;
                return (BUS_OK, if bits == 64 { (v as u64) << 32 } else { v as u64 });
            }
            hq4::WIDGET_ID => WIDGET_REV << 28 | WIDGET_PART << 12 | WIDGET_MFG << 1 | 1,
            // The reserved halves of the widget register slots.
            0x000020 => 0,
            hq4::WIDGET_CONTROL => s.control,
            0x000030 | 0x000034 => s.int_dest[0],
            0x000038 | 0x00003C => s.int_dest[1],
            hq4::BUS_STATE => 0,
            hq4::CONTEXT_WORD => s.context_word,
            hq4::PENDING => s.pending | s.status_extra,
            hq4::ENABLE_SET => s.enable,
            hq4::HQ_ENABLE => s.hq_enable,
            hq4::VECTOR_DEST_UPPER => s.vector_dest[0],
            hq4::VECTOR_DEST_LOWER => s.vector_dest[1],
            hq4::VECTORS => s.vectors,
            hq4::NIC => s.nic_control,
            hq4::RETRACE_STATUS_1 | hq4::RETRACE_STATUS_2 => 0,
            // FIFO occupancy reads zero: the port takes every word as it
            // arrives, into the HQ3 model's deep queue. An approximation;
            // the board's real occupancy while it works is unknown. The
            // drivers' waits for the pipe still wait, on the idle bits of
            // `0x020000`.
            hq4::FIFOSTATUS => 0,
            ge::READ_1 | ge::READ_2 => {
                drop(s);
                return (BUS_OK, self.shared.hq4.lock().ge.read(off == ge::READ_2) as u64);
            }
            hq4::RE_STATUS => {
                drop(s);
                let (st, v) = self.board_read(32, hq3::RASTER + (hq4::RE_STATUS - hq4::RASTER));
                if st != BUS_OK {
                    return (st, 0);
                }
                return (BUS_OK, v & !0xF0 | (b.re4_revision as u64 & 0xF) << 4);
            }
            // No kernel routine reads this; the value is an approximation.
            hq4::COMMAND => if s.ready { 0xF400 } else { 0 },
            hq4::MAPPING => return (BUS_OK, if bits == 64 { s.mapping } else { s.mapping as u32 as u64 }),
            hq4::SELECT => s.entry,
            hq4::I2C..hq4::I2C_END => {
                let v = s.i2c.get(&off).copied().unwrap_or(0) as u64;
                return (BUS_OK, if bits == 8 { v } else { v << 24 });
            }
            o if is_plain(o) => s.plain.get(&o).copied().unwrap_or(0),
            _ => {
                drop(s);
                // The CPU's PC at its last batch boundary: the caller, to within a batch.
                let pc = crate::ip30::LAST_PC[crate::cpu::smp::current_cpu()].load(Ordering::Relaxed);
                self.shared.note(format!("read of {off:#08x} at pc {pc:#x}"));
                0
            }
        };
        (BUS_OK, v as u64)
    }

    fn write(&self, addr: u64, bits: u32, val: u64) -> u32 {
        let off = self.offset(addr);
        match route(off) {
            Route::Hole => BUS_ERR,
            Route::Board(o) => match off {
                hq4::FLAGS_SET | hq4::FLAGS_CLEAR => {
                    let v = val as u32;
                    let st = self.board_write(32, o, (v & !hq3::FLAGS_INTERNAL) as u64);
                    if st == BUS_OK {
                        let mut s = self.shared.hq4.lock();
                        if off == hq4::FLAGS_SET {
                            s.flag_shadow |= v & hq3::FLAGS_INTERNAL;
                        } else {
                            s.flag_shadow &= !(v & hq3::FLAGS_INTERNAL);
                        }
                    }
                    st
                }
                // A write to the status register is accepted and ignored.
                hq4::STATUS => BUS_OK,
                _ => self.board_write(bits, o, val),
            },
            Route::Local => self.write_local(off, bits, val),
        }
    }

    fn write_local(&self, off: u32, bits: u32, val: u64) -> u32 {
        let v = val as u32;
        match off {
            hq4::I2C..hq4::I2C_END => {
                if bits == 8 {
                    self.shared.hq4.lock().i2c.insert(off, v as u8);
                } else {
                    self.shared.note(format!("{bits}-bit write to the I2C port {off:#08x}"));
                }
                return BUS_OK;
            }
            hq4::COMMAND => return self.command_write(bits, val),
            // The mapping port shares the command FIFO's framing. The HQ3
            // model takes its class 0x48 headers as the DMA register writes
            // they are (bit 22 is outside the command number it decodes),
            // and anything else as the command it is.
            hq4::MAPPING => {
                let st = self.board_write(bits, hq3::CFIFO_PRIVILEGED, val);
                if st == BUS_OK {
                    self.shared.hq4.lock().mapping = val;
                }
                return st;
            }
            hq4::SELECT => return self.select(bits, v),
            ge::DATA_A | ge::DATA_B | ge::CTRL_A | ge::CTRL_B if bits >= 32 => {
                let slot = (off == ge::DATA_B || off == ge::CTRL_B) as usize;
                let mut s = self.shared.hq4.lock();
                if off == ge::CTRL_A || off == ge::CTRL_B {
                    if let Some(fp) = s.ge.control(slot, v) {
                        drop(s);
                        // GE 0 loads through slot B, GE 1 through slot A.
                        self.mgras.set_ge_upload_fingerprint(1 - slot, fp);
                    }
                } else if !s.ge.data(slot, v) {
                    drop(s);
                    self.shared.note("GE program download past the end of its slot".into());
                }
                return BUS_OK;
            }
            _ => {}
        }
        // The rest are 32-bit registers; other widths are ignored.
        if bits != 32 {
            self.shared.note(format!("{bits}-bit write to {off:#08x}"));
            return BUS_OK;
        }
        // HQ3 writes, made once the lock is dropped.
        let mut board: Vec<(u32, u32)> = Vec::new();
        let known = self.shared.with_state(|s, out| {
            match off {
                hq4::WIDGET_ID_HI | 0x000020 | hq4::WIDGET_ID => {}
                hq4::WIDGET_CONTROL => s.control = v,
                // Both halves of each destination slot hold the same word.
                0x000030 | 0x000034 => s.int_dest[0] = v,
                0x000038 | 0x00003C => s.int_dest[1] = v,
                hq4::CONTEXT_WORD => {
                    s.context_word = v;
                    s.context_word_written = true;
                }
                hq4::PENDING => s.status_extra = v,
                hq4::ACK => {
                    crate::pf_instant!(crate::perfetto::events::MGRAS_HQ4_ACK, (off as u64) << 32 | v as u64);
                    s.pending &= !v;
                    if v & pending::FLAGS != 0 {
                        s.first_ack = true;
                    }
                    if v & pending::SWAP != 0 {
                        // The swap the HQ3 model reported is served.
                        board.push((hq3::CLEAR_FLAGS, hq3::FLAG_CP0));
                    }
                    s.update(out);
                }
                // A set port and a clear port for one enable word, inferred
                // from the kernel's use of them.
                hq4::ENABLE_SET => {
                    s.enable |= v;
                    s.update(out);
                }
                hq4::ENABLE_CLEAR => {
                    s.enable &= !v;
                    s.update(out);
                }
                // The low-water source rises as the enable does: the FIFO
                // reads empty (`FIFOSTATUS`), so it is always below any mark.
                // An approximation, as is taking `0x011058`, inferred as the
                // second-level clear, to clear both the source and the
                // enable bits written.
                hq4::HQ_ENABLE => {
                    s.plain.insert(off, v);
                    if v & !s.hq_enable & HQ_ENABLE_LOWATER != 0 {
                        s.pending |= pending::LOWATER;
                    }
                    s.hq_enable = v;
                    s.update(out);
                }
                hq4::HQ_ENABLE_CLEAR => {
                    s.plain.insert(off, v);
                    if v & s.hq_enable & HQ_ENABLE_LOWATER != 0 {
                        s.pending &= !pending::LOWATER;
                    }
                    s.hq_enable &= !v;
                    s.update(out);
                }
                hq4::VECTOR_DEST_UPPER => s.vector_dest[0] = v,
                hq4::VECTOR_DEST_LOWER => s.vector_dest[1] = v,
                hq4::VECTORS => s.vectors = v,
                // One bit slot, complete inside the write: real timing is
                // unknown, and the PROM and IDE poll the done bit first. A
                // long low is a reset, which presence answers by pulling the
                // wire low; a short one writes a 1 or reads (the thresholds
                // are the IOC3 model's, not hardware values).
                hq4::NIC => {
                    let (pulse, sample) = w1::slot::fields(v as u64);
                    let mut nic = self.shared.nic.lock();
                    let bus = nic.get_or_insert_with(|| nic_bus(&self.board));
                    let level = w1::slot::slot(bus, pulse, sample);
                    s.nic_control = v & NIC_WRITABLE | NIC_DONE | level as u32;
                }
                hq4::RETRACE_ACK => {
                    crate::pf_instant!(crate::perfetto::events::MGRAS_HQ4_ACK, (off as u64) << 32 | v as u64);
                }
                hq4::RE_STATUS => board.push((hq3::RASTER + (hq4::RE_STATUS - hq4::RASTER), v)),
                o if is_plain(o) => {
                    s.plain.insert(o, v);
                    match o {
                        hq4::CFIFO_HIGH => board.push((hq3::CFIFO_HIGH_MARK, v)),
                        hq4::CFIFO_LOW => board.push((hq3::CFIFO_LOW_MARK, v)),
                        _ => {}
                    }
                }
                _ => return false,
            }
            true
        });
        if !known {
            let pc = crate::ip30::LAST_PC[crate::cpu::smp::current_cpu()].load(Ordering::Relaxed);
            self.shared.note(format!("write of {v:#x} to {off:#08x} at pc {pc:#x}"));
            return BUS_OK;
        }
        for (o, v) in board {
            // The RE_STATUS forward goes through the command FIFO, so a full
            // FIFO holds the store back: the CPU retries it. It is the only
            // forward in this list that does and it is the only entry of its
            // write, so the retry repeats nothing.
            let st = self.board_write(32, o, v as u64);
            if st != BUS_OK {
                return st;
            }
        }
        BUS_OK
    }

    /// `0x130500`: the kernel selects the context-switch entry after writing
    /// the context word. The HQ3 model saves the outgoing context and takes
    /// the next 63 command words as the incoming image; the HQ4 raises
    /// source 19, which the kernel acknowledges before it sends them.
    fn select(&self, bits: u32, v: u32) -> u32 {
        let in_sequence = bits == 32 && v == SELECT_ENTRY && self.shared.hq4.lock().context_word_written;
        if !in_sequence {
            self.shared.note(format!("context-switch selection {v:#x} ({bits}-bit) out of sequence"));
            return BUS_OK;
        }
        let st = self.board_write(32, hq3::CONTEXT_SWITCH, 0);
        if st != BUS_OK {
            return st;
        }
        self.shared.with_state(|s, out| {
            s.entry = v;
            s.entry_selected = true;
            s.protocol = 1;
            self.shared.commands.store(false, Ordering::Release);
            s.first_ack = false;
            s.context_words = 0;
            s.pending |= pending::FLAGS;
            s.update(out);
        });
        BUS_OK
    }

    /// `0x120400`: during a switch, after the first acknowledgement, the 63
    /// context image words; once a switch has completed, command words of
    /// 32 or 64 bits.
    fn command_write(&self, bits: u32, val: u64) -> u32 {
        if self.shared.commands.load(Ordering::Acquire) {
            return self.board_write(bits, hq3::CFIFO, val);
        }
        let phase = {
            let s = self.shared.hq4.lock();
            if s.ready && s.protocol != 1 {
                Phase::Command
            } else if bits == 32 && s.entry_selected && s.protocol == 1 && s.first_ack && s.context_words < CONTEXT_WORDS {
                Phase::Image
            } else if bits == 64 && !s.entry_selected && is_whole_packet(val) {
                // The IP30 PROM's graphical console draws through this port
                // and never selects a context: direct raster writes and a
                // few other one-word packets. Before the first selection
                // only whole packets are taken as commands: the IP30
                // diagnostics write a bare bring-up word here first, whose
                // high word is zero, and parsed as a header its low word
                // would swallow the raster writes after it. That the board
                // takes them without a switch is inferred from the PROM's
                // command stream and OpenBSD's impact driver, which draws
                // with the same packets and no switch.
                Phase::Command
            } else if bits == 64 && !s.entry_selected {
                // Other traffic before any selection has no modelled effect.
                Phase::Ignore
            } else {
                Phase::Stray
            }
        };
        match phase {
            Phase::Command => self.board_write(bits, hq3::CFIFO, val),
            Phase::Image => {
                let st = self.board_write(32, hq3::CFIFO, val);
                if st != BUS_OK {
                    return st;
                }
                self.shared.with_state(|s, out| {
                    s.context_words += 1;
                    if s.context_words == CONTEXT_WORDS {
                        // The switch completes with the 63rd word: no
                        // closing word follows.
                        s.ready = true;
                        s.protocol = 3;
                        self.shared.commands.store(true, Ordering::Release);
                        s.pending |= pending::CONTEXT;
                        s.update(out);
                    }
                });
                BUS_OK
            }
            Phase::Ignore => BUS_OK,
            Phase::Stray => {
                self.shared.note(format!("{bits}-bit command word {val:#x} outside a context"));
                BUS_OK
            }
        }
    }
}

impl BusDevice for ImpactXio {
    fn read8(&self, addr: u64) -> BusRead8 {
        let (status, v) = self.read(addr, 8);
        BusRead8 { status, data: v as u8 }
    }
    fn read16(&self, addr: u64) -> BusRead16 {
        let (status, v) = self.read(addr, 16);
        BusRead16 { status, data: v as u16 }
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        let (status, v) = self.read(addr, 32);
        BusRead32 { status, data: v as u32 }
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        let (status, data) = self.read(addr, 64);
        BusRead64 { status, data }
    }
    fn write8(&self, addr: u64, val: u8) -> u32 { self.write(addr, 8, val as u64) }
    fn write16(&self, addr: u64, val: u16) -> u32 { self.write(addr, 16, val as u64) }
    fn write32(&self, addr: u64, val: u32) -> u32 { self.write(addr, 32, val as u64) }
    fn write64(&self, addr: u64, val: u64) -> u32 { self.write(addr, 64, val) }
}

impl Device for ImpactXio {
    fn step(&self, _cycles: u64) {}

    /// Starts the engine threads and the display refresh.
    fn start(&self) {
        self.mgras.start_display();
    }

    /// Stops and joins them; no interrupt or DMA follows.
    fn stop(&self) {
        self.mgras.stop_display();
    }

    fn is_running(&self) -> bool {
        self.mgras.is_running()
    }

    fn get_clock(&self) -> u64 {
        0
    }
}

impl Resettable for ImpactXio {
    fn power_on(&self) {
        self.reset();
    }
}

/// Register-contract tests for the Octane board, as far as one board on
/// its own reaches: identity, widths and holes, reset, DMA through HEART
/// memory and interrupt transport. The rest go through the raster model
/// with the Indigo2 tests' register setups.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::BUS_BUSY;

    fn gl() -> Box<Gl> {
        // SAFETY: Gl is plain data, valid zeroed.
        unsafe { Box::<Gl>::new_zeroed().assume_init() }
    }

    /// The HQ4 layout answers the Indigo2 layout's words 0x8C higher, the
    /// matrices and the modelview stack index where they are, and the
    /// stacks' depths as addresses in the moved stacks.
    #[test]
    fn hq4_readbacks_follow_the_shifted_layout() {
        let mut g = gl();
        for (hq3, hq4) in [(0xB91, 0xC1D), (0xB94, 0xC20), (0xB98, 0xC24), (0x1FC, 0x288), (0x2D0, 0x35C), (0xBBE, 0xC4A)] {
            assert_eq!(Hq4Quirks.ge_state(&mut g, hq4), g.state_word(hq3), "{hq4:#x}");
        }
        for addr in [0x29, 0x3A, 0x4A, 0x6A] {
            assert_eq!(Hq4Quirks.ge_state(&mut g, addr), g.state_word(addr), "{addr:#x}");
        }
        assert_eq!(Hq4Quirks.ge_state(&mut g, 0x83), Some(0x925), "modelview stack, depth 1");
        assert_eq!(Hq4Quirks.ge_state(&mut g, 0x922), Some(0xB25), "projection stack, depth 1");
        assert_eq!(Hq4Quirks.ge_state(&mut g, 0x923), Some(0xB45), "texture stack, depth 1");
    }

    #[test]
    fn raster_registers_past_0x1ff_are_dropped() {
        assert!(Hq4Quirks.drop_raster_write(0x200).is_some());
        assert!(Hq4Quirks.drop_raster_write(0x1FF).is_none());
    }

    const SH: u64 = widget_small_base(11);
    const CONTROL: u32 = 0x24;
    const COMMAND: u32 = 0x120400;
    const MAPPING: u32 = 0x130400;
    const SI_ID: u32 = 0x0C00_3555;
    const SI_HQ_REVISION: u64 = 0x2C00_3555_0000_0000;

    #[derive(Default)]
    struct Heart(Mutex<Vec<(u8, bool)>>);

    impl InterruptController for Heart {
        fn set_line(&self, vector: u32, set: bool) {
            self.0.lock().push((vector as u8, set));
        }
    }

    /// 1 MB of HEART memory at physical 0x20000000.
    struct Ram(Mutex<Vec<u8>>);
    const RAM: u64 = 0x2000_0000;

    impl Ram {
        fn span(&self, phys: u64, n: usize) -> Option<std::ops::Range<usize>> {
            let at = phys.checked_sub(RAM)? as usize;
            (at + n <= self.0.lock().len()).then_some(at..at + n)
        }
    }

    impl Ram {
        fn read(&self, phys: u64, buf: &mut [u8]) -> bool {
            match self.span(phys, buf.len()) {
                Some(r) => { buf.copy_from_slice(&self.0.lock()[r]); true }
                None => { buf.fill(0); false }
            }
        }
        fn write(&self, phys: u64, data: &[u8]) -> bool {
            match self.span(phys, data.len()) {
                Some(r) => { self.0.lock()[r].copy_from_slice(data); true }
                None => false,
            }
        }
        fn get<const N: usize>(&self, addr: u64) -> Option<[u8; N]> {
            let mut b = [0u8; N];
            self.read(addr, &mut b).then_some(b)
        }
        fn put(&self, addr: u64, b: &[u8]) -> u32 {
            if self.write(addr, b) { BUS_OK } else { BUS_ERR }
        }
    }

    /// The bus masters' memory view: big-endian accesses at physical
    /// addresses, an address outside the memory a bus error.
    impl BusDevice for Ram {
        fn read8(&self, addr: u64) -> BusRead8 {
            self.get::<1>(addr).map_or(BusRead8::err(), |b| BusRead8::ok(b[0]))
        }
        fn read16(&self, addr: u64) -> BusRead16 {
            self.get(addr).map_or(BusRead16::err(), |b| BusRead16::ok(u16::from_be_bytes(b)))
        }
        fn read32(&self, addr: u64) -> BusRead32 {
            self.get(addr).map_or(BusRead32::err(), |b| BusRead32::ok(u32::from_be_bytes(b)))
        }
        fn read64(&self, addr: u64) -> BusRead64 {
            self.get(addr).map_or(BusRead64::err(), |b| BusRead64::ok(u64::from_be_bytes(b)))
        }
        fn write8(&self, addr: u64, val: u8) -> u32 { self.put(addr, &[val]) }
        fn write16(&self, addr: u64, val: u16) -> u32 { self.put(addr, &val.to_be_bytes()) }
        fn write32(&self, addr: u64, val: u32) -> u32 { self.put(addr, &val.to_be_bytes()) }
        fn write64(&self, addr: u64, val: u64) -> u32 { self.put(addr, &val.to_be_bytes()) }
    }

    struct Rig {
        b: Arc<ImpactXio>,
        heart: Arc<Heart>,
        ram: Arc<Ram>,
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            self.b.mgras().stop_engines();
        }
    }

    /// A board at widget 11 with its engine threads running (no display).
    fn rig() -> Rig {
        let heart = Arc::new(Heart::default());
        let ram = Arc::new(Ram(Mutex::new(vec![0; 1 << 20])));
        let hb = Arc::new(AtomicU64::new(0));
        let b = ImpactXio::new(11, heart.clone(), ram.clone(), hb.clone(), hb);
        b.mgras().start_engines();
        Rig { b, heart, ram }
    }

    impl Rig {
        /// Store, retrying on back-pressure like the CPU does.
        fn w(&self, bits: u32, off: u32, val: u64) -> u32 {
            loop {
                let st = self.b.write(SH + off as u64, bits, val);
                if st != BUS_BUSY {
                    return st;
                }
                std::thread::yield_now();
            }
        }
        fn w32(&self, off: u32, val: u32) {
            assert_eq!(self.w(32, off, val as u64), BUS_OK, "write {off:#x}");
        }
        fn w64(&self, off: u32, val: u64) {
            assert_eq!(self.w(64, off, val), BUS_OK, "write {off:#x}");
        }
        fn r(&self, bits: u32, off: u32) -> (u32, u64) {
            loop {
                let r = self.b.read(SH + off as u64, bits);
                if r.0 != BUS_BUSY {
                    return r;
                }
                std::thread::yield_now();
            }
        }
        fn r32(&self, off: u32) -> u32 {
            let (st, v) = self.r(32, off);
            assert_eq!(st, BUS_OK, "read {off:#x}");
            v as u32
        }
        fn r8(&self, off: u32) -> u8 {
            let (st, v) = self.r(8, off);
            assert_eq!(st, BUS_OK, "read {off:#x}");
            v as u8
        }
        fn packets(&self) -> Vec<(u8, bool)> {
            std::mem::take(&mut *self.heart.0.lock())
        }
        fn poke(&self, phys: u64, bytes: &[u8]) {
            assert!(self.ram.write(phys, bytes));
        }
        fn peek(&self, phys: u64, n: usize) -> Vec<u8> {
            let mut b = vec![0; n];
            assert!(self.ram.read(phys, &mut b));
            b
        }

        /// The kernel's context switch, with no interrupt routing: the
        /// selection, the first acknowledgement, the image (a first load
        /// of context slot 0), the second.
        fn enter_context(&self) {
            self.w32(0x011000, 0);
            self.w32(0x130500, 0x4FC);
            self.w32(0x011028, 0x80000);
            for i in 0..63u32 {
                self.w32(COMMAND, if i == 1 { 0x8000_0000 } else { 0 });
            }
            self.w32(0x011028, 0x1000_0000);
        }

        /// A raster register write through the command port, in the
        /// 64-bit form the Octane drivers use.
        fn rss(&self, reg: u32, val: u32, exec: bool) {
            let sel = if exec { 0x70001 } else { 0x60001 } | reg << 6;
            self.w64(COMMAND, (sel as u64) << 34 | val as u64);
        }

        /// A raster register write through the direct window.
        fn direct(&self, reg: u32, val: u32) {
            self.w32(0x02C000 + 4 * reg, val);
        }

        /// A host DMA descriptor write through the command port.
        fn descriptor(&self, reg: u32, val: u32) {
            self.w64(COMMAND, ((0x0008_0004 | reg << 8) as u64) << 32 | val as u64);
        }

        /// The X server's raster setup (as the Indigo2 board tests):
        /// origin at the top row, colour index drawing into the PROM's
        /// 1280x1024 layout.
        fn x_server(&self) {
            self.direct(0x112, 0xCAC); // CONFIG
            self.direct(0x115, 1023 << 16); // XYWIN
            self.direct(0x16E, 0x31E); // DRBSIZE
            self.direct(0x16D, 0x240); // DRBPOINTERS
            self.direct(0x163, 0xFFF);
            self.direct(0x161, 0x0C00_4504); // PP1FILLMODE
        }

        fn block(&self, x0: u32, y0: u32, x1: u32, y1: u32) {
            self.rss(0x045, 0x18, false); // IR_ALIAS
            self.rss(0x046, x0 << 16 | y0, false);
            self.rss(0x047, x1 << 16 | y1, true);
        }
    }

    #[test]
    fn identity() {
        let r = rig();
        assert_eq!(r.r32(4), SI_ID);
        assert_eq!(r.r(64, 0), (BUS_OK, SI_HQ_REVISION));
        assert_eq!(r.r32(0x072488) >> 24, 0x45);
        assert_eq!(r.r8(0x072488), 0x45);
        assert_eq!(r.r32(0x072408) >> 24, 0x0F);
        // Monitor 0xF and CMAP revision 3; no video option, RA revision 0.
        assert_eq!(r.r8(0x071308), 0xF3);
        assert_eq!(r.r8(0x071708), 0x83);
        // The RE4's revision 2, an idle raster engine (bit 8), one RE.
        assert_eq!(r.r32(0x02C578) & 0x1F3, 0x120);
        // The GE revision until a query replaces it.
        assert_eq!(r.r32(0x028004), 2);
    }

    #[test]
    fn widths_and_holes() {
        let r = rig();
        r.w32(CONTROL, 0x39A5_62D7);
        r.w(8, CONTROL, 0xFF);
        r.w(16, CONTROL, 0xFFFF);
        assert_eq!(r.r32(CONTROL), 0x39A5_62D7);
        r.w32(0x040000, 0x1A2B_3C4D);
        r.w32(0x040004, 0x5E6F_7081);
        r.w64(0x040000, 0xA1B2_C3D4_E5F6_0718);
        assert_eq!(r.r32(0x040000), 0x1A2B_3C4D);
        assert_eq!(r.r32(0x040004), 0x5E6F_7081);
        r.w(8, 0x073C08, 0xA7);
        assert_eq!(r.r8(0x073C08), 0xA7);
        for off in [0x20_0000, 0x80_0000, 0xFF_FFFC] {
            assert_eq!(r.r(32, off).0, BUS_ERR);
            assert_eq!(r.w(32, off, 0xFFFF_FFFF), BUS_ERR);
            assert_eq!(r.r32(CONTROL), 0x39A5_62D7);
        }
    }

    #[test]
    fn reset_clears_the_hq4() {
        let r = rig();
        r.w32(CONTROL, 0xC1D2_8354);
        r.w32(0x040000, 0x75E6_8192);
        r.enter_context();
        assert!(r.b.shared.hq4.lock().first_ack);
        r.b.reset();
        assert_eq!(r.r32(4), SI_ID);
        assert_eq!(r.r32(CONTROL), 0);
        assert_eq!(r.r32(0x040000), 0);
        let s = r.b.shared.hq4.lock();
        assert!(!s.first_ack && !s.ready);
    }

    #[test]
    fn context_switch_handshake() {
        let r = rig();
        r.w32(0x011060, 8 << 24);
        r.w32(0x011068, 0x80);
        r.w32(0x011070, 18 << 8);
        r.w32(0x011040, 0x1008_A200);
        r.w32(0x011000, 0x41005);
        r.w32(0x130500, 0x4FC);
        assert_eq!(r.r32(0x011020) & 0x80000, 0x80000);
        assert_eq!(r.packets(), [(18, true)]);
        // The image waits for the first acknowledgement.
        r.w32(COMMAND, 0x1234);
        assert_eq!(r.b.shared.hq4.lock().context_words, 0);
        r.w32(0x011028, 0x80000);
        assert_eq!(r.packets(), [(18, false)]);
        for i in 0..63u32 {
            assert_eq!(r.r32(0x011020) & 0x1000_0000, 0, "before word {i}");
            r.w32(COMMAND, if i == 1 { 0x8000_0000 } else { 0 });
        }
        assert_eq!(r.r32(0x011020) & 0x1000_0000, 0x1000_0000);
        assert_eq!(r.packets(), [(18, true)]);
        r.w32(0x011028, 0x1000_0000);
        assert_eq!(r.packets(), [(18, false)]);
        assert_eq!(r.r32(COMMAND), 0xF400);
        // The HQ3 model's own switch flags stay out of the HQ flags.
        assert_eq!(r.r32(0x020010) & (1 << 19 | 1 << 6), 0);
    }

    /// Synthetic programmed vectors exercise transport, not the IP30
    /// allocation.
    #[test]
    fn interrupt_transport() {
        let r = rig();
        // No destination: nothing is sent.
        r.w32(0x011040, 0x80000);
        r.w32(0x011000, 0);
        r.w32(0x130500, 0x4FC);
        assert_eq!(r.packets(), []);
        r.w32(0x011028, 0x80000);
        r.w32(0x011060, 8 << 24);
        r.w32(0x011068, 0x80);
        r.w32(0x011070, 18 << 8);
        r.w32(0x130500, 0x4FC);
        assert_eq!(r.packets(), [(18, true)]);
        r.w32(0x011028, 0x80000);
        assert_eq!(r.packets(), [(18, false)]);
        r.w32(0x130500, 0x4FC);
        assert_eq!(r.packets(), [(18, true)]);
        r.w32(0x011028, 0x80000);
        r.w32(0x011070, 7 << 8);
        r.w32(0x130500, 0x4FC);
        assert_eq!(r.packets(), [(18, false), (7, true)]);
        // A disabled source is latched but not sent.
        r.w32(0x011048, 0x80000);
        assert_eq!(r.packets(), [(7, false)]);
        assert_eq!(r.r32(0x011020) & 0x80000, 0x80000);
        // A destination outside HEART is not sent either.
        r.w32(0x011060, 9 << 24);
        r.w32(0x011040, 0x80000);
        assert_eq!(r.packets(), []);
    }

    /// After the switch, command words reach the raster engine: through
    /// the command port, the normal FIFO and the direct window alike.
    #[test]
    fn commands_after_a_switch_draw() {
        let r = rig();
        r.enter_context();
        r.x_server();
        r.rss(0x110, 1 << 20, false); // FILLMODE: fast fill
        r.rss(0x176, 0x13, false); // FILL_COLOR_R
        r.block(10, 20, 12, 21);
        assert_eq!(r.b.mgras().fb_pixel(10, 20), 0x13);
        assert_eq!(r.b.mgras().fb_pixel(12, 21), 0x13);
        assert_eq!(r.b.mgras().fb_pixel(13, 21), 0);
        r.w64(0x020400, (0x60001u64 | 0x176 << 6) << 34 | 0x21);
        r.block(30, 5, 30, 5);
        assert_eq!(r.b.mgras().fb_pixel(30, 5), 0x21);
    }

    /// Before any context switch, as the IP30 PROM's console draws: whole
    /// packets reach the raster engine; another 64-bit word (the
    /// diagnostics' bring-up word) is dropped and does not swallow the
    /// writes after it.
    #[test]
    fn whole_packets_draw_before_any_switch() {
        let r = rig();
        r.w64(COMMAND, 0x65781);
        // The PROM's pixel-format packet, a whole packet but no raster write.
        r.w64(COMMAND, 0x000C_0004_0000_0E00);
        r.x_server();
        r.rss(0x110, 1 << 20, false); // FILLMODE: fast fill
        r.rss(0x176, 0x2A, false); // FILL_COLOR_R
        r.block(40, 7, 41, 7);
        assert_eq!(r.b.mgras().fb_pixel(40, 7), 0x2A);
        assert_eq!(r.b.mgras().fb_pixel(41, 7), 0x2A);
        assert_eq!(r.b.mgras().fb_pixel(42, 7), 0);
        assert!(!r.b.shared.hq4.lock().ready, "no switch happened");
        assert!(is_whole_packet(0x001C_1104_0000_0018));
        assert!(is_whole_packet(0x001A_6804_0000_0000));
        assert!(!is_whole_packet(0x65781));
        assert!(!is_whole_packet(0x8000_0004_0000_0000), "pixel data");
    }

    /// The IP30 PROM's pre-switch packet 0x001A6804 names a raster register
    /// past 0x1FF: it is dropped and reported, and leaves the raster
    /// registers alone.
    #[test]
    fn a_raster_write_past_the_register_file_is_dropped() {
        let r = rig();
        r.w64(COMMAND, 0x001A_6804_0000_0000);
        r.w64(COMMAND, 0x001A_6804_DEAD_BEEF);
        r.x_server();
        r.rss(0x110, 1 << 20, false);
        r.rss(0x176, 0x2A, false);
        r.block(3, 3, 3, 3);
        assert_eq!(r.b.mgras().fb_pixel(3, 3), 0x2A, "the pipe still draws");
        assert!(r.b.mgras().noted("past register 0x1FF"), "reported");
    }

    /// HQ4 DMA through HEART memory. On the Octane the page list holds
    /// 32-bit page-frame numbers from a list address that is only 4-byte
    /// aligned, as IRIX 6.5.30's IP30 X server lists them; the list here
    /// starts 12 bytes into its page and the transfer crosses a page, so
    /// 64-bit entries or a list rounded down to 8 bytes lose the second
    /// page.
    #[test]
    fn dma_through_heart_memory() {
        let r = rig();
        r.poke(0x2000_8008, &[0; 4]);
        r.poke(0x2000_800C, &0x20010u32.to_be_bytes());
        r.poke(0x2000_8010, &0x20011u32.to_be_bytes());
        r.poke(0x2001_0FFE, &[1, 2]);
        r.poke(0x2001_1000, &[3, 4]);
        r.enter_context();
        // Slot 0's page list, count and control (HEART's widget) through
        // the mapping port.
        r.w32(MAPPING, 0x0048_2008);
        r.w64(MAPPING, 0x2000_800C);
        r.w64(MAPPING, 0x0048_3004_0000_0002);
        r.w64(MAPPING, 0x0048_4B04_0008_0000);
        r.x_server();
        r.rss(0x110, 5 << 22, false); // FILLMODE: host data
        r.rss(0x159, 0, false); // XFRMODE
        r.rss(0x153, 1 << 16 | 4, false); // XFRSIZE: one line of four
        r.block(10, 10, 13, 10);
        r.descriptor(4, 4); // pitch
        r.descriptor(5, 2); // source offset, low bits
        r.descriptor(6, 0xFFC); // source offset, aligned
        r.descriptor(7, 1); // rows
        r.descriptor(8, 4); // bytes a row
        r.descriptor(11, 0xB1); // kick: slot 0, host to board
        let got: Vec<u32> = (10..14).map(|x| r.b.mgras().fb_pixel(x, 10)).collect();
        assert_eq!(got, [1, 2, 3, 4]);
        // And back: board to host, as the Octane X server reads: the
        // raster engine's read armed with XFRCONTROL 8, then the kick,
        // which starts it.
        r.poke(0x2001_0FFE, &[0xA5; 2]);
        r.poke(0x2001_1000, &[0xA5; 2]);
        r.rss(0x102, 8, false); // XFRCONTROL: DMA read
        r.rss(0x110, 4 << 22, false); // FILLMODE: DMA read
        r.block(10, 10, 13, 10);
        r.descriptor(11, 0xB9); // kick: slot 0, board to host
        r.b.mgras().state_hash();
        assert_eq!(r.peek(0x2001_0FFE, 2), [1, 2]);
        assert_eq!(r.peek(0x2001_1000, 2), [3, 4]);
        // A kick that comes before the raster side waits for XFRCONTROL.
        r.poke(0x2001_0FFE, &[0xA5; 2]);
        r.poke(0x2001_1000, &[0xA5; 2]);
        r.rss(0x102, 0, false);
        r.descriptor(11, 0xB9);
        r.block(10, 10, 13, 10);
        r.b.mgras().state_hash();
        assert_eq!(r.peek(0x2001_0FFE, 2), [0xA5; 2]);
        r.rss(0x102, 8, false);
        r.b.mgras().state_hash();
        assert_eq!(r.peek(0x2001_0FFE, 2), [1, 2]);
        assert_eq!(r.peek(0x2001_1000, 2), [3, 4]);
    }

    /// The swap command raises source 10 once the HQ reaches it; the
    /// handler's acknowledgement lets the next swap raise it again.
    #[test]
    fn swap_raises_source_10() {
        let r = rig();
        r.enter_context();
        r.w32(0x011060, 8 << 24);
        r.w32(0x011068, 0x80);
        r.w32(0x011070, 18 << 8);
        r.w32(0x011040, 0x400);
        assert_eq!(r.packets(), []);
        for _ in 0..2 {
            r.w32(COMMAND, 0x37 << 8); // SCHEDULE_SWAP
            // The HQ thread raises the source, then sends the packet.
            let mut got = Vec::new();
            for _ in 0..1_000_000 {
                got = r.packets();
                if !got.is_empty() {
                    break;
                }
                std::thread::yield_now();
            }
            assert_eq!(got, [(18, true)]);
            assert_eq!(r.r32(0x011020) & 0x400, 0x400);
            assert_eq!(r.r32(0x020010) & 0x400, 0);
            r.w32(0x011028, 0x401);
            assert_eq!(r.packets(), [(18, false)]);
            assert_eq!(r.r32(0x011020) & 0x400, 0);
        }
    }

    /// The kernel's FIFO flush with the pipe still busy: words queued that
    /// the engines have not taken. The FIFO reads empty, and arming the
    /// low-water interrupt (mark 1, then the second-level enable) raises it
    /// at once, so the flush's wait ends.
    #[test]
    fn low_water_rises_with_its_enable() {
        let r = rig();
        r.b.mgras().stop_engines();
        for i in 0..8u64 {
            r.w64(0x020400, (0x60001u64 | 0x176 << 6) << 34 | i);
        }
        assert_eq!(r.r32(0x020008) & 0xFF, 0);
        r.w32(0x011060, 8 << 24);
        r.w32(0x011068, 0x80);
        r.w32(0x011070, 5);
        r.w32(0x011058, 0xFFFF_FFEF);
        r.w32(0x040008, 1);
        assert_eq!(r.packets(), []);
        r.w32(0x011050, 0x10);
        assert_eq!(r.packets(), [(5, true)]);
        assert_eq!(r.r32(0x011020) & 4, 4);
        // An enable already set raises nothing new.
        r.w32(0x011050, 0x10);
        assert_eq!(r.packets(), []);
        // The handler's acknowledgement lowers it; the clear port takes the
        // enable away.
        r.w32(0x011028, 4);
        assert_eq!(r.packets(), [(5, false)]);
        r.w32(0x011058, 0x10);
        assert_eq!(r.r32(0x011050), 0);
        // Re-armed, it rises again; the clear port with the bit lowers it.
        r.w32(0x011050, 0x10);
        assert_eq!(r.packets(), [(5, true)]);
        r.w32(0x011058, 0x10);
        assert_eq!(r.packets(), [(5, false)]);
        assert_eq!(r.r32(0x011020) & 4, 0);
    }

    /// The swap source raised on an engine thread while the CPU acknowledges
    /// it: HEART must see its packets in the order they were decided, so its
    /// level ends where the board's is and sets and clears alternate.
    #[test]
    fn swap_packets_keep_their_order_across_threads() {
        let r = rig();
        r.w32(0x011060, 8 << 24);
        r.w32(0x011068, 0x80);
        r.w32(0x011070, 18 << 8);
        r.w32(0x011040, 0x400);
        let lines = Lines(r.b.shared.clone());
        std::thread::scope(|sc| {
            sc.spawn(|| {
                for _ in 0..200_000 {
                    lines.set_line(1, true);
                }
            });
            for _ in 0..200_000 {
                r.w32(0x011028, 0x400);
            }
        });
        let got = r.packets();
        assert!(got.iter().all(|&(v, _)| v == 18));
        for w in got.windows(2) {
            assert_ne!(w[0].1, w[1].1, "two packets of one kind in a row");
        }
        let heart = got.last().is_some_and(|p| p.1);
        let s = r.b.shared.hq4.lock();
        assert_eq!(heart, s.general_asserted);
        assert_eq!(heart, s.pending & 0x400 != 0);
    }

    /// A reset leaves no swap latched on the HQ3 model's side: an unserved
    /// swap before it must not keep the next one from being reported.
    #[test]
    fn reset_rearms_the_swap_source() {
        let r = rig();
        for round in 0..2 {
            r.enter_context();
            r.w32(0x011060, 8 << 24);
            r.w32(0x011068, 0x80);
            r.w32(0x011070, 18 << 8);
            r.w32(0x011040, 0x400);
            r.w32(COMMAND, 0x37 << 8); // SCHEDULE_SWAP
            let mut got = Vec::new();
            for _ in 0..1_000_000 {
                got = r.packets();
                if !got.is_empty() {
                    break;
                }
                std::thread::yield_now();
            }
            assert_eq!(got, [(18, true)], "round {round}");
            // Not acknowledged: the board is reset instead.
            r.b.reset();
        }
    }

    /// The pixel pipeline's ERAM blocks sit 0x8C higher on the Octane; GE
    /// memory below the ERAM and the program do not move.
    #[test]
    fn pixel_state_maps_to_the_indigo2_layout() {
        let q = Hq4Quirks;
        assert_eq!(q.pixel_state_address(0xE34), 0xDA8, "transfer size");
        assert_eq!(q.pixel_state_address(0xD91), 0xD05, "a module state");
        assert_eq!(q.pixel_state_address(0x1D6), 0x1D6, "GE memory");
        assert_eq!(q.pixel_state_address(0x13B), 0x13B, "the program");
    }

    #[test]
    fn retrace_is_a_message_a_frame() {
        let r = rig();
        r.w32(0x011060, 8 << 24);
        r.w32(0x011068, 0x80);
        r.w32(0x011070, 0x2A << 16);
        let lines = Lines(r.b.shared.clone());
        lines.set_line(2, true);
        lines.set_line(2, false);
        lines.set_line(2, true);
        assert_eq!(r.packets(), [(0x2A, true), (0x2A, true)]);
        assert_eq!(r.r32(0x011020) & 1, 1);
        r.w32(0x011028, 0x401);
        assert_eq!(r.r32(0x011020) & 1, 0);
    }

    /// The kernel's GE11 program download and its verification readback:
    /// slot B (GE 0) takes the words in order, and the two readback ports
    /// return them in pairs.
    #[test]
    fn ge_program_reads_back_in_pairs() {
        let r = rig();
        let prog: Vec<u32> = (0..8).map(|i| 0x0100_0000 * (i + 1) + i).collect();
        r.w32(0x058044, 0x20_0000);
        for &w in &prog {
            r.w32(0x058040, w);
        }
        // The download ends with any other control value.
        r.w32(0x058044, 0x8000_0000);
        r.w32(0x058040, 0xDEAD);
        for p in 0..4 {
            assert_eq!(r.r32(0x058230), prog[2 * p + 1], "pair {p}");
            assert_eq!(r.r32(0x05822C), prog[2 * p], "pair {p}");
        }
        assert_eq!(r.r32(0x05822C), 0, "past the download");
        // Bit 0 of the load value starts the readback a word early.
        r.w32(0x05804C, 0x20_0001);
        for &w in &prog[..4] {
            r.w32(0x058048, w);
        }
        r.w32(0x05804C, 0x8000_0000);
        assert_eq!(r.r32(0x058230), prog[0]);
        assert_eq!(r.r32(0x05822C), 0);
        assert_eq!(r.r32(0x058230), prog[2]);
        assert_eq!(r.r32(0x05822C), prog[1]);
        // Slot B's download is untouched.
        r.w32(0x058044, 0x8000_0000);
        assert_eq!(r.r32(0x05822C), prog[0]);
        // The GE revision stays until a query replaces it.
        assert_eq!(r.r32(0x028004), 2);
    }

    /// One MicroLAN bit slot on the serial-number chip.
    fn nic_slot(r: &Rig, pulse: u32, sample: u32) -> bool {
        r.w32(0x011098, pulse << 10 | sample << 2);
        let v = r.r32(0x011098);
        assert_eq!(v & 2, 2, "slot done");
        v & 1 != 0
    }

    fn nic_write_byte(r: &Rig, b: u8) {
        for bit in 0..8 {
            let one = b >> bit & 1 != 0;
            nic_slot(r, if one { 6 } else { 80 }, if one { 13 } else { 30 });
        }
    }

    fn nic_read_byte(r: &Rig) -> u8 {
        (0..8).fold(0, |v, bit| v | (nic_slot(r, 6, 13) as u8) << bit)
    }

    /// The board record as the PROM reads it for "Name:": skip ROM, read
    /// memory from 0, the barcode at byte 1 and the name at byte 48.
    #[test]
    fn nic_holds_the_board_record() {
        let r = rig();
        for pass in 0..2 {
            assert!(!nic_slot(&r, 500, 65), "presence");
            for b in [0xCC, 0xF0, 0, 0] {
                nic_write_byte(&r, b);
            }
            let rec: Vec<u8> = (0..64).map(|_| nic_read_byte(&r)).collect();
            assert_eq!(&rec[1..11], b"EMUG00    ", "pass {pass}");
            assert_eq!(&rec[48..62], b"GM10          ");
            assert_eq!(&rec[21..30], b"030-0938-");
            assert_eq!(&rec[35..39], b"003U");
            r.b.reset();
        }
    }

    #[test]
    fn a_status_store_into_a_full_fifo_is_held_back_and_lands_on_retry() {
        // No engine thread, so nothing drains the command FIFO.
        let heart = Arc::new(Heart::default());
        let ram = Arc::new(Ram(Mutex::new(vec![0; 1 << 20])));
        let hb = Arc::new(AtomicU64::new(0));
        let r = Rig { b: ImpactXio::new(11, heart.clone(), ram.clone(), hb.clone(), hb), heart, ram };
        let mut accepted = 0u32;
        let st = loop {
            let st = r.b.write(SH + hq4::RE_STATUS as u64, 32, 0);
            if st != BUS_OK {
                break st;
            }
            accepted += 1;
            assert!(accepted < 1 << 20, "the FIFO never filled");
        };
        assert_eq!(st, BUS_BUSY, "a full FIFO must back-pressure, not drop the store");
        assert!(accepted > 0);
        // Still full, still busy: nothing was consumed or lost silently.
        assert_eq!(r.b.write(SH + hq4::RE_STATUS as u64, 32, 0), BUS_BUSY);
        r.b.mgras().start_engines();
        let mut tries = 0u32;
        while r.b.write(SH + hq4::RE_STATUS as u64, 32, 0) == BUS_BUSY {
            tries += 1;
            assert!(tries < 10_000_000, "the store never landed after the engine drained");
            std::thread::yield_now();
        }
    }

    #[test]
    fn routes() {
        assert_eq!(route(0x020000), Route::Board(0x70000));
        assert_eq!(route(0x020008), Route::Local);
        assert_eq!(route(0x020400), Route::Board(0x70080));
        assert_eq!(route(0x02C1C4), Route::Board(0x7C1C4));
        assert_eq!(route(0x02D1C0), Route::Board(0x7D1C0));
        assert_eq!(route(0x050000), Route::Board(0x40000));
        assert_eq!(route(0x058040), Route::Local);
        assert_eq!(route(0x05822C), Route::Local);
        assert_eq!(route(0x058050), Route::Local);
        assert_eq!(route(0x071C88), Route::Board(0x61C88));
        assert_eq!(route(0x072038), Route::Board(0x62038));
        assert_eq!(route(0x073C08), Route::Local);
        assert_eq!(route(0x1F_FFFC), Route::Local);
        assert_eq!(route(0x20_0000), Route::Hole);
    }
}
