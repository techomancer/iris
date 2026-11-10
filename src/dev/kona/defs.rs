//! Every number the board's host interface and drawing back end consume: widget
//! register offsets and field layouts, host command tokens, display-bus
//! registers and the layouts of the multi-field words, each under one name.
//!
//! Names and values are transcribed from the board's published command and
//! register documentation; where it grades a value inferred or unknown, the
//! comment says so. A name is a role in plain words, not a vendor symbol. Nothing else
//! in the module spells these numbers. The register offsets and the first
//! token constants were ported from include/hw/misc/sgi-kona.h and
//! include/hw/misc/sgi-kona-ports.h in jamesbraid/qemu-sgi-origami; software
//! expectations they state are QEMU's model, not hardware facts.
//!
//! This is a table of facts, so it holds names no code uses yet; the module
//! allows dead code for that reason alone. Most of the unused ones are the
//! tokens, display-bus registers and field layouts that the steps still to
//! come (textures, pixel operations, queries, the display's cursor and video
//! paths) read.

#![allow(dead_code)]

/// One widget's worth of address space; the rest reads unmapped.
pub const APERTURE: u64 = 16 << 20;

// Crosstalk widget header.
pub const WIDGET_ID: u32 = 0x04;
pub const WIDGET_CONTROL: u32 = 0x24;
pub const INT_DEST_UPPER: u32 = 0x34;
pub const INT_DEST_LOWER: u32 = 0x3c;
pub const INT_VECTORS: u32 = 0x5c;
pub const PROBE: u32 = 0x64;
pub const LOW_WINDOW_WRITE: u32 = 0x7c;
pub const ONEWIRE: u32 = 0x98;
pub const NIC: u32 = 0x9c;

// Register window at +0x800000.
pub const CREDIT: u32 = 0x800208;
pub const CREDIT_CONFIG: u32 = 0x800210;
pub const RESET_ARG: u32 = 0x800220;
pub const INTR_END: u32 = 0x800228;
pub const INT_ENABLE: u32 = 0x800230;
pub const INT_CAUSE: u32 = 0x800238;
pub const MAILBOX: u32 = 0x800240;
pub const RETRACE_TAG: u32 = 0x800248;
pub const RESET_STROBE: u32 = 0x800300;
pub const EEPROM_ENABLE: u32 = 0x800308;
pub const LOADER_328: u32 = 0x800328;
pub const ACTIVITY: u32 = 0x8003a8;
pub const LOADER_3B0: u32 = 0x8003b0;
pub const INDIRECT_SELECT: u32 = 0x8003c0;
pub const INDIRECT_DATA: u32 = 0x8003d0;
pub const PORT_ADDR: u32 = 0x8003e8;
pub const PORT_DATA: u32 = 0x8003f0;
pub const PORT_WRITE_NEXT: u32 = 0x8003f8;
/// The kernel FIFO command push port. Approximated as a register-block
/// relative address: the driver's own citation is +0x7c, which the
/// low-window register claims.
pub const DMA_FIFO_PUSH: u32 = 0x80007c;

// Flow-controlled window and the command page.
pub const DMA_FIFO_KICK: u32 = 0xc00108;
pub const COMMAND_PAGE: u32 = 0xc10000;
pub const COMMAND_PAGE_SIZE: u32 = 0x4000;
pub const COMMAND_DATA_SLOT: u32 = 0xc;
pub const COMMAND_SLOT_SHIFT: u32 = 5;
pub const COMMAND_SLOT_MASK: u32 = 0x1f;
pub const COMMAND_TOKEN_MASK: u32 = 0x1ff;

// XG map RAM.
pub const MAPRAM_BASE: u32 = 0x100000;
pub const MAPRAM_ENTRIES: usize = (1 << 20) / 4;
pub const PROBE_MAPRAM_COUNT_SHIFT: u32 = 3;
pub const PROBE_MAPRAM_COUNT_MASK: u32 = 0x7 << PROBE_MAPRAM_COUNT_SHIFT;

// The widget id register.
pub const WIDGET_ID_REVISION_SHIFT: u32 = 28;
pub const WIDGET_ID_PART_SHIFT: u32 = 12;
pub const WIDGET_ID_MANUFACTURER_SHIFT: u32 = 1;
pub const WIDGET_ID_ALWAYS_ONE: u32 = 1;
pub const PART: u32 = 0xc102;
pub const MANUFACTURER: u32 = 0x2aa;

// The one-wire register, in the layout the Bridge and IOC3 share: a write
// starts a slot.
pub const ONEWIRE_LEVEL: u32 = 1 << 0;
pub const ONEWIRE_DONE: u32 = 1 << 1;
pub const ONEWIRE_SAMPLE_MASK: u32 = 0xff << 2;
pub const ONEWIRE_PULSE_MASK: u32 = 0x3ff << 10;
pub const ONEWIRE_WRITABLE: u32 = ONEWIRE_SAMPLE_MASK | ONEWIRE_PULSE_MASK;
/// The Dallas family code of the board's identity part (a DS2502).
pub const ONEWIRE_FAMILY: u8 = 0x09;

// Interrupt destination and vectors.
pub const INT_DEST_WIDGET_SHIFT: u32 = 16;
pub const INT_DEST_WIDGET_MASK: u32 = 0xf;
pub const INT_VECTORS_GE_SHIFT: u32 = 9;
pub const INT_VECTORS_GENERAL_MASK: u32 = 0xff;

/// The fill-level register's field, bits 10:1 on the second host-interface
/// revision: a level in 32-bit words at two-word resolution (inferred).
pub const CREDIT_MASK: u32 = 0x7fe;
/// The interrupt-enable register's bits on the second revision (the first
/// has 0x3ff): one per status bit.
pub const INT_ENABLE_MASK: u32 = 0xfff;

// Interrupt cause bits.
pub const CAUSE_LOWATER: u32 = 1 << 1;
pub const CAUSE_VIDEO: u32 = 1 << 3;
pub const CAUSE_MAILBOX: u32 = 1 << 4;
pub const CAUSE_GE: u32 = 1 << 5;

/// Bit 0 of the activity register: the kernel writes 1 once per video interrupt (established by
/// the kernel's use; nothing reads it back). Counted, with no effect.
pub const ACTIVITY_VIDEO: u32 = 1 << 0;
pub const ACTIVITY_BUS: u32 = 1 << 2;

// The indirect port's internal address space.
pub const FLASH_BASE: u32 = 0x800000;
pub const ICU_CONTROL: u32 = 0x200050;
pub const ARM_LOW_BASE: u32 = 0x0;
pub const ARM_LOW_WORDS: usize = 0x2c / 4;
pub const ARM_HIGH_BASE: u32 = 0x400_0000;
pub const ARM_HIGH_WORDS: usize = (0x414_e630 - 0x400_0000) / 4;
/// Sections of the firmware load the board records.
pub const LOAD_SECTIONS_MAX: usize = 32;

// Selectors the driver's attach reads its pipe class from through the
// indirect pair.
pub const SELECT_PRODUCT: u32 = 0x8800;
pub const SELECT_REDUCED: u32 = 0xb00;

// Bracket register bits.
pub const BRACKET_FORCE: u32 = 0x20000;
pub const BRACKET_PRIVILEGED: u32 = 0x10000;

// Page words: bit 28 marks a host physical page, otherwise the page word is
// a device address through the map RAM.
pub const PAGE_WORD_PHYSICAL: u32 = 1 << 28;
pub const PAGE_WORD_FRAME: u32 = 0x0fff_ffff;
pub const MAPRAM_KIND_MASK: u32 = 0xf000_0000;
pub const MAPRAM_KIND_16K: u32 = 0x2000_0000;
pub const MAPRAM_PAGE_SHIFT: u32 = 14;

// The textport grammar, the PROM's text console, drawn before any window
// exists. The table does not cover it; these are software expectations from
// QEMU's model (inferred).
/// The textport's sync word.
pub const TEXTPORT_SYNC_WORD: u32 = 1;
/// Commands framed and counted, with no drawing role known.
pub const TEXTPORT_UNMODELLED_1: u32 = 0x1;
pub const TEXTPORT_UNMODELLED_7: u32 = 0x7;
pub const TEXTPORT_UNMODELLED_8: u32 = 0x8;
pub const TEXTPORT_SET_COLOUR: u32 = 0x2;
pub const TEXTPORT_LOAD_COLOUR_MAP: u32 = 0x3;
pub const TEXTPORT_SET_RASTER_POSITION: u32 = 0x4;
pub const TEXTPORT_FILL_BOX: u32 = 0x5;
pub const TEXTPORT_PLOT_POINT: u32 = 0x6;
/// The sync probe, answered by the front end.
pub const TEXTPORT_SYNC_PROBE: u32 = 0x9;
/// Graphics info: the managed area, answered by the front end.
pub const TEXTPORT_GRAPHICS_INFO: u32 = 0xc;
/// The command that opens a bitmap blit.
pub const BITMAP_SELECTOR: u32 = 0xa;
pub const BITMAP_HEADER_WORDS: usize = 6;


/// The reasons the board posts on the reason page, and the one a context
/// clean-up carries when its window is destroyed (inferred).
pub const REASON_CONTEXT_CLEAN_UP: u32 = 0x103;
pub const REASON_CONTEXT_DEACTIVATED: u32 = 0x101;
pub const REASON_SWAP: u32 = 0x102;
pub const REASON_SUSPEND: u32 = 0x104;
/// The mailbox word a DMA packet's completion posts, which the packet asks
/// for with its last word.
pub const DMA_DONE_WORD: u32 = 0x100;

/// Words the GL texture table command carries after its slot word.
pub const GL_TEX_TABLE_WORDS: u32 = 33;
/// Data words behind a cursor load: the image and the colours.
pub const X_CURSOR_DATA_WORDS: u32 = 260;
/// Data words behind a cursor recolour.
pub const X_CURSOR_RECOLOUR_WORDS: u32 = 4;

/// The words a context's save area holds after a deactivate: word 0 is the
/// model's own marker ("KONA"), word 2 a no-op token the kernel sends back.
pub const SAVE_AREA_MARKER: u32 = 0x4b4f4e41;
pub const SAVE_AREA_TOKEN: u32 = 0x172;

/// The sync probe's answer.
pub const SYNC_PROBE_REPLY: u32 = 0x1234abcd;

/// Mailbox reply words the board queues at once.
pub const MAILBOX_MAX_WORDS: usize = 5;
/// Completion reasons that may wait behind an unread one.
pub const REASON_QUEUE: usize = 16;
/// The most words in one reason.
pub const REASON_WORDS: usize = 4;

/// Display-list memory.
pub const LIST_DEFAULT: usize = 15 << 20;
pub const LIST_GRANULE: u32 = 0x400;
/// The most words a list invocation fetches: the display-list class's size.
pub const LIST_MAX_WORDS: u32 = (LIST_DEFAULT / 4) as u32;
pub const LIST_CALLS_MAX: u32 = 8;
pub const LIST_CALLS_POOL_MAX: u32 = 4;
pub const DMA_ADDR_TAG: u32 = 0x9000;
pub const DMA_PAGE_SIZE: u64 = 4096;
pub const DMA_POOL_PAGE: u32 = 0x4000;
pub const DMA_POOL_MAX: u32 = 256 << 20;
/// The most a swap moves, a model bound.
pub const SWAP_MAX_BYTES: u64 = 256 << 20;

// Bit fields of the multi-field words. A field is `hi:lo`, inclusive, counted
// from zero at the least significant bit.

/// A run of bits in a 32-bit word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitField {
    pub hi: u8,
    pub lo: u8,
}

impl BitField {
    pub const fn new(hi: u8, lo: u8) -> Self {
        BitField { hi, lo }
    }

    /// The field's bits, in place.
    pub const fn mask(self) -> u32 {
        (((1u64 << (self.hi - self.lo + 1)) - 1) << self.lo) as u32
    }

    /// The field's value in `word`.
    pub const fn get(self, word: u32) -> u32 {
        (word & self.mask()) >> self.lo
    }

    /// `word` with the field set to `value` (truncated to its width).
    pub const fn put(self, word: u32, value: u32) -> u32 {
        (word & !self.mask()) | ((value << self.lo) & self.mask())
    }
}

/// The display-ID table entry (XMAP indirect memory, written at `XMAP_ID_TABLE_DATA`
/// with the ID in `XMAP_INDEX`). Established.
pub struct DisplayIdEntry;
impl DisplayIdEntry {
    /// The visual's colour-map slot number shifted left 7.
    pub const COLOUR_MAP_BASE: BitField = BitField::new(23, 7);
    /// The number of index planes minus 1: 7 for an 8-plane visual, 11 for 12.
    pub const INDEX_PLANES: BitField = BitField::new(6, 3);
    /// The luminance visual flag. What the XMAP does with it is unknown.
    pub const LUMINANCE: BitField = BitField::new(2, 2);
    /// The grey visual flag. What the XMAP does with it is unknown.
    pub const GREY: BitField = BitField::new(1, 1);
}

/// The per-display-ID mode byte, the third word of `DISPLAY_MODE_BYTE`.
/// Established as the X server builds it; what the XMAP does with bits 4:2 is
/// unknown.
pub struct ModeByte;
impl ModeByte {
    /// The back buffer is shown (the kernel rewrites it at each swap).
    pub const BACK_SHOWN: BitField = BitField::new(0, 0);
    pub const STEREO: BitField = BitField::new(1, 1);
    /// Non-colour-index RGB below 24 bits.
    pub const RGB: BitField = BitField::new(2, 2);
    /// Colour index or luminance.
    pub const COLOUR_INDEX: BitField = BitField::new(3, 3);
    /// Direct colour for a main window, opaque for an overlay.
    pub const DIRECT_OR_OPAQUE: BitField = BitField::new(4, 4);
    /// For an overlay, bit 0 is set for a non-RGB overlay (inferred from the X
    /// server's visuals; the table lists the bit only as the back buffer shown).
    pub const OVERLAY_NON_RGB: BitField = BitField::new(0, 0);
    /// For an overlay: 2, or 3 when double buffered and starting in the back
    /// buffer, plus 4 for an underlay.
    pub const OVERLAY_CODE: BitField = BitField::new(3, 1);
}

/// The channel set-up word (XMAP indirect memory, written at
/// `XMAP_CHANNEL_SETUP_DATA` with the channel in `XMAP_INDEX`). Established.
pub struct ChannelSetup;
impl ChannelSetup {
    /// The gamma page, a map index 0 to 15.
    pub const GAMMA_PAGE: BitField = BitField::new(3, 0);
    /// Use the external 1024-entry gamma RAM (the model treats it as no gamma).
    pub const EXTERNAL_GAMMA: BitField = BitField::new(4, 4);
    pub const NO_GAMMA: BitField = BitField::new(5, 5);
    pub const ALPHA_TO_BLUE: BitField = BitField::new(6, 6);
}

/// The gamma load address (written to `XMAP_INDEX` before gamma data).
pub struct GammaAddress;
impl GammaAddress {
    /// An address is 19 bits.
    pub const BITS: u32 = 19;
    pub const ENTRY: BitField = BitField::new(11, 0);
    /// 0 red, 1 green, 2 blue. A hardware test program's gamma loads rewrite 3 to green (1); what the board
    /// does with 3 is unknown, and the model keeps a fourth bank apart that nothing reads.
    pub const COLOUR: BitField = BitField::new(13, 12);
    pub const PAGE: BitField = BitField::new(17, 14);
    /// Set for one kind of map. What it selects is unknown.
    pub const FLAG: BitField = BitField::new(18, 18);
}

/// The cursor position register. Established.
pub struct CursorPosition;
impl CursorPosition {
    /// x less the hot spot and the rectangle origin, plus the bias.
    pub const X: BitField = BitField::new(12, 0);
    pub const Y: BitField = BitField::new(25, 13);
    /// The offset that lets the cursor sit partly off the top-left corner
    /// (inferred).
    pub const BIAS: i64 = 32;
}

/// The window-ID test register (raster register 0x1c06). Established. The
/// board encodes the window-placement request's word 9 into it: ordinary IDs
/// 0 to `WINDOW_ID_ORDINARY_LAST` as reference = ID, mask 0x3f; `WINDOW_ID_POPUP`
/// as reference 0x800, mask 0x800; the group IDs `WINDOW_ID_GROUP_FIRST` to
/// `WINDOW_ID_GROUP_LAST` as reference `(ID - 0x40) << 6`, mask 0x7c0; all ones
/// (the kernel's null clip, which makes a pipe own nothing) and anything else as
/// reference 0x3f, mask 0x3f. Which polarity a set mask bit has is inferred: a set bit
/// is compared. The compared value is the painted value shifted right by 2 (inferred):
/// the X server paints main ID n as `n << PAINT_MAIN_SHIFT` under mask 0xfc and an
/// overlay or popup group g as `g << PAINT_GROUP_SHIFT` under mask 0x1f00. Word 10 of the
/// placement, the "obscured" flag, sets the enable bit.
#[allow(non_snake_case)]
pub mod WINDOW_ID_TEST {
    use super::BitField;
    pub const ENABLE: BitField = BitField::new(0, 0);
    pub const REFERENCE_ID: BitField = BitField::new(12, 1);
    pub const MASK: BitField = BitField::new(24, 13);
}
/// Where a display ID sits in a painted pixel value (established): a main ID from bit 2, an
/// overlay or popup group (the ID less 0x40) from bit 8.
pub const PAINT_MAIN_SHIFT: u32 = 2;
pub const PAINT_GROUP_SHIFT: u32 = 8;
pub const WINDOW_ID_ORDINARY_LAST: i32 = 0x3e;
pub const WINDOW_ID_POPUP: i32 = 0x3f;
pub const WINDOW_ID_GROUP_FIRST: i32 = 0x40;
pub const WINDOW_ID_GROUP_LAST: i32 = 0x5f;

// XMAP registers on the display bus (0x100 + n), as the indirect store names
// them.
/// The cursor image base: the address of the 260-word block.
pub const XMAP_CURSOR_IMAGE_BASE: u32 = 0x101;
pub const XMAP_CURSOR_POSITION: u32 = 0x102;
pub const XMAP_CURSOR_MODE: u32 = 0x103;
/// The address for the data ports: a memory address, a channel number or a
/// display ID.
pub const XMAP_INDEX: u32 = 0x118;
pub const XMAP_COLOUR_DATA: u32 = 0x119;
pub const XMAP_GAMMA_DATA: u32 = 0x11b;
/// Display-ID table data (the ID is in `XMAP_INDEX`).
pub const XMAP_ID_TABLE_DATA: u32 = 0x11c;
/// Channel set-up word (the channel is in `XMAP_INDEX`).
pub const XMAP_CHANNEL_SETUP_DATA: u32 = 0x11d;

/// The display request that loads the DAC cursor (263 words; established: the channel, the six
/// colours as 10-bit values, and 256 image words of two bits a pixel, `pixel = mask << 1 | source`).
/// The model decodes and keeps the load (`display::dac_cursor`) and does not draw it.
pub const VIDEO_DAC_CURSOR_LOAD: u32 = 56;

// The display generator's video commands the model gives behaviour (the command number is the first
// word after the video token; `tables::VIDEO` frames all 93).
pub const VIDEO_SETUP_FRAME_BUFFER: u32 = 0x08;
pub const VIDEO_CHANNEL_GAMMA_MAP: u32 = 0x0f;
pub const VIDEO_GET_BOARD_CONFIG: u32 = 0x12;
pub const VIDEO_GET_CHANNEL_CONFIG: u32 = 0x13;
pub const VIDEO_RESIZE: u32 = 0x14;
pub const VIDEO_GET_RESIZE: u32 = 0x15;
pub const VIDEO_LOAD_FORMAT: u32 = 0x16;
pub const VIDEO_ENABLE_CHANNEL: u32 = 0x17;
pub const VIDEO_DISABLE_CHANNEL: u32 = 0x18;
pub const VIDEO_CURSOR_PRIORITY: u32 = 0x1e;
pub const VIDEO_CHANNEL_BLANKING: u32 = 0x1f;
pub const VIDEO_GET_CHANNEL_BLANKING: u32 = 0x20;
pub const VIDEO_BIND_TO_X_WINDOW: u32 = 0x34;
pub const VIDEO_GET_VAMPIRE: u32 = 0x3a;
pub const VIDEO_TEMP_BLANK: u32 = 0x3c;
pub const VIDEO_GET_CHANNEL_ENABLES: u32 = 0x42;
pub const VIDEO_PLACEMENT_RES: u32 = 0x43;
pub const VIDEO_GET_FORMAT_INFO: u32 = 0x4a;
pub const VIDEO_VAMPIRE: u32 = 0x4f;
pub const VIDEO_X_SERVER_INIT: u32 = 0x53;

/// The gamma-map word that means the linear map.
pub const VIDEO_GAMMA_MAP_LINEAR: u32 = 0xffff_fffe;

/// The channels a DAC cursor load can name: 0 to 7, or 0xde (established).
pub const DAC_CURSOR_CHANNELS: std::ops::RangeInclusive<u32> = 0..=7;
/// The one value outside 0 to 7 a load may name. The table gives the value and not what it selects.
pub const DAC_CURSOR_CHANNEL_DE: u32 = 0xde;

/// The video DAC's cursor control register (indexed register 0x204; the bit meanings are the
/// DAC part's published register description, established; that the part is that DAC is inferred).
pub mod dac_cursor_control {
    use super::BitField;
    /// The cursor mode: 1 is X11, 2 is XGA.
    pub const MODE: BitField = BitField::new(1, 0);
    pub const MODE_X11: u32 = 1;
    pub const MODE_XGA: u32 = 2;
    pub const ENABLE: BitField = BitField::new(2, 2);
    pub const INTERLACE: BitField = BitField::new(3, 3);
    /// What the kernel writes for the channel that owns the cursor (X11, enabled, not interlaced);
    /// it writes 0 for the others (established).
    pub const KERNEL_OWNER: u32 = 5;

    /// The control value in words, for the monitor.
    pub fn describe(v: u32) -> String {
        let mode = match MODE.get(v) {
            MODE_X11 => "X11".to_string(),
            MODE_XGA => "XGA".to_string(),
            n => format!("mode {n}"),
        };
        format!("{v:#x} = {mode} cursor, {}, {}", if ENABLE.get(v) != 0 { "enabled" } else { "disabled" }, if INTERLACE.get(v) != 0 { "interlaced" } else { "not interlaced" })
    }
}

// Cursor modes (the value written to `XMAP_CURSOR_MODE`).
pub const CURSOR_MODE_OFF: u32 = 0;
pub const CURSOR_MODE_CROSSHAIR: u32 = 1;
pub const CURSOR_MODE_GLYPH: u32 = 2;
/// A glyph slot: 64 rows of four words (plane A then plane B), then the four
/// colour words 10:10:10 at `CURSOR_COLOUR_WORDS` (pixel values 3, 2, 1, 0).
pub const CURSOR_SLOT_WORDS: usize = 260;
pub const CURSOR_SLOTS: usize = 39;
pub const CURSOR_COLOUR_WORDS: usize = 256;

// Display IDs. Established: main-plane windows 0 to 62, the popup at 63,
// overlays 64 to 95.
pub const POPUP_ID: usize = 63;
/// A point up to this size is one pixel; a larger one is a disc.
pub const POINT_PIXEL_MAX: f32 = 3.5;
pub const OVERLAY_ID_FIRST: usize = 64;
pub const OVERLAY_ID_LAST: usize = 95;
/// Display IDs the model's tables hold.
pub const DISPLAY_IDS: usize = OVERLAY_ID_LAST + 1;

/// OpenGL enumerants the pipe's state tokens carry. Values are OpenGL's.
pub mod gl {
    pub const NEVER: u32 = 0x200;
    pub const LESS: u32 = 0x201;
    pub const EQUAL: u32 = 0x202;
    pub const LEQUAL: u32 = 0x203;
    pub const GREATER: u32 = 0x204;
    pub const NOTEQUAL: u32 = 0x205;
    pub const GEQUAL: u32 = 0x206;
    pub const ALWAYS: u32 = 0x207;
    pub const DEPTH_BUFFER_BIT: u32 = 0x100;
    pub const FRONT_LEFT: u32 = 0x400;
    pub const FRONT_RIGHT: u32 = 0x401;
    pub const BACK_LEFT: u32 = 0x402;
    pub const FRONT: u32 = 0x404;
    pub const BACK: u32 = 0x405;
    pub const RIGHT: u32 = 0x406;
    pub const FRONT_AND_BACK: u32 = 0x408;
    pub const INVALID_ENUM: u32 = 0x500;
    pub const INVALID_OPERATION: u32 = 0x502;
    pub const STACK_OVERFLOW: u32 = 0x503;
    pub const STACK_UNDERFLOW: u32 = 0x504;
    pub const CW: u32 = 0x900;
    pub const CULL_FACE: u32 = 0xb44;
    pub const LIGHTING: u32 = 0xb50;
    pub const LIGHT_MODEL_LOCAL_VIEWER: u32 = 0xb51;
    pub const LIGHT_MODEL_TWO_SIDE: u32 = 0xb52;
    pub const LIGHT_MODEL_AMBIENT: u32 = 0xb53;
    pub const COLOR_MATERIAL: u32 = 0xb57;
    pub const DEPTH_TEST: u32 = 0xb71;
    pub const NORMALIZE: u32 = 0xba1;
    pub const SCISSOR_TEST: u32 = 0xc11;
    pub const TEXTURE_2D: u32 = 0xde1;
    pub const TEXTURE_3D: u32 = 0x806f;
    /// The two-bit RGB internal format, EXT's name for 0x804e.
    pub const RGB2_EXT: u32 = 0x804e;
    pub const FLAT: u32 = 0x1d00;
    pub const COLOR_BUFFER_BIT: u32 = 0x4000;
    /// Light n is `LIGHT0 + n`.
    pub const LIGHT0: u32 = 0x4000;
    pub const LIGHT7: u32 = 0x4007;
    /// User clip plane n is `CLIP_PLANE0 + n`, for n = 0 to 5.
    pub const CLIP_PLANE0: u32 = 0x3000;
    pub const CLIP_PLANE5: u32 = 0x3005;
    /// The rest of the enable tokens' clip-plane range, 0x3000 to 0x3fff: an invalid enumerant.
    pub const CLIP_PLANE_RANGE_FIRST: u32 = 0x3006;
    pub const CLIP_PLANE_RANGE_LAST: u32 = 0x3fff;
    /// Texture coordinate generation: the coordinates s, t, r, q are `TEXTURE_COORD_S + i`,
    /// their enables `TEXTURE_GEN_S + i`, and the parameter names the mode and the two planes.
    pub const TEXTURE_COORD_S: u32 = 0x2000;
    pub const TEXTURE_GEN_S: u32 = 0xc60;
    pub const TEXTURE_GEN_Q: u32 = 0xc63;
    pub const TEXTURE_GEN_MODE: u32 = 0x2500;
    pub const OBJECT_PLANE: u32 = 0x2501;
    pub const EYE_PLANE: u32 = 0x2502;
    /// The point and line distance planes of the texgen extension.
    pub const DISTANCE_PLANE_FIRST: u32 = 0x81f4;
    pub const DISTANCE_PLANE_LAST: u32 = 0x81f7;
    /// The generation modes of the point and line distance planes.
    pub const DISTANCE_MODE_FIRST: u32 = 0x81f0;
    pub const DISTANCE_MODE_LAST: u32 = 0x81f3;
    pub const CLIP_PLANES: usize = 6;
    // Capabilities and clear bits of the fragment operations.
    pub const STENCIL_BUFFER_BIT: u32 = 0x400;
    pub const ACCUM_BUFFER_BIT: u32 = 0x200;
    // The multisample enables.
    pub const MULTISAMPLE: u32 = 0x809d;
    pub const SAMPLE_ALPHA_TO_MASK: u32 = 0x809e;
    pub const SAMPLE_ALPHA_TO_ONE: u32 = 0x809f;
    pub const SAMPLE_MASK: u32 = 0x80a0;
    pub const ALPHA_TEST: u32 = 0xbc0;
    pub const LINE_STIPPLE: u32 = 0xb24;
    pub const POLYGON_STIPPLE: u32 = 0xb42;
    pub const BLEND: u32 = 0xbe2;
    pub const STENCIL_TEST: u32 = 0xb90;
    pub const COLOR_LOGIC_OP: u32 = 0xbf2;
    pub const DITHER: u32 = 0xbd0;
    pub const POLYGON_OFFSET_FILL: u32 = 0x8037;
    // Fog: the capability, the parameter names the setter carries (the same numbers the
    // generic state query lists) and the three modes.
    // The polygon modes a face can be drawn in.
    pub const POINT: u32 = 0x1b00;
    pub const LINE: u32 = 0x1b01;
    pub const FILL: u32 = 0x1b02;
    // The smoothing enables.
    pub const POINT_SMOOTH: u32 = 0xb10;
    pub const LINE_SMOOTH: u32 = 0xb20;
    pub const POLYGON_SMOOTH: u32 = 0xb41;
    pub const FOG: u32 = 0xb60;
    /// The seven names the fog setter accepts (the index and the offset extension are kept by the pipe,
    /// and not modelled).
    pub const FOG_INDEX: u32 = 0xb61;
    pub const FOG_OFFSET: u32 = 0x8199;
    pub const FOG_DENSITY: u32 = 0xb62;
    pub const FOG_START: u32 = 0xb63;
    pub const FOG_END: u32 = 0xb64;
    pub const FOG_MODE: u32 = 0xb65;
    pub const FOG_COLOR: u32 = 0xb66;
    pub const LINEAR: u32 = 0x2601;
    pub const EXP: u32 = 0x800;
    pub const EXP2: u32 = 0x801;
    /// The fog function table extension's mode, accepted and not modelled.
    pub const FOG_FUNC: u32 = 0x812a;
    // Stencil operations.
    pub const KEEP: u32 = 0x1e00;
    pub const REPLACE: u32 = 0x1e01;
    pub const INCR: u32 = 0x1e02;
    pub const DECR: u32 = 0x1e03;
    pub const INVERT: u32 = 0x150a;
    pub const ZERO: u32 = 0;
    // Blend factors (`ZERO` and `ONE` are 0 and 1).
    pub const ONE: u32 = 1;
    pub const SRC_COLOR: u32 = 0x300;
    pub const ONE_MINUS_SRC_COLOR: u32 = 0x301;
    pub const SRC_ALPHA: u32 = 0x302;
    pub const ONE_MINUS_SRC_ALPHA: u32 = 0x303;
    pub const DST_ALPHA: u32 = 0x304;
    pub const ONE_MINUS_DST_ALPHA: u32 = 0x305;
    pub const DST_COLOR: u32 = 0x306;
    pub const ONE_MINUS_DST_COLOR: u32 = 0x307;
    pub const SRC_ALPHA_SATURATE: u32 = 0x308;
    pub const CONSTANT_COLOR: u32 = 0x8001;
    pub const ONE_MINUS_CONSTANT_COLOR: u32 = 0x8002;
    pub const CONSTANT_ALPHA: u32 = 0x8003;
    pub const ONE_MINUS_CONSTANT_ALPHA: u32 = 0x8004;
    /// The logic operations are `LOGIC_OP_CLEAR + n` for n = 0 to 15, in the order
    /// the X server numbers them (clear, and, and-reverse, copy, ...).
    pub const LOGIC_OP_CLEAR: u32 = 0x1500;
    // Matrix modes.
    pub const MODELVIEW: u32 = 0x1700;
    pub const PROJECTION: u32 = 0x1701;
    pub const TEXTURE: u32 = 0x1702;
    // Light and material parameters.
    pub const AMBIENT: u32 = 0x1200;
    pub const DIFFUSE: u32 = 0x1201;
    pub const SPECULAR: u32 = 0x1202;
    pub const POSITION: u32 = 0x1203;
    pub const SPOT_DIRECTION: u32 = 0x1204;
    pub const SPOT_EXPONENT: u32 = 0x1205;
    pub const SPOT_CUTOFF: u32 = 0x1206;
    pub const CONSTANT_ATTENUATION: u32 = 0x1207;
    pub const LINEAR_ATTENUATION: u32 = 0x1208;
    pub const QUADRATIC_ATTENUATION: u32 = 0x1209;
    pub const EMISSION: u32 = 0x1600;
    pub const SHININESS: u32 = 0x1601;
    pub const AMBIENT_AND_DIFFUSE: u32 = 0x1602;
}

// The X graphics-context load (token WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, thirteen
// words), by word. The token is routed in hardware; the engine program keeps the first four
// words in its own memory and writes words 4 to 12, one each, to nine raster registers, under
// the masks `GC_WORD_MASKS` (read from the instruction operands: the pairing of word to
// register and the masks are established). What the engine does with word 4 in its special
// register 0x23 is unknown. A zero in the background high word makes the engine skip the
// raster-register part of the load.
pub const GC_WORDS: usize = 13;
/// Foreground colour, high half.
pub const GC_FOREGROUND_HIGH: usize = 0;
/// Foreground colour, low half.
pub const GC_FOREGROUND_LOW: usize = 1;
/// Background colour, high half.
pub const GC_BACKGROUND_HIGH: usize = 2;
/// Background colour, low half.
pub const GC_BACKGROUND_LOW: usize = 3;
/// Plane write mask, the red and green 12-bit fields: raster register 0x1c14 (and special register 0x23).
pub const GC_WRITE_MASK_FIRST: usize = 4;
/// Plane write mask, the blue and alpha fields: raster register 0x1c15.
pub const GC_WRITE_MASK_SECOND: usize = 5;
/// The colour format (bits 5:2) and read format (bits 13:11) of the frame-buffer configuration, raster
/// register 0x1c00. The pixel size and the four write enables are not written.
pub const GC_FRAMEBUFFER_CONFIG: usize = 6;
/// The draw buffer (bits 3:0), read-back format (17:14) and read base (23:18) of the operation register, 0x1c02.
pub const GC_OPERATION: usize = 7;
/// The colour base address only, the first of the six 6-bit fields of raster register 0x1c0b.
pub const GC_FRAMEBUFFER_BASES: usize = 8;
/// The logic-operation code (bits 4:1) only of raster register 0x1c0d: no blend field, no enable bit.
pub const GC_BLEND_AND_LOGIC: usize = 9;
/// The bus pixel-format code (5 bits), raster register 0x3ff8.
pub const GC_PIXEL_PACK_CODE: usize = 10;
/// The luminance-alpha colour flag for the texture filter, raster register 0x140f.
pub const GC_TEXTURE_FILTER_COLOUR_FLAG: usize = 11;
/// The luminance-alpha pixel flag for the texture filter, raster register 0x1410.
pub const GC_TEXTURE_FILTER_PIXEL_FLAG: usize = 12;
/// The masks the engine applies to words 4 to 12 (established).
pub const GC_WORD_MASKS: [u32; 9] = [0xff_ffff, 0xff_ffff, 0x383c, 0xff_c00f, 0x3f, 0x1e, 0x1f, 0x1, 0x1];

// The X pixel path's seven-word set-up (token WINDOW_SYSTEM_PIXEL_PATH_SET_UP), by
// word. Word 0's low four bits are the transfer kind (0: frame buffer only;
// nonzero also sets up the texture-filter chip; 2 also loads the host-interface
// packer from word 2) and bit 4 the direct-write flag, kept until the finish. Clear,
// the write's data words are counted through the command stream; set, they are not,
// so the rows come by DMA. That the flag exists, that the bitmap loaders send it
// clear and that the count is off when it is set are established; that set means
// DMA is inferred from the recording. Word 1 names the engine's pixel-output module (4: bitmap
// and glyph output with a one-bit-to-pixel conversion; 5: opaque bitmap draw with a
// 16- or 32-bit conversion chosen by word 2); the model also reads word 1 as the
// source pixel packing. Words 3 and 6 are passed to the engine and distributor
// set-up and the X module sends 0: their meaning is unknown, as are the transfer
// kinds other than 0 and 2 and the output modules other than 4 and 5.
pub const PIXEL_SETUP_WORDS: usize = 7;
pub const PIXEL_SETUP_TRANSFER_KIND: usize = 0;
pub const PIXEL_SETUP_OUTPUT_MODULE: usize = 1;
pub const PIXEL_SETUP_CONVERSION_OR_PACKER: usize = 2;
pub const PIXEL_SETUP_ENGINE_VALUE: usize = 3;
pub const PIXEL_SETUP_WIDTH: usize = 4;
pub const PIXEL_SETUP_HEIGHT: usize = 5;
pub const PIXEL_SETUP_DISTRIBUTOR_VALUE: usize = 6;
pub const PIXEL_SETUP_TRANSFER_KIND_MASK: u32 = 0xf;
pub const PIXEL_SETUP_DIRECT_WRITE_FLAG: u32 = 0x10;
pub const PIXEL_OUTPUT_MODULE_BITMAP: u32 = 4;
pub const PIXEL_OUTPUT_MODULE_OPAQUE_BITMAP: u32 = 5;
// The pixel transfer's payloads (established): a read's x and y, measured from the
// bottom; a copy's source x and y and destination x and y; a finish's one word, zero.
pub const PIXEL_READ_WORDS: usize = 2;
pub const PIXEL_COPY_WORDS: usize = 4;

/// Entries in the front end's context table: 64, and one more for its transient entry.
pub const CONTEXT_SLOTS: usize = 65;

// The host command tokens, by the role the definitions table gives each. The
// grade is the table's: established unless the comment says otherwise.

/// Raw data word.
pub const RAW_DATA_WORD: u32 = 0x000;
/// Configure the frame-buffer layout (colour, depth, stencil, accumulation, multisample) (inferred).
pub const CONFIGURE_FRAME_BUFFER_LAYOUT: u32 = 0x001;
/// Flush and rebuild the hardware token map for packed-vertex mode (inferred).
pub const FLUSH_AND_REBUILD_HARDWARE_TOKEN_MAP_FOR_PACKED_VERTEX: u32 = 0x002;
/// Begin primitive: rectangle (shares the quad entry).
pub const BEGIN_PRIMITIVE_RECTANGLE: u32 = 0x003;
/// Configure the read-side frame-buffer layout (inferred).
pub const CONFIGURE_READ_SIDE_FRAME_BUFFER_LAYOUT: u32 = 0x004;
/// Draw a bitmap from a display list (inferred).
pub const DRAW_BITMAP_FROM_DISPLAY_LIST: u32 = 0x005;
/// Draw a bitmap (inferred).
pub const DRAW_BITMAP: u32 = 0x006;
/// Set a light parameter (float form) (inferred).
pub const SET_LIGHT_PARAMETER_FLOAT_FORM: u32 = 0x007;
/// Set a light parameter (integer form) (inferred).
pub const SET_LIGHT_PARAMETER_INTEGER_FORM: u32 = 0x008;
/// Set a lighting-model parameter (float form) (inferred).
pub const SET_LIGHTING_MODEL_PARAMETER_FLOAT_FORM: u32 = 0x009;
/// Set a lighting-model parameter (integer form) (inferred).
pub const SET_LIGHTING_MODEL_PARAMETER_INTEGER_FORM: u32 = 0x00a;
/// Load the polygon stipple pattern (inferred).
pub const LOAD_POLYGON_STIPPLE_PATTERN: u32 = 0x00b;
/// Set the scissor box (inferred).
pub const SET_SCISSOR_BOX: u32 = 0x00c;
/// Texture parameter, float.
pub const TEXTURE_PARAMETER_FLOAT: u32 = 0x00d;
/// Texture parameter, integer.
pub const TEXTURE_PARAMETER_INTEGER: u32 = 0x00e;
/// 1D texture image load.
pub const T_1D_TEXTURE_IMAGE_LOAD: u32 = 0x00f;
/// 2D texture image load.
pub const T_2D_TEXTURE_IMAGE_LOAD: u32 = 0x010;
/// Set a texture-environment parameter (float form) (inferred).
pub const SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM: u32 = 0x011;
/// Set a texture-environment parameter (integer form) (inferred).
pub const SET_TEXTURE_ENVIRONMENT_PARAMETER_INTEGER_FORM: u32 = 0x012;
/// Texture-coordinate generation, double form (unknown).
pub const TEXTURE_COORDINATE_GENERATION_DOUBLE_FORM: u32 = 0x013;
/// Texture-coordinate generation, float form (inferred).
pub const TEXTURE_COORDINATE_GENERATION_FLOAT_FORM: u32 = 0x014;
/// Texture-coordinate generation, integer form (unknown).
pub const TEXTURE_COORDINATE_GENERATION_INTEGER_FORM: u32 = 0x015;
/// Bind a texture by old-style binding (unknown).
pub const BIND_TEXTURE_BY_OLD_STYLE_BINDING: u32 = 0x016;
/// Select the feedback record type (inferred).
pub const SELECT_FEEDBACK_RECORD_TYPE: u32 = 0x017;
/// Select-buffer registration.
pub const SELECT_BUFFER_REGISTRATION: u32 = 0x018;
/// Switch between render, select and feedback modes (inferred).
pub const SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES: u32 = 0x019;
/// Clear the selected buffers (inferred).
pub const CLEAR_SELECTED_BUFFERS: u32 = 0x01a;
/// Set the accumulation-buffer clear value (inferred).
pub const SET_ACCUMULATION_BUFFER_CLEAR_VALUE: u32 = 0x01b;
/// Set the colour-index clear value (inferred).
pub const SET_COLOUR_INDEX_CLEAR_VALUE: u32 = 0x01c;
/// Set the colour clear value (inferred).
pub const SET_COLOUR_CLEAR_VALUE: u32 = 0x01d;
/// Set the stencil clear value (inferred).
pub const SET_STENCIL_CLEAR_VALUE: u32 = 0x01e;
/// Set the depth clear value (inferred).
pub const SET_DEPTH_CLEAR_VALUE: u32 = 0x01f;
/// Set the stencil write mask (inferred).
pub const SET_STENCIL_WRITE_MASK: u32 = 0x020;
/// Set the colour write mask (inferred).
pub const SET_COLOUR_WRITE_MASK: u32 = 0x021;
/// Set the depth write mask (inferred).
pub const SET_DEPTH_WRITE_MASK: u32 = 0x022;
/// Set the colour-index write mask (inferred).
pub const SET_COLOUR_INDEX_WRITE_MASK: u32 = 0x023;
/// Enable a capability (inferred).
pub const ENABLE_CAPABILITY: u32 = 0x024;
/// Disable a capability (inferred).
pub const DISABLE_CAPABILITY: u32 = 0x025;
/// Evaluate a 1D grid strip (inferred).
pub const EVALUATE_1D_GRID_STRIP: u32 = 0x026;
/// Evaluate a 2D grid strip (inferred).
pub const EVALUATE_2D_GRID_STRIP: u32 = 0x027;
/// Set up a 1D evaluator grid.
pub const SET_UP_1D_EVALUATOR_GRID: u32 = 0x028;
/// Set up a 2D evaluator grid.
pub const SET_UP_2D_EVALUATOR_GRID: u32 = 0x029;
/// Load 1D evaluator control points (inferred).
pub const LOAD_1D_EVALUATOR_CONTROL_POINTS: u32 = 0x02a;
/// Load 2D evaluator control points (inferred).
pub const LOAD_2D_EVALUATOR_CONTROL_POINTS: u32 = 0x02b;
/// Set up a 1D evaluator map specification (inferred).
pub const SET_UP_1D_EVALUATOR_MAP_SPECIFICATION: u32 = 0x02c;
/// Set up a 2D evaluator map specification (inferred).
pub const SET_UP_2D_EVALUATOR_MAP_SPECIFICATION: u32 = 0x02d;
/// Set the logic operation (inferred).
pub const SET_LOGIC_OPERATION: u32 = 0x02e;
/// Set the pixel zoom factors (inferred).
pub const SET_PIXEL_ZOOM_FACTORS: u32 = 0x02f;
/// Set a pixel-transfer parameter (float form) (inferred).
pub const SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM: u32 = 0x030;
/// Set a pixel-transfer parameter (integer form).
pub const SET_PIXEL_TRANSFER_PARAMETER_INTEGER_FORM: u32 = 0x031;
/// Set a pixel-store parameter (float form).
pub const SET_PIXEL_STORE_PARAMETER_FLOAT_FORM: u32 = 0x032;
/// Set a pixel-store parameter (integer form) (inferred).
pub const SET_PIXEL_STORE_PARAMETER_INTEGER_FORM: u32 = 0x033;
/// Load a pixel map (float values) (unknown).
pub const LOAD_PIXEL_MAP_FLOAT_VALUES: u32 = 0x034;
/// Load a pixel map (unsigned 32-bit values) (unknown).
pub const LOAD_PIXEL_MAP_UNSIGNED_32_BIT_VALUES: u32 = 0x035;
/// Load a pixel map (unsigned 16-bit values) (unknown).
pub const LOAD_PIXEL_MAP_UNSIGNED_16_BIT_VALUES: u32 = 0x036;
/// Select the read buffer (inferred).
pub const SELECT_READ_BUFFER: u32 = 0x037;
/// Copy a pixel rectangle within the frame buffer (inferred).
pub const COPY_PIXEL_RECTANGLE_WITHIN_FRAME_BUFFER: u32 = 0x038;
/// Read a pixel rectangle back to the host (inferred).
pub const READ_PIXEL_RECTANGLE_BACK_TO_HOST: u32 = 0x039;
/// Draw a pixel rectangle from host data (inferred).
pub const DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA: u32 = 0x03a;
/// Query current vertex attributes (inferred).
pub const QUERY_CURRENT_VERTEX_ATTRIBUTES: u32 = 0x03b;
/// Query a clip plane (inferred).
pub const QUERY_CLIP_PLANE: u32 = 0x03c;
/// Query the raster position state (inferred).
pub const QUERY_RASTER_POSITION_STATE: u32 = 0x03d;
/// Query the pending error (inferred).
pub const QUERY_PENDING_ERROR: u32 = 0x03e;
/// Query state (float).
pub const QUERY_STATE_FLOAT: u32 = 0x03f;
/// Query state (integer).
pub const QUERY_STATE_INTEGER: u32 = 0x040;
/// Query a light (float) (inferred).
pub const QUERY_LIGHT_FLOAT: u32 = 0x041;
/// Query a light (integer).
pub const QUERY_LIGHT_INTEGER: u32 = 0x042;
/// Query an evaluator map (double).
pub const QUERY_EVALUATOR_MAP_DOUBLE: u32 = 0x043;
/// Query an evaluator map (float).
pub const QUERY_EVALUATOR_MAP_FLOAT: u32 = 0x044;
/// Query an evaluator map (integer).
pub const QUERY_EVALUATOR_MAP_INTEGER: u32 = 0x045;
/// Query a material (float) (inferred).
pub const QUERY_MATERIAL_FLOAT: u32 = 0x046;
/// Query a material (integer).
pub const QUERY_MATERIAL_INTEGER: u32 = 0x047;
/// Query a pixel map (float) (inferred).
pub const QUERY_PIXEL_MAP_FLOAT: u32 = 0x048;
/// Query a pixel map (unsigned 32-bit).
pub const QUERY_PIXEL_MAP_UNSIGNED_32_BIT: u32 = 0x049;
/// Query a pixel map (unsigned 16-bit).
pub const QUERY_PIXEL_MAP_UNSIGNED_16_BIT: u32 = 0x04a;
/// Query the polygon stipple (inferred).
pub const QUERY_POLYGON_STIPPLE: u32 = 0x04b;
/// Query state (generic) (inferred).
pub const QUERY_STATE_GENERIC: u32 = 0x04c;
/// Query texture environment (float) (inferred).
pub const QUERY_TEXTURE_ENVIRONMENT_FLOAT: u32 = 0x04d;
/// Query texture environment (integer).
pub const QUERY_TEXTURE_ENVIRONMENT_INTEGER: u32 = 0x04e;
/// Query texture generation (double).
pub const QUERY_TEXTURE_GENERATION_DOUBLE: u32 = 0x04f;
/// Query texture generation (float) (inferred).
pub const QUERY_TEXTURE_GENERATION_FLOAT: u32 = 0x050;
/// Query texture generation (integer).
pub const QUERY_TEXTURE_GENERATION_INTEGER: u32 = 0x051;
/// Query a texture image (inferred).
pub const QUERY_TEXTURE_IMAGE: u32 = 0x052;
/// Query a texture parameter (float).
pub const QUERY_TEXTURE_PARAMETER_FLOAT: u32 = 0x053;
/// Query a texture parameter (integer).
pub const QUERY_TEXTURE_PARAMETER_INTEGER: u32 = 0x054;
/// Query a texture-level parameter (float).
pub const QUERY_TEXTURE_LEVEL_PARAMETER_FLOAT: u32 = 0x055;
/// Query a texture-level parameter (integer).
pub const QUERY_TEXTURE_LEVEL_PARAMETER_INTEGER: u32 = 0x056;
/// Begin primitive: points.
pub const BEGIN_PRIMITIVE_POINTS: u32 = 0x057;
/// Begin primitive: independent lines.
pub const BEGIN_PRIMITIVE_INDEPENDENT_LINES: u32 = 0x058;
/// Begin primitive: line loop.
pub const BEGIN_PRIMITIVE_LINE_LOOP: u32 = 0x059;
/// Begin primitive: line strip.
pub const BEGIN_PRIMITIVE_LINE_STRIP: u32 = 0x05a;
/// Begin primitive: independent triangles.
pub const BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES: u32 = 0x05b;
/// Begin primitive: triangle strip.
pub const BEGIN_PRIMITIVE_TRIANGLE_STRIP: u32 = 0x05c;
/// Begin primitive: triangle fan.
pub const BEGIN_PRIMITIVE_TRIANGLE_FAN: u32 = 0x05d;
/// Begin primitive: independent quads.
pub const BEGIN_PRIMITIVE_INDEPENDENT_QUADS: u32 = 0x05e;
/// Begin primitive: quad strip.
pub const BEGIN_PRIMITIVE_QUAD_STRIP: u32 = 0x05f;
/// Begin primitive: polygon.
pub const BEGIN_PRIMITIVE_POLYGON: u32 = 0x060;
/// Set current colour, 3 components, signed byte (inferred).
pub const SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_BYTE: u32 = 0x061;
/// Set current colour, 3 components, double (inferred).
pub const SET_CURRENT_COLOUR_3_COMPONENTS_DOUBLE: u32 = 0x062;
/// Set current colour, 3 components, single float (inferred).
pub const SET_CURRENT_COLOUR_3_COMPONENTS_SINGLE_FLOAT: u32 = 0x063;
/// Set current colour, 3 components, signed 32-bit integer (inferred).
pub const SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_32_BIT_INTEGER: u32 = 0x064;
/// Set current colour, 3 components, signed short (inferred).
pub const SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_SHORT: u32 = 0x065;
/// Set current colour, 3 components, unsigned byte (inferred).
pub const SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_BYTE: u32 = 0x066;
/// Set current colour, 3 components, unsigned 32-bit integer (inferred).
pub const SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_32_BIT_INTEGER: u32 = 0x067;
/// Set current colour, 3 components, unsigned short (inferred).
pub const SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_SHORT: u32 = 0x068;
/// Set current colour, 4 components, signed byte (inferred).
pub const SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_BYTE: u32 = 0x069;
/// Set current colour, 4 components, double (inferred).
pub const SET_CURRENT_COLOUR_4_COMPONENTS_DOUBLE: u32 = 0x06a;
/// Set current colour, 4 components, single float (inferred).
pub const SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT: u32 = 0x06b;
/// Set current colour, 4 components, signed 32-bit integer (inferred).
pub const SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_32_BIT_INTEGER: u32 = 0x06c;
/// Set current colour, 4 components, signed short (inferred).
pub const SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_SHORT: u32 = 0x06d;
/// Set current colour, 4 components, unsigned byte (inferred).
pub const SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_BYTE: u32 = 0x06e;
/// Set current colour, 4 components, unsigned 32-bit integer (inferred).
pub const SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_32_BIT_INTEGER: u32 = 0x06f;
/// Set current colour, 4 components, unsigned short (inferred).
pub const SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_SHORT: u32 = 0x070;
/// End the open primitive.
pub const END_OPEN_PRIMITIVE: u32 = 0x071;
/// Set current colour index, double (inferred).
pub const SET_CURRENT_COLOUR_INDEX_DOUBLE: u32 = 0x072;
/// Set current colour index, single float (inferred).
pub const SET_CURRENT_COLOUR_INDEX_SINGLE_FLOAT: u32 = 0x073;
/// Set current colour index, signed 32-bit integer (inferred).
pub const SET_CURRENT_COLOUR_INDEX_SIGNED_32_BIT_INTEGER: u32 = 0x074;
/// Set current colour index, signed short (inferred).
pub const SET_CURRENT_COLOUR_INDEX_SIGNED_SHORT: u32 = 0x075;
/// Set current normal, 3 components, signed byte (inferred).
pub const SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_BYTE: u32 = 0x076;
/// Set current normal, 3 components, double (inferred).
pub const SET_CURRENT_NORMAL_3_COMPONENTS_DOUBLE: u32 = 0x077;
/// Set current normal, 3 components, single float (inferred).
pub const SET_CURRENT_NORMAL_3_COMPONENTS_SINGLE_FLOAT: u32 = 0x078;
/// Set current normal, 3 components, signed 32-bit integer (inferred).
pub const SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_32_BIT_INTEGER: u32 = 0x079;
/// Set current normal, 3 components, signed short (inferred).
pub const SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_SHORT: u32 = 0x07a;
/// Set current texture coordinate, 1 components, double (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_DOUBLE: u32 = 0x07b;
/// Set current texture coordinate, 1 components, single float (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT: u32 = 0x07c;
/// Set current texture coordinate, 1 components, signed 32-bit integer (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SIGNED_32_BIT: u32 = 0x07d;
/// Set current texture coordinate, 1 components, signed short (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SIGNED_SHORT: u32 = 0x07e;
/// Set current texture coordinate, 2 components, double (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_DOUBLE: u32 = 0x07f;
/// Set current texture coordinate, 2 components, single float (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT: u32 = 0x080;
/// Set current texture coordinate, 2 components, signed 32-bit integer (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_32_BIT: u32 = 0x081;
/// Set current texture coordinate, 2 components, signed short (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_SHORT: u32 = 0x082;
/// Set current texture coordinate, 3 components, double (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_DOUBLE: u32 = 0x083;
/// Set current texture coordinate, 3 components, single float (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT: u32 = 0x084;
/// Set current texture coordinate, 3 components, signed 32-bit integer (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SIGNED_32_BIT: u32 = 0x085;
/// Set current texture coordinate, 3 components, signed short (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SIGNED_SHORT: u32 = 0x086;
/// Set current texture coordinate, 4 components, double (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_DOUBLE: u32 = 0x087;
/// Set current texture coordinate, 4 components, single float (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT: u32 = 0x088;
/// Set current texture coordinate, 4 components, signed 32-bit integer (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SIGNED_32_BIT: u32 = 0x089;
/// Set current texture coordinate, 4 components, signed short (inferred).
pub const SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SIGNED_SHORT: u32 = 0x08a;
/// Submit a vertex, 2 components, double (inferred).
pub const SUBMIT_VERTEX_2_COMPONENTS_DOUBLE: u32 = 0x08b;
/// Submit a vertex, 2 components, single float (inferred).
pub const SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT: u32 = 0x08c;
/// Submit a vertex, 2 components, signed 32-bit integer (inferred).
pub const SUBMIT_VERTEX_2_COMPONENTS_SIGNED_32_BIT_INTEGER: u32 = 0x08d;
/// Submit a vertex, 2 components, signed short (inferred).
pub const SUBMIT_VERTEX_2_COMPONENTS_SIGNED_SHORT: u32 = 0x08e;
/// Submit a vertex, 3 components, double (inferred).
pub const SUBMIT_VERTEX_3_COMPONENTS_DOUBLE: u32 = 0x08f;
/// Submit a vertex, 3 components, single float (inferred).
pub const SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT: u32 = 0x090;
/// Submit a vertex, 3 components, signed 32-bit integer (inferred).
pub const SUBMIT_VERTEX_3_COMPONENTS_SIGNED_32_BIT_INTEGER: u32 = 0x091;
/// Submit a vertex, 3 components, signed short (inferred).
pub const SUBMIT_VERTEX_3_COMPONENTS_SIGNED_SHORT: u32 = 0x092;
/// Submit a vertex, 4 components, double (inferred).
pub const SUBMIT_VERTEX_4_COMPONENTS_DOUBLE: u32 = 0x093;
/// Submit a vertex, 4 components, single float (inferred).
pub const SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT: u32 = 0x094;
/// Submit a vertex, 4 components, signed 32-bit integer (inferred).
pub const SUBMIT_VERTEX_4_COMPONENTS_SIGNED_32_BIT_INTEGER: u32 = 0x095;
/// Submit a vertex, 4 components, signed short (inferred).
pub const SUBMIT_VERTEX_4_COMPONENTS_SIGNED_SHORT: u32 = 0x096;
/// Set a material parameter (float form) (inferred).
pub const SET_MATERIAL_PARAMETER_FLOAT_FORM: u32 = 0x097;
/// Continuation entry of the float material parameter (inferred).
pub const CONTINUATION_ENTRY_OF_FLOAT_MATERIAL_PARAMETER: u32 = 0x098;
/// Set a material parameter (integer form) (inferred).
pub const SET_MATERIAL_PARAMETER_INTEGER_FORM: u32 = 0x099;
/// Continuation entry of the integer material parameter (inferred).
pub const CONTINUATION_ENTRY_OF_INTEGER_MATERIAL_PARAMETER: u32 = 0x09a;
/// Evaluator coordinate, 1D, double (inferred).
pub const EVALUATOR_COORDINATE_1D_DOUBLE: u32 = 0x09b;
/// Continuation entry of the evaluator eval coord1d (inferred).
pub const CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD1D: u32 = 0x09c;
/// Evaluator coordinate, 1D, single float (inferred).
pub const EVALUATOR_COORDINATE_1D_SINGLE_FLOAT: u32 = 0x09d;
/// Continuation entry of the evaluator eval coord1f (inferred).
pub const CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD1F: u32 = 0x09e;
/// Evaluator coordinate, 2D, double (inferred).
pub const EVALUATOR_COORDINATE_2D_DOUBLE: u32 = 0x09f;
/// Continuation entry of the evaluator eval coord2d (inferred).
pub const CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD2D: u32 = 0x0a0;
/// Evaluator coordinate, 2D, single float (inferred).
pub const EVALUATOR_COORDINATE_2D_SINGLE_FLOAT: u32 = 0x0a1;
/// Continuation entry of the evaluator eval coord2f (inferred).
pub const CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD2F: u32 = 0x0a2;
/// Evaluator grid point, 1D (inferred).
pub const EVALUATOR_GRID_POINT_1D: u32 = 0x0a3;
/// Continuation entry of the evaluator eval point1 (inferred).
pub const CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_POINT1: u32 = 0x0a4;
/// Evaluator grid point, 2D (inferred).
pub const EVALUATOR_GRID_POINT_2D: u32 = 0x0a5;
/// Continuation entry of the evaluator eval point2 (inferred).
pub const CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_POINT2: u32 = 0x0a6;
/// Set the current edge flag (inferred).
pub const SET_CURRENT_EDGE_FLAG: u32 = 0x0a7;
/// Set the raster position, 2 components, double (inferred).
pub const SET_RASTER_POSITION_2_COMPONENTS_DOUBLE: u32 = 0x0a8;
/// Set the raster position, 2 components, float (inferred).
pub const SET_RASTER_POSITION_2_COMPONENTS_FLOAT: u32 = 0x0a9;
/// Set the raster position, 2 components, 32-bit integer (inferred).
pub const SET_RASTER_POSITION_2_COMPONENTS_32_BIT_INTEGER: u32 = 0x0aa;
/// Set the raster position, 2 components, short (inferred).
pub const SET_RASTER_POSITION_2_COMPONENTS_SHORT: u32 = 0x0ab;
/// Set the raster position, 3 components, double (inferred).
pub const SET_RASTER_POSITION_3_COMPONENTS_DOUBLE: u32 = 0x0ac;
/// Set the raster position, 3 components, float (inferred).
pub const SET_RASTER_POSITION_3_COMPONENTS_FLOAT: u32 = 0x0ad;
/// Set the raster position, 3 components, 32-bit integer (inferred).
pub const SET_RASTER_POSITION_3_COMPONENTS_32_BIT_INTEGER: u32 = 0x0ae;
/// Set the raster position, 3 components, short (inferred).
pub const SET_RASTER_POSITION_3_COMPONENTS_SHORT: u32 = 0x0af;
/// Set the raster position, 4 components, double (inferred).
pub const SET_RASTER_POSITION_4_COMPONENTS_DOUBLE: u32 = 0x0b0;
/// Set the raster position, 4 components, float (inferred).
pub const SET_RASTER_POSITION_4_COMPONENTS_FLOAT: u32 = 0x0b1;
/// Set the raster position, 4 components, 32-bit integer (inferred).
pub const SET_RASTER_POSITION_4_COMPONENTS_32_BIT_INTEGER: u32 = 0x0b2;
/// Set the raster position, 4 components, short (inferred).
pub const SET_RASTER_POSITION_4_COMPONENTS_SHORT: u32 = 0x0b3;
/// Set a user clip plane (inferred).
pub const SET_USER_CLIP_PLANE: u32 = 0x0b4;
/// Set colour-material tracking (inferred).
pub const SET_COLOUR_MATERIAL_TRACKING: u32 = 0x0b5;
/// Set face culling (inferred).
pub const SET_FACE_CULLING: u32 = 0x0b6;
/// Set a fog parameter (float form).
pub const SET_FOG_PARAMETER_FLOAT_FORM: u32 = 0x0b7;
/// Set a fog parameter (integer form) (inferred).
pub const SET_FOG_PARAMETER_INTEGER_FORM: u32 = 0x0b8;
/// Continuation entry of the integer fog parameter (inferred).
pub const CONTINUATION_ENTRY_OF_INTEGER_FOG_PARAMETER: u32 = 0x0b9;
/// Set front-face winding (inferred).
pub const SET_FRONT_FACE_WINDING: u32 = 0x0ba;
/// Set a quality hint (inferred).
pub const SET_QUALITY_HINT: u32 = 0x0bb;
/// Set the line stipple (inferred).
pub const SET_LINE_STIPPLE: u32 = 0x0bc;
/// Set the line width (inferred).
pub const SET_LINE_WIDTH: u32 = 0x0bd;
/// Set the point size (inferred).
pub const SET_POINT_SIZE: u32 = 0x0be;
/// Set the polygon fill mode (inferred).
pub const SET_POLYGON_FILL_MODE: u32 = 0x0bf;
/// Set flat or smooth shading (inferred).
pub const SET_FLAT_OR_SMOOTH_SHADING: u32 = 0x0c0;
/// Insert a feedback pass-through marker (inferred).
pub const INSERT_FEEDBACK_PASS_THROUGH_MARKER: u32 = 0x0c1;
/// Pop a selection name (inferred).
pub const POP_SELECTION_NAME: u32 = 0x0c2;
/// Push a selection name (inferred).
pub const PUSH_SELECTION_NAME: u32 = 0x0c3;
/// Select the draw buffer (inferred).
pub const SELECT_DRAW_BUFFER: u32 = 0x0c4;
/// Accumulation-buffer operation (inferred).
pub const ACCUMULATION_BUFFER_OPERATION: u32 = 0x0c5;
/// Finish: drain the pipe and signal completion (inferred).
pub const FINISH_DRAIN_PIPE_AND_SIGNAL_COMPLETION: u32 = 0x0c6;
/// Flush the vertex pipeline (inferred).
pub const FLUSH_VERTEX_PIPELINE: u32 = 0x0c7;
/// Pop the attribute stack (inferred).
pub const POP_ATTRIBUTE_STACK: u32 = 0x0c8;
/// Push the attribute stack (inferred).
pub const PUSH_ATTRIBUTE_STACK: u32 = 0x0c9;
/// Set the alpha test (inferred).
pub const SET_ALPHA_TEST: u32 = 0x0ca;
/// Set the blend factors (inferred).
pub const SET_BLEND_FACTORS: u32 = 0x0cb;
/// Set the stencil test (inferred).
pub const SET_STENCIL_TEST: u32 = 0x0cc;
/// Set the stencil operations (inferred).
pub const SET_STENCIL_OPERATIONS: u32 = 0x0cd;
/// Set the depth test (inferred).
pub const SET_DEPTH_TEST: u32 = 0x0ce;
/// Set the depth range (inferred).
pub const SET_DEPTH_RANGE: u32 = 0x0cf;
/// Select the current matrix stack (inferred).
pub const SELECT_CURRENT_MATRIX_STACK: u32 = 0x0d0;
/// Multiply the current matrix by a frustum.
pub const MULTIPLY_CURRENT_MATRIX_BY_FRUSTUM: u32 = 0x0d1;
/// Load the identity matrix.
pub const LOAD_IDENTITY_MATRIX: u32 = 0x0d2;
/// Load a matrix (16 floats).
pub const LOAD_MATRIX_16_FLOATS: u32 = 0x0d3;
/// Load a matrix (16 doubles).
pub const LOAD_MATRIX_16_DOUBLES: u32 = 0x0d4;
/// Multiply by a matrix (16 floats).
pub const MULTIPLY_BY_MATRIX_16_FLOATS: u32 = 0x0d5;
/// Multiply by a matrix (16 doubles).
pub const MULTIPLY_BY_MATRIX_16_DOUBLES: u32 = 0x0d6;
/// Multiply the current matrix by an orthographic projection.
pub const MULTIPLY_CURRENT_MATRIX_BY_ORTHOGRAPHIC_PROJECTION: u32 = 0x0d7;
/// Pop the matrix stack.
pub const POP_MATRIX_STACK: u32 = 0x0d8;
/// Push the matrix stack.
pub const PUSH_MATRIX_STACK: u32 = 0x0d9;
/// Rotate (doubles).
pub const ROTATE_DOUBLES: u32 = 0x0da;
/// Rotate (floats).
pub const ROTATE_FLOATS: u32 = 0x0db;
/// Scale (doubles).
pub const SCALE_DOUBLES: u32 = 0x0dc;
/// Scale (floats).
pub const SCALE_FLOATS: u32 = 0x0dd;
/// Translate (doubles).
pub const TRANSLATE_DOUBLES: u32 = 0x0de;
/// Translate (floats).
pub const TRANSLATE_FLOATS: u32 = 0x0df;
/// Copy engine work memory to control memory (no handler bound) (inferred).
pub const COPY_ENGINE_WORK_MEMORY_TO_CONTROL_MEMORY: u32 = 0x0e0;
/// Pass one command word through (null handler) (inferred).
pub const PASS_ONE_COMMAND_WORD_THROUGH: u32 = 0x0e1;
/// Write one raster-bus word (mode code, low half, high half).
pub const WRITE_ONE_RASTER_BUS_WORD: u32 = 0x0e2;
/// Set the viewport (inferred).
pub const SET_VIEWPORT: u32 = 0x0e3;
/// Record an error code (inferred).
pub const RECORD_ERROR_CODE: u32 = 0x0e4;
/// End-of-simulation report (simulation only) (inferred).
pub const END_OF_SIMULATION_REPORT: u32 = 0x0e5;
/// Set the constant blend colour (inferred).
pub const SET_CONSTANT_BLEND_COLOUR: u32 = 0x0e6;
/// Set the blend equation (unknown).
pub const SET_BLEND_EQUATION: u32 = 0x0e7;
/// Load a colour table (inferred).
pub const LOAD_COLOUR_TABLE: u32 = 0x0e8;
/// Copy a colour table from the frame buffer.
pub const COPY_COLOUR_TABLE_FROM_FRAME_BUFFER: u32 = 0x0e9;
/// Query a colour table (inferred).
pub const QUERY_COLOUR_TABLE: u32 = 0x0ea;
/// Set colour-table parameters (inferred).
pub const SET_COLOUR_TABLE_PARAMETERS: u32 = 0x0eb;
/// Query colour-table parameters (inferred).
pub const QUERY_COLOUR_TABLE_PARAMETERS: u32 = 0x0ec;
/// Load the post-texture scale and bias.
pub const LOAD_POST_TEXTURE_SCALE_AND_BIAS: u32 = 0x0ed;
/// Query the line-mode limit table (inferred).
pub const QUERY_LINE_MODE_LIMIT_TABLE: u32 = 0x0ee;
/// Load a 1D convolution filter (inferred).
pub const LOAD_1D_CONVOLUTION_FILTER: u32 = 0x0ef;
/// Load a 2D convolution filter (inferred).
pub const LOAD_2D_CONVOLUTION_FILTER: u32 = 0x0f0;
/// 3D convolution filter.
pub const T_3D_CONVOLUTION_FILTER: u32 = 0x0f1;
/// Copy a 1D convolution filter from the frame buffer (inferred).
pub const COPY_1D_CONVOLUTION_FILTER_FROM_FRAME_BUFFER: u32 = 0x0f2;
/// Copy a 2D convolution filter from the frame buffer (inferred).
pub const COPY_2D_CONVOLUTION_FILTER_FROM_FRAME_BUFFER: u32 = 0x0f3;
/// Query a convolution filter (inferred).
pub const QUERY_CONVOLUTION_FILTER: u32 = 0x0f4;
/// Load a separable 2D filter (inferred).
pub const LOAD_SEPARABLE_2D_FILTER: u32 = 0x0f5;
/// 3D separable filter.
pub const T_3D_SEPARABLE_FILTER: u32 = 0x0f6;
/// Query a separable filter (inferred).
pub const QUERY_SEPARABLE_FILTER: u32 = 0x0f7;
/// Set pixel-transform parameters (integer) (inferred).
pub const SET_PIXEL_TRANSFORM_PARAMETERS_INTEGER: u32 = 0x0f8;
/// Set pixel-transform parameters (float) (inferred).
pub const SET_PIXEL_TRANSFORM_PARAMETERS_FLOAT: u32 = 0x0f9;
/// Query pixel-transform parameters (integer) (inferred).
pub const QUERY_PIXEL_TRANSFORM_PARAMETERS_INTEGER: u32 = 0x0fa;
/// Query pixel-transform parameters (float).
pub const QUERY_PIXEL_TRANSFORM_PARAMETERS_FLOAT: u32 = 0x0fb;
/// Copy a 1D texture image from the frame buffer.
pub const COPY_1D_TEXTURE_IMAGE_FROM_FRAME_BUFFER: u32 = 0x0fc;
/// Copy a 2D texture image from the frame buffer (inferred).
pub const COPY_2D_TEXTURE_IMAGE_FROM_FRAME_BUFFER: u32 = 0x0fd;
/// Copy a 1D texture sub-image from the frame buffer.
pub const COPY_1D_TEXTURE_SUB_IMAGE_FROM_FRAME_BUFFER: u32 = 0x0fe;
/// Write-gather flush, first slot.
pub const WRITE_GATHER_FLUSH_FIRST_SLOT: u32 = 0x0ff;
/// Raw data word, second form.
pub const RAW_DATA_WORD_SECOND_FORM: u32 = 0x100;
/// Copy a 2D texture sub-image from the frame buffer.
pub const COPY_2D_TEXTURE_SUB_IMAGE_FROM_FRAME_BUFFER: u32 = 0x101;
/// Set up a histogram (inferred).
pub const SET_UP_HISTOGRAM: u32 = 0x102;
/// Reset a histogram (inferred).
pub const RESET_HISTOGRAM: u32 = 0x103;
/// Query a histogram (inferred).
pub const QUERY_HISTOGRAM: u32 = 0x104;
/// Query histogram parameters (inferred).
pub const QUERY_HISTOGRAM_PARAMETERS: u32 = 0x105;
/// Bind a linear texture (inferred).
pub const BIND_LINEAR_TEXTURE: u32 = 0x106;
/// Set up min/max tracking (inferred).
pub const SET_UP_MIN_MAX_TRACKING: u32 = 0x107;
/// Reset min/max tracking (inferred).
pub const RESET_MIN_MAX_TRACKING: u32 = 0x108;
/// Query min/max results (inferred).
pub const QUERY_MIN_MAX_RESULTS: u32 = 0x109;
/// Query min/max parameters (inferred).
pub const QUERY_MIN_MAX_PARAMETERS: u32 = 0x10a;
/// Save light attributes on the engine attribute stack (ARM-injected) (inferred).
pub const SAVE_LIGHT_ATTRIBUTES_ON_ENGINE_ATTRIBUTE_STACK: u32 = 0x10b;
/// Clear the selection name stack (inferred).
pub const CLEAR_SELECTION_NAME_STACK: u32 = 0x10c;
/// Replace the top selection name (inferred).
pub const REPLACE_TOP_SELECTION_NAME: u32 = 0x10d;
/// Set the current raster position (fifteen words) (inferred).
pub const SET_CURRENT_RASTER_POSITION: u32 = 0x10e;
/// Set the fog function table (inferred).
pub const SET_FOG_FUNCTION_TABLE: u32 = 0x10f;
/// Ask whether a copy source overlaps its destination (inferred).
pub const ASK_WHETHER_COPY_SOURCE_OVERLAPS_ITS_DESTINATION: u32 = 0x110;
/// Suspend pixel-mode state around a transfer (inferred).
pub const SUSPEND_PIXEL_MODE_STATE_AROUND_TRANSFER: u32 = 0x111;
/// Switch the legacy-API compatibility mode (inferred).
pub const SWITCH_LEGACY_API_COMPATIBILITY_MODE: u32 = 0x112;
/// Set the colour-material value (inferred).
pub const SET_COLOUR_MATERIAL_VALUE: u32 = 0x113;
/// Select the video source (inferred).
pub const SELECT_VIDEO_SOURCE: u32 = 0x114;
/// Ask whether a convolution copy overlaps (inferred).
pub const ASK_WHETHER_CONVOLUTION_COPY_OVERLAPS: u32 = 0x115;
/// Resolve whether a pixel format is single-component (inferred).
pub const RESOLVE_WHETHER_PIXEL_FORMAT_IS_SINGLE_COMPONENT: u32 = 0x116;
/// Set the multisample pattern (inferred).
pub const SET_MULTISAMPLE_PATTERN: u32 = 0x117;
/// Set the multisample coverage mask (inferred).
pub const SET_MULTISAMPLE_COVERAGE_MASK: u32 = 0x118;
/// Pop saved light state (inferred).
pub const POP_SAVED_LIGHT_STATE: u32 = 0x119;
/// Set pixel texture generation.
pub const SET_PIXEL_TEXTURE_GENERATION: u32 = 0x11a;
/// Set polygon offset (extension form) (inferred).
pub const SET_POLYGON_OFFSET_EXTENSION_FORM: u32 = 0x11b;
/// Load a 1D texture sub-image.
pub const LOAD_1D_TEXTURE_SUB_IMAGE: u32 = 0x11c;
/// Load a 2D texture sub-image.
pub const LOAD_2D_TEXTURE_SUB_IMAGE: u32 = 0x11d;
/// Load a 3D texture sub-image.
pub const LOAD_3D_TEXTURE_SUB_IMAGE: u32 = 0x11e;
/// Load a 3D texture image.
pub const LOAD_3D_TEXTURE_IMAGE: u32 = 0x11f;
/// 4D texture image.
pub const T_4D_TEXTURE_IMAGE: u32 = 0x120;
/// 4D texture sub-image.
pub const T_4D_TEXTURE_SUB_IMAGE: u32 = 0x121;
/// Allocate 1D texture names.
pub const ALLOCATE_1D_TEXTURE_NAMES: u32 = 0x122;
/// Allocate 2D texture names.
pub const ALLOCATE_2D_TEXTURE_NAMES: u32 = 0x123;
/// Allocate 3D texture names.
pub const ALLOCATE_3D_TEXTURE_NAMES: u32 = 0x124;
/// Query whether a name is a texture.
pub const QUERY_WHETHER_NAME_IS_TEXTURE: u32 = 0x125;
/// Delete texture names.
pub const DELETE_TEXTURE_NAMES: u32 = 0x126;
/// Bind a texture name.
pub const BIND_TEXTURE_NAME: u32 = 0x127;
/// Window-system context initialise (screen size and depth) (inferred).
pub const WINDOW_SYSTEM_CONTEXT_INITIALISE: u32 = 0x128;
/// Window-system graphics-context load (13 words).
pub const WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD: u32 = 0x129;
/// Window-system drawable origin and size load.
pub const WINDOW_SYSTEM_DRAWABLE_ORIGIN_AND_SIZE_LOAD: u32 = 0x12a;
/// Window-system clip load.
pub const WINDOW_SYSTEM_CLIP_LOAD: u32 = 0x12b;
/// Window-system clip list load (ARM-run form) (inferred).
pub const WINDOW_SYSTEM_CLIP_LIST_LOAD: u32 = 0x12c;
/// Window-system scissor load.
pub const WINDOW_SYSTEM_SCISSOR_LOAD: u32 = 0x12d;
/// Load colour-map entries (inferred).
pub const LOAD_COLOUR_MAP_ENTRIES: u32 = 0x12e;
/// Load gamma-table entries (inferred).
pub const LOAD_GAMMA_TABLE_ENTRIES: u32 = 0x12f;
/// Load a cursor image block (inferred).
pub const LOAD_CURSOR_IMAGE_BLOCK: u32 = 0x130;
/// Load the cursor colour words (inferred).
pub const LOAD_CURSOR_COLOUR_WORDS: u32 = 0x131;
/// Load one entry of the display-id table (inferred).
pub const LOAD_ONE_ENTRY_OF_DISPLAY_ID_TABLE: u32 = 0x132;
/// Window created or destroyed notice to the colour-map chip (inferred).
pub const WINDOW_CREATED_OR_DESTROYED_NOTICE_TO_COLOUR_MAP_CHIP: u32 = 0x133;
/// Query engine-side texture attributes (inferred).
pub const QUERY_ENGINE_SIDE_TEXTURE_ATTRIBUTES: u32 = 0x134;
/// Set the pixel-transform mode (inferred).
pub const SET_PIXEL_TRANSFORM_MODE: u32 = 0x135;
/// Window-system pixel path set-up (inferred).
pub const WINDOW_SYSTEM_PIXEL_PATH_SET_UP: u32 = 0x136;
/// Window-system pixel copy (inferred).
pub const WINDOW_SYSTEM_PIXEL_COPY: u32 = 0x137;
/// Window-system pixel read (inferred).
pub const WINDOW_SYSTEM_PIXEL_READ: u32 = 0x138;
/// Window-system pixel write (inferred).
pub const WINDOW_SYSTEM_PIXEL_WRITE: u32 = 0x139;
/// Window-system pixel path finish (inferred).
pub const WINDOW_SYSTEM_PIXEL_PATH_FINISH: u32 = 0x13a;
/// Unused window-system slot 3.
pub const UNUSED_WINDOW_SYSTEM_SLOT_3: u32 = 0x13b;
/// Load a window-system stipple (inferred).
pub const LOAD_WINDOW_SYSTEM_STIPPLE: u32 = 0x13c;
/// Store a window-system stipple (inferred).
pub const STORE_WINDOW_SYSTEM_STIPPLE: u32 = 0x13d;
/// Rotate a window-system stipple (inferred).
pub const ROTATE_WINDOW_SYSTEM_STIPPLE: u32 = 0x13e;
/// Unused window-system slot 4.
pub const UNUSED_WINDOW_SYSTEM_SLOT_4: u32 = 0x13f;
/// Unused window-system slot 5.
pub const UNUSED_WINDOW_SYSTEM_SLOT_5: u32 = 0x140;
/// Window-system 1-word data preload (inferred).
pub const WINDOW_SYSTEM_1_WORD_DATA_PRELOAD: u32 = 0x141;
/// Window-system 2-word data preload (inferred).
pub const WINDOW_SYSTEM_2_WORD_DATA_PRELOAD: u32 = 0x142;
/// Window-system 3-word data preload (inferred).
pub const WINDOW_SYSTEM_3_WORD_DATA_PRELOAD: u32 = 0x143;
/// Window-system 4-word data preload (inferred).
pub const WINDOW_SYSTEM_4_WORD_DATA_PRELOAD: u32 = 0x144;
/// Window-system four-word data preload, byte form (inferred).
pub const WINDOW_SYSTEM_FOUR_WORD_DATA_PRELOAD_BYTE_FORM: u32 = 0x145;
/// Window-system 5-word data preload (inferred).
pub const WINDOW_SYSTEM_5_WORD_DATA_PRELOAD: u32 = 0x146;
/// Window-system 6-word data preload (inferred).
pub const WINDOW_SYSTEM_6_WORD_DATA_PRELOAD: u32 = 0x147;
/// Window-system 7-word data preload (inferred).
pub const WINDOW_SYSTEM_7_WORD_DATA_PRELOAD: u32 = 0x148;
/// Window-system 8-word data preload (inferred).
pub const WINDOW_SYSTEM_8_WORD_DATA_PRELOAD: u32 = 0x149;
/// Window-system 9-word data preload (inferred).
pub const WINDOW_SYSTEM_9_WORD_DATA_PRELOAD: u32 = 0x14a;
/// Window-system 10-word data preload (inferred).
pub const WINDOW_SYSTEM_10_WORD_DATA_PRELOAD: u32 = 0x14b;
/// Draw absolute points.
pub const DRAW_ABSOLUTE_POINTS: u32 = 0x14c;
/// Draw absolute points (burst of 6).
pub const DRAW_ABSOLUTE_POINTS_BURST_OF_6: u32 = 0x14d;
/// Draw absolute points (burst of 24).
pub const DRAW_ABSOLUTE_POINTS_BURST_OF_24: u32 = 0x14e;
/// Draw relative points.
pub const DRAW_RELATIVE_POINTS: u32 = 0x14f;
/// Draw relative points (burst of 6).
pub const DRAW_RELATIVE_POINTS_BURST_OF_6: u32 = 0x150;
/// Draw relative points (burst of 24).
pub const DRAW_RELATIVE_POINTS_BURST_OF_24: u32 = 0x151;
/// Fill rectangles (origin, width, height).
pub const FILL_RECTANGLES_ORIGIN_WIDTH_HEIGHT: u32 = 0x152;
/// Fill rectangles (origin, width, height) (burst of 6).
pub const FILL_RECTANGLES_ORIGIN_WIDTH_HEIGHT_BURST_OF_6: u32 = 0x153;
/// Fill boxes (two corners).
pub const FILL_BOXES_TWO_CORNERS: u32 = 0x154;
/// Fill boxes (two corners) (burst of 6).
pub const FILL_BOXES_TWO_CORNERS_BURST_OF_6: u32 = 0x155;
/// Fill spans.
pub const FILL_SPANS: u32 = 0x156;
/// Fill spans (burst of 2).
pub const FILL_SPANS_BURST_OF_2: u32 = 0x157;
/// Fill spans (burst of 6).
pub const FILL_SPANS_BURST_OF_6: u32 = 0x158;
/// Draw line segments.
pub const DRAW_LINE_SEGMENTS: u32 = 0x159;
/// Draw line segments (burst of 6) (inferred).
pub const DRAW_LINE_SEGMENTS_BURST_OF_6: u32 = 0x15a;
/// Draw connected polyline.
pub const DRAW_CONNECTED_POLYLINE: u32 = 0x15b;
/// Draw connected polyline (burst of 6).
pub const DRAW_CONNECTED_POLYLINE_BURST_OF_6: u32 = 0x15c;
/// Draw connected polyline, relative points.
pub const DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS: u32 = 0x15d;
/// Draw connected polyline, relative points (burst of 6).
pub const DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS_BURST_OF_6: u32 = 0x15e;
/// Draw points in the background colour.
pub const DRAW_POINTS_IN_BACKGROUND_COLOUR: u32 = 0x15f;
/// Draw points in the background colour (burst of 6).
pub const DRAW_POINTS_IN_BACKGROUND_COLOUR_BURST_OF_6: u32 = 0x160;
/// Draw points in the background colour (burst of 24).
pub const DRAW_POINTS_IN_BACKGROUND_COLOUR_BURST_OF_24: u32 = 0x161;
/// Unused window-system slot 6.
pub const UNUSED_WINDOW_SYSTEM_SLOT_6: u32 = 0x162;
/// Unused window-system slot 7.
pub const UNUSED_WINDOW_SYSTEM_SLOT_7: u32 = 0x163;
/// Unused window-system slot 8.
pub const UNUSED_WINDOW_SYSTEM_SLOT_8: u32 = 0x164;
/// Unused window-system slot 9.
pub const UNUSED_WINDOW_SYSTEM_SLOT_9: u32 = 0x165;
/// Swap request for a multi-pipe display option (inferred).
pub const SWAP_REQUEST_FOR_MULTI_PIPE_DISPLAY_OPTION: u32 = 0x166;
/// Copy a user list by list DMA (inferred).
pub const COPY_USER_LIST_BY_LIST_DMA: u32 = 0x167;
/// Set polygon offset (inferred).
pub const SET_POLYGON_OFFSET: u32 = 0x168;
/// Diagnostic sub-command dispatch (inferred).
pub const DIAGNOSTIC_SUB_COMMAND_DISPATCH: u32 = 0x169;
/// Reserved diagnostic token 0 (inferred).
pub const RESERVED_DIAGNOSTIC_TOKEN_0: u32 = 0x16a;
/// Reserved diagnostic token 1 (inferred).
pub const RESERVED_DIAGNOSTIC_TOKEN_1: u32 = 0x16b;
/// Copy a list by list DMA (inferred).
pub const COPY_LIST_BY_LIST_DMA: u32 = 0x16c;
/// Execute a list by list DMA (inferred).
pub const EXECUTE_LIST_BY_LIST_DMA: u32 = 0x16d;
/// Execute a list by input DMA (inferred).
pub const EXECUTE_LIST_BY_INPUT_DMA: u32 = 0x16e;
/// Move a list by list DMA (inferred).
pub const MOVE_LIST_BY_LIST_DMA: u32 = 0x16f;
/// Execute a list by list DMA, repeated (inferred).
pub const EXECUTE_LIST_BY_LIST_DMA_REPEATED: u32 = 0x170;
/// Execute a list by input DMA, repeated (inferred).
pub const EXECUTE_LIST_BY_INPUT_DMA_REPEATED: u32 = 0x171;
/// Distributor no-op padding word (inferred).
pub const DISTRIBUTOR_NO_OP_PADDING_WORD: u32 = 0x172;
/// Texture-load set-up (image size and format) (inferred).
pub const TEXTURE_LOAD_SET_UP: u32 = 0x173;
/// Draw texels into texture memory (inferred).
pub const DRAW_TEXELS_INTO_TEXTURE_MEMORY: u32 = 0x174;
/// Texture-load set-up, second form (unknown).
pub const TEXTURE_LOAD_SET_UP_SECOND_FORM: u32 = 0x175;
/// Texture-load by identifier (inferred).
pub const TEXTURE_LOAD_BY_IDENTIFIER: u32 = 0x176;
/// Select the active texture unit (inferred).
pub const SELECT_ACTIVE_TEXTURE_UNIT: u32 = 0x177;
/// Bind a texture by identifier (inferred).
pub const BIND_TEXTURE_BY_IDENTIFIER: u32 = 0x178;
/// Bind a 2D mip-mapped texture (22 words).
pub const BIND_2D_MIP_MAPPED_TEXTURE: u32 = 0x179;
/// Bind a bicubic texture (inferred).
pub const BIND_BICUBIC_TEXTURE: u32 = 0x17a;
/// Bind a 3D mip-mapped texture (inferred).
pub const BIND_3D_MIP_MAPPED_TEXTURE: u32 = 0x17b;
/// Bind a 3D texture (inferred).
pub const BIND_3D_TEXTURE: u32 = 0x17c;
/// Bind a detail texture (inferred).
pub const BIND_DETAIL_TEXTURE: u32 = 0x17d;
/// Load one entry of the ARM texture table (33 words) (inferred).
pub const LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE: u32 = 0x17e;
/// Multi-purpose diagnostic entry (inferred).
pub const MULTI_PURPOSE_DIAGNOSTIC_ENTRY: u32 = 0x17f;
/// Set the vertex-array command table (inferred).
pub const SET_VERTEX_ARRAY_COMMAND_TABLE: u32 = 0x180;
/// Vertex-array set-up (inferred).
pub const VERTEX_ARRAY_SET_UP: u32 = 0x181;
/// Begin a vertex-array loop (inferred).
pub const BEGIN_VERTEX_ARRAY_LOOP: u32 = 0x182;
/// Vertex-array loop entry 0 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_0: u32 = 0x183;
/// Vertex-array loop entry 1 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_1: u32 = 0x184;
/// Vertex-array loop entry 2 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_2: u32 = 0x185;
/// Vertex-array loop entry 3 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_3: u32 = 0x186;
/// Vertex-array loop entry 4 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_4: u32 = 0x187;
/// Vertex-array loop entry 5 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_5: u32 = 0x188;
/// Vertex-array loop entry 6 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_6: u32 = 0x189;
/// Vertex-array loop entry 7 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_7: u32 = 0x18a;
/// Vertex-array loop entry 8 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_8: u32 = 0x18b;
/// Vertex-array loop entry 9 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_9: u32 = 0x18c;
/// Vertex-array loop entry 10 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_10: u32 = 0x18d;
/// Vertex-array loop entry 11 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_11: u32 = 0x18e;
/// Vertex-array loop entry 12 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_12: u32 = 0x18f;
/// Vertex-array loop entry 13 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_13: u32 = 0x190;
/// Vertex-array loop entry 14 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_14: u32 = 0x191;
/// Vertex-array loop entry 15 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_15: u32 = 0x192;
/// Vertex-array loop entry 16 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_16: u32 = 0x193;
/// Vertex-array loop entry 17 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_17: u32 = 0x194;
/// Vertex-array loop entry 18 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_18: u32 = 0x195;
/// Vertex-array loop entry 19 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_19: u32 = 0x196;
/// Vertex-array loop entry 20 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_20: u32 = 0x197;
/// Vertex-array loop entry 21 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_21: u32 = 0x198;
/// Vertex-array loop entry 22 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_22: u32 = 0x199;
/// Vertex-array loop entry 23 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_23: u32 = 0x19a;
/// Vertex-array loop entry 24 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_24: u32 = 0x19b;
/// Vertex-array loop entry 25 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_25: u32 = 0x19c;
/// Vertex-array loop entry 26 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_26: u32 = 0x19d;
/// Vertex-array loop entry 27 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_27: u32 = 0x19e;
/// Vertex-array loop entry 28 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_28: u32 = 0x19f;
/// Vertex-array loop entry 29 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_29: u32 = 0x1a0;
/// Vertex-array loop entry 30 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_30: u32 = 0x1a1;
/// Vertex-array loop entry 31 (inferred).
pub const VERTEX_ARRAY_LOOP_ENTRY_31: u32 = 0x1a2;
/// Vertex-array tear-down (inferred).
pub const VERTEX_ARRAY_TEAR_DOWN: u32 = 0x1a3;
/// Set a sprite parameter (integer) (inferred).
pub const SET_SPRITE_PARAMETER: u32 = 0x1a4;
/// Set sprite parameters (float vector) (inferred).
pub const SET_SPRITE_PARAMETERS: u32 = 0x1a5;
/// Display-generator request carrier (number, payload, checksum) (inferred).
pub const DISPLAY_GENERATOR_REQUEST_CARRIER: u32 = 0x1a6;
/// Set a point parameter (float) (inferred).
pub const SET_POINT_PARAMETER: u32 = 0x1a7;
/// Set point parameters (float vector) (inferred).
pub const SET_POINT_PARAMETERS: u32 = 0x1a8;
/// Swap triangle-mesh vertex (begin/end grammar) (inferred).
pub const SWAP_TRIANGLE_MESH_VERTEX: u32 = 0x1a9;
/// Bind a 2D clip-map texture (inferred).
pub const BIND_2D_CLIP_MAP_TEXTURE: u32 = 0x1aa;
/// Packed vertex: position (3 floats) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS: u32 = 0x1ab;
/// Packed vertex: position (3 floats), colour (4 unsigned bytes in one word) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1AC: u32 = 0x1ac;
/// Packed vertex: position (3 floats), normal (3 signed shorts packed) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED: u32 = 0x1ad;
/// Packed vertex: position (3 floats), colour (4 unsigned bytes in one word), normal (3 signed shorts packed) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1AE: u32 = 0x1ae;
/// Packed vertex: position (3 floats), texture coordinate (2 floats) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_TEXTURE_COORDINATE_2_FLOATS: u32 = 0x1af;
/// Packed vertex: position (3 floats), colour (4 unsigned bytes in one word), texture coordinate (2 floats) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B0: u32 = 0x1b0;
/// Packed vertex: position (3 floats), normal (3 signed shorts packed), texture coordinate (2 floats) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED_TEXTURE_1B1: u32 = 0x1b1;
/// Packed vertex: position (3 floats), colour (4 unsigned bytes in one word), normal (3 signed shorts packed), texture coordinate (2 floats) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B2: u32 = 0x1b2;
/// Packed vertex: position (3 floats), texture coordinate (2 signed shorts in one word) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_TEXTURE_COORDINATE_2_SIGNED_SHORTS_IN: u32 = 0x1b3;
/// Packed vertex: position (3 floats), colour (4 unsigned bytes in one word), texture coordinate (2 signed shorts in one word) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B4: u32 = 0x1b4;
/// Packed vertex: position (3 floats), normal (3 signed shorts packed), texture coordinate (2 signed shorts in one word) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED_TEXTURE_1B5: u32 = 0x1b5;
/// Packed vertex: position (3 floats), colour (4 unsigned bytes in one word), normal (3 signed shorts packed), texture coordinate (2 signed shorts in one word) (inferred).
pub const PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B6: u32 = 0x1b6;
/// Packed-vertex set-up (inferred).
pub const PACKED_VERTEX_SET_UP: u32 = 0x1b7;
/// Set the instrumentation buffer (inferred).
pub const SET_INSTRUMENTATION_BUFFER: u32 = 0x1b8;
/// Start instrumentation (inferred).
pub const START_INSTRUMENTATION: u32 = 0x1b9;
/// Stop instrumentation (inferred).
pub const STOP_INSTRUMENTATION: u32 = 0x1ba;
/// Set a texel-store parameter (inferred).
pub const SET_TEXEL_STORE_PARAMETER: u32 = 0x1bb;
/// Restart select or feedback (inferred).
pub const RESTART_SELECT_OR_FEEDBACK: u32 = 0x1bc;
/// Set the reference plane.
pub const SET_REFERENCE_PLANE: u32 = 0x1bd;
/// Set colour/normal mode value (distributor register 0x293) (inferred).
pub const SET_COLOUR_NORMAL_MODE_VALUE: u32 = 0x1be;
/// Deformation control (inferred).
pub const DEFORMATION_CONTROL: u32 = 0x1bf;
/// Deformation map, float form (inferred).
pub const DEFORMATION_MAP_FLOAT_FORM: u32 = 0x1c0;
/// Deformation map, double form (inferred).
pub const DEFORMATION_MAP_DOUBLE_FORM: u32 = 0x1c1;
/// Continuation entry of the double deformation map (inferred).
pub const CONTINUATION_ENTRY_OF_DOUBLE_DEFORMATION_MAP: u32 = 0x1c2;
/// Load identity deformation (inferred).
pub const LOAD_IDENTITY_DEFORMATION: u32 = 0x1c3;
/// Bind an ARM texture-cache entry (inferred).
pub const BIND_ARM_TEXTURE_CACHE_ENTRY: u32 = 0x1c4;
/// Free an ARM texture-cache entry (inferred).
pub const FREE_ARM_TEXTURE_CACHE_ENTRY: u32 = 0x1c5;
/// Query the deformation state (inferred).
pub const QUERY_DEFORMATION_STATE: u32 = 0x1c6;
/// Set shadow-texture state (inferred).
pub const SET_SHADOW_TEXTURE_STATE: u32 = 0x1c7;
/// Copy a list by input DMA then list DMA (inferred).
pub const COPY_LIST_BY_INPUT_DMA_THEN_LIST_DMA: u32 = 0x1c8;
/// Set triangle-mesh mode (inferred).
pub const SET_TRIANGLE_MESH_MODE: u32 = 0x1c9;
/// Load engine transform attributes (inferred).
pub const LOAD_ENGINE_TRANSFORM_ATTRIBUTES: u32 = 0x1ca;
/// Query engine transform attributes (inferred).
pub const QUERY_ENGINE_TRANSFORM_ATTRIBUTES: u32 = 0x1cb;
/// Flush the raster boards (no completion reply) (inferred).
pub const FLUSH_RASTER_BOARDS: u32 = 0x1cc;
/// Read instrumentation counters (inferred).
pub const READ_INSTRUMENTATION_COUNTERS: u32 = 0x1cd;
/// Set lazy-validation mode (inferred).
pub const SET_LAZY_VALIDATION_MODE: u32 = 0x1ce;
/// Packed-vertex clean-up (inferred).
pub const PACKED_VERTEX_CLEAN_UP: u32 = 0x1cf;
/// Subdivision patch (inferred).
pub const SUBDIVISION_PATCH: u32 = 0x1d0;
/// Subdivision patch set-up (inferred).
pub const SUBDIVISION_PATCH_SET_UP: u32 = 0x1d1;
/// Subdivision patch clean-up (inferred).
pub const SUBDIVISION_PATCH_CLEAN_UP: u32 = 0x1d2;
/// Subdivision patch parameter (inferred).
pub const SUBDIVISION_PATCH_PARAMETER: u32 = 0x1d3;
/// Begin primitive: subdivision patch (inferred).
pub const BEGIN_PRIMITIVE_SUBDIVISION_PATCH: u32 = 0x1d4;
/// Bind a texture, current form (inferred).
pub const BIND_TEXTURE_CURRENT_FORM: u32 = 0x1d5;
/// Pop texture-generation state (inferred).
pub const POP_TEXTURE_GENERATION_STATE: u32 = 0x1d6;
/// Change the post-texture scale and bias (inferred).
pub const CHANGE_POST_TEXTURE_SCALE_AND_BIAS: u32 = 0x1d7;
/// Skip 2 words (not data) (inferred).
pub const SKIP_2_WORDS: u32 = 0x1d8;
/// Skip 4 words (not data) (inferred).
pub const SKIP_4_WORDS: u32 = 0x1d9;
/// Skip 5 words (not data) (inferred).
pub const SKIP_5_WORDS: u32 = 0x1da;
/// Swap control: write the swap strobe (inferred).
pub const SWAP_CONTROL_WRITE_SWAP_STROBE: u32 = 0x1db;
/// Set the input-DMA wait-mode flag (inferred).
pub const SET_INPUT_DMA_WAIT_MODE_FLAG: u32 = 0x1dc;
/// Raise a host interrupt with an event code (privileged) (inferred).
pub const RAISE_HOST_INTERRUPT_WITH_EVENT_CODE: u32 = 0x1dd;
/// Pipe-memory resume (privileged) (inferred).
pub const PIPE_MEMORY_RESUME: u32 = 0x1de;
/// Pipe-memory suspend with an event code (privileged) (inferred).
pub const PIPE_MEMORY_SUSPEND_WITH_EVENT_CODE: u32 = 0x1df;
/// Invalidate an ARM texture-cache index (inferred).
pub const INVALIDATE_ARM_TEXTURE_CACHE_INDEX: u32 = 0x1e0;
/// Pipe-memory copy-down (privileged) (inferred).
pub const PIPE_MEMORY_COPY_DOWN: u32 = 0x1e1;
/// Pipe-memory swap-in (privileged) (inferred).
pub const PIPE_MEMORY_SWAP_IN: u32 = 0x1e2;
/// Pipe-memory swap-out (privileged) (inferred).
pub const PIPE_MEMORY_SWAP_OUT: u32 = 0x1e3;
/// Pipe-memory reserve (privileged) (inferred).
pub const PIPE_MEMORY_RESERVE: u32 = 0x1e4;
/// Roll call of every engine (inferred).
pub const ROLL_CALL_OF_EVERY_ENGINE: u32 = 0x1e5;
/// Set the per-display-id mode byte (inferred).
pub const DISPLAY_MODE_BYTE: u32 = 0x1e6;
/// Publish a retrace sequence number (inferred).
pub const PUBLISH_RETRACE_SEQUENCE_NUMBER: u32 = 0x1e7;
/// Payload-path self-test (inferred).
pub const PAYLOAD_PATH_SELF_TEST: u32 = 0x1e8;
/// Query the board configuration block (478 words) (inferred).
pub const QUERY_BOARD_CONFIGURATION_BLOCK: u32 = 0x1e9;
/// Synchronise the pipe (inferred).
pub const SYNCHRONISE_PIPE: u32 = 0x1ea;
/// Start an input DMA (inferred).
pub const START_INPUT_DMA: u32 = 0x1eb;
/// EEPROM operation (flash load, read, write) (inferred).
pub const EEPROM_OPERATION: u32 = 0x1ec;
/// Describe the DMA page pool (privileged) (inferred).
pub const DESCRIBE_DMA_PAGE_POOL: u32 = 0x1ed;
/// Window read-clip set-up (three words) (inferred).
pub const WINDOW_READ_CLIP_SET_UP: u32 = 0x1ee;
/// Load the distributor verification program (inferred).
pub const LOAD_DISTRIBUTOR_VERIFICATION_PROGRAM: u32 = 0x1ef;
/// Dump the distributor verification state (inferred).
pub const DUMP_DISTRIBUTOR_VERIFICATION_STATE: u32 = 0x1f0;
/// Pipe set-up (privileged) (inferred).
pub const PIPE_SET_UP: u32 = 0x1f1;
/// Set up output DMA for a host read (inferred).
pub const SET_UP_OUTPUT_DMA_FOR_HOST_READ: u32 = 0x1f2;
/// Window and clip set-up (eleven words) (inferred).
pub const WINDOW_CLIP_SETUP: u32 = 0x1f3;
/// Display-bank set-up (two words) (inferred).
pub const DISPLAY_BANK_SET_UP: u32 = 0x1f4;
/// Start select/feedback return DMA (six words) (inferred).
pub const START_SELECT_FEEDBACK_RETURN_DMA: u32 = 0x1f5;
/// Finish select/feedback with an event code (inferred).
pub const FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE: u32 = 0x1f6;
/// Context destroy (inferred).
pub const CONTEXT_DESTROY: u32 = 0x1f7;
/// Context clean-up with an event code (inferred).
pub const CONTEXT_CLEAN_UP_WITH_EVENT_CODE: u32 = 0x1f8;
/// Context image load (inferred).
pub const CONTEXT_IMAGE_LOAD: u32 = 0x1f9;
/// Context activate (inferred).
pub const CONTEXT_ACTIVATE: u32 = 0x1fa;
/// Context FIFO synchronise (unknown).
pub const CONTEXT_FIFO_SYNCHRONISE: u32 = 0x1fb;
/// Context deactivate (inferred).
pub const CONTEXT_DEACTIVATE: u32 = 0x1fc;
/// Context initialise (inferred).
pub const CONTEXT_INITIALISE: u32 = 0x1fd;
/// Kernel-private command (ignored) (unknown).
pub const KERNEL_PRIVATE_COMMAND: u32 = 0x1fe;
/// Write-gather flush, last slot.
pub const WRITE_GATHER_FLUSH_LAST_SLOT: u32 = 0x1ff;

// The name and the family of every host token, for the monitor and the replay census.
// The family follows the table's section: vertex tokens, matrices, arrays, textures,
// queries, and everything else.

/// The role name of host token `token`, or "" for a number past the token map.
pub fn token_name(token: u32) -> &'static str {
    match token {
        RAW_DATA_WORD => "RAW_DATA_WORD",
        CONFIGURE_FRAME_BUFFER_LAYOUT => "CONFIGURE_FRAME_BUFFER_LAYOUT",
        FLUSH_AND_REBUILD_HARDWARE_TOKEN_MAP_FOR_PACKED_VERTEX => "FLUSH_AND_REBUILD_HARDWARE_TOKEN_MAP_FOR_PACKED_VERTEX",
        BEGIN_PRIMITIVE_RECTANGLE => "BEGIN_PRIMITIVE_RECTANGLE",
        CONFIGURE_READ_SIDE_FRAME_BUFFER_LAYOUT => "CONFIGURE_READ_SIDE_FRAME_BUFFER_LAYOUT",
        DRAW_BITMAP_FROM_DISPLAY_LIST => "DRAW_BITMAP_FROM_DISPLAY_LIST",
        DRAW_BITMAP => "DRAW_BITMAP",
        SET_LIGHT_PARAMETER_FLOAT_FORM => "SET_LIGHT_PARAMETER_FLOAT_FORM",
        SET_LIGHT_PARAMETER_INTEGER_FORM => "SET_LIGHT_PARAMETER_INTEGER_FORM",
        SET_LIGHTING_MODEL_PARAMETER_FLOAT_FORM => "SET_LIGHTING_MODEL_PARAMETER_FLOAT_FORM",
        SET_LIGHTING_MODEL_PARAMETER_INTEGER_FORM => "SET_LIGHTING_MODEL_PARAMETER_INTEGER_FORM",
        LOAD_POLYGON_STIPPLE_PATTERN => "LOAD_POLYGON_STIPPLE_PATTERN",
        SET_SCISSOR_BOX => "SET_SCISSOR_BOX",
        TEXTURE_PARAMETER_FLOAT => "TEXTURE_PARAMETER_FLOAT",
        TEXTURE_PARAMETER_INTEGER => "TEXTURE_PARAMETER_INTEGER",
        T_1D_TEXTURE_IMAGE_LOAD => "T_1D_TEXTURE_IMAGE_LOAD",
        T_2D_TEXTURE_IMAGE_LOAD => "T_2D_TEXTURE_IMAGE_LOAD",
        SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM => "SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM",
        SET_TEXTURE_ENVIRONMENT_PARAMETER_INTEGER_FORM => "SET_TEXTURE_ENVIRONMENT_PARAMETER_INTEGER_FORM",
        TEXTURE_COORDINATE_GENERATION_DOUBLE_FORM => "TEXTURE_COORDINATE_GENERATION_DOUBLE_FORM",
        TEXTURE_COORDINATE_GENERATION_FLOAT_FORM => "TEXTURE_COORDINATE_GENERATION_FLOAT_FORM",
        TEXTURE_COORDINATE_GENERATION_INTEGER_FORM => "TEXTURE_COORDINATE_GENERATION_INTEGER_FORM",
        BIND_TEXTURE_BY_OLD_STYLE_BINDING => "BIND_TEXTURE_BY_OLD_STYLE_BINDING",
        SELECT_FEEDBACK_RECORD_TYPE => "SELECT_FEEDBACK_RECORD_TYPE",
        SELECT_BUFFER_REGISTRATION => "SELECT_BUFFER_REGISTRATION",
        SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES => "SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES",
        CLEAR_SELECTED_BUFFERS => "CLEAR_SELECTED_BUFFERS",
        SET_ACCUMULATION_BUFFER_CLEAR_VALUE => "SET_ACCUMULATION_BUFFER_CLEAR_VALUE",
        SET_COLOUR_INDEX_CLEAR_VALUE => "SET_COLOUR_INDEX_CLEAR_VALUE",
        SET_COLOUR_CLEAR_VALUE => "SET_COLOUR_CLEAR_VALUE",
        SET_STENCIL_CLEAR_VALUE => "SET_STENCIL_CLEAR_VALUE",
        SET_DEPTH_CLEAR_VALUE => "SET_DEPTH_CLEAR_VALUE",
        SET_STENCIL_WRITE_MASK => "SET_STENCIL_WRITE_MASK",
        SET_COLOUR_WRITE_MASK => "SET_COLOUR_WRITE_MASK",
        SET_DEPTH_WRITE_MASK => "SET_DEPTH_WRITE_MASK",
        SET_COLOUR_INDEX_WRITE_MASK => "SET_COLOUR_INDEX_WRITE_MASK",
        ENABLE_CAPABILITY => "ENABLE_CAPABILITY",
        DISABLE_CAPABILITY => "DISABLE_CAPABILITY",
        EVALUATE_1D_GRID_STRIP => "EVALUATE_1D_GRID_STRIP",
        EVALUATE_2D_GRID_STRIP => "EVALUATE_2D_GRID_STRIP",
        SET_UP_1D_EVALUATOR_GRID => "SET_UP_1D_EVALUATOR_GRID",
        SET_UP_2D_EVALUATOR_GRID => "SET_UP_2D_EVALUATOR_GRID",
        LOAD_1D_EVALUATOR_CONTROL_POINTS => "LOAD_1D_EVALUATOR_CONTROL_POINTS",
        LOAD_2D_EVALUATOR_CONTROL_POINTS => "LOAD_2D_EVALUATOR_CONTROL_POINTS",
        SET_UP_1D_EVALUATOR_MAP_SPECIFICATION => "SET_UP_1D_EVALUATOR_MAP_SPECIFICATION",
        SET_UP_2D_EVALUATOR_MAP_SPECIFICATION => "SET_UP_2D_EVALUATOR_MAP_SPECIFICATION",
        SET_LOGIC_OPERATION => "SET_LOGIC_OPERATION",
        SET_PIXEL_ZOOM_FACTORS => "SET_PIXEL_ZOOM_FACTORS",
        SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM => "SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM",
        SET_PIXEL_TRANSFER_PARAMETER_INTEGER_FORM => "SET_PIXEL_TRANSFER_PARAMETER_INTEGER_FORM",
        SET_PIXEL_STORE_PARAMETER_FLOAT_FORM => "SET_PIXEL_STORE_PARAMETER_FLOAT_FORM",
        SET_PIXEL_STORE_PARAMETER_INTEGER_FORM => "SET_PIXEL_STORE_PARAMETER_INTEGER_FORM",
        LOAD_PIXEL_MAP_FLOAT_VALUES => "LOAD_PIXEL_MAP_FLOAT_VALUES",
        LOAD_PIXEL_MAP_UNSIGNED_32_BIT_VALUES => "LOAD_PIXEL_MAP_UNSIGNED_32_BIT_VALUES",
        LOAD_PIXEL_MAP_UNSIGNED_16_BIT_VALUES => "LOAD_PIXEL_MAP_UNSIGNED_16_BIT_VALUES",
        SELECT_READ_BUFFER => "SELECT_READ_BUFFER",
        COPY_PIXEL_RECTANGLE_WITHIN_FRAME_BUFFER => "COPY_PIXEL_RECTANGLE_WITHIN_FRAME_BUFFER",
        READ_PIXEL_RECTANGLE_BACK_TO_HOST => "READ_PIXEL_RECTANGLE_BACK_TO_HOST",
        DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA => "DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA",
        QUERY_CURRENT_VERTEX_ATTRIBUTES => "QUERY_CURRENT_VERTEX_ATTRIBUTES",
        QUERY_CLIP_PLANE => "QUERY_CLIP_PLANE",
        QUERY_RASTER_POSITION_STATE => "QUERY_RASTER_POSITION_STATE",
        QUERY_PENDING_ERROR => "QUERY_PENDING_ERROR",
        QUERY_STATE_FLOAT => "QUERY_STATE_FLOAT",
        QUERY_STATE_INTEGER => "QUERY_STATE_INTEGER",
        QUERY_LIGHT_FLOAT => "QUERY_LIGHT_FLOAT",
        QUERY_LIGHT_INTEGER => "QUERY_LIGHT_INTEGER",
        QUERY_EVALUATOR_MAP_DOUBLE => "QUERY_EVALUATOR_MAP_DOUBLE",
        QUERY_EVALUATOR_MAP_FLOAT => "QUERY_EVALUATOR_MAP_FLOAT",
        QUERY_EVALUATOR_MAP_INTEGER => "QUERY_EVALUATOR_MAP_INTEGER",
        QUERY_MATERIAL_FLOAT => "QUERY_MATERIAL_FLOAT",
        QUERY_MATERIAL_INTEGER => "QUERY_MATERIAL_INTEGER",
        QUERY_PIXEL_MAP_FLOAT => "QUERY_PIXEL_MAP_FLOAT",
        QUERY_PIXEL_MAP_UNSIGNED_32_BIT => "QUERY_PIXEL_MAP_UNSIGNED_32_BIT",
        QUERY_PIXEL_MAP_UNSIGNED_16_BIT => "QUERY_PIXEL_MAP_UNSIGNED_16_BIT",
        QUERY_POLYGON_STIPPLE => "QUERY_POLYGON_STIPPLE",
        QUERY_STATE_GENERIC => "QUERY_STATE_GENERIC",
        QUERY_TEXTURE_ENVIRONMENT_FLOAT => "QUERY_TEXTURE_ENVIRONMENT_FLOAT",
        QUERY_TEXTURE_ENVIRONMENT_INTEGER => "QUERY_TEXTURE_ENVIRONMENT_INTEGER",
        QUERY_TEXTURE_GENERATION_DOUBLE => "QUERY_TEXTURE_GENERATION_DOUBLE",
        QUERY_TEXTURE_GENERATION_FLOAT => "QUERY_TEXTURE_GENERATION_FLOAT",
        QUERY_TEXTURE_GENERATION_INTEGER => "QUERY_TEXTURE_GENERATION_INTEGER",
        QUERY_TEXTURE_IMAGE => "QUERY_TEXTURE_IMAGE",
        QUERY_TEXTURE_PARAMETER_FLOAT => "QUERY_TEXTURE_PARAMETER_FLOAT",
        QUERY_TEXTURE_PARAMETER_INTEGER => "QUERY_TEXTURE_PARAMETER_INTEGER",
        QUERY_TEXTURE_LEVEL_PARAMETER_FLOAT => "QUERY_TEXTURE_LEVEL_PARAMETER_FLOAT",
        QUERY_TEXTURE_LEVEL_PARAMETER_INTEGER => "QUERY_TEXTURE_LEVEL_PARAMETER_INTEGER",
        BEGIN_PRIMITIVE_POINTS => "BEGIN_PRIMITIVE_POINTS",
        BEGIN_PRIMITIVE_INDEPENDENT_LINES => "BEGIN_PRIMITIVE_INDEPENDENT_LINES",
        BEGIN_PRIMITIVE_LINE_LOOP => "BEGIN_PRIMITIVE_LINE_LOOP",
        BEGIN_PRIMITIVE_LINE_STRIP => "BEGIN_PRIMITIVE_LINE_STRIP",
        BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES => "BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES",
        BEGIN_PRIMITIVE_TRIANGLE_STRIP => "BEGIN_PRIMITIVE_TRIANGLE_STRIP",
        BEGIN_PRIMITIVE_TRIANGLE_FAN => "BEGIN_PRIMITIVE_TRIANGLE_FAN",
        BEGIN_PRIMITIVE_INDEPENDENT_QUADS => "BEGIN_PRIMITIVE_INDEPENDENT_QUADS",
        BEGIN_PRIMITIVE_QUAD_STRIP => "BEGIN_PRIMITIVE_QUAD_STRIP",
        BEGIN_PRIMITIVE_POLYGON => "BEGIN_PRIMITIVE_POLYGON",
        SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_BYTE => "SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_BYTE",
        SET_CURRENT_COLOUR_3_COMPONENTS_DOUBLE => "SET_CURRENT_COLOUR_3_COMPONENTS_DOUBLE",
        SET_CURRENT_COLOUR_3_COMPONENTS_SINGLE_FLOAT => "SET_CURRENT_COLOUR_3_COMPONENTS_SINGLE_FLOAT",
        SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_32_BIT_INTEGER => "SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_32_BIT_INTEGER",
        SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_SHORT => "SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_SHORT",
        SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_BYTE => "SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_BYTE",
        SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_32_BIT_INTEGER => "SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_32_BIT_INTEGER",
        SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_SHORT => "SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_SHORT",
        SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_BYTE => "SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_BYTE",
        SET_CURRENT_COLOUR_4_COMPONENTS_DOUBLE => "SET_CURRENT_COLOUR_4_COMPONENTS_DOUBLE",
        SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT => "SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT",
        SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_32_BIT_INTEGER => "SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_32_BIT_INTEGER",
        SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_SHORT => "SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_SHORT",
        SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_BYTE => "SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_BYTE",
        SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_32_BIT_INTEGER => "SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_32_BIT_INTEGER",
        SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_SHORT => "SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_SHORT",
        END_OPEN_PRIMITIVE => "END_OPEN_PRIMITIVE",
        SET_CURRENT_COLOUR_INDEX_DOUBLE => "SET_CURRENT_COLOUR_INDEX_DOUBLE",
        SET_CURRENT_COLOUR_INDEX_SINGLE_FLOAT => "SET_CURRENT_COLOUR_INDEX_SINGLE_FLOAT",
        SET_CURRENT_COLOUR_INDEX_SIGNED_32_BIT_INTEGER => "SET_CURRENT_COLOUR_INDEX_SIGNED_32_BIT_INTEGER",
        SET_CURRENT_COLOUR_INDEX_SIGNED_SHORT => "SET_CURRENT_COLOUR_INDEX_SIGNED_SHORT",
        SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_BYTE => "SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_BYTE",
        SET_CURRENT_NORMAL_3_COMPONENTS_DOUBLE => "SET_CURRENT_NORMAL_3_COMPONENTS_DOUBLE",
        SET_CURRENT_NORMAL_3_COMPONENTS_SINGLE_FLOAT => "SET_CURRENT_NORMAL_3_COMPONENTS_SINGLE_FLOAT",
        SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_32_BIT_INTEGER => "SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_32_BIT_INTEGER",
        SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_SHORT => "SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_SHORT",
        SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_DOUBLE => "SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_DOUBLE",
        SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT => "SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT",
        SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SIGNED_32_BIT => "SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SIGNED_32_BIT",
        SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SIGNED_SHORT => "SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SIGNED_SHORT",
        SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_DOUBLE => "SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_DOUBLE",
        SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT => "SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT",
        SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_32_BIT => "SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_32_BIT",
        SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_SHORT => "SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_SHORT",
        SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_DOUBLE => "SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_DOUBLE",
        SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT => "SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT",
        SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SIGNED_32_BIT => "SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SIGNED_32_BIT",
        SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SIGNED_SHORT => "SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SIGNED_SHORT",
        SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_DOUBLE => "SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_DOUBLE",
        SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT => "SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT",
        SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SIGNED_32_BIT => "SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SIGNED_32_BIT",
        SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SIGNED_SHORT => "SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SIGNED_SHORT",
        SUBMIT_VERTEX_2_COMPONENTS_DOUBLE => "SUBMIT_VERTEX_2_COMPONENTS_DOUBLE",
        SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT => "SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT",
        SUBMIT_VERTEX_2_COMPONENTS_SIGNED_32_BIT_INTEGER => "SUBMIT_VERTEX_2_COMPONENTS_SIGNED_32_BIT_INTEGER",
        SUBMIT_VERTEX_2_COMPONENTS_SIGNED_SHORT => "SUBMIT_VERTEX_2_COMPONENTS_SIGNED_SHORT",
        SUBMIT_VERTEX_3_COMPONENTS_DOUBLE => "SUBMIT_VERTEX_3_COMPONENTS_DOUBLE",
        SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT => "SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT",
        SUBMIT_VERTEX_3_COMPONENTS_SIGNED_32_BIT_INTEGER => "SUBMIT_VERTEX_3_COMPONENTS_SIGNED_32_BIT_INTEGER",
        SUBMIT_VERTEX_3_COMPONENTS_SIGNED_SHORT => "SUBMIT_VERTEX_3_COMPONENTS_SIGNED_SHORT",
        SUBMIT_VERTEX_4_COMPONENTS_DOUBLE => "SUBMIT_VERTEX_4_COMPONENTS_DOUBLE",
        SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT => "SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT",
        SUBMIT_VERTEX_4_COMPONENTS_SIGNED_32_BIT_INTEGER => "SUBMIT_VERTEX_4_COMPONENTS_SIGNED_32_BIT_INTEGER",
        SUBMIT_VERTEX_4_COMPONENTS_SIGNED_SHORT => "SUBMIT_VERTEX_4_COMPONENTS_SIGNED_SHORT",
        SET_MATERIAL_PARAMETER_FLOAT_FORM => "SET_MATERIAL_PARAMETER_FLOAT_FORM",
        CONTINUATION_ENTRY_OF_FLOAT_MATERIAL_PARAMETER => "CONTINUATION_ENTRY_OF_FLOAT_MATERIAL_PARAMETER",
        SET_MATERIAL_PARAMETER_INTEGER_FORM => "SET_MATERIAL_PARAMETER_INTEGER_FORM",
        CONTINUATION_ENTRY_OF_INTEGER_MATERIAL_PARAMETER => "CONTINUATION_ENTRY_OF_INTEGER_MATERIAL_PARAMETER",
        EVALUATOR_COORDINATE_1D_DOUBLE => "EVALUATOR_COORDINATE_1D_DOUBLE",
        CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD1D => "CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD1D",
        EVALUATOR_COORDINATE_1D_SINGLE_FLOAT => "EVALUATOR_COORDINATE_1D_SINGLE_FLOAT",
        CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD1F => "CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD1F",
        EVALUATOR_COORDINATE_2D_DOUBLE => "EVALUATOR_COORDINATE_2D_DOUBLE",
        CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD2D => "CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD2D",
        EVALUATOR_COORDINATE_2D_SINGLE_FLOAT => "EVALUATOR_COORDINATE_2D_SINGLE_FLOAT",
        CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD2F => "CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_COORD2F",
        EVALUATOR_GRID_POINT_1D => "EVALUATOR_GRID_POINT_1D",
        CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_POINT1 => "CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_POINT1",
        EVALUATOR_GRID_POINT_2D => "EVALUATOR_GRID_POINT_2D",
        CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_POINT2 => "CONTINUATION_ENTRY_OF_EVALUATOR_EVAL_POINT2",
        SET_CURRENT_EDGE_FLAG => "SET_CURRENT_EDGE_FLAG",
        SET_RASTER_POSITION_2_COMPONENTS_DOUBLE => "SET_RASTER_POSITION_2_COMPONENTS_DOUBLE",
        SET_RASTER_POSITION_2_COMPONENTS_FLOAT => "SET_RASTER_POSITION_2_COMPONENTS_FLOAT",
        SET_RASTER_POSITION_2_COMPONENTS_32_BIT_INTEGER => "SET_RASTER_POSITION_2_COMPONENTS_32_BIT_INTEGER",
        SET_RASTER_POSITION_2_COMPONENTS_SHORT => "SET_RASTER_POSITION_2_COMPONENTS_SHORT",
        SET_RASTER_POSITION_3_COMPONENTS_DOUBLE => "SET_RASTER_POSITION_3_COMPONENTS_DOUBLE",
        SET_RASTER_POSITION_3_COMPONENTS_FLOAT => "SET_RASTER_POSITION_3_COMPONENTS_FLOAT",
        SET_RASTER_POSITION_3_COMPONENTS_32_BIT_INTEGER => "SET_RASTER_POSITION_3_COMPONENTS_32_BIT_INTEGER",
        SET_RASTER_POSITION_3_COMPONENTS_SHORT => "SET_RASTER_POSITION_3_COMPONENTS_SHORT",
        SET_RASTER_POSITION_4_COMPONENTS_DOUBLE => "SET_RASTER_POSITION_4_COMPONENTS_DOUBLE",
        SET_RASTER_POSITION_4_COMPONENTS_FLOAT => "SET_RASTER_POSITION_4_COMPONENTS_FLOAT",
        SET_RASTER_POSITION_4_COMPONENTS_32_BIT_INTEGER => "SET_RASTER_POSITION_4_COMPONENTS_32_BIT_INTEGER",
        SET_RASTER_POSITION_4_COMPONENTS_SHORT => "SET_RASTER_POSITION_4_COMPONENTS_SHORT",
        SET_USER_CLIP_PLANE => "SET_USER_CLIP_PLANE",
        SET_COLOUR_MATERIAL_TRACKING => "SET_COLOUR_MATERIAL_TRACKING",
        SET_FACE_CULLING => "SET_FACE_CULLING",
        SET_FOG_PARAMETER_FLOAT_FORM => "SET_FOG_PARAMETER_FLOAT_FORM",
        SET_FOG_PARAMETER_INTEGER_FORM => "SET_FOG_PARAMETER_INTEGER_FORM",
        CONTINUATION_ENTRY_OF_INTEGER_FOG_PARAMETER => "CONTINUATION_ENTRY_OF_INTEGER_FOG_PARAMETER",
        SET_FRONT_FACE_WINDING => "SET_FRONT_FACE_WINDING",
        SET_QUALITY_HINT => "SET_QUALITY_HINT",
        SET_LINE_STIPPLE => "SET_LINE_STIPPLE",
        SET_LINE_WIDTH => "SET_LINE_WIDTH",
        SET_POINT_SIZE => "SET_POINT_SIZE",
        SET_POLYGON_FILL_MODE => "SET_POLYGON_FILL_MODE",
        SET_FLAT_OR_SMOOTH_SHADING => "SET_FLAT_OR_SMOOTH_SHADING",
        INSERT_FEEDBACK_PASS_THROUGH_MARKER => "INSERT_FEEDBACK_PASS_THROUGH_MARKER",
        POP_SELECTION_NAME => "POP_SELECTION_NAME",
        PUSH_SELECTION_NAME => "PUSH_SELECTION_NAME",
        SELECT_DRAW_BUFFER => "SELECT_DRAW_BUFFER",
        ACCUMULATION_BUFFER_OPERATION => "ACCUMULATION_BUFFER_OPERATION",
        FINISH_DRAIN_PIPE_AND_SIGNAL_COMPLETION => "FINISH_DRAIN_PIPE_AND_SIGNAL_COMPLETION",
        FLUSH_VERTEX_PIPELINE => "FLUSH_VERTEX_PIPELINE",
        POP_ATTRIBUTE_STACK => "POP_ATTRIBUTE_STACK",
        PUSH_ATTRIBUTE_STACK => "PUSH_ATTRIBUTE_STACK",
        SET_ALPHA_TEST => "SET_ALPHA_TEST",
        SET_BLEND_FACTORS => "SET_BLEND_FACTORS",
        SET_STENCIL_TEST => "SET_STENCIL_TEST",
        SET_STENCIL_OPERATIONS => "SET_STENCIL_OPERATIONS",
        SET_DEPTH_TEST => "SET_DEPTH_TEST",
        SET_DEPTH_RANGE => "SET_DEPTH_RANGE",
        SELECT_CURRENT_MATRIX_STACK => "SELECT_CURRENT_MATRIX_STACK",
        MULTIPLY_CURRENT_MATRIX_BY_FRUSTUM => "MULTIPLY_CURRENT_MATRIX_BY_FRUSTUM",
        LOAD_IDENTITY_MATRIX => "LOAD_IDENTITY_MATRIX",
        LOAD_MATRIX_16_FLOATS => "LOAD_MATRIX_16_FLOATS",
        LOAD_MATRIX_16_DOUBLES => "LOAD_MATRIX_16_DOUBLES",
        MULTIPLY_BY_MATRIX_16_FLOATS => "MULTIPLY_BY_MATRIX_16_FLOATS",
        MULTIPLY_BY_MATRIX_16_DOUBLES => "MULTIPLY_BY_MATRIX_16_DOUBLES",
        MULTIPLY_CURRENT_MATRIX_BY_ORTHOGRAPHIC_PROJECTION => "MULTIPLY_CURRENT_MATRIX_BY_ORTHOGRAPHIC_PROJECTION",
        POP_MATRIX_STACK => "POP_MATRIX_STACK",
        PUSH_MATRIX_STACK => "PUSH_MATRIX_STACK",
        ROTATE_DOUBLES => "ROTATE_DOUBLES",
        ROTATE_FLOATS => "ROTATE_FLOATS",
        SCALE_DOUBLES => "SCALE_DOUBLES",
        SCALE_FLOATS => "SCALE_FLOATS",
        TRANSLATE_DOUBLES => "TRANSLATE_DOUBLES",
        TRANSLATE_FLOATS => "TRANSLATE_FLOATS",
        COPY_ENGINE_WORK_MEMORY_TO_CONTROL_MEMORY => "COPY_ENGINE_WORK_MEMORY_TO_CONTROL_MEMORY",
        PASS_ONE_COMMAND_WORD_THROUGH => "PASS_ONE_COMMAND_WORD_THROUGH",
        WRITE_ONE_RASTER_BUS_WORD => "WRITE_ONE_RASTER_BUS_WORD",
        SET_VIEWPORT => "SET_VIEWPORT",
        RECORD_ERROR_CODE => "RECORD_ERROR_CODE",
        END_OF_SIMULATION_REPORT => "END_OF_SIMULATION_REPORT",
        SET_CONSTANT_BLEND_COLOUR => "SET_CONSTANT_BLEND_COLOUR",
        SET_BLEND_EQUATION => "SET_BLEND_EQUATION",
        LOAD_COLOUR_TABLE => "LOAD_COLOUR_TABLE",
        COPY_COLOUR_TABLE_FROM_FRAME_BUFFER => "COPY_COLOUR_TABLE_FROM_FRAME_BUFFER",
        QUERY_COLOUR_TABLE => "QUERY_COLOUR_TABLE",
        SET_COLOUR_TABLE_PARAMETERS => "SET_COLOUR_TABLE_PARAMETERS",
        QUERY_COLOUR_TABLE_PARAMETERS => "QUERY_COLOUR_TABLE_PARAMETERS",
        LOAD_POST_TEXTURE_SCALE_AND_BIAS => "LOAD_POST_TEXTURE_SCALE_AND_BIAS",
        QUERY_LINE_MODE_LIMIT_TABLE => "QUERY_LINE_MODE_LIMIT_TABLE",
        LOAD_1D_CONVOLUTION_FILTER => "LOAD_1D_CONVOLUTION_FILTER",
        LOAD_2D_CONVOLUTION_FILTER => "LOAD_2D_CONVOLUTION_FILTER",
        T_3D_CONVOLUTION_FILTER => "T_3D_CONVOLUTION_FILTER",
        COPY_1D_CONVOLUTION_FILTER_FROM_FRAME_BUFFER => "COPY_1D_CONVOLUTION_FILTER_FROM_FRAME_BUFFER",
        COPY_2D_CONVOLUTION_FILTER_FROM_FRAME_BUFFER => "COPY_2D_CONVOLUTION_FILTER_FROM_FRAME_BUFFER",
        QUERY_CONVOLUTION_FILTER => "QUERY_CONVOLUTION_FILTER",
        LOAD_SEPARABLE_2D_FILTER => "LOAD_SEPARABLE_2D_FILTER",
        T_3D_SEPARABLE_FILTER => "T_3D_SEPARABLE_FILTER",
        QUERY_SEPARABLE_FILTER => "QUERY_SEPARABLE_FILTER",
        SET_PIXEL_TRANSFORM_PARAMETERS_INTEGER => "SET_PIXEL_TRANSFORM_PARAMETERS_INTEGER",
        SET_PIXEL_TRANSFORM_PARAMETERS_FLOAT => "SET_PIXEL_TRANSFORM_PARAMETERS_FLOAT",
        QUERY_PIXEL_TRANSFORM_PARAMETERS_INTEGER => "QUERY_PIXEL_TRANSFORM_PARAMETERS_INTEGER",
        QUERY_PIXEL_TRANSFORM_PARAMETERS_FLOAT => "QUERY_PIXEL_TRANSFORM_PARAMETERS_FLOAT",
        COPY_1D_TEXTURE_IMAGE_FROM_FRAME_BUFFER => "COPY_1D_TEXTURE_IMAGE_FROM_FRAME_BUFFER",
        COPY_2D_TEXTURE_IMAGE_FROM_FRAME_BUFFER => "COPY_2D_TEXTURE_IMAGE_FROM_FRAME_BUFFER",
        COPY_1D_TEXTURE_SUB_IMAGE_FROM_FRAME_BUFFER => "COPY_1D_TEXTURE_SUB_IMAGE_FROM_FRAME_BUFFER",
        WRITE_GATHER_FLUSH_FIRST_SLOT => "WRITE_GATHER_FLUSH_FIRST_SLOT",
        RAW_DATA_WORD_SECOND_FORM => "RAW_DATA_WORD_SECOND_FORM",
        COPY_2D_TEXTURE_SUB_IMAGE_FROM_FRAME_BUFFER => "COPY_2D_TEXTURE_SUB_IMAGE_FROM_FRAME_BUFFER",
        SET_UP_HISTOGRAM => "SET_UP_HISTOGRAM",
        RESET_HISTOGRAM => "RESET_HISTOGRAM",
        QUERY_HISTOGRAM => "QUERY_HISTOGRAM",
        QUERY_HISTOGRAM_PARAMETERS => "QUERY_HISTOGRAM_PARAMETERS",
        BIND_LINEAR_TEXTURE => "BIND_LINEAR_TEXTURE",
        SET_UP_MIN_MAX_TRACKING => "SET_UP_MIN_MAX_TRACKING",
        RESET_MIN_MAX_TRACKING => "RESET_MIN_MAX_TRACKING",
        QUERY_MIN_MAX_RESULTS => "QUERY_MIN_MAX_RESULTS",
        QUERY_MIN_MAX_PARAMETERS => "QUERY_MIN_MAX_PARAMETERS",
        SAVE_LIGHT_ATTRIBUTES_ON_ENGINE_ATTRIBUTE_STACK => "SAVE_LIGHT_ATTRIBUTES_ON_ENGINE_ATTRIBUTE_STACK",
        CLEAR_SELECTION_NAME_STACK => "CLEAR_SELECTION_NAME_STACK",
        REPLACE_TOP_SELECTION_NAME => "REPLACE_TOP_SELECTION_NAME",
        SET_CURRENT_RASTER_POSITION => "SET_CURRENT_RASTER_POSITION",
        SET_FOG_FUNCTION_TABLE => "SET_FOG_FUNCTION_TABLE",
        ASK_WHETHER_COPY_SOURCE_OVERLAPS_ITS_DESTINATION => "ASK_WHETHER_COPY_SOURCE_OVERLAPS_ITS_DESTINATION",
        SUSPEND_PIXEL_MODE_STATE_AROUND_TRANSFER => "SUSPEND_PIXEL_MODE_STATE_AROUND_TRANSFER",
        SWITCH_LEGACY_API_COMPATIBILITY_MODE => "SWITCH_LEGACY_API_COMPATIBILITY_MODE",
        SET_COLOUR_MATERIAL_VALUE => "SET_COLOUR_MATERIAL_VALUE",
        SELECT_VIDEO_SOURCE => "SELECT_VIDEO_SOURCE",
        ASK_WHETHER_CONVOLUTION_COPY_OVERLAPS => "ASK_WHETHER_CONVOLUTION_COPY_OVERLAPS",
        RESOLVE_WHETHER_PIXEL_FORMAT_IS_SINGLE_COMPONENT => "RESOLVE_WHETHER_PIXEL_FORMAT_IS_SINGLE_COMPONENT",
        SET_MULTISAMPLE_PATTERN => "SET_MULTISAMPLE_PATTERN",
        SET_MULTISAMPLE_COVERAGE_MASK => "SET_MULTISAMPLE_COVERAGE_MASK",
        POP_SAVED_LIGHT_STATE => "POP_SAVED_LIGHT_STATE",
        SET_PIXEL_TEXTURE_GENERATION => "SET_PIXEL_TEXTURE_GENERATION",
        SET_POLYGON_OFFSET_EXTENSION_FORM => "SET_POLYGON_OFFSET_EXTENSION_FORM",
        LOAD_1D_TEXTURE_SUB_IMAGE => "LOAD_1D_TEXTURE_SUB_IMAGE",
        LOAD_2D_TEXTURE_SUB_IMAGE => "LOAD_2D_TEXTURE_SUB_IMAGE",
        LOAD_3D_TEXTURE_SUB_IMAGE => "LOAD_3D_TEXTURE_SUB_IMAGE",
        LOAD_3D_TEXTURE_IMAGE => "LOAD_3D_TEXTURE_IMAGE",
        T_4D_TEXTURE_IMAGE => "T_4D_TEXTURE_IMAGE",
        T_4D_TEXTURE_SUB_IMAGE => "T_4D_TEXTURE_SUB_IMAGE",
        ALLOCATE_1D_TEXTURE_NAMES => "ALLOCATE_1D_TEXTURE_NAMES",
        ALLOCATE_2D_TEXTURE_NAMES => "ALLOCATE_2D_TEXTURE_NAMES",
        ALLOCATE_3D_TEXTURE_NAMES => "ALLOCATE_3D_TEXTURE_NAMES",
        QUERY_WHETHER_NAME_IS_TEXTURE => "QUERY_WHETHER_NAME_IS_TEXTURE",
        DELETE_TEXTURE_NAMES => "DELETE_TEXTURE_NAMES",
        BIND_TEXTURE_NAME => "BIND_TEXTURE_NAME",
        WINDOW_SYSTEM_CONTEXT_INITIALISE => "WINDOW_SYSTEM_CONTEXT_INITIALISE",
        WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD => "WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD",
        WINDOW_SYSTEM_DRAWABLE_ORIGIN_AND_SIZE_LOAD => "WINDOW_SYSTEM_DRAWABLE_ORIGIN_AND_SIZE_LOAD",
        WINDOW_SYSTEM_CLIP_LOAD => "WINDOW_SYSTEM_CLIP_LOAD",
        WINDOW_SYSTEM_CLIP_LIST_LOAD => "WINDOW_SYSTEM_CLIP_LIST_LOAD",
        WINDOW_SYSTEM_SCISSOR_LOAD => "WINDOW_SYSTEM_SCISSOR_LOAD",
        LOAD_COLOUR_MAP_ENTRIES => "LOAD_COLOUR_MAP_ENTRIES",
        LOAD_GAMMA_TABLE_ENTRIES => "LOAD_GAMMA_TABLE_ENTRIES",
        LOAD_CURSOR_IMAGE_BLOCK => "LOAD_CURSOR_IMAGE_BLOCK",
        LOAD_CURSOR_COLOUR_WORDS => "LOAD_CURSOR_COLOUR_WORDS",
        LOAD_ONE_ENTRY_OF_DISPLAY_ID_TABLE => "LOAD_ONE_ENTRY_OF_DISPLAY_ID_TABLE",
        WINDOW_CREATED_OR_DESTROYED_NOTICE_TO_COLOUR_MAP_CHIP => "WINDOW_CREATED_OR_DESTROYED_NOTICE_TO_COLOUR_MAP_CHIP",
        QUERY_ENGINE_SIDE_TEXTURE_ATTRIBUTES => "QUERY_ENGINE_SIDE_TEXTURE_ATTRIBUTES",
        SET_PIXEL_TRANSFORM_MODE => "SET_PIXEL_TRANSFORM_MODE",
        WINDOW_SYSTEM_PIXEL_PATH_SET_UP => "WINDOW_SYSTEM_PIXEL_PATH_SET_UP",
        WINDOW_SYSTEM_PIXEL_COPY => "WINDOW_SYSTEM_PIXEL_COPY",
        WINDOW_SYSTEM_PIXEL_READ => "WINDOW_SYSTEM_PIXEL_READ",
        WINDOW_SYSTEM_PIXEL_WRITE => "WINDOW_SYSTEM_PIXEL_WRITE",
        WINDOW_SYSTEM_PIXEL_PATH_FINISH => "WINDOW_SYSTEM_PIXEL_PATH_FINISH",
        UNUSED_WINDOW_SYSTEM_SLOT_3 => "UNUSED_WINDOW_SYSTEM_SLOT_3",
        LOAD_WINDOW_SYSTEM_STIPPLE => "LOAD_WINDOW_SYSTEM_STIPPLE",
        STORE_WINDOW_SYSTEM_STIPPLE => "STORE_WINDOW_SYSTEM_STIPPLE",
        ROTATE_WINDOW_SYSTEM_STIPPLE => "ROTATE_WINDOW_SYSTEM_STIPPLE",
        UNUSED_WINDOW_SYSTEM_SLOT_4 => "UNUSED_WINDOW_SYSTEM_SLOT_4",
        UNUSED_WINDOW_SYSTEM_SLOT_5 => "UNUSED_WINDOW_SYSTEM_SLOT_5",
        WINDOW_SYSTEM_1_WORD_DATA_PRELOAD => "WINDOW_SYSTEM_1_WORD_DATA_PRELOAD",
        WINDOW_SYSTEM_2_WORD_DATA_PRELOAD => "WINDOW_SYSTEM_2_WORD_DATA_PRELOAD",
        WINDOW_SYSTEM_3_WORD_DATA_PRELOAD => "WINDOW_SYSTEM_3_WORD_DATA_PRELOAD",
        WINDOW_SYSTEM_4_WORD_DATA_PRELOAD => "WINDOW_SYSTEM_4_WORD_DATA_PRELOAD",
        WINDOW_SYSTEM_FOUR_WORD_DATA_PRELOAD_BYTE_FORM => "WINDOW_SYSTEM_FOUR_WORD_DATA_PRELOAD_BYTE_FORM",
        WINDOW_SYSTEM_5_WORD_DATA_PRELOAD => "WINDOW_SYSTEM_5_WORD_DATA_PRELOAD",
        WINDOW_SYSTEM_6_WORD_DATA_PRELOAD => "WINDOW_SYSTEM_6_WORD_DATA_PRELOAD",
        WINDOW_SYSTEM_7_WORD_DATA_PRELOAD => "WINDOW_SYSTEM_7_WORD_DATA_PRELOAD",
        WINDOW_SYSTEM_8_WORD_DATA_PRELOAD => "WINDOW_SYSTEM_8_WORD_DATA_PRELOAD",
        WINDOW_SYSTEM_9_WORD_DATA_PRELOAD => "WINDOW_SYSTEM_9_WORD_DATA_PRELOAD",
        WINDOW_SYSTEM_10_WORD_DATA_PRELOAD => "WINDOW_SYSTEM_10_WORD_DATA_PRELOAD",
        DRAW_ABSOLUTE_POINTS => "DRAW_ABSOLUTE_POINTS",
        DRAW_ABSOLUTE_POINTS_BURST_OF_6 => "DRAW_ABSOLUTE_POINTS_BURST_OF_6",
        DRAW_ABSOLUTE_POINTS_BURST_OF_24 => "DRAW_ABSOLUTE_POINTS_BURST_OF_24",
        DRAW_RELATIVE_POINTS => "DRAW_RELATIVE_POINTS",
        DRAW_RELATIVE_POINTS_BURST_OF_6 => "DRAW_RELATIVE_POINTS_BURST_OF_6",
        DRAW_RELATIVE_POINTS_BURST_OF_24 => "DRAW_RELATIVE_POINTS_BURST_OF_24",
        FILL_RECTANGLES_ORIGIN_WIDTH_HEIGHT => "FILL_RECTANGLES_ORIGIN_WIDTH_HEIGHT",
        FILL_RECTANGLES_ORIGIN_WIDTH_HEIGHT_BURST_OF_6 => "FILL_RECTANGLES_ORIGIN_WIDTH_HEIGHT_BURST_OF_6",
        FILL_BOXES_TWO_CORNERS => "FILL_BOXES_TWO_CORNERS",
        FILL_BOXES_TWO_CORNERS_BURST_OF_6 => "FILL_BOXES_TWO_CORNERS_BURST_OF_6",
        FILL_SPANS => "FILL_SPANS",
        FILL_SPANS_BURST_OF_2 => "FILL_SPANS_BURST_OF_2",
        FILL_SPANS_BURST_OF_6 => "FILL_SPANS_BURST_OF_6",
        DRAW_LINE_SEGMENTS => "DRAW_LINE_SEGMENTS",
        DRAW_LINE_SEGMENTS_BURST_OF_6 => "DRAW_LINE_SEGMENTS_BURST_OF_6",
        DRAW_CONNECTED_POLYLINE => "DRAW_CONNECTED_POLYLINE",
        DRAW_CONNECTED_POLYLINE_BURST_OF_6 => "DRAW_CONNECTED_POLYLINE_BURST_OF_6",
        DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS => "DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS",
        DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS_BURST_OF_6 => "DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS_BURST_OF_6",
        DRAW_POINTS_IN_BACKGROUND_COLOUR => "DRAW_POINTS_IN_BACKGROUND_COLOUR",
        DRAW_POINTS_IN_BACKGROUND_COLOUR_BURST_OF_6 => "DRAW_POINTS_IN_BACKGROUND_COLOUR_BURST_OF_6",
        DRAW_POINTS_IN_BACKGROUND_COLOUR_BURST_OF_24 => "DRAW_POINTS_IN_BACKGROUND_COLOUR_BURST_OF_24",
        UNUSED_WINDOW_SYSTEM_SLOT_6 => "UNUSED_WINDOW_SYSTEM_SLOT_6",
        UNUSED_WINDOW_SYSTEM_SLOT_7 => "UNUSED_WINDOW_SYSTEM_SLOT_7",
        UNUSED_WINDOW_SYSTEM_SLOT_8 => "UNUSED_WINDOW_SYSTEM_SLOT_8",
        UNUSED_WINDOW_SYSTEM_SLOT_9 => "UNUSED_WINDOW_SYSTEM_SLOT_9",
        SWAP_REQUEST_FOR_MULTI_PIPE_DISPLAY_OPTION => "SWAP_REQUEST_FOR_MULTI_PIPE_DISPLAY_OPTION",
        COPY_USER_LIST_BY_LIST_DMA => "COPY_USER_LIST_BY_LIST_DMA",
        SET_POLYGON_OFFSET => "SET_POLYGON_OFFSET",
        DIAGNOSTIC_SUB_COMMAND_DISPATCH => "DIAGNOSTIC_SUB_COMMAND_DISPATCH",
        RESERVED_DIAGNOSTIC_TOKEN_0 => "RESERVED_DIAGNOSTIC_TOKEN_0",
        RESERVED_DIAGNOSTIC_TOKEN_1 => "RESERVED_DIAGNOSTIC_TOKEN_1",
        COPY_LIST_BY_LIST_DMA => "COPY_LIST_BY_LIST_DMA",
        EXECUTE_LIST_BY_LIST_DMA => "EXECUTE_LIST_BY_LIST_DMA",
        EXECUTE_LIST_BY_INPUT_DMA => "EXECUTE_LIST_BY_INPUT_DMA",
        MOVE_LIST_BY_LIST_DMA => "MOVE_LIST_BY_LIST_DMA",
        EXECUTE_LIST_BY_LIST_DMA_REPEATED => "EXECUTE_LIST_BY_LIST_DMA_REPEATED",
        EXECUTE_LIST_BY_INPUT_DMA_REPEATED => "EXECUTE_LIST_BY_INPUT_DMA_REPEATED",
        DISTRIBUTOR_NO_OP_PADDING_WORD => "DISTRIBUTOR_NO_OP_PADDING_WORD",
        TEXTURE_LOAD_SET_UP => "TEXTURE_LOAD_SET_UP",
        DRAW_TEXELS_INTO_TEXTURE_MEMORY => "DRAW_TEXELS_INTO_TEXTURE_MEMORY",
        TEXTURE_LOAD_SET_UP_SECOND_FORM => "TEXTURE_LOAD_SET_UP_SECOND_FORM",
        TEXTURE_LOAD_BY_IDENTIFIER => "TEXTURE_LOAD_BY_IDENTIFIER",
        SELECT_ACTIVE_TEXTURE_UNIT => "SELECT_ACTIVE_TEXTURE_UNIT",
        BIND_TEXTURE_BY_IDENTIFIER => "BIND_TEXTURE_BY_IDENTIFIER",
        BIND_2D_MIP_MAPPED_TEXTURE => "BIND_2D_MIP_MAPPED_TEXTURE",
        BIND_BICUBIC_TEXTURE => "BIND_BICUBIC_TEXTURE",
        BIND_3D_MIP_MAPPED_TEXTURE => "BIND_3D_MIP_MAPPED_TEXTURE",
        BIND_3D_TEXTURE => "BIND_3D_TEXTURE",
        BIND_DETAIL_TEXTURE => "BIND_DETAIL_TEXTURE",
        LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE => "LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE",
        MULTI_PURPOSE_DIAGNOSTIC_ENTRY => "MULTI_PURPOSE_DIAGNOSTIC_ENTRY",
        SET_VERTEX_ARRAY_COMMAND_TABLE => "SET_VERTEX_ARRAY_COMMAND_TABLE",
        VERTEX_ARRAY_SET_UP => "VERTEX_ARRAY_SET_UP",
        BEGIN_VERTEX_ARRAY_LOOP => "BEGIN_VERTEX_ARRAY_LOOP",
        VERTEX_ARRAY_LOOP_ENTRY_0 => "VERTEX_ARRAY_LOOP_ENTRY_0",
        VERTEX_ARRAY_LOOP_ENTRY_1 => "VERTEX_ARRAY_LOOP_ENTRY_1",
        VERTEX_ARRAY_LOOP_ENTRY_2 => "VERTEX_ARRAY_LOOP_ENTRY_2",
        VERTEX_ARRAY_LOOP_ENTRY_3 => "VERTEX_ARRAY_LOOP_ENTRY_3",
        VERTEX_ARRAY_LOOP_ENTRY_4 => "VERTEX_ARRAY_LOOP_ENTRY_4",
        VERTEX_ARRAY_LOOP_ENTRY_5 => "VERTEX_ARRAY_LOOP_ENTRY_5",
        VERTEX_ARRAY_LOOP_ENTRY_6 => "VERTEX_ARRAY_LOOP_ENTRY_6",
        VERTEX_ARRAY_LOOP_ENTRY_7 => "VERTEX_ARRAY_LOOP_ENTRY_7",
        VERTEX_ARRAY_LOOP_ENTRY_8 => "VERTEX_ARRAY_LOOP_ENTRY_8",
        VERTEX_ARRAY_LOOP_ENTRY_9 => "VERTEX_ARRAY_LOOP_ENTRY_9",
        VERTEX_ARRAY_LOOP_ENTRY_10 => "VERTEX_ARRAY_LOOP_ENTRY_10",
        VERTEX_ARRAY_LOOP_ENTRY_11 => "VERTEX_ARRAY_LOOP_ENTRY_11",
        VERTEX_ARRAY_LOOP_ENTRY_12 => "VERTEX_ARRAY_LOOP_ENTRY_12",
        VERTEX_ARRAY_LOOP_ENTRY_13 => "VERTEX_ARRAY_LOOP_ENTRY_13",
        VERTEX_ARRAY_LOOP_ENTRY_14 => "VERTEX_ARRAY_LOOP_ENTRY_14",
        VERTEX_ARRAY_LOOP_ENTRY_15 => "VERTEX_ARRAY_LOOP_ENTRY_15",
        VERTEX_ARRAY_LOOP_ENTRY_16 => "VERTEX_ARRAY_LOOP_ENTRY_16",
        VERTEX_ARRAY_LOOP_ENTRY_17 => "VERTEX_ARRAY_LOOP_ENTRY_17",
        VERTEX_ARRAY_LOOP_ENTRY_18 => "VERTEX_ARRAY_LOOP_ENTRY_18",
        VERTEX_ARRAY_LOOP_ENTRY_19 => "VERTEX_ARRAY_LOOP_ENTRY_19",
        VERTEX_ARRAY_LOOP_ENTRY_20 => "VERTEX_ARRAY_LOOP_ENTRY_20",
        VERTEX_ARRAY_LOOP_ENTRY_21 => "VERTEX_ARRAY_LOOP_ENTRY_21",
        VERTEX_ARRAY_LOOP_ENTRY_22 => "VERTEX_ARRAY_LOOP_ENTRY_22",
        VERTEX_ARRAY_LOOP_ENTRY_23 => "VERTEX_ARRAY_LOOP_ENTRY_23",
        VERTEX_ARRAY_LOOP_ENTRY_24 => "VERTEX_ARRAY_LOOP_ENTRY_24",
        VERTEX_ARRAY_LOOP_ENTRY_25 => "VERTEX_ARRAY_LOOP_ENTRY_25",
        VERTEX_ARRAY_LOOP_ENTRY_26 => "VERTEX_ARRAY_LOOP_ENTRY_26",
        VERTEX_ARRAY_LOOP_ENTRY_27 => "VERTEX_ARRAY_LOOP_ENTRY_27",
        VERTEX_ARRAY_LOOP_ENTRY_28 => "VERTEX_ARRAY_LOOP_ENTRY_28",
        VERTEX_ARRAY_LOOP_ENTRY_29 => "VERTEX_ARRAY_LOOP_ENTRY_29",
        VERTEX_ARRAY_LOOP_ENTRY_30 => "VERTEX_ARRAY_LOOP_ENTRY_30",
        VERTEX_ARRAY_LOOP_ENTRY_31 => "VERTEX_ARRAY_LOOP_ENTRY_31",
        VERTEX_ARRAY_TEAR_DOWN => "VERTEX_ARRAY_TEAR_DOWN",
        SET_SPRITE_PARAMETER => "SET_SPRITE_PARAMETER",
        SET_SPRITE_PARAMETERS => "SET_SPRITE_PARAMETERS",
        DISPLAY_GENERATOR_REQUEST_CARRIER => "DISPLAY_GENERATOR_REQUEST_CARRIER",
        SET_POINT_PARAMETER => "SET_POINT_PARAMETER",
        SET_POINT_PARAMETERS => "SET_POINT_PARAMETERS",
        SWAP_TRIANGLE_MESH_VERTEX => "SWAP_TRIANGLE_MESH_VERTEX",
        BIND_2D_CLIP_MAP_TEXTURE => "BIND_2D_CLIP_MAP_TEXTURE",
        PACKED_VERTEX_POSITION_3_FLOATS => "PACKED_VERTEX_POSITION_3_FLOATS",
        PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1AC => "PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1AC",
        PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED => "PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED",
        PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1AE => "PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1AE",
        PACKED_VERTEX_POSITION_3_FLOATS_TEXTURE_COORDINATE_2_FLOATS => "PACKED_VERTEX_POSITION_3_FLOATS_TEXTURE_COORDINATE_2_FLOATS",
        PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B0 => "PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B0",
        PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED_TEXTURE_1B1 => "PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED_TEXTURE_1B1",
        PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B2 => "PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B2",
        PACKED_VERTEX_POSITION_3_FLOATS_TEXTURE_COORDINATE_2_SIGNED_SHORTS_IN => "PACKED_VERTEX_POSITION_3_FLOATS_TEXTURE_COORDINATE_2_SIGNED_SHORTS_IN",
        PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B4 => "PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B4",
        PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED_TEXTURE_1B5 => "PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED_TEXTURE_1B5",
        PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B6 => "PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B6",
        PACKED_VERTEX_SET_UP => "PACKED_VERTEX_SET_UP",
        SET_INSTRUMENTATION_BUFFER => "SET_INSTRUMENTATION_BUFFER",
        START_INSTRUMENTATION => "START_INSTRUMENTATION",
        STOP_INSTRUMENTATION => "STOP_INSTRUMENTATION",
        SET_TEXEL_STORE_PARAMETER => "SET_TEXEL_STORE_PARAMETER",
        RESTART_SELECT_OR_FEEDBACK => "RESTART_SELECT_OR_FEEDBACK",
        SET_REFERENCE_PLANE => "SET_REFERENCE_PLANE",
        SET_COLOUR_NORMAL_MODE_VALUE => "SET_COLOUR_NORMAL_MODE_VALUE",
        DEFORMATION_CONTROL => "DEFORMATION_CONTROL",
        DEFORMATION_MAP_FLOAT_FORM => "DEFORMATION_MAP_FLOAT_FORM",
        DEFORMATION_MAP_DOUBLE_FORM => "DEFORMATION_MAP_DOUBLE_FORM",
        CONTINUATION_ENTRY_OF_DOUBLE_DEFORMATION_MAP => "CONTINUATION_ENTRY_OF_DOUBLE_DEFORMATION_MAP",
        LOAD_IDENTITY_DEFORMATION => "LOAD_IDENTITY_DEFORMATION",
        BIND_ARM_TEXTURE_CACHE_ENTRY => "BIND_ARM_TEXTURE_CACHE_ENTRY",
        FREE_ARM_TEXTURE_CACHE_ENTRY => "FREE_ARM_TEXTURE_CACHE_ENTRY",
        QUERY_DEFORMATION_STATE => "QUERY_DEFORMATION_STATE",
        SET_SHADOW_TEXTURE_STATE => "SET_SHADOW_TEXTURE_STATE",
        COPY_LIST_BY_INPUT_DMA_THEN_LIST_DMA => "COPY_LIST_BY_INPUT_DMA_THEN_LIST_DMA",
        SET_TRIANGLE_MESH_MODE => "SET_TRIANGLE_MESH_MODE",
        LOAD_ENGINE_TRANSFORM_ATTRIBUTES => "LOAD_ENGINE_TRANSFORM_ATTRIBUTES",
        QUERY_ENGINE_TRANSFORM_ATTRIBUTES => "QUERY_ENGINE_TRANSFORM_ATTRIBUTES",
        FLUSH_RASTER_BOARDS => "FLUSH_RASTER_BOARDS",
        READ_INSTRUMENTATION_COUNTERS => "READ_INSTRUMENTATION_COUNTERS",
        SET_LAZY_VALIDATION_MODE => "SET_LAZY_VALIDATION_MODE",
        PACKED_VERTEX_CLEAN_UP => "PACKED_VERTEX_CLEAN_UP",
        SUBDIVISION_PATCH => "SUBDIVISION_PATCH",
        SUBDIVISION_PATCH_SET_UP => "SUBDIVISION_PATCH_SET_UP",
        SUBDIVISION_PATCH_CLEAN_UP => "SUBDIVISION_PATCH_CLEAN_UP",
        SUBDIVISION_PATCH_PARAMETER => "SUBDIVISION_PATCH_PARAMETER",
        BEGIN_PRIMITIVE_SUBDIVISION_PATCH => "BEGIN_PRIMITIVE_SUBDIVISION_PATCH",
        BIND_TEXTURE_CURRENT_FORM => "BIND_TEXTURE_CURRENT_FORM",
        POP_TEXTURE_GENERATION_STATE => "POP_TEXTURE_GENERATION_STATE",
        CHANGE_POST_TEXTURE_SCALE_AND_BIAS => "CHANGE_POST_TEXTURE_SCALE_AND_BIAS",
        SKIP_2_WORDS => "SKIP_2_WORDS",
        SKIP_4_WORDS => "SKIP_4_WORDS",
        SKIP_5_WORDS => "SKIP_5_WORDS",
        SWAP_CONTROL_WRITE_SWAP_STROBE => "SWAP_CONTROL_WRITE_SWAP_STROBE",
        SET_INPUT_DMA_WAIT_MODE_FLAG => "SET_INPUT_DMA_WAIT_MODE_FLAG",
        RAISE_HOST_INTERRUPT_WITH_EVENT_CODE => "RAISE_HOST_INTERRUPT_WITH_EVENT_CODE",
        PIPE_MEMORY_RESUME => "PIPE_MEMORY_RESUME",
        PIPE_MEMORY_SUSPEND_WITH_EVENT_CODE => "PIPE_MEMORY_SUSPEND_WITH_EVENT_CODE",
        INVALIDATE_ARM_TEXTURE_CACHE_INDEX => "INVALIDATE_ARM_TEXTURE_CACHE_INDEX",
        PIPE_MEMORY_COPY_DOWN => "PIPE_MEMORY_COPY_DOWN",
        PIPE_MEMORY_SWAP_IN => "PIPE_MEMORY_SWAP_IN",
        PIPE_MEMORY_SWAP_OUT => "PIPE_MEMORY_SWAP_OUT",
        PIPE_MEMORY_RESERVE => "PIPE_MEMORY_RESERVE",
        ROLL_CALL_OF_EVERY_ENGINE => "ROLL_CALL_OF_EVERY_ENGINE",
        DISPLAY_MODE_BYTE => "DISPLAY_MODE_BYTE",
        PUBLISH_RETRACE_SEQUENCE_NUMBER => "PUBLISH_RETRACE_SEQUENCE_NUMBER",
        PAYLOAD_PATH_SELF_TEST => "PAYLOAD_PATH_SELF_TEST",
        QUERY_BOARD_CONFIGURATION_BLOCK => "QUERY_BOARD_CONFIGURATION_BLOCK",
        SYNCHRONISE_PIPE => "SYNCHRONISE_PIPE",
        START_INPUT_DMA => "START_INPUT_DMA",
        EEPROM_OPERATION => "EEPROM_OPERATION",
        DESCRIBE_DMA_PAGE_POOL => "DESCRIBE_DMA_PAGE_POOL",
        WINDOW_READ_CLIP_SET_UP => "WINDOW_READ_CLIP_SET_UP",
        LOAD_DISTRIBUTOR_VERIFICATION_PROGRAM => "LOAD_DISTRIBUTOR_VERIFICATION_PROGRAM",
        DUMP_DISTRIBUTOR_VERIFICATION_STATE => "DUMP_DISTRIBUTOR_VERIFICATION_STATE",
        PIPE_SET_UP => "PIPE_SET_UP",
        SET_UP_OUTPUT_DMA_FOR_HOST_READ => "SET_UP_OUTPUT_DMA_FOR_HOST_READ",
        WINDOW_CLIP_SETUP => "WINDOW_CLIP_SETUP",
        DISPLAY_BANK_SET_UP => "DISPLAY_BANK_SET_UP",
        START_SELECT_FEEDBACK_RETURN_DMA => "START_SELECT_FEEDBACK_RETURN_DMA",
        FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE => "FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE",
        CONTEXT_DESTROY => "CONTEXT_DESTROY",
        CONTEXT_CLEAN_UP_WITH_EVENT_CODE => "CONTEXT_CLEAN_UP_WITH_EVENT_CODE",
        CONTEXT_IMAGE_LOAD => "CONTEXT_IMAGE_LOAD",
        CONTEXT_ACTIVATE => "CONTEXT_ACTIVATE",
        CONTEXT_FIFO_SYNCHRONISE => "CONTEXT_FIFO_SYNCHRONISE",
        CONTEXT_DEACTIVATE => "CONTEXT_DEACTIVATE",
        CONTEXT_INITIALISE => "CONTEXT_INITIALISE",
        KERNEL_PRIVATE_COMMAND => "KERNEL_PRIVATE_COMMAND",
        WRITE_GATHER_FLUSH_LAST_SLOT => "WRITE_GATHER_FLUSH_LAST_SLOT",
        _ => "",
    }
}

/// The family of host token `token`, by the table's section.
pub fn token_family(token: u32) -> &'static str {
    match token {
        BEGIN_PRIMITIVE_RECTANGLE | BEGIN_PRIMITIVE_POINTS | BEGIN_PRIMITIVE_INDEPENDENT_LINES | BEGIN_PRIMITIVE_LINE_LOOP | BEGIN_PRIMITIVE_LINE_STRIP | BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES | BEGIN_PRIMITIVE_TRIANGLE_STRIP | BEGIN_PRIMITIVE_TRIANGLE_FAN | BEGIN_PRIMITIVE_INDEPENDENT_QUADS | BEGIN_PRIMITIVE_QUAD_STRIP | BEGIN_PRIMITIVE_POLYGON | SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_BYTE | SET_CURRENT_COLOUR_3_COMPONENTS_DOUBLE | SET_CURRENT_COLOUR_3_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_32_BIT_INTEGER | SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_SHORT | SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_BYTE | SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_32_BIT_INTEGER | SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_SHORT | SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_BYTE | SET_CURRENT_COLOUR_4_COMPONENTS_DOUBLE | SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_32_BIT_INTEGER | SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_SHORT | SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_BYTE | SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_32_BIT_INTEGER | SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_SHORT | END_OPEN_PRIMITIVE | SET_CURRENT_COLOUR_INDEX_DOUBLE | SET_CURRENT_COLOUR_INDEX_SINGLE_FLOAT | SET_CURRENT_COLOUR_INDEX_SIGNED_32_BIT_INTEGER | SET_CURRENT_COLOUR_INDEX_SIGNED_SHORT | SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_BYTE | SET_CURRENT_NORMAL_3_COMPONENTS_DOUBLE | SET_CURRENT_NORMAL_3_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_32_BIT_INTEGER | SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_SHORT | SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_DOUBLE | SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SIGNED_32_BIT | SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SIGNED_SHORT | SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_DOUBLE | SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_32_BIT | SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_SHORT | SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_DOUBLE | SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SIGNED_32_BIT | SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SIGNED_SHORT | SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_DOUBLE | SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT | SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SIGNED_32_BIT | SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SIGNED_SHORT | SUBMIT_VERTEX_2_COMPONENTS_DOUBLE | SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT | SUBMIT_VERTEX_2_COMPONENTS_SIGNED_32_BIT_INTEGER | SUBMIT_VERTEX_2_COMPONENTS_SIGNED_SHORT | SUBMIT_VERTEX_3_COMPONENTS_DOUBLE | SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT | SUBMIT_VERTEX_3_COMPONENTS_SIGNED_32_BIT_INTEGER | SUBMIT_VERTEX_3_COMPONENTS_SIGNED_SHORT | SUBMIT_VERTEX_4_COMPONENTS_DOUBLE | SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT | SUBMIT_VERTEX_4_COMPONENTS_SIGNED_32_BIT_INTEGER | SUBMIT_VERTEX_4_COMPONENTS_SIGNED_SHORT | SWAP_TRIANGLE_MESH_VERTEX | PACKED_VERTEX_POSITION_3_FLOATS | PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1AC | PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED | PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1AE | PACKED_VERTEX_POSITION_3_FLOATS_TEXTURE_COORDINATE_2_FLOATS | PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B0 | PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED_TEXTURE_1B1 | PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B2 | PACKED_VERTEX_POSITION_3_FLOATS_TEXTURE_COORDINATE_2_SIGNED_SHORTS_IN | PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B4 | PACKED_VERTEX_POSITION_3_FLOATS_NORMAL_3_SIGNED_SHORTS_PACKED_TEXTURE_1B5 | PACKED_VERTEX_POSITION_3_FLOATS_COLOUR_4_UNSIGNED_BYTES_IN_ONE_WORD_1B6 | BEGIN_PRIMITIVE_SUBDIVISION_PATCH => "vertex",
        SET_SCISSOR_BOX | SET_USER_CLIP_PLANE | SET_DEPTH_RANGE | SELECT_CURRENT_MATRIX_STACK | MULTIPLY_CURRENT_MATRIX_BY_FRUSTUM | LOAD_IDENTITY_MATRIX | LOAD_MATRIX_16_FLOATS | LOAD_MATRIX_16_DOUBLES | MULTIPLY_BY_MATRIX_16_FLOATS | MULTIPLY_BY_MATRIX_16_DOUBLES | MULTIPLY_CURRENT_MATRIX_BY_ORTHOGRAPHIC_PROJECTION | POP_MATRIX_STACK | PUSH_MATRIX_STACK | ROTATE_DOUBLES | ROTATE_FLOATS | SCALE_DOUBLES | SCALE_FLOATS | TRANSLATE_DOUBLES | TRANSLATE_FLOATS | SET_VIEWPORT => "matrix",
        SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM | SET_TEXTURE_ENVIRONMENT_PARAMETER_INTEGER_FORM | TEXTURE_COORDINATE_GENERATION_DOUBLE_FORM | TEXTURE_COORDINATE_GENERATION_FLOAT_FORM | TEXTURE_COORDINATE_GENERATION_INTEGER_FORM | BIND_TEXTURE_BY_OLD_STYLE_BINDING | COPY_2D_TEXTURE_IMAGE_FROM_FRAME_BUFFER | BIND_LINEAR_TEXTURE | TEXTURE_LOAD_SET_UP | DRAW_TEXELS_INTO_TEXTURE_MEMORY | TEXTURE_LOAD_BY_IDENTIFIER | SELECT_ACTIVE_TEXTURE_UNIT | BIND_TEXTURE_BY_IDENTIFIER | BIND_BICUBIC_TEXTURE | BIND_3D_MIP_MAPPED_TEXTURE | BIND_3D_TEXTURE | BIND_DETAIL_TEXTURE | LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE | BIND_2D_CLIP_MAP_TEXTURE | SET_TEXEL_STORE_PARAMETER | BIND_ARM_TEXTURE_CACHE_ENTRY | SET_SHADOW_TEXTURE_STATE | BIND_TEXTURE_CURRENT_FORM | POP_TEXTURE_GENERATION_STATE | CHANGE_POST_TEXTURE_SCALE_AND_BIAS | INVALIDATE_ARM_TEXTURE_CACHE_INDEX => "texture",
        EVALUATE_1D_GRID_STRIP | EVALUATE_2D_GRID_STRIP | LOAD_1D_EVALUATOR_CONTROL_POINTS | LOAD_2D_EVALUATOR_CONTROL_POINTS | SET_RASTER_POSITION_2_COMPONENTS_DOUBLE | SET_RASTER_POSITION_2_COMPONENTS_FLOAT | SET_RASTER_POSITION_2_COMPONENTS_32_BIT_INTEGER | SET_RASTER_POSITION_2_COMPONENTS_SHORT | SET_RASTER_POSITION_3_COMPONENTS_DOUBLE | SET_RASTER_POSITION_3_COMPONENTS_FLOAT | SET_RASTER_POSITION_3_COMPONENTS_32_BIT_INTEGER | SET_RASTER_POSITION_3_COMPONENTS_SHORT | SET_RASTER_POSITION_4_COMPONENTS_DOUBLE | SET_RASTER_POSITION_4_COMPONENTS_FLOAT | SET_RASTER_POSITION_4_COMPONENTS_32_BIT_INTEGER | SET_RASTER_POSITION_4_COMPONENTS_SHORT | SET_CURRENT_RASTER_POSITION | SET_VERTEX_ARRAY_COMMAND_TABLE | VERTEX_ARRAY_SET_UP | BEGIN_VERTEX_ARRAY_LOOP | VERTEX_ARRAY_LOOP_ENTRY_0 | VERTEX_ARRAY_LOOP_ENTRY_1 | VERTEX_ARRAY_LOOP_ENTRY_2 | VERTEX_ARRAY_LOOP_ENTRY_3 | VERTEX_ARRAY_LOOP_ENTRY_4 | VERTEX_ARRAY_LOOP_ENTRY_5 | VERTEX_ARRAY_LOOP_ENTRY_6 | VERTEX_ARRAY_LOOP_ENTRY_7 | VERTEX_ARRAY_LOOP_ENTRY_8 | VERTEX_ARRAY_LOOP_ENTRY_9 | VERTEX_ARRAY_LOOP_ENTRY_10 | VERTEX_ARRAY_LOOP_ENTRY_11 | VERTEX_ARRAY_LOOP_ENTRY_12 | VERTEX_ARRAY_LOOP_ENTRY_13 | VERTEX_ARRAY_LOOP_ENTRY_14 | VERTEX_ARRAY_LOOP_ENTRY_15 | VERTEX_ARRAY_LOOP_ENTRY_16 | VERTEX_ARRAY_LOOP_ENTRY_17 | VERTEX_ARRAY_LOOP_ENTRY_18 | VERTEX_ARRAY_LOOP_ENTRY_19 | VERTEX_ARRAY_LOOP_ENTRY_20 | VERTEX_ARRAY_LOOP_ENTRY_21 | VERTEX_ARRAY_LOOP_ENTRY_22 | VERTEX_ARRAY_LOOP_ENTRY_23 | VERTEX_ARRAY_LOOP_ENTRY_24 | VERTEX_ARRAY_LOOP_ENTRY_25 | VERTEX_ARRAY_LOOP_ENTRY_26 | VERTEX_ARRAY_LOOP_ENTRY_27 | VERTEX_ARRAY_LOOP_ENTRY_28 | VERTEX_ARRAY_LOOP_ENTRY_29 | VERTEX_ARRAY_LOOP_ENTRY_30 | VERTEX_ARRAY_LOOP_ENTRY_31 | VERTEX_ARRAY_TEAR_DOWN | PACKED_VERTEX_SET_UP | PACKED_VERTEX_CLEAN_UP | SUBDIVISION_PATCH | SUBDIVISION_PATCH_SET_UP | SUBDIVISION_PATCH_CLEAN_UP | SUBDIVISION_PATCH_PARAMETER => "array",
        QUERY_CURRENT_VERTEX_ATTRIBUTES | QUERY_CLIP_PLANE | QUERY_RASTER_POSITION_STATE | QUERY_PENDING_ERROR | QUERY_LIGHT_FLOAT | QUERY_MATERIAL_FLOAT | QUERY_PIXEL_MAP_FLOAT | QUERY_POLYGON_STIPPLE | QUERY_STATE_GENERIC | QUERY_TEXTURE_ENVIRONMENT_FLOAT | QUERY_TEXTURE_GENERATION_FLOAT | QUERY_TEXTURE_IMAGE | QUERY_COLOUR_TABLE | QUERY_COLOUR_TABLE_PARAMETERS | QUERY_LINE_MODE_LIMIT_TABLE | QUERY_CONVOLUTION_FILTER | QUERY_SEPARABLE_FILTER | QUERY_PIXEL_TRANSFORM_PARAMETERS_INTEGER | QUERY_HISTOGRAM | QUERY_HISTOGRAM_PARAMETERS | QUERY_MIN_MAX_RESULTS | QUERY_MIN_MAX_PARAMETERS | QUERY_ENGINE_SIDE_TEXTURE_ATTRIBUTES | QUERY_DEFORMATION_STATE | QUERY_ENGINE_TRANSFORM_ATTRIBUTES | QUERY_BOARD_CONFIGURATION_BLOCK => "query",
        _ => "state and other",
    }
}

/// Query reply layouts (section 7.16 of the definitions table). A reply is the completion
/// word at byte 0, written last (established), the count of data words at byte 4, then the data
/// from byte 8; the offsets here are into the data, in 32-bit words.
pub mod reply {
    /// The current-value reply (0x03b), 13 words: colour (4 floats), colour index, normal (3
    /// floats), texture coordinates (4 floats), edge flag. Established; the edge flag is a float
    /// (the client truncates it to an integer, so set is 1.0), its exact bit pattern inferred.
    pub const CURRENT_WORDS: usize = 13;
    pub const CURRENT_COLOUR: usize = 0;
    pub const CURRENT_INDEX: usize = 4;
    pub const CURRENT_NORMAL: usize = 5;
    pub const CURRENT_TEXTURE: usize = 8;
    pub const CURRENT_EDGE_FLAG: usize = 12;
    /// The raster-position reply (0x03d), 15 words with no gaps: colour (4 floats), colour
    /// index, texture coordinates (4 floats), position (4 floats), valid flag, distance.
    /// Established as to the client's offsets and the tiling of the count; the engine
    /// handler's push order is not separately decoded.
    pub const RASTER_WORDS: usize = 15;
    pub const RASTER_COLOUR: usize = 0;
    pub const RASTER_INDEX: usize = 4;
    pub const RASTER_TEXTURE: usize = 5;
    pub const RASTER_POSITION: usize = 9;
    pub const RASTER_VALID: usize = 13;
    pub const RASTER_DISTANCE: usize = 14;
    /// Feedback vertices: the pipe's formatter scales window x and y by 0.0625 after subtracting
    /// the window origin and z, after the depth offset, by 2^-20 (constants 0x3d800000 and
    /// 0x35800000). Inferred: the order of the values inside a vertex follows GL's by count only,
    /// and the engine's own units are taken to be sixteenths of a pixel and the 20-bit window
    /// depth range below.
    pub const FEEDBACK_XY_SCALE: f32 = 0.0625;
    pub const FEEDBACK_Z_SCALE: f32 = 1.0 / 1_048_576.0;
    /// The window-depth range constant (0x497fffff) the pipe loads: the engine's depth for a
    /// fraction of 1. Inferred to be the range the z scale above divides out.
    pub const WINDOW_DEPTH_RANGE: f32 = 1_048_575.0;
    /// A select record's depth words are unsigned 32-bit integers, the pipe's depth bounds scaled
    /// and clamped at this float (0x4f7fffff). Inferred; which constant multiplies and which
    /// clamps, and whether 0 is the near plane, is read from constants alone.
    pub const SELECT_DEPTH_MAX: f32 = 4_294_967_040.0;
}

/// Texture memory addressing (sections 7.12 and 7.13 of the definitions table).
pub mod texture {
    use super::BitField;
    /// The texel size code in header word 17: the log2 of the texel's width in 16-bit lanes (0 for
    /// 16-bit texels, 1 for 32-bit, 2 for a 64-bit slot of which 48 bits are used). Established
    /// by two shipped programs; code 3 is not exercised and is read as the same log2 rule.
    pub const SIZE_CODE: BitField = BitField::new(1, 0);
    /// Bit 3 of header word 17 selects the bank (and with it which of words 18 and 19 holds the
    /// level's address). Established.
    pub const BANK: BitField = BitField::new(3, 3);
    /// A level address counts 256 bits (sixteen 16-bit lanes) per bank. Established for the
    /// vendor allocator, inferred for the hardware; the unit of the global-offset registers is
    /// unknown.
    pub const ADDRESS_UNIT_LANES: u32 = 16;
    /// The detail image's texture target, the public enumerant GL_DETAIL_TEXTURE_2D_SGIS.
    pub const DETAIL_TARGET: u32 = 0x8095;
    /// The volume texture target, the public enumerant GL_TEXTURE_3D (load header word 0). Established.
    pub const VOLUME_TARGET: u32 = 0x806f;

    /// Whether a storage code holds one colour channel (luminance, intensity or alpha, alone or with
    /// alpha) and not red, green and blue. The pipe's read-back of colour as luminance sums red, green
    /// and blue unless its source is such a texture (established as the stage the pipe picks; the
    /// clamped sum is inferred).
    pub const fn storage_is_one_channel(code: u32) -> bool {
        matches!(code, 0x08..=0x0f | 0x20 | 0x28)
    }
}

/// How the pipe treats a colour index (section 7.6 of the definitions table, from the firmware): as
/// a 12-bit component value, left-justified. A write mask for a `w`-bit index is `(2^w - 1) << (12 - w)`
/// on the main level and `(2^w - 1) << 4` on an overlay or popup level, written to raster register
/// 0x1c14 as `m | m << 12`, and the clear index is scaled by `1 << (12 - w)`. So an 8-bit index is
/// its value shifted left 4 (bits 11:4), a 12-bit one fills bits 11:0, and overlay planes sit in bits
/// `[3 + w : 4]`. Established; which component lane holds the index and its offset in the stored bank
/// word are unknown. The draw-pixels path puts an index of 8 bits or fewer at bits 15:8 of each 16-bit
/// lane on a board mode other than 2 and at bits 15:4 (11:4 for an 8-bit index) on board mode 2 or
/// for an index above 8 bits (the shift model is inferred; what board mode 2 is, is unknown).
pub mod colour_index {
    /// The write mask of a `width`-bit index (1 to 12) in component space.
    pub const fn write_mask(width: u32, overlay: bool) -> u32 {
        let ones = (1u32 << width) - 1;
        if overlay { ones << 4 } else { ones << (12 - width) }
    }

    /// The value written to the colour write-mask register for a component-space mask.
    pub const fn write_mask_register(mask: u32) -> u32 {
        mask | mask << 12
    }

    /// The factor a clear index is scaled by.
    pub const fn clear_scale(width: u32) -> u32 {
        1 << (12 - width)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn an_index_is_a_left_justified_twelve_bit_component() {
            assert_eq!(write_mask(8, false), 0xff0, "CI8 is its value shifted left 4");
            assert_eq!(write_mask(12, false), 0xfff);
            assert_eq!(write_mask(2, true), 0x30, "overlay planes sit from bit 4");
            assert_eq!(write_mask(8, true), 0xff0);
            assert_eq!(write_mask_register(0xff0), 0xff0_ff0);
            assert_eq!([clear_scale(8), clear_scale(12), clear_scale(4)], [16, 1, 256]);
        }
    }
}

/// The generic state query's rows (token 0x04c, section 7.16 of the definitions table): the 171 names
/// the pipe's firmware answers, with the words of each reply. A name with no row is answered with an
/// empty reply and an invalid-enumerant error. State rows send one word for each element (a byte or
/// halfword zero-extended); a matrix is 16 words; the depth range (0xb70) is two doubles, 4 words, and
/// the depth clear value (0xb73) one double, 2. The eleven callback rows write their own count: the
/// stack depths and polygon offset values are one word, the reference-plane equation (0x817e) four,
/// and the fog function (0x812a) takes its count from the word before the reply (listed as 1).
/// Established.
pub mod state_query {
    pub const ROWS: [(u32, u8); 171] = [
        (0x0b10, 1),
        (0x0b11, 1),
        (0x0b20, 1),
        (0x0b21, 1),
        (0x0b24, 1),
        (0x0b25, 1),
        (0x0b26, 1),
        (0x0b40, 2),
        (0x0b41, 1),
        (0x0b42, 1),
        (0x0b44, 1),
        (0x0b45, 1),
        (0x0b46, 1),
        (0x0b50, 1),
        (0x0b51, 1),
        (0x0b52, 1),
        (0x0b53, 4),
        (0x0b54, 1),
        (0x0b55, 1),
        (0x0b56, 1),
        (0x0b57, 1),
        (0x0b60, 1),
        (0x0b61, 1),
        (0x0b62, 1),
        (0x0b63, 1),
        (0x0b64, 1),
        (0x0b65, 1),
        (0x0b66, 4),
        (0x0b70, 4),
        (0x0b71, 1),
        (0x0b72, 1),
        (0x0b73, 2),
        (0x0b74, 1),
        (0x0b80, 4),
        (0x0b90, 1),
        (0x0b91, 1),
        (0x0b92, 1),
        (0x0b93, 1),
        (0x0b94, 1),
        (0x0b95, 1),
        (0x0b96, 1),
        (0x0b97, 1),
        (0x0b98, 1),
        (0x0ba0, 1),
        (0x0ba1, 1),
        (0x0ba2, 4),
        (0x0ba3, 1),
        (0x0ba4, 1),
        (0x0ba5, 1),
        (0x0ba6, 16),
        (0x0ba7, 16),
        (0x0ba8, 16),
        (0x0bb0, 1),
        (0x0bc0, 1),
        (0x0bc1, 1),
        (0x0bc2, 1),
        (0x0bd0, 1),
        (0x0be0, 1),
        (0x0be1, 1),
        (0x0be2, 1),
        (0x0bf0, 1),
        (0x0bf1, 1),
        (0x0bf2, 1),
        (0x0c01, 1),
        (0x0c02, 1),
        (0x0c10, 4),
        (0x0c11, 1),
        (0x0c20, 1),
        (0x0c21, 1),
        (0x0c22, 4),
        (0x0c23, 4),
        (0x0c50, 1),
        (0x0c51, 1),
        (0x0c52, 1),
        (0x0c53, 1),
        (0x0c54, 1),
        (0x0c60, 1),
        (0x0c61, 1),
        (0x0c62, 1),
        (0x0c63, 1),
        (0x0cb0, 1),
        (0x0cb1, 1),
        (0x0cb2, 1),
        (0x0cb3, 1),
        (0x0cb4, 1),
        (0x0cb5, 1),
        (0x0cb6, 1),
        (0x0cb7, 1),
        (0x0cb8, 1),
        (0x0cb9, 1),
        (0x0d70, 1),
        (0x0d80, 1),
        (0x0de0, 1),
        (0x0de1, 1),
        (0x2a00, 1),
        (0x2a01, 1),
        (0x2a02, 1),
        (0x3000, 1),
        (0x3001, 1),
        (0x3002, 1),
        (0x3003, 1),
        (0x3004, 1),
        (0x3005, 1),
        (0x4000, 1),
        (0x4001, 1),
        (0x4002, 1),
        (0x4003, 1),
        (0x4004, 1),
        (0x4005, 1),
        (0x4006, 1),
        (0x4007, 1),
        (0x8005, 4),
        (0x8009, 1),
        (0x8010, 1),
        (0x8011, 1),
        (0x8012, 1),
        (0x801c, 1),
        (0x801d, 1),
        (0x801e, 1),
        (0x801f, 1),
        (0x8020, 1),
        (0x8021, 1),
        (0x8022, 1),
        (0x8023, 1),
        (0x8024, 1),
        (0x802e, 1),
        (0x8037, 1),
        (0x8038, 1),
        (0x8039, 1),
        (0x806f, 1),
        (0x8094, 1),
        (0x809d, 1),
        (0x809e, 1),
        (0x809f, 1),
        (0x80a0, 1),
        (0x80aa, 1),
        (0x80ab, 1),
        (0x80ac, 1),
        (0x80b1, 16),
        (0x80b2, 1),
        (0x80b4, 1),
        (0x80b5, 1),
        (0x80b6, 1),
        (0x80b7, 1),
        (0x80b8, 1),
        (0x80b9, 1),
        (0x80ba, 1),
        (0x80bb, 1),
        (0x80bc, 1),
        (0x80d0, 1),
        (0x80d1, 1),
        (0x80d2, 1),
        (0x8126, 1),
        (0x8127, 1),
        (0x8128, 1),
        (0x8129, 3),
        (0x812a, 1),
        (0x812b, 1),
        (0x8148, 1),
        (0x8149, 1),
        (0x814a, 3),
        (0x814b, 3),
        (0x817d, 1),
        (0x817e, 4),
        (0x8181, 1),
        (0x8183, 1),
        (0x8196, 1),
        (0x8198, 1),
        (0x8199, 4),
        (0x81c4, 1),
        (0x81c7, 1),
    ];

    /// The words of the reply to a name, or None for a name the firmware has no row for.
    pub fn words(name: u32) -> Option<usize> {
        ROWS.binary_search_by_key(&name, |r| r.0).ok().map(|i| ROWS[i].1 as usize)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_rows_are_sorted_and_carry_the_tables_counts() {
            assert!(ROWS.windows(2).all(|w| w[0].0 < w[1].0));
            assert_eq!([words(0xba6), words(0xb70), words(0xb73), words(0xc22), words(0x0b44), words(0x817e)], [Some(16), Some(4), Some(2), Some(4), Some(1), Some(4)]);
            assert_eq!(words(0xb14), None, "the one name the client asks and the firmware has no row for");
            assert_eq!(words(0x4000), Some(1), "light 0's enable");
            assert_eq!(ROWS.iter().filter(|r| r.1 == 16).count(), 4, "four matrices");
        }
    }
}

/// The multisample registers of the raster back end (section 7.17 of the definitions table). The
/// model draws multisampled windows from its context state, not from these words, so they are the
/// board's register images: what a context's state makes the firmware write, for the monitor and
/// for tests that tie the model's state to the table.
pub mod multisample {
    use super::BitField;

    /// The image-processor multisample control word (raster register 0x1c03). Layout established
    /// (the vendor diagnostic names the fields); the effects are inferred.
    pub const CONTROL_REGISTER: u32 = 0x1c03;
    pub const CONTROL_ENABLE: BitField = BitField::new(0, 0);
    /// 0 for 4 samples, 1 for 8.
    pub const CONTROL_EIGHT_SAMPLES: BitField = BitField::new(1, 1);
    /// 0 AND, 1 replace.
    pub const CONTROL_MASK_REPLACE: BitField = BitField::new(2, 2);
    pub const CONTROL_MASK: BitField = BitField::new(10, 3);
    pub const CONTROL_MAINTAIN_TWO: BitField = BitField::new(11, 11);
    pub const CONTROL_OPTIMISE: BitField = BitField::new(13, 12);
    /// The optimisation field's values: 0 none, 2 disable, 3 enable. The firmware writes 3 with a
    /// multisample buffer, 0 for the 12-bit-colour, 8-sample case, and 2 while blending, a logic
    /// operation or a partial colour mask is on (and again for the length of a masked clear); why
    /// the hardware needs that is not read (values established, meaning inferred).
    pub const OPTIMISE_NONE: u32 = 0;
    pub const OPTIMISE_DISABLE: u32 = 2;
    pub const OPTIMISE_ENABLE: u32 = 3;

    /// The control word the firmware writes for a window: the mask is always all ones and the
    /// operation AND; the maintain-two bit is never written (read as zero).
    pub const fn control_word(eight_samples: bool, enabled: bool, optimisation: u32) -> u32 {
        let w = CONTROL_ENABLE.put(0, enabled as u32);
        let w = CONTROL_EIGHT_SAMPLES.put(w, eight_samples as u32);
        let w = CONTROL_MASK.put(w, 0xff);
        CONTROL_OPTIMISE.put(w, optimisation)
    }

    /// The optimisation value (`OPTIMISE_*`) the firmware's multisample configuration write carries. With no
    /// multisample buffer it is 0 (none). With one it is 2 (disable) while `restricted` -- blending, a logic
    /// operation or a partial colour mask being on -- and otherwise 3 (enable), except 0 for the 12-bit-colour,
    /// 8-sample case. That exception belongs to the configuration write alone: the write that undoes a blend,
    /// logic-operation or colour-mask disable puts 3 back whatever the colour size, and this function does not
    /// describe that write (the table: values established, the effect inferred). The masked-clear case and the
    /// colour-mask flag's exception are not parameters.
    pub const fn configuration_optimisation(samples: u32, colour_12_bit: bool, restricted: bool) -> u32 {
        match (samples, restricted) {
            (0, _) => OPTIMISE_NONE,
            (_, true) => OPTIMISE_DISABLE,
            (8, _) if colour_12_bit => OPTIMISE_NONE,
            _ => OPTIMISE_ENABLE,
        }
    }

    /// The primitive sampling word (raster register 0x0806). The values are established, and the
    /// vendor diagnostic's field names settle which field is which class: the polygon fields are
    /// bits 3:2 and 15:14, the line fields bits 1:0, 9:8 and 13:12, and the enable bits 17:16; bits
    /// 11:10 are never written.
    pub const SAMPLING_REGISTER: u32 = 0x0806;
    pub const SAMPLING_LINE_MASK: BitField = BitField::new(1, 0);
    pub const SAMPLING_POLYGON_MASK: BitField = BitField::new(3, 2);
    pub const SAMPLING_SMOOTH_LINE_MASK: BitField = BitField::new(9, 8);
    pub const SAMPLING_LINE_SAMPLE: BitField = BitField::new(13, 12);
    pub const SAMPLING_POLYGON_SAMPLE: BitField = BitField::new(15, 14);
    pub const SAMPLING_ENABLE: BitField = BitField::new(17, 16);

    /// The word as the firmware leaves it for a context: the enable and the two classes. `buffer` is
    /// a multisample buffer being present. The firmware writes the line fields only when smoothing
    /// is on, or multisampling is and smoothing is not; otherwise it leaves them, and here they
    /// read as zero (a stateless image).
    pub const fn sampling_word(enabled: bool, buffer: bool, polygon_smooth: bool, line_smooth: bool) -> u32 {
        let mut w = SAMPLING_ENABLE.put(0, enabled as u32);
        w = SAMPLING_POLYGON_MASK.put(w, 1);
        let ms = enabled && buffer;
        w = SAMPLING_POLYGON_SAMPLE.put(w, if ms { 1 } else if polygon_smooth { 2 } else { 0 });
        if line_smooth {
            w = SAMPLING_LINE_MASK.put(w, 1);
            w = SAMPLING_SMOOTH_LINE_MASK.put(w, 0);
            w = SAMPLING_LINE_SAMPLE.put(w, 2);
        } else if ms {
            w = SAMPLING_LINE_MASK.put(w, 2);
            w = SAMPLING_SMOOTH_LINE_MASK.put(w, 1);
            w = SAMPLING_LINE_SAMPLE.put(w, 0);
        }
        w
    }

    /// The most leading ones the depth clear word's count field holds.
    pub const DEPTH_CLEAR_MAX_ONES: u32 = 14;

    /// The 32-bit fraction the clear word is built from: the depth times 2^32, less one, which is all
    /// ones for 1.0 and above and zero for zero and below. Established; the rounding is not read, so a
    /// fraction is truncated here (inferred).
    pub fn depth_clear_fraction(depth: f64) -> u32 {
        if depth >= 1.0 {
            u32::MAX
        } else {
            (depth * 4_294_967_296.0 - 1.0).clamp(0.0, 4_294_967_295.0) as u32
        }
    }

    /// The depth clear value (raster register 0x1c09) for a depth other than 32 bits, from the clear
    /// depth's 32-bit fraction `f` (`depth_clear_fraction`). The word's top four bits count the leading
    /// ones of `f`, at most 14. After those ones the terminating zero is dropped, unless the count
    /// reached 14, and the next bits are the mantissa; its split depends on the sample count. With a
    /// nonzero sample count the next 12 bits go to 27:16, `f[20:13]` to 7:0 and 0xf000 is set; with
    /// none the next 11 bits go to 27:17 and `f[20:13]` to 15:8. Established.
    pub const fn depth_clear_word(f: u32, sampled: bool) -> u32 {
        let ones = if f.leading_ones() < DEPTH_CLEAR_MAX_ONES { f.leading_ones() } else { DEPTH_CLEAR_MAX_ONES };
        let after = if ones < DEPTH_CLEAR_MAX_ONES { f << (ones + 1) } else { f << DEPTH_CLEAR_MAX_ONES };
        let low = f >> 13 & 0xff;
        let w = ones << 28;
        if sampled {
            w | (after >> 20) << 16 | 0xf000 | low
        } else {
            w | (after >> 21) << 17 | low << 8
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_control_word_follows_the_firmwares_writes() {
            // Four samples, enabled, optimisation 3: bit 0, mask 0xff in 10:3, 3 in 13:12.
            assert_eq!(control_word(false, true, OPTIMISE_ENABLE), 1 | 0xff << 3 | 3 << 12);
            assert_eq!(control_word(true, true, OPTIMISE_DISABLE), 1 | 2 | 0xff << 3 | 2 << 12);
            assert_eq!(control_word(false, false, 0), 0xff << 3);
        }

        #[test]
        fn the_configuration_optimisation_is_forced_off_under_blend_logic_op_and_colour_mask() {
            assert_eq!(configuration_optimisation(4, false, false), OPTIMISE_ENABLE);
            assert_eq!(configuration_optimisation(4, false, true), OPTIMISE_DISABLE);
            assert_eq!(configuration_optimisation(8, true, false), OPTIMISE_NONE, "the 12-bit-colour, 8-sample case");
            assert_eq!(configuration_optimisation(8, true, true), OPTIMISE_DISABLE, "blending writes 2 whatever the colour size");
            assert_eq!(configuration_optimisation(8, false, true), OPTIMISE_DISABLE);
            assert_eq!(configuration_optimisation(0, false, true), OPTIMISE_NONE, "no multisample buffer");
        }

        #[test]
        fn the_primitive_sampling_word_follows_the_enable_and_smoothing() {
            // Multisampled, no smoothing: polygons 1 and 1, lines 2, 1 and 0, enable 1.
            let w = sampling_word(true, true, false, false);
            assert_eq!([SAMPLING_ENABLE.get(w), SAMPLING_POLYGON_MASK.get(w), SAMPLING_POLYGON_SAMPLE.get(w)], [1, 1, 1]);
            assert_eq!([SAMPLING_LINE_MASK.get(w), SAMPLING_SMOOTH_LINE_MASK.get(w), SAMPLING_LINE_SAMPLE.get(w)], [2, 1, 0]);
            // Not multisampled, smooth polygons and lines.
            let w = sampling_word(false, true, true, true);
            assert_eq!([SAMPLING_ENABLE.get(w), SAMPLING_POLYGON_SAMPLE.get(w)], [0, 2]);
            assert_eq!([SAMPLING_LINE_MASK.get(w), SAMPLING_SMOOTH_LINE_MASK.get(w), SAMPLING_LINE_SAMPLE.get(w)], [1, 0, 2]);
            // Nothing on: only the polygon mask.
            assert_eq!(sampling_word(false, false, false, false), 1 << 2);
            // Enabled but no buffer behaves as not multisampled.
            assert_eq!(SAMPLING_POLYGON_SAMPLE.get(sampling_word(true, false, false, false)), 0);
        }

        #[test]
        fn the_depth_clear_fraction_is_the_depth_times_two_to_the_32_less_one() {
            assert_eq!(depth_clear_fraction(1.0), u32::MAX);
            assert_eq!(depth_clear_fraction(2.0), u32::MAX);
            assert_eq!(depth_clear_fraction(0.0), 0);
            assert_eq!(depth_clear_fraction(0.5), 0x7fff_ffff);
            assert_eq!(depth_clear_fraction(0.75), 0xbfff_ffff);
        }

        /// The count is the leading ones of the fraction, capped at 14; the terminating zero goes unless the
        /// cap was reached; then the mantissa splits by the sample count.
        #[test]
        fn the_depth_clear_word_counts_leading_ones_and_splits_the_mantissa() {
            // 0xdeadbeef starts 110: two ones, the zero dropped, then 1 1110 1010 1101 ...; f[20:13] is 0x6d.
            assert_eq!(depth_clear_word(0xdead_beef, true), 2 << 28 | 0xf56 << 16 | 0xf000 | 0x6d);
            assert_eq!(depth_clear_word(0xdead_beef, false), 2 << 28 | 0x7ab << 17 | 0x6d << 8);
            // No leading one: the count is zero and only the zero goes.
            assert_eq!(depth_clear_word(0x7fff_ffff, true), 0xfff << 16 | 0xf000 | 0xff);
            // Depth 1.0 is all ones: the count caps at 14 and no zero is dropped, so the next 12 bits are ones.
            assert_eq!(depth_clear_word(depth_clear_fraction(1.0), true), 14 << 28 | 0xfff << 16 | 0xf000 | 0xff);
            assert_eq!(depth_clear_word(depth_clear_fraction(1.0), false), 14 << 28 | 0x7ff << 17 | 0xff << 8);
            // Exactly 14 ones then a zero keeps the zero in the mantissa, since the cap was reached.
            assert_eq!(depth_clear_word(0xfffc_0000, true), 14 << 28 | 0xf000 | 0xe0);
            // Fifteen ones also cap at 14, and the next bit (a one) is mantissa.
            assert_eq!(depth_clear_word(0xfffe_0000, true), 14 << 28 | 0x800 << 16 | 0xf000 | 0xf0);
            // Zero is no ones, and nothing after.
            assert_eq!(depth_clear_word(0, true), 0xf000);
            assert_eq!(depth_clear_word(0, false), 0);
        }
    }
}

/// GL pixel operations: parameter names, format and type codes and the accumulation
/// operations, as the pixel-state tokens and the pixel headers carry them (section 7.15 of the
/// definitions table). The packed type enumerants are GL's public ones.
pub mod pixel {
    /// Pixel-store names: the unpack group starts at `UNPACK_BASE`, the pack group at
    /// `PACK_BASE`, each six names in the order swap bytes, least-significant-bit first, row
    /// length, skip rows, skip pixels, alignment; the volume names follow.
    pub const UNPACK_BASE: u32 = 0xcf0;
    pub const PACK_BASE: u32 = 0xd00;
    pub const STORE_NAMES: u32 = 6;
    pub const UNPACK_SKIP_IMAGES: u32 = 0x806d;
    pub const UNPACK_IMAGE_HEIGHT: u32 = 0x806e;
    pub const UNPACK_SKIP_VOLUMES: u32 = 0x8132;
    pub const UNPACK_IMAGE_DEPTH: u32 = 0x8133;
    pub const PACK_SKIP_IMAGES: u32 = 0x806b;
    pub const PACK_IMAGE_HEIGHT: u32 = 0x806c;
    pub const PACK_SKIP_VOLUMES: u32 = 0x8130;
    pub const PACK_IMAGE_DEPTH: u32 = 0x8131;
    /// Pixel-transfer names (value a float unless noted).
    pub const MAP_COLOUR: u32 = 0xd10;
    pub const MAP_STENCIL: u32 = 0xd11;
    pub const INDEX_SHIFT: u32 = 0xd12;
    pub const INDEX_OFFSET: u32 = 0xd13;
    pub const RED_SCALE: u32 = 0xd14;
    pub const RED_BIAS: u32 = 0xd15;
    pub const GREEN_SCALE: u32 = 0xd18;
    pub const GREEN_BIAS: u32 = 0xd19;
    pub const BLUE_SCALE: u32 = 0xd1a;
    pub const BLUE_BIAS: u32 = 0xd1b;
    pub const ALPHA_SCALE: u32 = 0xd1c;
    pub const ALPHA_BIAS: u32 = 0xd1d;
    pub const DEPTH_SCALE: u32 = 0xd1e;
    pub const DEPTH_BIAS: u32 = 0xd1f;
    /// Post-convolution scales (red to alpha) then biases, and post-colour-matrix scales then biases.
    pub const POST_CONVOLUTION_SCALE: u32 = 0x801c;
    pub const POST_CONVOLUTION_BIAS: u32 = 0x8020;
    pub const POST_COLOUR_MATRIX_SCALE: u32 = 0x80b4;
    pub const POST_COLOUR_MATRIX_BIAS: u32 = 0x80b8;
    /// Pixel maps, by name.
    pub const MAP_INDEX_TO_INDEX: u32 = 0xc70;
    pub const MAP_STENCIL_TO_STENCIL: u32 = 0xc71;
    pub const MAP_INDEX_TO_RED: u32 = 0xc72;
    pub const MAP_RED_TO_RED: u32 = 0xc76;
    /// The pixel-mode suspend token's selector values and mode bits.
    /// (The pixel-mode word of this token is the colour-matrix classification: a 16-bit map with bit
    /// 4 x row + column set when that element is exactly 1.0 and every other is 0, 0x8421 for the
    /// identity, 0xffff for any other matrix; the suspend token sets 0x8421. Established, orientation inferred.)
    pub const SUSPEND_USE_TEXEL_COPIES: u32 = 2;
    pub const SUSPEND_USE_DEFAULT_RECORDS: u32 = 4;
    pub const SUSPEND_USE_LIST_RECORD: u32 = 0x20;
    pub const SUSPEND_SWITCH_RECORDS: u32 = 1 << 29;
    pub const SUSPEND_SAVE: u32 = 1 << 30;
    /// Format codes.
    pub const FORMAT_RED: u32 = 0;
    pub const FORMAT_GREEN: u32 = 1;
    pub const FORMAT_BLUE: u32 = 2;
    pub const FORMAT_ALPHA: u32 = 3;
    pub const FORMAT_RGB: u32 = 4;
    pub const FORMAT_RGBA: u32 = 5;
    pub const FORMAT_LUMINANCE: u32 = 6;
    pub const FORMAT_LUMINANCE_ALPHA: u32 = 7;
    /// Codes 8 to 10 are the extension formats 0x8000 (ABGR, four components), 0x800c (CMYK, four) and
    /// 0x800d (CMYKA, five). The client's format table is indexed by the public format less 0x1900 for
    /// 0x1900 to 0x190a, and the three extension enumerants index its slots 11 to 13, whose entries are
    /// these codes. The pipe's component table by format code is 1, 1, 1, 1, 3, 4, 1, 2, 4, 4, 5, 1, 1,
    /// 1 and 0 for code 14. Established.
    pub const FORMAT_ABGR: u32 = 8;
    pub const FORMAT_CMYK: u32 = 9;
    pub const FORMAT_CMYKA: u32 = 10;
    pub const FORMAT_COLOUR_INDEX: u32 = 11;
    pub const FORMAT_STENCIL: u32 = 12;
    pub const FORMAT_DEPTH: u32 = 13;
    /// Type codes: 0 to 6 are signed byte, unsigned byte, signed short, unsigned short, signed
    /// int, unsigned int and float; 7 is the one-bit bitmap; 8 to 12 are the packed types.
    pub const TYPE_BYTE: u32 = 0;
    pub const TYPE_UNSIGNED_BYTE: u32 = 1;
    pub const TYPE_SHORT: u32 = 2;
    pub const TYPE_UNSIGNED_SHORT: u32 = 3;
    pub const TYPE_INT: u32 = 4;
    pub const TYPE_UNSIGNED_INT: u32 = 5;
    pub const TYPE_FLOAT: u32 = 6;
    pub const TYPE_BITMAP: u32 = 7;
    pub const TYPE_3_3_2: u32 = 8;
    pub const TYPE_4_4_4_4: u32 = 9;
    pub const TYPE_5_5_5_1: u32 = 10;
    pub const TYPE_8_8_8_8: u32 = 11;
    pub const TYPE_10_10_10_2: u32 = 12;
    /// Copy types.
    pub const COPY_COLOUR: u32 = 0x1800;
    pub const COPY_DEPTH: u32 = 0x1801;
    pub const COPY_STENCIL: u32 = 0x1802;
    /// Accumulation operations, GL's enumerants. The engine's accumulation operation register
    /// (raster register 0x1c0e) holds the value in bits 12:0, the operation (the enumerant minus
    /// 0x100: accumulate 0, load 1, return 2, multiply 3, add 4) in bits 15:13, a return-by-four
    /// flag in bit 16, a precision flag in bit 17, a speed-up flag in bit 18 and a three-bit
    /// low-order extension of the value in bits 21:19 (established from the diagnostic's table).
    /// A value of exactly 1.0 is two passes, since it does not fit the value field (inferred).
    pub const ACCUM: u32 = 0x100;
    pub const ACCUM_LOAD: u32 = 0x101;
    pub const ACCUM_RETURN: u32 = 0x102;
    pub const ACCUM_MULT: u32 = 0x103;
    pub const ACCUM_ADD: u32 = 0x104;
    /// A small read's reply goes in the return page when it is not larger than this.
    pub const READ_REPLY_LIMIT: usize = 0x3ff0;
    /// A small bitmap's pixel count limit (token 0x005).
    pub const SMALL_BITMAP_PIXELS: u32 = 0xc40;
    /// The overlap query's reply bit.
    pub const OVERLAP_BIT: u32 = 1 << 2;
}
