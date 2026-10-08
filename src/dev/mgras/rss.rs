//! The raster subsystem: the raster engine's register file, its indirect
//! device space, pixel transfers, and the framebuffer it draws into.
//!
//! Registers are numbered 0..0x3FF. A register write may carry an "execute"
//! flag, which runs the primitive held in the instruction register (IR) once
//! the write has landed.
//!
//! Coordinates: primitives give block corners in window coordinates. The
//! window origin (`xywin`: y in the high half, x in the low) is added, and
//! with Y-flip set in `config` y runs downward from it. The PROM draws with no
//! origin and no flip; the X server sets the origin to the top row and flips,
//! so it draws top-down. Framebuffer row 0 is at the bottom.
//!
//! Pixels live in page memory (`pixmem`): drawing goes to the buffer whose
//! page pointer is in `DRBpointers` bits 9:0, and the PP1 fill mode's
//! draw-buffer field says what kind it is (overlay or 36-bit). Both X
//! servers switch the pointer to the overlay's pages before overlay
//! drawing (6.5.0: 0x101C0 / 0x90240, 6.5.22: 0x38120 / 0xB81C0).
//!
//! A block runs one of several ways, chosen by the fill mode's block type:
//! fill it (fast fill uses the fill colour registers, others the red
//! iterator), stipple it with character data, or move pixels in or out of it
//! (by PIO through the character registers, or by DMA).

use super::pixmem::{Buffer, Kind, PixMem};
use super::plain::RegMap;
use super::te1::{reg as te_reg, tex_env, Sampler, Te1, ITER_ONE, TEXMODE1_ENABLE};

/// Framebuffer coordinates the model accepts (x and y, 0..2048); where a
/// pixel lands in memory depends on its buffer (see `pixmem`).
pub const WIDTH: usize = 2048;
pub const HEIGHT: usize = 2048;

/// Rss registers. Those OpenBSD's impact(4) driver also uses carry its
/// names; the rest are named for what they do here.
pub mod reg {
    /// The instruction register: the primitive a write with "execute" runs.
    pub const IR: u32 = 0x013;
    pub const LINE_START: u32 = 0x040;
    pub const LINE_END: u32 = 0x041;
    pub const IR_ALIAS: u32 = 0x045;
    pub const BLOCKXYSTARTI: u32 = 0x046;
    pub const BLOCKXYENDI: u32 = 0x047;
    /// Packed RGB colour for character and line drawing in RGB modes.
    pub const PACKEDCOLOR: u32 = 0x05B;
    pub const RED: u32 = 0x05C;
    pub const CHAR_H: u32 = 0x070;
    pub const CHAR_L: u32 = 0x071;
    pub const XFRCONTROL: u32 = 0x102;
    /// Fog colour: red 11:0, green 23:12; blue 11:0 (12 bits each).
    pub const FOG_RG: u32 = 0x144;
    pub const FOG_B: u32 = 0x145;
    /// Model-private (our GE HLE and this RSS agree on them; the GE11's
    /// own way of fogging after texturing is not visible on the bus): the
    /// fog factor's plane, 64-bit pairs at 2^32 (start, step along the span
    /// or line, step down the edge), and whether it applies.
    pub const FOG_F: u32 = 0x3F0;
    pub const FOG_ON: u32 = 0x3F6;
    /// Model-private: a TE read returns TRAM's bytes as they lie (the
    /// texture manager's save of pages), not texels.
    pub const TE_RAW: u32 = 0x3F7;
    pub const FILLMODE: u32 = 0x110;
    pub const CONFIG: u32 = 0x112;
    pub const XYWIN: u32 = 0x115;
    /// Background colour, where an opaque stipple draws its 0 bits: a colour
    /// index, or in RGB modes blue (bits 23:12) and green (11:0), 12 bits
    /// each, with red in the next register.
    pub const BG_COLOR: u32 = 0x140;
    pub const BG_COLOR_RED: u32 = 0x141;
    /// The clip rectangle: x and y ranges, each `min << 16 | max`, and its
    /// control (bit 0 enable, bit 4 keep the inside rather than the outside).
    pub const CLIP_X: u32 = 0x147;
    pub const CLIP_Y: u32 = 0x148;
    pub const CLIP_MODE: u32 = 0x14F;
    pub const XFRSIZE: u32 = 0x153;
    pub const XFRMODE: u32 = 0x159;
    pub const LINE_STIPPLE: u32 = 0x15A;
    /// The indirect device space: an address, then its data.
    pub const INDIRECT_ADDR: u32 = 0x15C;
    pub const INDIRECT_DATA: u32 = 0x15D;
    pub const STATUS: u32 = 0x15E;
    pub const PP1FILLMODE: u32 = 0x161;
    pub const PP1WINMODE: u32 = 0x17B;
    /// Plane write mask (low planes, buffer A).
    pub const COLORMASKLSBSA: u32 = 0x163;
    /// Plane write mask (low planes, buffer B).
    pub const COLORMASKLSBSB: u32 = 0x164;
    /// Page pointer of the buffer drawn into (bits 9:0).
    pub const DRBPOINTERS: u32 = 0x16D;
    /// Tiles per row: bits 5:2 for 36-bit buffers, 1:0 for the overlay.
    pub const DRBSIZE: u32 = 0x16E;
    pub const ZMODE: u32 = 0x168;
    pub const BLENDFACTOR: u32 = 0x165;
    pub const STENCILMODE: u32 = 0x166;
    pub const STENCILMASK: u32 = 0x167;
    pub const AFUNCMODE: u32 = 0x169;
    pub const COLORMASKMSBS: u32 = 0x162;
    pub const GLINECONFIG: u32 = 0x146;
    pub const LSCRL: u32 = 0x15B;
    /// Fast-fill colour: one 12-bit component each, or the index in R.
    pub const FILL_COLOR_R: u32 = 0x176;
    pub const FILL_COLOR_G: u32 = 0x177;
    pub const FILL_COLOR_B: u32 = 0x178;
}

/// Raster register names (MGRAS.h `rss_single`; TE registers 0x80-0xCB in
/// their non-warp meaning). Unnamed registers are empty.
pub const REG_NAMES: [&str; 0x200] = {
    let mut n = [""; 0x200];
    n[0x000] = "tri_x0";
    n[0x001] = "tri_x1";
    n[0x002] = "tri_x2";
    n[0x003] = "tri_ymax";
    n[0x004] = "tri_ymid";
    n[0x005] = "tri_ymin";
    n[0x006] = "tri_dxdy0";
    n[0x007] = "tri_dxdy0_low";
    n[0x008] = "tri_dxdy1_hi";
    n[0x009] = "tri_dxdy1_low";
    n[0x00A] = "tri_dxdy2_hi";
    n[0x00B] = "tri_dxdy2_low";
    n[0x00C] = "gline_xstartf";
    n[0x00D] = "gline_ystartf";
    n[0x00E] = "gline_xendf";
    n[0x00F] = "gline_yendf";
    n[0x010] = "gline_dx";
    n[0x011] = "gline_dy";
    n[0x012] = "gline_adjust";
    n[0x013] = "ir";
    n[0x040] = "xline_xystarti";
    n[0x041] = "xline_xyendi";
    n[0x042] = "xline_inc1";
    n[0x043] = "xline_inc2";
    n[0x044] = "xline_error_oct";
    n[0x045] = "ir_alias";
    n[0x046] = "block_xystarti";
    n[0x047] = "block_xyendi";
    n[0x048] = "block_xymove";
    n[0x049] = "block_xsaveoct";
    n[0x05B] = "packed_color";
    n[0x05C] = "red";
    n[0x05D] = "green";
    n[0x05E] = "blue";
    n[0x05F] = "alpha";
    n[0x060] = "dre";
    n[0x061] = "drx";
    n[0x062] = "dge";
    n[0x063] = "dgx";
    n[0x064] = "dbe";
    n[0x065] = "dbx";
    n[0x066] = "dae";
    n[0x067] = "dax";
    n[0x068] = "z_hi";
    n[0x069] = "z_low";
    n[0x06A] = "dzx_hi";
    n[0x06B] = "dzx_low";
    n[0x06C] = "dze_hi";
    n[0x06D] = "dze_low";
    n[0x070] = "char_h";
    n[0x071] = "char_l";
    n[0x080] = "sw_u";
    n[0x081] = "sw_l";
    n[0x082] = "tw_dtx_u";
    n[0x083] = "tw_dtx_l";
    n[0x084] = "wi_u";
    n[0x085] = "wi_l";
    n[0x086] = "dwie_u";
    n[0x087] = "dwie_l";
    n[0x088] = "dwix_u";
    n[0x089] = "dwix_l";
    n[0x08A] = "dwiy_u";
    n[0x08B] = "dwiy_l";
    n[0x08C] = "txscale";
    n[0x0C0] = "dswe_u";
    n[0x0C1] = "dswe_l";
    n[0x0C2] = "dtwe_u";
    n[0x0C3] = "dtwe_l";
    n[0x0C4] = "dswx_u";
    n[0x0C5] = "dswx_l";
    n[0x0C6] = "dtwx_u";
    n[0x0C7] = "dtwx_l";
    n[0x0C8] = "dswy_u";
    n[0x0C9] = "dswy_l";
    n[0x0CA] = "dtwy_u";
    n[0x0CB] = "dtwy_l";
    n[0x0CC] = "dsex_u";
    n[0x0CD] = "dsex_l";
    n[0x0CE] = "dtex_u";
    n[0x0CF] = "dtex_l";
    n[0x100] = "xfrabort";
    n[0x102] = "xfrcontrol";
    n[0x110] = "fillmode";
    n[0x111] = "texmode1";
    n[0x112] = "config";
    n[0x113] = "scissorx";
    n[0x114] = "scissory";
    n[0x115] = "xywin";
    n[0x140] = "bkgrd_rg";
    n[0x141] = "bkgrd_ba";
    n[0x142] = "txenv_rg";
    n[0x143] = "txenv_b";
    n[0x144] = "fog_rg";
    n[0x145] = "fog_b";
    n[0x146] = "glineconfig";
    n[0x147] = "scrmsk1x";
    n[0x148] = "scrmsk1y";
    n[0x149] = "scrmsk2x";
    n[0x14A] = "scrmsk2y";
    n[0x14B] = "scrmsk3x";
    n[0x14C] = "scrmsk3y";
    n[0x14D] = "scrmsk4x";
    n[0x14E] = "scrmsk4y";
    n[0x14F] = "winmode";
    n[0x153] = "xfrsize";
    n[0x154] = "xfrinitfactor";
    n[0x155] = "xfrfactor";
    n[0x156] = "xfrmasklow";
    n[0x157] = "xfrmaskhigh";
    n[0x158] = "xfrcounters";
    n[0x159] = "xfrmode";
    n[0x15A] = "lspat";
    n[0x15B] = "lscrl";
    n[0x15C] = "device_addr";
    n[0x15D] = "device_data";
    n[0x15E] = "status";
    n[0x15F] = "re_togglecntx";
    n[0x160] = "PixCmd";
    n[0x161] = "pp1fillmode";
    n[0x162] = "ColorMaskMSBs";
    n[0x163] = "ColorMaskLSBsA";
    n[0x164] = "ColorMaskLSBsB";
    n[0x165] = "BlendFactor";
    n[0x166] = "Stencilmode";
    n[0x167] = "Stencilmask";
    n[0x168] = "Zmode";
    n[0x169] = "Afuncmode";
    n[0x16A] = "Accmode";
    n[0x16B] = "BlendcolorRG";
    n[0x16C] = "BlendcolorBA";
    n[0x16D] = "DRBpointers";
    n[0x16E] = "DRBsize";
    n[0x16F] = "TAGmode";
    n[0x170] = "TAGdata_R";
    n[0x171] = "TAGdata_G";
    n[0x172] = "TAGdata_B";
    n[0x173] = "TAGdata_A";
    n[0x174] = "TAGdata_Z";
    n[0x175] = "PixCoordinate";
    n[0x176] = "PIXcolorR";
    n[0x177] = "PIXcolorG";
    n[0x178] = "PIXcolorB";
    n[0x179] = "PIXcolorA";
    n[0x17A] = "PIXcolorAcc";
    n[0x17B] = "pp1winmode";
    n[0x17E] = "MainBufSel";
    n[0x17F] = "OlayBufSel";
    n[0x180] = "texmode2";
    n[0x181] = "txsize";
    n[0x182] = "txtile";
    n[0x183] = "txlod";
    n[0x184] = "txbcolor_rg";
    n[0x185] = "txbcolor_ba";
    n[0x186] = "txbase";
    n[0x187] = "tram_cntrl";
    n[0x188] = "max_pixel_rg";
    n[0x189] = "max_pixel_ba";
    n[0x18A] = "mddma_cntrl";
    n[0x18B] = "texrbuffer";
    n[0x18C] = "txclampsize";
    n[0x190] = "txaddr";
    n[0x191] = "txmipmap";
    n[0x192] = "txdetail";
    n[0x193] = "txborder";
    n[0x194] = "detailscale";
    n[0x195] = "te_togglecntx";
    n[0x196] = "teversion";
    n[0x1A0] = "tl_wbuffer";
    n[0x1A1] = "tl_mode";
    n[0x1A2] = "tl_spec";
    n[0x1A3] = "tl_addr";
    n[0x1A4] = "tl_mipmap";
    n[0x1A5] = "tl_border";
    n[0x1A6] = "tl_vidcntrl";
    n[0x1A7] = "tl_base";
    n[0x1A8] = "tl_vidabort";
    n[0x1A9] = "tl_s_size";
    n[0x1AA] = "tl_t_size";
    n[0x1AB] = "tl_s_left";
    n[0x1AC] = "tl_t_bottom";
    n[0x1AF] = "reserved";
    n
};

/// IR opcodes: a line between two points, and a block (rectangle).
const OP_LINE: u32 = 0x5;
const OP_BLOCK: u32 = 0x8;
/// IR opcode: a point at xline_xystarti, drawn by each execute of it (X
/// PolyPoint).
const OP_POINT: u32 = 0x4;
/// IR opcodes: a triangle (area) whose spans run left to right (the major
/// edge on the left) or right to left. SGI's names are IR_OP_AREA_LTOR /
/// IR_OP_AREA_RTOL; their numbers are not known (only GE11 microcode writes
/// them), these are provisional and used by our GE11 HLE alone.
pub const OP_AREA_LTOR: u32 = 0x0;
pub const OP_AREA_RTOL: u32 = 0x1;
/// IR opcode: an OpenGL line (IR_OP_GL_LINE; number provisional, ours).
pub const OP_GL_LINE: u32 = 0x2;

/// The OpenGL per-fragment state our GE11 HLE programs into the PP1
/// registers. The real layouts are not known (only GE11 microcode writes
/// them); these follow the shapes SGI's diagnostics' reset values suggest
/// (Afuncmode 0x7, Stencilmode 0x7, Stencilmask 0xFFFF, Zmode 0x0FFFFFF1:
/// compare function in bits 2:0, masks) and are provisional:
///   Afuncmode (0x169)   bits 2:0 compare, bit 3 enable, bits 15:4 ref * 4096
///   Stencilmode (0x166) bits 2:0 compare, bit 3 enable, ops (KEEP ZERO
///                       REPLACE INCR DECR INVERT) fail 6:4, zfail 10:8,
///                       zpass 14:12, ref 23:16
///   Stencilmask (0x167) compare mask 7:0, write mask 15:8
///   BlendFactor (0x165) source 3:0, destination 7:4 (ZERO ONE SRC_COLOR
///                       ONE_MINUS_SRC_COLOR SRC_ALPHA ONE_MINUS_SRC_ALPHA
///                       DST_ALPHA ONE_MINUS_DST_ALPHA DST_COLOR
///                       ONE_MINUS_DST_COLOR SRC_ALPHA_SATURATE), bit 8 enable
///   ColorMaskLSBsA / MSBs  RGB / alpha plane write masks
///   fill mode bit 2     polygon stipple, rows at device address POLY_STIPPLE_RAM
/// Compare functions in OpenGL order: NEVER LESS EQUAL LEQUAL GREATER
/// NOTEQUAL GEQUAL ALWAYS, "incoming OP stored".
const TEST_ENABLE: u32 = 1 << 3;
const BLEND_ENABLE: u32 = 1 << 8;
const FILL_POLY_STIPPLE: u32 = 1 << 2;
/// Polygon stipple RAM in the indirect device space (SGI's POLYSTIP_RAM).
pub const POLY_STIPPLE_RAM: u32 = 0xA000_0000;

/// `a` OP `b` for an OpenGL compare function (0 NEVER .. 7 ALWAYS).
fn compare(func: u32, a: f64, b: f64) -> bool {
    match func & 7 {
        0 => false,
        1 => a < b,
        2 => a == b,
        3 => a <= b,
        4 => a > b,
        5 => a != b,
        6 => a >= b,
        _ => true,
    }
}
/// Fill mode: a line leaves out its last pixel (X's CapNotLast).
const FILL_LINE_SKIP_LAST: u32 = 1 << 10;
/// Fill mode: lines follow the 32-bit line stipple pattern.
const FILL_LINE_STIPPLE: u32 = 1 << 5;
/// Fill mode: the line stipple is opaque, its 0 bits drawn in the background
/// colour instead of left alone. The desktop shades icons this way: a 50%
/// pattern of the foreground and background colours.
const FILL_LINE_STIPPLE_OPAQUE: u32 = 1 << 6;
/// Status: command FIFO empty, engine and pixel processors idle, revision 1.
const STATUS_IDLE: u32 = 0x100 | (1 << 4);
/// Config: Y-flip.
const CONFIG_YFLIP: u32 = 1 << 3;
/// Fill mode: fast fill (solid, from the fill colour registers).
const FILL_FAST: u32 = 1 << 20;
/// Fill mode: char data stipples the block (with block type 1), and the
/// stipple is opaque, its 0 bits drawn in the background colour.
const FILL_CHAR_STIPPLE: u32 = 1 << 3;
const FILL_CHAR_STIPPLE_OPAQUE: u32 = 1 << 4;
/// PP1 fill mode draw-buffer field (bits 20:14), as the drivers use it:
/// 0x01 the main colour buffer (A), 0x02 the second (B), 0x03 both (see
/// `Rss::target`), 0x4F the overlay planes (4Dwm menus, overlay
/// clears), 0x50 the clip-ID planes (see `Rss::cid`). The 6.5.22
/// TrueColor server keeps `DRBpointers` at 0xB81C0
/// (A 0x1C0, B 0x2E0) for all drawing and picks the buffer here.
fn draw_buffer(pp1fillmode: u32) -> u32 {
    (pp1fillmode >> 14) & 0x7F
}
const DRAW_B: u32 = 0x02;
const DRAW_A_AND_B: u32 = 0x03;
const DRAW_CID: u32 = 0x50;

/// PP1 window mode (`pp1winmode`; SGI's fields WINxLSBs, WINyLSBs,
/// CIDmatch, CIDdata, CIDmask). Bits 3:0 are the window origin's low x and
/// y bits (not modelled). Bits 7:4 (CIDmatch) are one bit per clip ID:
/// bit 4 + n lets a pixel whose clip ID is n be drawn; all clear, no
/// check. The kernel (MgrasValidateClip) sets 1 << (4 + n) for a window
/// that X gave clip ID n, else 0. Bits 11:10 (CIDmask, our reading) write
/// enable the two clip-ID planes: the X server (mgrasDrawCID) sets 0xC00
/// to draw clip IDs and leaves it set.
fn cid_match(pp1winmode: u32) -> u32 {
    (pp1winmode >> 4) & 0xF
}
fn cid_write_mask(pp1winmode: u32) -> u8 {
    ((pp1winmode >> 10) & 3) as u8
}

/// PP1 fill mode read-buffer field (bits 25:21), whatever the draw field:
/// 0 the first buffer, DRBpointers bits 9:0; 1 the second buffer, DRBpointers bits 19:10
/// (the file manager scrolls its double-buffered 12-bit window, drawn with
/// draw field 2, by reading it with read field 1 and draw field 0 and
/// writing it back a line up); 4 the overlay, when the X server reads it
/// back (with draw field 0 and DRBpointers at the overlay's pages: it
/// copies popup menu pixels through host memory).
fn read_buffer(pp1fillmode: u32) -> u32 {
    (pp1fillmode >> 21) & 0x1F
}
const READ_B: u32 = 1;
const READ_OVERLAY: u32 = 4;

/// Block types (fill mode bits 24:22).
mod block {
    pub const NORMAL: u32 = 1;
    pub const PIO_READ: u32 = 2;
    pub const PIO_WRITE: u32 = 3;
    pub const DMA_READ: u32 = 4;
    pub const DMA_WRITE: u32 = 5;
}

/// Whether the pixel processors' pixel type (fill mode bits 10:8) is an RGB
/// one; the others are colour index.
fn rgb_pixtype(pp1fillmode: u32) -> bool {
    matches!((pp1fillmode >> 8) & 7, 0 | 1 | 2 | 4)
}

/// Pixel processor logic op (fill mode bit 2 enables it; bits 29:26 hold
/// the X11 function number) applied to source `s` and destination `d`.
fn logic_op(op: u32, s: u32, d: u32) -> u32 {
    match op & 0xF {
        0x0 => 0,
        0x1 => s & d,
        0x2 => s & !d,
        0x3 => s,
        0x4 => !s & d,
        0x5 => d,
        0x6 => s ^ d,
        0x7 => s | d,
        0x8 => !(s | d),
        0x9 => !(s ^ d),
        0xA => !d,
        0xB => s | !d,
        0xC => !s,
        0xD => !s | d,
        0xE => !(s & d),
        _ => !0,
    }
}
const PP1_LOGIC_OP_ENABLE: u32 = 1 << 2;

/// Framebuffer pixels: colour indices as they are, RGB as `0x00BBGGRR` with
/// eight bits per component.
fn pack_rgb(r: u32, g: u32, b: u32) -> u32 {
    (r & 0xFF) | (g & 0xFF) << 8 | (b & 0xFF) << 16
}

/// A host pixel of transfer format (PixelFormat, CompType) to a framebuffer
/// pixel, and back. Only the RGB formats convert.
/// Transfer format (PixelFormat, CompType) of depth pixels.
const DEPTH_FORMAT: (u32, u32) = (2, 3);

fn from_host(format: (u32, u32), v: u64) -> u32 {
    // RGB with 16-bit components (glCopyPixels' round trip through the
    // host, 6 bytes a pixel, red first): the top 8 bits of each.
    if format == (7, 1) {
        return pack_rgb((v >> 40) as u32, (v >> 24) as u32, (v >> 8) as u32);
    }
    // RGBA with 16-bit components (glCopyPixels draws back so).
    if format == (8, 1) {
        return pack_rgb((v >> 56) as u32, (v >> 40) as u32, (v >> 24) as u32);
    }
    // RGB, a byte each, red first (glCopyTexImage reads the screen so).
    if format == (7, 0) {
        return pack_rgb((v >> 16) as u32, (v >> 8) as u32, v as u32);
    }
    // Depth (glReadPixels / glCopyPixels of GL_DEPTH_COMPONENT): 32 bits
    // full scale on the host (libGLcore's scale is 1 / 2^31), 24 here.
    if format == DEPTH_FORMAT {
        return (v as u32 >> 8) & 0xFF_FFFF;
    }
    let v = v as u32;
    let c4 = |s: u32| ((v >> s) & 0xF) * 0x11;
    let c5 = |s: u32| ((v >> s) & 0x1F) << 3 | ((v >> s) & 0x1F) >> 2;
    match format {
        (8, 8) => pack_rgb(c4(0), c4(4), c4(8)),
        (8, 10) => pack_rgb(c5(0), c5(5), c5(10)),
        (8, 0) => v,
        (0, 1) => v & 0xFFF,
        _ => v,
    }
}

fn to_host(format: (u32, u32), v: u32) -> u64 {
    let c = |s: u32| (v >> s) & 0xFF;
    if format == (7, 1) {
        let w = |s: u32| c(s) as u64 * 0x101;
        return w(0) << 32 | w(8) << 16 | w(16);
    }
    if format == (7, 0) {
        return (c(0) << 16 | c(8) << 8 | c(16)) as u64;
    }
    if format == (8, 1) {
        let w = |s: u32| c(s) as u64 * 0x101;
        return w(0) << 48 | w(8) << 32 | w(16) << 16 | 0xFFFF;
    }
    if format == DEPTH_FORMAT {
        let z = v & 0xFF_FFFF;
        return (z << 8 | z >> 16) as u64;
    }
    (match format {
        (8, 8) => c(0) >> 4 | (c(8) >> 4) << 4 | (c(16) >> 4) << 8,
        (8, 10) => c(0) >> 3 | (c(8) >> 3) << 5 | (c(16) >> 3) << 10,
        _ => v,
    }) as u64
}

fn signed16(v: u32) -> i32 {
    v as u16 as i16 as i32
}

/// A block, in window coordinates, with its colour.
#[derive(Clone, Copy)]
struct Block {
    xs: i32,
    ys: i32,
    xe: i32,
    ye: i32,
    color: u32,
}

impl Block {
    fn dx(&self) -> i32 {
        if self.xe < self.xs { -1 } else { 1 }
    }
    fn dy(&self) -> i32 {
        if self.ye < self.ys { -1 } else { 1 }
    }
    fn rows(&self) -> i32 {
        (self.ye - self.ys).abs() + 1
    }
    fn cols(&self) -> i32 {
        (self.xe - self.xs).abs() + 1
    }
}

/// A character block being filled with stipple data, one row at a time.
#[derive(Clone, Copy)]
struct Stipple {
    block: Block,
    col: i32,
    row: i32,
    opaque: bool,
}

/// An armed pixel transfer.
#[derive(Clone, Copy)]
struct Xfer {
    block: Block,
    read: bool,
    /// Armed by a PIO read block: the host reads it through the char
    /// registers.
    pio_read: bool,
    /// Pixels per line, bytes per pixel, and (PixelFormat, CompType).
    width: u32,
    bpp: u32,
    format: (u32, u32),
    begin_skip: u32,
    stride_skip: u32,
    /// PIO write stream: the line being assembled, its byte offset within
    /// its first doubleword, the bytes collected (in `Rss::line_buf`), and
    /// filler still to skip.
    line: u32,
    line_begin: u32,
    pending: u32,
    skip: u32,
    /// PIO read stream, produced as it is read: per line, filler up to its
    /// begin offset, the pixels, padding to the doubleword. The line, the
    /// byte within it, its begin offset, and the low half of the doubleword
    /// the last high read took.
    rd_line: u32,
    rd_pos: u32,
    rd_begin: u32,
    out_lo: u32,
}

impl Xfer {
    fn line_bytes(&self) -> u32 {
        self.width * self.bpp
    }

    /// Where the line after one starting at `b` starts within its first
    /// doubleword.
    fn next_begin(&self, b: u32) -> u32 {
        (b + self.line_bytes() + self.stride_skip) & 7
    }

    /// PIO read: bytes the current line takes in the stream (filler, pixels,
    /// padding to the doubleword).
    fn rd_line_len(&self) -> u32 {
        (self.rd_begin + self.line_bytes() + 7) & !7
    }

    /// PIO read: move past finished (or empty) lines.
    fn rd_settle(&mut self) {
        while self.rd_line < self.block.rows() as u32 && self.rd_pos >= self.rd_line_len() {
            self.rd_pos = 0;
            self.rd_line += 1;
            self.rd_begin = self.next_begin(self.rd_begin);
        }
    }
}

/// Bytes per pixel for an xfrmode (PixelFormat, PixelCompType) pair.
pub(crate) fn bytes_per_pixel(xfrmode: u32) -> u32 {
    match ((xfrmode >> 4) & 0xF, xfrmode & 0xF) {
        (0, 0) => 1,
        (0, 1) | (8, 8) | (8, 10) => 2,
        (8, 0) | (2, 3) => 4,
        (7, 0) => 3,
        (8, 1) => 8,
        (7, 1) => 6,
        _ => 1,
    }
}

/// What the raster subsystem has done, for `mgras stats`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RssStats {
    /// Executed primitives by IR opcode (bits 3:0).
    pub prims: [u64; 16],
    /// Blocks by kind (fill mode bits 24:22), fast fills counted apart.
    pub blocks: [u64; 8],
    pub fast_fills: u64,
    pub stipple_chunks: u64,
    pub pio_write_dw: u64,
    pub pio_read_dw: u64,
    pub dma_lines_in: u64,
}

/// Registers a GL batch leaves as it set them when it ends (`gl_leave`
/// puts every other register back): the texture engine's, which the GE
/// keeps loaded across batches (it reloads them only when another context
/// may have used the TE), and the model-private ones from 0x3F0.
fn gl_keeps(r: usize) -> bool {
    r >= 0x3F0 || r == te_reg::TXADDR as usize || crate::dev::mgras::te1::CONTEXT_REGS.contains(&(r as u32))
}

/// PP1 Zmode, as our GE11 HLE uses it (the real layout is not known; SGI's
/// diagnostics write it as value | 0x0FFFFFF0, which fits bits 27:4 being
/// a 24-bit Z write mask): bits 2:0 the compare function in OpenGL's order
/// (NEVER, LESS, EQUAL, LEQUAL, GREATER, NOTEQUAL, GEQUAL, ALWAYS), bit 3
/// test enable (provisional), bits 27:4 the Z write mask.
const ZMODE_TEST: u32 = 1 << 3;
/// Page pointer of the depth (ZST: Z 23:0, stencil 31:24) buffer: the
/// buffer the PROM's 1280x1024 layout calls "aux" (page 0; unused by the
/// X server at 1024x768). Provisional: how the PP1 is told is not known.
pub const ZST_PAGE: u32 = 0;

/// Longest PIO line, in bytes (2048 pixels of 6 bytes, with room to spare).
const MAX_LINE_BYTES: usize = 16384;

/// The raster subsystem's state. Plain data apart from the `Option`s, which
/// `init` writes; build with `Rss::init` on zeroed memory or `new_boxed`.
#[repr(C)]
pub struct Rss {
    regs: [u32; 0x400],
    device: RegMap<256>,
    /// Pixel memory. In colour-index modes a pixel's low byte is the index.
    pub mem: PixMem,
    /// The PIO write line being assembled.
    line_buf: [u8; MAX_LINE_BYTES],
    stipple: Option<Stipple>,
    xfer: Option<Xfer>,
    pub stats: RssStats,
    /// Pixels drawn since the GL line stipple was last restarted (lscrl).
    gl_line_stipple_pos: u32,
    /// The raster registers as `gl_enter` found them.
    gl_stash: [u32; 0x400],
    /// Fill modes seen with a block, for bring-up logging.
    fillmodes: [u32; 32],
    fillmodes_n: usize,
    /// The texture engine and TRAM.
    pub te: Te1,
    /// A texture load transfer is armed (XFRCONTROL to the texture side):
    /// DMA lines go to the texture loader.
    te_load: u32,
    /// A texture read transfer is armed: armed, texels a line, lines.
    te_read: [u32; 3],
    /// The clip-ID planes: two bits a framebuffer pixel, index `y * WIDTH +
    /// x`. The X server paints a window's visible region with an ID (1-3)
    /// when its clip is too complex for the four screen masks, and the
    /// kernel has that window's GL drawing match it (`pp1winmode`). Where
    /// the board keeps them in RDRAM is not known (X draws them with
    /// DRBpointers at the main buffer); nothing reads them back, so they
    /// are kept apart here. Separate from the VC3's display IDs.
    pub cid: [u8; WIDTH * HEIGHT],
}

/// XFRCONTROL (provisional layout, from the IDE's TRAM load writing 5):
/// start by DMA or PIO (bits 0, 1), device (bit 2: 0 PP1, 1 texture),
/// direction (bit 3: read).
const XFR_START: u32 = 3;
const XFR_DEVICE_TE: u32 = 1 << 2;
const XFR_READ: u32 = 1 << 3;

impl Rss {
    /// Make zeroed memory a valid, reset `Rss`.
    ///
    /// # Safety
    /// `p` points at a zeroed, exclusively owned `Rss`.
    pub unsafe fn init(p: *mut Rss) {
        use std::ptr::addr_of_mut;
        addr_of_mut!((*p).stipple).write(None);
        addr_of_mut!((*p).xfer).write(None);
    }

    /// A GL batch begins (our GE11 HLE): save the raster registers and
    /// turn Y-flip off (GL draws y up). The X server keeps its origin,
    /// masks, flip, fill modes, colours and instruction in the same
    /// registers and sets them only when they change: after a GL batch it
    /// executes with what it left (a line with the IR as GL left it drew
    /// GL's last triangle in X's window, traced with gltest). The real RE4
    /// has a second register context for this (re_togglecntx), which is not
    /// modelled, so `gl_leave` puts the registers back.
    pub fn gl_enter(&mut self) {
        self.gl_stash = self.regs;
        self.regs[reg::CONFIG as usize] &= !CONFIG_YFLIP;
    }

    /// A GL batch ends: restore what `gl_enter` saved, but for what the GE
    /// keeps loaded (`gl_keeps`).
    pub fn gl_leave(&mut self) {
        for (r, s) in self.gl_stash.iter().enumerate() {
            if !gl_keeps(r) {
                self.regs[r] = *s;
            }
        }
    }

    /// A reset `Rss` on the heap (tests).
    #[cfg(test)]
    pub fn new_boxed() -> Box<Rss> {
        let mut b = Box::<Rss>::new_zeroed();
        // SAFETY: zeroed and exclusively owned; `init` writes the fields that
        // are not valid zeroed.
        unsafe {
            Rss::init(b.as_mut_ptr());
            b.assume_init()
        }
    }

    pub fn fillmodes_seen(&self) -> &[u32] {
        &self.fillmodes[..self.fillmodes_n]
    }
}

impl Rss {
    pub fn read(&self, r: u32) -> u32 {
        match r & 0x3FF {
            reg::STATUS => STATUS_IDLE,
            reg::INDIRECT_DATA => {
                self.device.get(self.regs[reg::INDIRECT_ADDR as usize])
            }
            r => self.regs[r as usize],
        }
    }

    pub fn reg(&self, r: u32) -> u32 {
        self.regs[r as usize]
    }

    /// Write register `r`; `exec` runs the primitive afterwards. Returns true
    /// when the framebuffer changed.
    pub fn write(&mut self, r: u32, val: u32, exec: bool) -> bool {
        let r = r & 0x3FF;
        self.regs[r as usize] = val;
        match r {
            reg::INDIRECT_DATA => {
                self.device.insert(self.regs[reg::INDIRECT_ADDR as usize], val);
                if self.device.overflowed {
                    self.device.overflowed = false;
                    eprintln!("mgras: raster device space full, write to {:#x} dropped", self.regs[reg::INDIRECT_ADDR as usize]);
                }
            }
            reg::IR_ALIAS => self.regs[reg::IR as usize] = val,
            reg::LSCRL => self.gl_line_stipple_pos = 0,
            reg::XFRCONTROL if val == 0 => {
                self.xfer = None;
                self.te_load = 0;
                self.te_read[0] = 0;
            }
            reg::XFRCONTROL if val & XFR_DEVICE_TE != 0 && val & XFR_START != 0 => {
                self.xfer = None;
                let read = val & XFR_READ != 0;
                self.te_load = (!read) as u32;
                let size = self.regs[reg::XFRSIZE as usize];
                self.te_read = [read as u32, size & 0xFFFF, size >> 16];
            }
            te_reg::TXMIPMAP | te_reg::TXBORDER | te_reg::DETAILSCALE => {
                let i = &mut self.regs[te_reg::TXADDR as usize];
                self.te.table_write(r, *i, val);
                *i = i.wrapping_add(1);
            }
            _ => {}
        }
        if !exec {
            return false;
        }
        let pio_write = self.xfer.as_ref().is_some_and(|x| !x.read);
        match r {
            reg::CHAR_L if pio_write => {
                let dw = ((self.reg(reg::CHAR_H) as u64) << 32) | val as u64;
                self.pio_write(dw)
            }
            reg::CHAR_H if pio_write => self.pio_write((val as u64) << 32),
            reg::CHAR_H => self.stipple_bits((val as u64) << 32, 32),
            reg::CHAR_L => {
                let bits = ((self.reg(reg::CHAR_H) as u64) << 32) | val as u64;
                self.stipple_bits(bits, 64)
            }
            _ => self.execute(),
        }
    }

    /// Whether pixels are RGB rather than colour indices: an RGB pixel type,
    /// or a write mask of exactly the 24 RGB planes (24-bit windows draw
    /// with other pixel types; colour-index drawing masks 8 or 12 planes, or
    /// all 32 when the PROM and kernel draw).
    fn rgb_mode(&self) -> bool {
        rgb_pixtype(self.reg(reg::PP1FILLMODE)) || self.reg(reg::COLORMASKLSBSA) == 0xFF_FFFF
    }

    /// Window coordinates to framebuffer coordinates.
    fn to_fb(&self, x: i32, y: i32) -> (i32, i32) {
        let win = self.reg(reg::XYWIN);
        let (ox, oy) = (signed16(win), signed16(win >> 16));
        if self.reg(reg::CONFIG) & CONFIG_YFLIP != 0 {
            (ox + x, oy - y)
        } else {
            (ox + x, oy + y)
        }
    }

    /// Whether a framebuffer pixel may be written: on screen, passing
    /// every enabled screen mask, and its clip ID matching (`cid_match`).
    /// Window mode bit `n - 1` enables mask `n` (1..4, `scrmsk{n}x` /
    /// `scrmsk{n}y`, each `min << 16 | max`), and bit `n + 3` keeps the
    /// pixels inside it rather than outside.
    fn visible(&self, x: i32, y: i32) -> bool {
        if !(0..WIDTH as i32).contains(&x) || !(0..HEIGHT as i32).contains(&y) {
            return false;
        }
        let m = cid_match(self.reg(reg::PP1WINMODE));
        if m != 0 && (m >> self.cid[y as usize * WIDTH + x as usize]) & 1 == 0 {
            return false;
        }
        let mode = self.reg(reg::CLIP_MODE);
        for n in 0..4 {
            if mode & (1 << n) == 0 {
                continue;
            }
            let (mx, my) = (self.reg(reg::CLIP_X + 2 * n), self.reg(reg::CLIP_Y + 2 * n));
            let inside = (signed16(mx >> 16)..=signed16(mx)).contains(&x)
                && (signed16(my >> 16)..=signed16(my)).contains(&y);
            if inside != (mode & (0x10 << n) != 0) {
                return false;
            }
        }
        true
    }

    /// The buffer drawing goes to: the page pointer in
    /// `DRBpointers`, its kind from the draw-buffer field.
    ///
    /// Main-buffer drawing names its buffer absolutely in the draw-buffer
    /// field: 1 the page in DRBpointers bits 9:0 (A), 2 the page in bits
    /// 19:10 (B), 3 both. The X server keeps DRBpointers at A | B << 10
    /// and tracks which one a double-buffered window shows itself (traced:
    /// the Icon Catalog draws with 2 before the kernel's swap shows B, 1
    /// before the swap back). 8/8/8/8 double buffering is two 36-bit
    /// buffers (Octane report 4.4.14).
    pub fn target(&self) -> Buffer {
        let kind = if self.draws_overlay() { Kind::Overlay } else { Kind::Wide };
        let drb = self.reg(reg::DRBPOINTERS);
        let ptr = match self.second_buffer() {
            Some(b) if kind == Kind::Wide && draw_buffer(self.reg(reg::PP1FILLMODE)) == DRAW_B => b,
            _ => drb,
        };
        Buffer::new(ptr, kind, self.reg(reg::DRBSIZE))
    }

    /// DRBpointers' second buffer page (bits 19:10), if any.
    fn second_buffer(&self) -> Option<u32> {
        let b = (self.reg(reg::DRBPOINTERS) >> 10) & 0x3FF;
        (b != 0).then_some(b)
    }

    /// Store `v` at framebuffer `(x, y)` in the buffer (or both buffers)
    /// the pixel processors are drawing to, or its low two bits in the
    /// clip-ID planes.
    fn put(&mut self, x: i32, y: i32, v: u32) {
        let field = draw_buffer(self.reg(reg::PP1FILLMODE));
        if !self.visible(x, y) {
            return;
        }
        if field == DRAW_CID {
            let m = cid_write_mask(self.reg(reg::PP1WINMODE));
            let c = &mut self.cid[y as usize * WIDTH + x as usize];
            *c = (*c & !m) | (v as u8 & m);
            return;
        }
        let b = self.target();
        self.put_in(b, x, y, v, field == DRAW_B);
        if field == DRAW_A_AND_B {
            if let Some(p) = self.second_buffer() {
                let b2 = Buffer::new(p, Kind::Wide, self.reg(reg::DRBSIZE));
                // The 12-bit X visuals can give A and B the same page.
                // Applying an XOR twice there would erase the drawing.
                if b2 != b {
                    self.put_in(b2, x, y, v, true);
                }
            }
        }
    }

    fn put_in(&mut self, b: Buffer, x: i32, y: i32, v: u32, back: bool) {
        let pp1 = self.reg(reg::PP1FILLMODE);
        let lsb = self.reg(if back { reg::COLORMASKLSBSB } else { reg::COLORMASKLSBSA });
        // A write through all planes (window moves copy the screen that way,
        // 24 bits a pixel) keeps the whole value; so does RGB.
        let wide = rgb_pixtype(pp1) || lsb == 0xFF_FFFF || lsb == u32::MAX;
        let old = self.mem.get(&b, x as u32, y as u32) as u32;
        let v = if pp1 & PP1_LOGIC_OP_ENABLE != 0 {
            let width = if (pp1 >> 8) & 7 == 2 { u32::MAX } else if wide { 0xFF_FFFF } else { 0xFFF };
            logic_op(pp1 >> 26, v, old) & width
        } else {
            v
        };
        // X's 12-bit TrueColor visual uses RGB444 (pixel type 0, buffer
        // size 0). Keep expanded nibbles in our RGB888 storage, whether
        // the pixel came from a fill, an iterator, or a host transfer.
        // Otherwise a scroll through RGBA4444 changes a fresh fill's
        // 0x20/0x60/0x50 into a different shade on every round trip.
        let v = if b.kind == Kind::Wide && (pp1 >> 8) & 7 == 0 && pp1 & (1 << 13) == 0 {
            let nibbles = v & 0xF0_F0F0;
            nibbles | nibbles >> 4
        } else {
            v
        };
        // Plane write masks where the pixel format matches our storage
        // (pixel type 2, RGBA8888: R G B in the low 24 planes, alpha in the
        // top 8): ColorMaskLSBsA the low 24 planes (all 32 when it is all
        // ones), ColorMaskMSBs the top 8. Overlay writes: the X server
        // sends ColorMaskLSBsA 0 and ColorMaskMSBs 0xF0 (overlay visuals,
        // values 0-15) or 0x70 (popup menus, values 0-3, colormap entries
        // 0-3); read as the high nibble masking overlay planes 3:0 (our
        // reading: the stored values stay as the X server wrote them).
        // Other formats keep their decoded values: their masks describe
        // a packed storage layout this model does not keep.
        let v = if b.kind == Kind::Overlay {
            // Native GL CI8 overlays select the upper overlay planes
            // (DRAW_BUFFER 0x48). X's 4-bit overlay/popup selector 0x4f
            // presents those planes as indices 0..15.
            let mask = if draw_buffer(pp1) == 0x48 {
                self.reg(reg::COLORMASKMSBS) & 0xFF
            } else {
                (self.reg(reg::COLORMASKMSBS) >> 4) & 0xF
            };
            (old & !mask) | (v & mask)
        } else if b.kind == Kind::Wide && (pp1 >> 8) & 7 == 2 {
            let mask = if lsb == u32::MAX { lsb } else { lsb & 0xFF_FFFF | (self.reg(reg::COLORMASKMSBS) & 0xFF) << 24 };
            (old & !mask) | (v & mask)
        } else if b.kind == Kind::Wide && (pp1 >> 8) & 7 == 6 {
            let mask = lsb & 0xFFF;
            (old & !mask) | (v & mask)
        } else {
            v
        };
        self.mem.put(&b, x as u32, y as u32, v as u64);
    }

    /// Drawing goes to the overlay planes.
    fn draws_overlay(&self) -> bool {
        draw_buffer(self.reg(reg::PP1FILLMODE)) & 0x70 == 0x40
    }

    /// Pixel reads select A, B, or the overlay independently of drawing.
    fn source(&self) -> Buffer {
        let drb = self.reg(reg::DRBPOINTERS);
        let (ptr, kind) = match read_buffer(self.reg(reg::PP1FILLMODE)) {
            READ_OVERLAY => (drb, Kind::Overlay),
            READ_B => (self.second_buffer().unwrap_or(drb), Kind::Wide),
            _ => (drb, Kind::Wide),
        };
        Buffer::new(ptr, kind, self.reg(reg::DRBSIZE))
    }

    fn get(&self, x: i32, y: i32) -> u32 {
        if !(0..WIDTH as i32).contains(&x) || !(0..HEIGHT as i32).contains(&y) {
            return 0;
        }
        self.mem.get(&self.source(), x as u32, y as u32) as u32
    }

    /// Block pixel `(col, row)` in framebuffer coordinates.
    fn block_px(&self, b: &Block, col: i32, row: i32) -> (i32, i32) {
        self.to_fb(b.xs + col * b.dx(), b.ys + row * b.dy())
    }

    fn current_block(&self) -> Block {
        let s = self.reg(reg::BLOCKXYSTARTI);
        let e = self.reg(reg::BLOCKXYENDI);
        // Colour index modes: the index, from the fill colour register for
        // fast fills and from the red iterator (12 fraction bits) otherwise.
        // RGB modes: three 12-bit components for fast fills, a packed
        // 8-8-8 colour otherwise.
        let rgb = self.rgb_mode();
        let fast = self.reg(reg::FILLMODE) & FILL_FAST != 0;
        let color = match (rgb, fast) {
            (false, true) => self.reg(reg::FILL_COLOR_R),
            (false, false) => (self.reg(reg::RED) >> 12) & 0xFFF,
            (true, true) => pack_rgb(
                self.reg(reg::FILL_COLOR_R) >> 4,
                self.reg(reg::FILL_COLOR_G) >> 4,
                self.reg(reg::FILL_COLOR_B) >> 4,
            ) | ((self.reg(reg::FILL_COLOR_B + 1) >> 4) & 0xFF) << 24,
            (true, false) => self.reg(reg::PACKEDCOLOR) & 0xFF_FFFF,
        };
        Block { xs: signed16(s >> 16), ys: signed16(s), xe: signed16(e >> 16), ye: signed16(e), color }
    }

    /// Run the primitive in the IR.
    fn execute(&mut self) -> bool {
        self.stats.prims[(self.reg(reg::IR) & 0xF) as usize] += 1;
        match self.reg(reg::IR) & 0xF {
            OP_BLOCK => {}
            OP_LINE => return self.line(),
            OP_POINT => return self.point(),
            OP_AREA_LTOR => return self.triangle(true),
            OP_AREA_RTOL => return self.triangle(false),
            OP_GL_LINE => return self.gl_line(),
            _ => return false,
        }
        let fm = self.reg(reg::FILLMODE);
        if !self.fillmodes_seen().contains(&fm) && self.fillmodes_n < 32 {
            self.fillmodes[self.fillmodes_n] = fm;
            self.fillmodes_n += 1;
        }
        let b = self.current_block();
        // A new block ends whatever the last one was doing; a write transfer
        // in particular is never disarmed explicitly.
        self.stipple = None;
        self.xfer = None;
        let kind = (fm >> 22) & 7;
        if fm & FILL_FAST != 0 {
            self.stats.fast_fills += 1;
            self.fill(&b);
            return true;
        }
        self.stats.blocks[kind as usize] += 1;
        match kind {
            block::NORMAL if fm & FILL_CHAR_STIPPLE != 0 => {
                // Character block: the char data that follows stipples it.
                let opaque = fm & FILL_CHAR_STIPPLE_OPAQUE != 0;
                self.stipple = Some(Stipple { block: b, col: 0, row: 0, opaque });
                false
            }
            block::PIO_READ | block::PIO_WRITE | block::DMA_READ | block::DMA_WRITE => {
                let read = kind == block::PIO_READ || kind == block::DMA_READ;
                self.arm_transfer(b, read, kind == block::PIO_READ);
                false
            }
            _ => {
                // Block type 0 (and 1 without char stipple): a fill in the
                // iterated colour. The kernel's boot gradient draws a block
                // per line this way; char data never follows one.
                self.fill(&b);
                true
            }
        }
    }

    /// The primitive in the IR as the registers describe it, for the trace.
    pub fn describe_ir(&self) -> String {
        let ir = self.reg(reg::IR);
        let fm = self.reg(reg::FILLMODE);
        let pp1 = self.reg(reg::PP1FILLMODE);
        let xy = |v: u32| format!("({}, {})", signed16(v >> 16), signed16(v));
        let target = format!(
            "{}{} mask={:#x} pp1={pp1:#x}{} win=(x {}, y {}){}",
            if self.rgb_mode() { "RGB" } else { "CI" },
            if self.draws_overlay() { " overlay" } else { "" },
            self.reg(reg::COLORMASKLSBSA),
            if pp1 & PP1_LOGIC_OP_ENABLE != 0 { format!(" lop={:#x}", (pp1 >> 26) & 0xF) } else { String::new() },
            signed16(self.reg(reg::XYWIN)),
            signed16(self.reg(reg::XYWIN) >> 16),
            if self.reg(reg::CONFIG) & CONFIG_YFLIP != 0 { " yflip" } else { "" },
        );
        match ir & 0xF {
            OP_BLOCK => {
                let b = self.current_block();
                let kind = if fm & FILL_FAST != 0 {
                    "fast fill".to_string()
                } else {
                    match (fm >> 22) & 7 {
                        block::NORMAL if fm & FILL_CHAR_STIPPLE != 0 => "char block".into(),
                        block::PIO_READ => format!("PIO read {}", xfer_desc(self)),
                        block::PIO_WRITE => format!("PIO write {}", xfer_desc(self)),
                        block::DMA_READ => format!("DMA read {}", xfer_desc(self)),
                        block::DMA_WRITE => format!("DMA write {}", xfer_desc(self)),
                        k => format!("fill (block type {k})"),
                    }
                };
                format!("BLOCK ({}, {})-({}, {}) {kind} color={:#x} fm={fm:#x} {target}", b.xs, b.ys, b.xe, b.ye, b.color)
            }
            OP_LINE => format!(
                "LINE {}-{} color={:#x} fm={fm:#x}{}{} {target}",
                xy(self.reg(reg::LINE_START)),
                xy(self.reg(reg::LINE_END)),
                self.current_block().color,
                if fm & FILL_LINE_STIPPLE != 0 { format!(" stipple={:#010x}", self.reg(reg::LINE_STIPPLE)) } else { String::new() },
                if fm & FILL_LINE_SKIP_LAST != 0 { " skip-last" } else { "" },
            ),
            OP_POINT => format!("POINT {} color={:#x} {target}", xy(self.reg(reg::LINE_START)), self.current_block().color),
            op @ (OP_AREA_LTOR | OP_AREA_RTOL) => {
                let pos = |n: u32| f32::from_bits(self.reg(n)) - 49151.5;
                format!("TRIANGLE {} ymax {} ymid {} ymin {} x0 {} {target}",
                    if op == OP_AREA_LTOR { "left to right" } else { "right to left" },
                    pos(0x003), pos(0x004), pos(0x005), pos(0x000))
            }
            op => format!("IR {ir:#x} (opcode {op:#x} not modelled) fm={fm:#x} {target}"),
        }
    }

    /// A point at the line start register, in the current colour.
    fn point(&mut self) -> bool {
        let s = self.reg(reg::LINE_START);
        let color = self.current_block().color;
        let (fx, fy) = self.to_fb(signed16(s >> 16), signed16(s));
        self.put(fx, fy, color);
        true
    }

    /// A line from the start to the end point, both included unless the fill
    /// mode skips the last, in the current colour; with line stipple on,
    /// pixel `k` is drawn only where bit `31 - k % 32` of the pattern is set.
    fn line(&mut self) -> bool {
        let s = self.reg(reg::LINE_START);
        let e = self.reg(reg::LINE_END);
        let (mut x, mut y) = (signed16(s >> 16), signed16(s));
        let (x1, y1) = (signed16(e >> 16), signed16(e));
        let color = self.current_block().color;
        let fm = self.reg(reg::FILLMODE);
        let stipple = (fm & FILL_LINE_STIPPLE != 0).then(|| self.reg(reg::LINE_STIPPLE));
        let background = (stipple.is_some() && fm & FILL_LINE_STIPPLE_OPAQUE != 0).then(|| self.background());
        let (dx, dy) = ((x1 - x).abs(), -(y1 - y).abs());
        let (sx, sy) = (if x < x1 { 1 } else { -1 }, if y < y1 { 1 } else { -1 });
        let mut err = dx + dy;
        let mut k = 0u32;
        let skip_last = fm & FILL_LINE_SKIP_LAST != 0;
        loop {
            if skip_last && x == x1 && y == y1 {
                break;
            }
            let lit = stipple.map_or(true, |p| p & (1 << (31 - k % 32)) != 0);
            if let Some(c) = if lit { Some(color) } else { background } {
                let (fx, fy) = self.to_fb(x, y);
                self.put(fx, fy, c);
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
            k += 1;
        }
        true
    }

    /// The background colour, in the same form as the drawing colour: packed
    /// RGB in RGB modes (from three 12-bit components, like the fast-fill
    /// colour), a colour index otherwise.
    fn background(&self) -> u32 {
        let bg = self.reg(reg::BG_COLOR);
        if self.rgb_mode() {
            let red = self.reg(reg::BG_COLOR_RED) & 0xFFF;
            pack_rgb(red >> 4, (bg & 0xFFF) >> 4, ((bg >> 12) & 0xFFF) >> 4)
        } else {
            bg & 0xFFF
        }
    }

    /// A triangle from the area registers, as SGI's diagnostic triangle
    /// setup (_mg0_FillTriangle) loads them: vertices sorted top (max y),
    /// middle, bottom (min y); positions as floats biased by 49151.5
    /// (window coordinates, y up); edge slopes dx per unit of y going down,
    /// s.24 fixed point in hi/lo pairs:
    ///   tri_x0/x1   major / upper minor edge x at y = floor(ymax)
    ///   tri_x2      lower minor edge x at y = floor(ymid)
    ///   tri_ymax, tri_ymid, tri_ymin
    ///   tri_dxdy0/1/2  major, upper minor, lower minor slopes
    /// Colours: 12.12 fixed (1.0 = 0xFFF000), the value at the first pixel
    /// of the first span (x = ceil(x0) left to right, floor(x0) right to
    /// left; y = floor(ymax)), dx per pixel along the span (negated right to
    /// left) and de per scanline down the major edge, corrected for the
    /// edge's integer step (de = dc/dy + trunc(dxdy0) * dx, less dx when the
    /// span direction and the slope's sign disagree).
    ///
    /// Pixels are sampled at their centres with OpenGL's half-open rule
    /// (bottom and left edges in). The real RE4's edge and tie rules are
    /// not known; this is the GL rule.
    fn triangle(&mut self, ltor: bool) -> bool {
        let pos = |r: &Self, n: u32| f32::from_bits(r.reg(n)) as f64 - 49151.5;
        let slope = |r: &Self, n: u32| ((r.reg(n) as u64) << 32 | r.reg(n + 1) as u64) as i64 as f64 / (1u64 << 24) as f64;
        let (x0, x1, x2) = (pos(self, 0x000), pos(self, 0x001), pos(self, 0x002));
        let (ymax, ymid, ymin) = (pos(self, 0x003), pos(self, 0x004), pos(self, 0x005));
        let (s0, s1, s2) = (slope(self, 0x006), slope(self, 0x008), slope(self, 0x00A));
        if !(ymax > ymin) || ![x0, x1, x2, ymax, ymid, ymin, s0, s1, s2].iter().all(|v| v.is_finite()) {
            return true;
        }
        let (yref, yref2) = (ymax.floor(), ymid.floor());
        // Colour planes c(x, y) = c0 + dx * (x - xs) + dy * (yref - y).
        let xs = if ltor { x0.ceil() } else { x0.floor() };
        let n1 = s0.trunc();
        let fix = |v: u32| v as i32 as f64 / 0xFF_F000 as f64;
        let plane = |r: &Self, c: u32, de: u32, dx: u32| -> (f64, f64, f64) {
            let dx_reg = fix(r.reg(dx));
            let dx = if ltor { dx_reg } else { -dx_reg };
            let mut de = fix(r.reg(de));
            if ltor ^ (s0 >= 0.0) {
                de += dx_reg;
            }
            (fix(r.reg(c)), dx, de - n1 * dx)
        };
        let planes = [
            plane(self, 0x05C, 0x060, 0x061),
            plane(self, 0x05D, 0x062, 0x063),
            plane(self, 0x05E, 0x064, 0x065),
            plane(self, 0x05F, 0x066, 0x067),
        ];
        // Z: 64-bit hi/lo registers, value * 0x1000, the same rule.
        let fix64 = |r: &Self, n: u32| ((r.reg(n) as u64) << 32 | r.reg(n + 1) as u64) as i64 as f64 / 4096.0;
        let zplane = {
            let dzx_reg = fix64(self, 0x06A);
            let dzx = if ltor { dzx_reg } else { -dzx_reg };
            let mut dze = fix64(self, 0x06C);
            if ltor ^ (s0 >= 0.0) {
                dze += dzx_reg;
            }
            (fix64(self, 0x068), dzx, dze - n1 * dzx)
        };
        // Texture: S/W, T/W and 1/W, the same plane rule at 2^32 (te1.rs).
        let texture = self.reg(te_reg::TEXMODE1) & TEXMODE1_ENABLE != 0;
        let fixt = |r: &Self, n: u32| ((r.reg(n) as u64) << 32 | r.reg(n + 1) as u64) as i64 as f64 / ITER_ONE;
        let tplane = |r: &Self, c: u32, de: u32, dx: u32| -> (f64, f64, f64) {
            let dx_reg = fixt(r, dx);
            let dx = if ltor { dx_reg } else { -dx_reg };
            let mut de = fixt(r, de);
            if ltor ^ (s0 >= 0.0) {
                de += dx_reg;
            }
            (fixt(r, c), dx, de - n1 * dx)
        };
        let tex = texture.then(|| {
            let planes = [
                tplane(self, te_reg::SW, te_reg::DSWE, te_reg::DSWX),
                tplane(self, te_reg::TW, te_reg::DTWE, te_reg::DTWX),
                tplane(self, te_reg::WI, te_reg::DWIE, te_reg::DWIX),
            ];
            {
                if let Some((want, have)) = self.te.stale(&self.regs) {
                    self.te.stale_events += 1;
                    self.te.stale_last = [want, have];
                }
                (planes, self.te.sampler(&self.regs))
            }
        });
        let fog = (self.reg(reg::FOG_ON) != 0).then(|| tplane(self, reg::FOG_F, reg::FOG_F + 4, reg::FOG_F + 2));
        let stipple = self.reg(reg::FILLMODE) & FILL_POLY_STIPPLE != 0;
        let j0 = (ymin - 0.5).ceil() as i32;
        let j1 = (ymax - 0.5).ceil() as i32;
        for j in j0..j1 {
            let yc = j as f64 + 0.5;
            let major = x0 + s0 * (yref - yc);
            let minor = if yc >= ymid { x1 + s1 * (yref - yc) } else { x2 + s2 * (yref2 - yc) };
            let (l, r) = if ltor { (major, minor) } else { (minor, major) };
            let i0 = (l - 0.5).ceil() as i32;
            let i1 = (r - 0.5).ceil() as i32;
            // Polygon stipple: 32 rows in the device space, window y mod 32,
            // the most significant bit at window x mod 32 = 0.
            let stip_row = stipple.then(|| self.device.get(POLY_STIPPLE_RAM + (j & 31) as u32));
            for i in i0..i1 {
                if let Some(row) = stip_row {
                    if (row >> (31 - (i & 31))) & 1 == 0 {
                        continue;
                    }
                }
                let xc = i as f64 + 0.5;
                let at = |p: (f64, f64, f64)| p.0 + p.1 * (xc - xs) + p.2 * (yref - yc);
                let mut rgba = [at(planes[0]), at(planes[1]), at(planes[2]), at(planes[3])];
                if let Some((p, smp)) = &tex {
                    rgba = self.textured(p, smp, at, rgba);
                }
                if let Some(fp) = fog {
                    rgba = self.fogged(rgba, at(fp));
                }
                self.gl_fragment(i, j, rgba, at(zplane));
            }
        }
        true
    }

    /// A fragment's colour textured: S/W, T/W, 1/W from their planes (value,
    /// d/dx, d/(-y)) evaluated by `at`; the level of detail from the
    /// derivatives of s and t in level-0 texels; the texel through the
    /// texture environment.
    fn textured(&self, p: &[(f64, f64, f64); 3], smp: &Sampler, at: impl Fn((f64, f64, f64)) -> f64, rgba: [f64; 4]) -> [f64; 4] {
        let (sw, tw, wi) = (at(p[0]), at(p[1]), at(p[2]));
        if wi == 0.0 || !wi.is_finite() {
            return rgba;
        }
        let (s, t) = (sw / wi, tw / wi);
        // d(a/w) = (da' - a dw') / w for a' = a/w, w' = 1/w; y runs down
        // the plane's third term.
        let d = |q: &(f64, f64, f64), v: f64| ((q.1 - v * p[2].1) / wi, (v * p[2].2 - q.2) / wi);
        let ((dsx, dsy), (dtx, dty)) = (d(&p[0], s), d(&p[1], t));
        let (w, h) = smp.size();
        let rho = (dsx * w).hypot(dtx * h).max((dsy * w).hypot(dty * h));
        let lambda = if rho > 0.0 { rho.log2() } else { f64::NEG_INFINITY };
        let texel = smp.sample(&self.te, s, t, lambda);
        tex_env(self.reg(te_reg::TEXMODE1), self.reg(te_reg::TXENV_RG), self.reg(te_reg::TXENV_B), rgba, texel)
    }

    /// Fog applied to a fragment: factor `f` (1 = none) toward the fog
    /// colour (FOG_RG / FOG_B), alpha kept.
    fn fogged(&self, rgba: [f64; 4], f: f64) -> [f64; 4] {
        let f = f.clamp(0.0, 1.0);
        let c = |v: u32| (v & 0xFFF) as f64 / 4095.0;
        let (rg, b) = (self.reg(reg::FOG_RG), self.reg(reg::FOG_B));
        let fc = [c(rg), c(rg >> 12), c(b)];
        [0, 1, 2, 3].map(|k| if k < 3 { f * rgba[k] + (1.0 - f) * fc[k] } else { rgba[3] })
    }

    /// A GL line (our GE11 HLE, IR `OP_GL_LINE`): from gline_xstartf/ystartf
    /// to gline_xendf/yendf (biased floats, as the triangle positions);
    /// colours (red..alpha) and Z (z_hi/low) at the start, their steps per
    /// pixel along the major axis in dre/dge/dbe/dae and dze. Pixels whose
    /// centre's major coordinate lies in [start, end) along the direction
    /// of travel (the GL rule for non-antialiased lines), `glineconfig` + 1
    /// pixels wide across the major axis. With line stipple (fill mode bit
    /// 5), lspat holds the 16-bit pattern, first bit in bit 15, and lscrl
    /// the repeat count less one; the position carries on until lscrl is
    /// written again.
    fn gl_line(&mut self) -> bool {
        let pos = |r: &Self, n: u32| f32::from_bits(r.reg(n)) as f64 - 49151.5;
        let (x0, y0, x1, y1) = (pos(self, 0x00C), pos(self, 0x00D), pos(self, 0x00E), pos(self, 0x00F));
        if ![x0, y0, x1, y1].iter().all(|v| v.is_finite()) {
            return true;
        }
        let fix = |r: &Self, n: u32| r.reg(n) as i32 as f64 / 0xFF_F000 as f64;
        let fix64 = |r: &Self, n: u32| ((r.reg(n) as u64) << 32 | r.reg(n + 1) as u64) as i64 as f64 / 4096.0;
        let c0 = [fix(self, 0x05C), fix(self, 0x05D), fix(self, 0x05E), fix(self, 0x05F)];
        let dc = [fix(self, 0x060), fix(self, 0x062), fix(self, 0x064), fix(self, 0x066)];
        let (z0, dz) = (fix64(self, 0x068), fix64(self, 0x06C));
        let width = (self.reg(reg::GLINECONFIG) & 0xFF) as i32 + 1;
        let stipple = self.reg(reg::FILLMODE) & FILL_LINE_STIPPLE != 0;
        let (pattern, repeat) = (self.reg(reg::LINE_STIPPLE) & 0xFFFF, (self.reg(reg::LSCRL) & 0xFF) + 1);
        // Texture along the line: S/W, T/W, Q/W at the start, steps per
        // pixel in the edge step registers; the fog factor likewise.
        let fix64t = |r: &Self, n: u32| ((r.reg(n) as u64) << 32 | r.reg(n + 1) as u64) as i64 as f64 / ITER_ONE;
        let tex = (self.reg(te_reg::TEXMODE1) & TEXMODE1_ENABLE != 0).then(|| {
            let q = |c: u32, d: u32| (fix64t(self, c), fix64t(self, d), 0.0);
            ([q(te_reg::SW, te_reg::DSWE), q(te_reg::TW, te_reg::DTWE), q(te_reg::WI, te_reg::DWIE)], self.te.sampler(&self.regs))
        });
        let fog = (self.reg(reg::FOG_ON) != 0).then(|| (fix64t(self, reg::FOG_F), fix64t(self, reg::FOG_F + 2)));
        let xmajor = (x1 - x0).abs() >= (y1 - y0).abs();
        let (a0, a1, b0, b1) = if xmajor { (x0, x1, y0, y1) } else { (y0, y1, x0, x1) };
        if a0 == a1 {
            return true;
        }
        let slope = (b1 - b0) / (a1 - a0);
        let dir = if a1 > a0 { 1 } else { -1 };
        // First and one-past-last pixel along the major axis.
        let (p0, p1) = if dir > 0 { ((a0 - 0.5).ceil() as i32, (a1 - 0.5).ceil() as i32) } else { ((a0 - 0.5).floor() as i32, (a1 - 0.5).floor() as i32) };
        let mut p = p0;
        while p != p1 {
            let t = ((p as f64 + 0.5) - a0).abs();
            let draw = !stipple || {
                let bit = (self.gl_line_stipple_pos / repeat) % 16;
                self.gl_line_stipple_pos += 1;
                (pattern >> (15 - bit)) & 1 != 0
            };
            if draw {
                let b = b0 + slope * ((p as f64 + 0.5) - a0);
                let q0 = (b - 0.5 * width as f64 + 0.5).floor() as i32;
                let mut rgba = [c0[0] + dc[0] * t, c0[1] + dc[1] * t, c0[2] + dc[2] * t, c0[3] + dc[3] * t];
                if let Some((p, smp)) = &tex {
                    // Planes (value, d/major, 0) evaluated at t along it.
                    rgba = self.textured(p, smp, |q: (f64, f64, f64)| q.0 + q.1 * t, rgba);
                }
                if let Some((f0, df)) = fog {
                    rgba = self.fogged(rgba, f0 + df * t);
                }
                for q in q0..q0 + width {
                    let (wx, wy) = if xmajor { (p, q) } else { (q, p) };
                    self.gl_fragment(wx, wy, rgba, z0 + dz * t);
                }
            }
            p += dir;
        }
        true
    }

    /// One GL fragment at window (wx, wy): the per-fragment pipeline our
    /// GE11 HLE programs (layouts provisional, see GL_RASTER_STATE): screen
    /// masks, alpha test, stencil test, depth test, stencil update, then
    /// blending or the logic op, and the colour write masks. `rgba` is
    /// 0..1 (red alone is the index in colour-index windows), `z` window
    /// depth (0..2^24 - 1).
    fn gl_fragment(&mut self, wx: i32, wy: i32, rgba: [f64; 4], z: f64) {
        let (fx, fy) = self.to_fb(wx, wy);
        if draw_buffer(self.reg(reg::PP1FILLMODE)) == DRAW_CID || !self.visible(fx, fy) {
            return;
        }
        let rgba = rgba.map(|c| c.clamp(0.0, 1.0));
        let af = self.reg(reg::AFUNCMODE);
        if af & TEST_ENABLE != 0 && !compare(af & 7, rgba[3], ((af >> 4) & 0xFFF) as f64 / 4096.0) {
            return;
        }
        let st = self.reg(reg::STENCILMODE);
        let zm = self.reg(reg::ZMODE);
        let (sten, ztest) = (st & TEST_ENABLE != 0, zm & ZMODE_TEST != 0);
        let zbuf = Buffer::new(ZST_PAGE, Kind::Wide, self.reg(reg::DRBSIZE));
        let (ux, uy) = (fx as u32, fy as u32);
        if sten || ztest {
            let zst = self.mem.get(&zbuf, ux, uy);
            let s_old = ((zst >> 24) & 0xFF) as u32;
            let masks = self.reg(reg::STENCILMASK);
            let (cmask, wmask) = (masks & 0xFF, (masks >> 8) & 0xFF);
            let sref = (st >> 16) & 0xFF;
            let z = z.round().clamp(0.0, 16_777_215.0) as u64;
            let zold = zst & 0xFF_FFFF;
            let spass = !sten || compare(st & 7, (sref & cmask) as f64, (s_old & cmask) as f64);
            let zpass = spass && (!ztest || compare(zm & 7, z as f64, zold as f64));
            let mut new = zst;
            if sten {
                let op = if !spass { (st >> 4) & 7 } else if !zpass { (st >> 8) & 7 } else { (st >> 12) & 7 };
                let s_new = match op {
                    1 => 0,
                    2 => sref,
                    3 => (s_old + 1).min(0xFF),
                    4 => s_old.saturating_sub(1),
                    5 => !s_old & 0xFF,
                    _ => s_old,
                };
                let s = (s_old & !wmask) | (s_new & wmask);
                new = (new & !(0xFF << 24)) | (s as u64) << 24;
            }
            if zpass && ztest {
                let m = ((zm >> 4) & 0xFF_FFFF) as u64;
                new = (new & !m) | (z & m);
            }
            if new != zst {
                self.mem.put(&zbuf, ux, uy, new);
            }
            if !zpass {
                return;
            }
        }
        let pp1 = self.reg(reg::PP1FILLMODE);
        let b = self.target();
        self.fragment_color(b, ux, uy, rgba, pp1, draw_buffer(pp1) == DRAW_B);
        if draw_buffer(pp1) == DRAW_A_AND_B {
            if let Some(p) = self.second_buffer() {
                let b2 = Buffer::new(p, Kind::Wide, self.reg(reg::DRBSIZE));
                if b2 != b {
                    self.fragment_color(b2, ux, uy, rgba, pp1, true);
                }
            }
        }
    }

    fn fragment_color(&mut self, b: Buffer, ux: u32, uy: u32, rgba: [f64; 4], pp1: u32, back: bool) {
        let dst = self.mem.get(&b, ux, uy) as u32;
        // A logic op other than copy replaces blending (OpenGL).
        let logic = pp1 & PP1_LOGIC_OP_ENABLE != 0 && (pp1 >> 26) & 0xF != 3;
        let rgb = self.rgb_mode();
        let src = if rgb {
            let blend = self.reg(reg::BLENDFACTOR);
            let c = if blend & BLEND_ENABLE != 0 && !logic {
                let d = [dst & 0xFF, (dst >> 8) & 0xFF, (dst >> 16) & 0xFF, dst >> 24].map(|v| v as f64 / 255.0);
                let f = |code: u32, k: usize| -> f64 {
                    let sat = rgba[3].min(1.0 - d[3]);
                    match code {
                        0 => 0.0,
                        1 => 1.0,
                        2 => rgba[k],
                        3 => 1.0 - rgba[k],
                        4 => rgba[3],
                        5 => 1.0 - rgba[3],
                        6 => d[3],
                        7 => 1.0 - d[3],
                        8 => d[k],
                        9 => 1.0 - d[k],
                        10 => if k == 3 { 1.0 } else { sat },
                        _ => 1.0,
                    }
                };
                let (sf, df) = (blend & 0xF, (blend >> 4) & 0xF);
                [0, 1, 2, 3].map(|k| (rgba[k] * f(sf, k) + d[k] * f(df, k)).clamp(0.0, 1.0))
            } else {
                rgba
            };
            let q = |v: f64| (v * 255.0).round() as u32;
            q(c[0]) | q(c[1]) << 8 | q(c[2]) << 16 | q(c[3]) << 24
        } else {
            (rgba[0] * 4095.0).round() as u32 & 0xFFF
        };
        self.put_in(b, ux as i32, uy as i32, src, back);
    }

    fn fill(&mut self, b: &Block) {
        for row in 0..b.rows() {
            for col in 0..b.cols() {
                let (x, y) = self.block_px(b, col, row);
                self.put(x, y, b.color);
            }
        }
    }

    /// Consume `n` stipple bits (most significant first) into the current
    /// character block: 1 bits take the colour, 0 bits leave the pixel (or
    /// take the background colour when the stipple is opaque). A row ends at
    /// the block's edge, discarding the rest of the chunk.
    fn stipple_bits(&mut self, bits: u64, n: u32) -> bool {
        self.stats.stipple_chunks += 1;
        let Some(mut s) = self.stipple else { return false };
        if s.row >= s.block.rows() {
            return false;
        }
        let background = s.opaque.then(|| self.background());
        for i in 0..n {
            let c = if bits & (1u64 << (63 - i)) != 0 { Some(s.block.color) } else { background };
            if let Some(c) = c {
                let (x, y) = self.block_px(&s.block, s.col, s.row);
                self.put(x, y, c);
            }
            s.col += 1;
            if s.col >= s.block.cols() {
                s.col = 0;
                s.row += 1;
                break;
            }
        }
        self.stipple = Some(s);
        true
    }

    // ---- pixel transfers ----

    fn arm_transfer(&mut self, block: Block, read: bool, pio_read: bool) {
        let mode = self.reg(reg::XFRMODE);
        let begin = (mode >> 8) & 7;
        let mut x = Xfer {
            block,
            read,
            pio_read,
            width: self.reg(reg::XFRSIZE) & 0xFFFF,
            bpp: bytes_per_pixel(mode),
            format: ((mode >> 4) & 0xF, mode & 0xF),
            begin_skip: begin,
            stride_skip: (mode >> 14) & 0x1FF,
            line: 0,
            line_begin: begin,
            pending: 0,
            skip: begin,
            rd_line: 0,
            rd_pos: 0,
            rd_begin: begin,
            out_lo: 0,
        };
        x.rd_settle();
        self.xfer = Some(x);
    }

    /// `Some(read)` while a transfer is armed.
    pub fn transfer_armed(&self) -> Option<bool> {
        if self.te_read[0] != 0 {
            return Some(true);
        }
        self.xfer.as_ref().map(|x| x.read)
    }

    /// Lines and bytes per line of the armed transfer.
    pub fn transfer_shape(&self) -> Option<(u32, u32)> {
        if self.te_read[0] != 0 {
            return Some((self.te_read[2], self.te_read[1] * Te1::read_texel_bytes(self.regs[reg::XFRMODE as usize])));
        }
        self.xfer.as_ref().map(|x| (x.block.rows() as u32, x.line_bytes()))
    }

    fn put_line(&mut self, x: &Xfer, line: u32, bytes: &[u8]) {
        let bpp = x.bpp as usize;
        for (k, px) in bytes.chunks(bpp).enumerate().take(x.width as usize) {
            // Big-endian bytes.
            let v = px.iter().fold(0u64, |a, &b| (a << 8) | b as u64);
            let (fx, fy) = self.block_px(&x.block, k as i32, line as i32);
            if x.format == DEPTH_FORMAT {
                // Depth into the depth buffer, its stencil kept.
                if self.visible(fx, fy) {
                    let zbuf = Buffer::new(ZST_PAGE, Kind::Wide, self.reg(reg::DRBSIZE));
                    let old = self.mem.get(&zbuf, fx as u32, fy as u32);
                    self.mem.put(&zbuf, fx as u32, fy as u32, (old & !0xFF_FFFF) | from_host(x.format, v) as u64);
                }
                continue;
            }
            self.put(fx, fy, from_host(x.format, v));
        }
    }

    /// Byte `i` (big-endian) of transfer pixel `k` of `line`, as the host sees it.
    fn line_byte(&self, x: &Xfer, line: u32, k: u32, i: u32) -> u8 {
        let (fx, fy) = self.block_px(&x.block, k as i32, line as i32);
        let v = to_host(x.format, self.get(fx, fy));
        (v >> (8 * (x.bpp - 1 - i))) as u8
    }

    fn get_line(&self, x: &Xfer, line: u32) -> Vec<u8> {
        (0..x.width).flat_map(|k| (0..x.bpp).map(move |i| (k, i))).map(|(k, i)| self.line_byte(x, line, k, i)).collect()
    }

    /// A PIO write doubleword. Each line starts in a fresh doubleword, after
    /// its begin offset; the rest of a line's last doubleword is dropped.
    fn pio_write(&mut self, dw: u64) -> bool {
        let Some(mut x) = self.xfer.take() else { return false };
        self.stats.pio_write_dw += 1;
        let mut changed = false;
        for i in 0..8 {
            if x.line >= x.block.rows() as u32 {
                break;
            }
            if x.skip > 0 {
                x.skip -= 1;
                continue;
            }
            if let Some(b) = self.line_buf.get_mut(x.pending as usize) {
                *b = (dw >> (56 - 8 * i)) as u8;
            }
            x.pending += 1;
            if x.pending == x.line_bytes() {
                let n = (x.pending as usize).min(MAX_LINE_BYTES);
                let bytes = self.line_buf[..n].to_vec();
                self.put_line(&x, x.line, &bytes);
                changed = true;
                x.pending = 0;
                x.line += 1;
                x.line_begin = x.next_begin(x.line_begin);
                x.skip = x.line_begin;
                break;
            }
        }
        self.xfer = Some(x);
        changed
    }

    /// The PIO read doubleword at the stream's current position (0 past the
    /// end, or when the transfer is not a PIO read).
    fn pio_doubleword(&self) -> u64 {
        let Some(x) = self.xfer.as_ref() else { return 0 };
        if !x.pio_read || x.rd_line >= x.block.rows() as u32 {
            return 0;
        }
        (0..8).fold(0u64, |a, j| {
            let pos = x.rd_pos + j;
            let b = match pos.checked_sub(x.rd_begin) {
                Some(p) if p < x.line_bytes() => self.line_byte(x, x.rd_line, p / x.bpp, p % x.bpp),
                _ => 0,
            };
            (a << 8) | b as u64
        })
    }

    /// PIO read, high half, without taking it (`pio_read_hi` takes it).
    pub fn pio_peek_hi(&self) -> u32 {
        (self.pio_doubleword() >> 32) as u32
    }

    /// PIO read, high half: takes the next doubleword.
    pub fn pio_read_hi(&mut self) -> u32 {
        let dw = self.pio_doubleword();
        self.stats.pio_read_dw += 1;
        let Some(x) = self.xfer.as_mut() else { return 0 };
        x.out_lo = dw as u32;
        if x.pio_read && x.rd_line < x.block.rows() as u32 {
            x.rd_pos += 8;
            x.rd_settle();
        }
        (dw >> 32) as u32
    }

    /// PIO read, low half of the doubleword the last high read took.
    pub fn pio_read_lo(&self) -> u32 {
        self.xfer.as_ref().map(|x| x.out_lo).unwrap_or(0)
    }

    /// DMA into the armed block: line `line`'s pixel bytes.
    pub fn dma_write_line(&mut self, line: u32, bytes: &[u8]) {
        self.stats.dma_lines_in += 1;
        if self.te_load != 0 {
            self.te.load_line(&self.regs, line, bytes);
            return;
        }
        if let Some(x) = self.xfer.take() {
            self.put_line(&x, line, bytes);
            self.xfer = Some(x);
        }
    }

    /// DMA out of the armed block: line `line`'s pixel bytes.
    pub fn dma_read_line(&self, line: u32) -> Vec<u8> {
        if self.te_read[0] != 0 {
            if self.regs[reg::TE_RAW as usize] != 0 {
                return self.te.read_raw(&self.regs, line, self.te_read[1]);
            }
            return self.te.read_line(&self.regs, line, self.te_read[1]);
        }
        self.xfer.as_ref().map(|x| self.get_line(x, line)).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The X server's screen height in these tests (window origin row 1023).
    const SCREEN_H: usize = 1024;

    /// The PROM's 1280x1024 main buffer.
    fn main_buffer() -> Buffer {
        Buffer::new(0x240, Kind::Wide, 0x31E)
    }

    fn px(r: &Rss, x: usize, y_top: usize) -> u32 {
        r.mem.get(&main_buffer(), x as u32, (SCREEN_H - 1 - y_top) as u32) as u32
    }

    /// The X server's setup: window origin at the top row, Y-flip on, so
    /// block coordinates are X's top-down ones.
    fn x_server() -> Box<Rss> {
        let mut r = Rss::new_boxed();
        r.write(reg::CONFIG, 0xCAC, false);
        r.write(reg::XYWIN, 1023 << 16, false);
        r.write(reg::DRBSIZE, 0x31E, false);
        r.write(reg::DRBPOINTERS, 0x240, false);
        // All planes writable, as the X server sets them.
        r.write(reg::COLORMASKLSBSA, u32::MAX, false);
        r.write(reg::PP1FILLMODE, 0x0C00_4504, false);
        r
    }

    fn block(r: &mut Rss, x0: u32, y0: u32, x1: u32, y1: u32) {
        r.write(reg::IR_ALIAS, 0x18, false);
        r.write(reg::BLOCKXYSTARTI, x0 << 16 | y0, false);
        r.write(reg::BLOCKXYENDI, x1 << 16 | y1, true);
    }

    #[test]
    fn fast_fill_lands_top_down_with_yflip() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::FILL_COLOR_R, 0x13, false);
        block(&mut r, 10, 20, 12, 21);
        assert_eq!(px(&r, 10, 20), 0x13);
        assert_eq!(px(&r, 12, 21), 0x13);
        assert_eq!(px(&r, 13, 21), 0);
        assert_eq!(px(&r, 10, 22), 0);
    }

    /// The draw-buffer field picks DRBpointers' first page (1), its second
    /// (2) or both (3): an X double-buffered window draws its back buffer
    /// with 2 while the screen shows A.
    #[test]
    fn draw_buffer_field_picks_a_b_or_both() {
        let mut r = x_server();
        let a = r.reg(reg::DRBPOINTERS) & 0x3FF;
        let b_page = a + 0x100;
        r.write(reg::DRBPOINTERS, (r.reg(reg::DRBPOINTERS) & !0xF_FFFF) | b_page << 10 | a, false);
        let fm = r.reg(reg::PP1FILLMODE) & !(0x7F << 14);
        let read = |r: &Rss, page: u32, x: u32, y: u32| {
            let buf = Buffer::new(page, Kind::Wide, r.reg(reg::DRBSIZE));
            r.mem.get(&buf, x, (SCREEN_H - 1 - y as usize) as u32) as u32 & 0xFFF
        };
        r.write(reg::FILLMODE, FILL_FAST, false);
        for (field, x, c) in [(1u32, 10, 0x11), (2, 20, 0x22), (3, 30, 0x33)] {
            r.write(reg::PP1FILLMODE, fm | field << 14, false);
            r.write(reg::FILL_COLOR_R, c, false);
            block(&mut r, x, 5, x, 5);
        }
        assert_eq!([read(&r, a, 10, 5), read(&r, b_page, 10, 5)], [0x11, 0], "field 1: A");
        assert_eq!([read(&r, a, 20, 5), read(&r, b_page, 20, 5)], [0, 0x22], "field 2: B");
        assert_eq!([read(&r, a, 30, 5), read(&r, b_page, 30, 5)], [0x33, 0x33], "field 3: both");
    }

    #[test]
    fn pixel_reads_select_a_or_b_independently_of_drawing() {
        let mut r = x_server();
        let a = 0x240;
        let b = 0x140;
        r.write(reg::DRBPOINTERS, a | b << 10, false);
        let ba = Buffer::new(a, Kind::Wide, r.reg(reg::DRBSIZE));
        let bb = Buffer::new(b, Kind::Wide, r.reg(reg::DRBSIZE));
        r.mem.put(&ba, 10, 1018, 0x123456);
        r.mem.put(&bb, 10, 1018, 0x654321);
        r.write(reg::FILLMODE, 2 << 22, false);
        r.write(reg::XFRMODE, 0x80, false);
        r.write(reg::XFRSIZE, 1 << 16 | 1, false);
        for (read, draw, expected) in [(0, 2, 0x123456), (1, 1, 0x654321), (0, 3, 0x123456), (1, 3, 0x654321)] {
            r.write(reg::PP1FILLMODE, 0x0C00_0204 | read << 21 | draw << 14, false);
            block(&mut r, 10, 5, 10, 5);
            assert_eq!(r.pio_read_hi(), expected, "read {read}, draw {draw}");
        }
    }

    /// File Manager scrolls its depth-12 TrueColor child by reading
    /// RGBA4444 with pp1fillmode 0, then uploading with 0x0c004004.
    /// Newly exposed rows are filled with 0x0c00c804. All three paths
    /// must agree for every background, icon, and text colour.
    #[test]
    fn rgb12_scroll_preserves_every_colour() {
        let mut r = x_server();
        r.write(reg::DRBPOINTERS, 0x240 | 0x240 << 10, false);
        r.write(reg::XFRMODE, 0x88, false);
        r.write(reg::XFRSIZE, 1 << 16 | 1, false);
        for colour in 0..0x1000 {
            r.write(reg::PP1FILLMODE, 0x0C00_C804, false);
            r.write(reg::COLORMASKLSBSA, 0xFF_FFFF, false);
            r.write(reg::COLORMASKLSBSB, 0xFF_FFFF, false);
            r.write(reg::FILLMODE, FILL_FAST, false);
            r.write(reg::FILL_COLOR_R, (colour & 0xF) << 8, false);
            r.write(reg::FILL_COLOR_G, (colour & 0xF0) << 4, false);
            r.write(reg::FILL_COLOR_B, colour & 0xF00, false);
            block(&mut r, 10, 5, 10, 5);
            let expected = from_host((8, 8), colour as u64);
            assert_eq!(px(&r, 10, 5), expected, "fresh fill {colour:#x}");
            for _ in 0..3 {
                r.write(reg::PP1FILLMODE, 0, false);
                r.write(reg::COLORMASKLSBSA, 0xFFF, false);
                r.write(reg::FILLMODE, 4 << 22, false);
                block(&mut r, 10, 5, 10, 5);
                let bytes = r.dma_read_line(0);
                assert_eq!(bytes, (colour as u16).to_be_bytes());
                r.write(reg::PP1FILLMODE, 0x0C00_4004, false);
                r.write(reg::FILLMODE, 5 << 22, false);
                block(&mut r, 10, 5, 10, 5);
                r.dma_write_line(0, &bytes);
                assert_eq!(px(&r, 10, 5), expected, "scroll {colour:#x}");
            }
        }
    }

    #[test]
    fn drawing_both_aliased_buffers_applies_xor_once() {
        let mut r = x_server();
        r.write(reg::DRBPOINTERS, 0x240 | 0x240 << 10, false);
        r.write(reg::PP1FILLMODE, 6 << 26 | PP1_LOGIC_OP_ENABLE | 3 << 14 | 0x500, false);
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::FILL_COLOR_R, 0xA5, false);
        block(&mut r, 10, 5, 10, 5);
        assert_eq!(px(&r, 10, 5), 0xA5);
        block(&mut r, 10, 5, 10, 5);
        assert_eq!(px(&r, 10, 5), 0);
    }

    #[test]
    fn fragments_replicate_to_both_buffers_with_their_own_masks() {
        let mut r = x_server();
        r.write(reg::DRBPOINTERS, 0x240 | 0x140 << 10, false);
        r.write(reg::PP1FILLMODE, 0x0C00_C204, false);
        r.write(reg::COLORMASKLSBSA, 0xFF, false);
        r.write(reg::COLORMASKLSBSB, 0xFF_0000, false);
        let a = Buffer::new(0x240, Kind::Wide, r.reg(reg::DRBSIZE));
        let b = Buffer::new(0x140, Kind::Wide, r.reg(reg::DRBSIZE));
        r.mem.put(&a, 10, 1018, 0x123456);
        r.mem.put(&b, 10, 1018, 0x654321);
        r.gl_fragment(10, 5, [1.0, 1.0, 1.0, 1.0], 0.0);
        assert_eq!(r.mem.get(&a, 10, 1018), 0x1234FF);
        assert_eq!(r.mem.get(&b, 10, 1018), 0xFF4321);
    }

    #[test]
    fn rgb_fast_fill_packs_components() {
        let mut r = x_server();
        r.write(reg::PP1FILLMODE, 3 << 26 | 0x104, false); // RGB pixel type, copy
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::FILL_COLOR_R, 0xF00, false);
        r.write(reg::FILL_COLOR_G, 0x800, false);
        r.write(reg::FILL_COLOR_B, 0x100, false);
        block(&mut r, 0, 0, 0, 0);
        assert_eq!(px(&r, 0, 0), 0x10_80F0);
    }

    #[test]
    fn pio_write_frames_each_line_in_a_fresh_doubleword() {
        let mut r = x_server();
        // Three 1-byte pixels per line, two lines, begin skip 2: line 0 is
        // bytes 2..5 of its doubleword; line 1 starts at (2 + 3) & 7 = 5.
        r.write(reg::FILLMODE, 3 << 22, false);
        r.write(reg::XFRMODE, 2 << 8, false);
        r.write(reg::XFRSIZE, 2 << 16 | 3, false);
        block(&mut r, 100, 50, 102, 51);
        assert_eq!(r.transfer_armed(), Some(false));
        r.write(reg::CHAR_H, 0x0000_0102, false);
        r.write(reg::CHAR_L, 0x03FF_FFFF, true);
        r.write(reg::CHAR_H, 0x0000_0000, false);
        r.write(reg::CHAR_L, 0x0004_0506, true);
        assert_eq!([px(&r, 100, 50), px(&r, 101, 50), px(&r, 102, 50)], [1, 2, 3]);
        assert_eq!([px(&r, 100, 51), px(&r, 101, 51), px(&r, 102, 51)], [4, 5, 6]);
    }

    #[test]
    fn pio_read_frames_each_line_in_a_fresh_doubleword() {
        let mut r = x_server();
        for (k, v) in [1u32, 2, 3, 4, 5, 6].into_iter().enumerate() {
            let (x, y) = (100 + k % 3, 50 + k / 3);
            r.mem.put(&main_buffer(), x as u32, (SCREEN_H - 1 - y) as u32, v as u64);
        }
        // Three 1-byte pixels per line, two lines, begin skip 2: line 0 is
        // bytes 2..5 of its doubleword; line 1 starts at (2 + 3) & 7 = 5.
        r.write(reg::FILLMODE, 2 << 22, false);
        r.write(reg::XFRMODE, 2 << 8, false);
        r.write(reg::XFRSIZE, 2 << 16 | 3, false);
        block(&mut r, 100, 50, 102, 51);
        assert_eq!(r.transfer_armed(), Some(true));
        let mut dw = || {
            let hi = r.pio_peek_hi();
            assert_eq!(r.pio_read_hi(), hi, "peek and take agree");
            (hi as u64) << 32 | r.pio_read_lo() as u64
        };
        assert_eq!(dw(), 0x0000_0102_0300_0000);
        assert_eq!(dw(), 0x0000_0000_0004_0506);
        assert_eq!(dw(), 0, "past the end");
    }

    #[test]
    fn a_new_block_disarms_a_finished_write_transfer() {
        let mut r = x_server();
        r.write(reg::FILLMODE, 5 << 22, false);
        r.write(reg::XFRSIZE, 1 << 16 | 1, false);
        block(&mut r, 0, 0, 0, 0);
        assert!(r.transfer_armed().is_some());
        // A glyph block (block type 1 + char stipple, as X sends it): its
        // char data must stipple, not feed the transfer.
        r.write(reg::FILLMODE, 1 << 22 | FILL_CHAR_STIPPLE, false);
        r.write(reg::RED, 0x7 << 12, false);
        block(&mut r, 200, 10, 203, 10);
        assert_eq!(r.transfer_armed(), None);
        r.write(reg::CHAR_H, 0xA000_0000, true);
        assert_eq!([px(&r, 200, 10), px(&r, 201, 10), px(&r, 202, 10)], [7, 0, 7]);
    }

    #[test]
    fn rgb_host_formats_round_trip() {
        for (fmt, v) in [((8, 8), 0x0ABCu64), ((8, 10), 0x7FFF), ((8, 0), 0x00C0_FFEE), ((0, 1), 0xFFF), ((7, 1), 0x1212_3434_5656), ((7, 0), 0x12_3456), ((2, 3), 0x1234_5612), ((8, 1), 0x1212_3434_5656_FFFF)] {
            assert_eq!(to_host(fmt, from_host(fmt, v)), v, "{fmt:?}");
        }
        assert_eq!(from_host((8, 8), 0x0F0), pack_rgb(0, 0xFF, 0));
        // A 12-bit visual's pixel reads back as its top nibbles.
        assert_eq!(to_host((8, 8), 0x50_2020), 0x522);
        assert_eq!(from_host((8, 8), 0x522), 0x55_2222);
        // 8-8-8 host pixels are X pixel values of the visuals, red in 7:0.
        assert_eq!(from_host((8, 0), 0x00_00FF), pack_rgb(0xFF, 0, 0));
    }

    #[test]
    fn a_24_plane_mask_means_rgb_fills() {
        let mut r = x_server();
        r.write(reg::PP1FILLMODE, 0x0C00_6304, false);
        r.write(reg::COLORMASKLSBSA, 0xFF_FFFF, false);
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::FILL_COLOR_R, 0x380, false);
        r.write(reg::FILL_COLOR_G, 0x8E0, false);
        r.write(reg::FILL_COLOR_B, 0x8E0, false);
        block(&mut r, 1, 1, 1, 1);
        assert_eq!(px(&r, 1, 1), 0x8E_8E38);
    }

    #[test]
    fn an_all_planes_copy_keeps_24_bit_pixels() {
        let mut r = x_server();
        // A window move's write-back: pixel type 2, every plane enabled,
        // 4-byte host pixels.
        r.write(reg::PP1FILLMODE, 0x0C00_6204, false);
        r.write(reg::COLORMASKLSBSA, 0xFFFF_FFFF, false);
        r.write(reg::FILLMODE, 5 << 22, false);
        r.write(reg::XFRMODE, 0x80, false);
        r.write(reg::XFRSIZE, 1 << 16 | 1, false);
        block(&mut r, 7, 7, 7, 7);
        r.dma_write_line(0, &[0x00, 0x50, 0x50, 0x50]);
        assert_eq!(px(&r, 7, 7), 0x50_5050);
    }

    #[test]
    fn xor_fill_toggles_and_restores() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::FILL_COLOR_R, 0x13, false);
        block(&mut r, 5, 5, 5, 5);
        r.write(reg::PP1FILLMODE, 0x0C00_4504 & !(0xF << 26) | 6 << 26, false);
        r.write(reg::FILL_COLOR_R, 0x0F, false);
        block(&mut r, 5, 5, 5, 5);
        assert_eq!(px(&r, 5, 5), 0x13 ^ 0x0F);
        block(&mut r, 5, 5, 5, 5);
        assert_eq!(px(&r, 5, 5), 0x13);
    }

    #[test]
    fn stippled_line_skips_zero_bits() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_LINE_STIPPLE, false);
        r.write(reg::RED, 0x9 << 12, false);
        r.write(reg::LINE_STIPPLE, 0xAAAA_AAAA, false);
        r.write(reg::IR_ALIAS, 0x15, false);
        r.write(reg::LINE_START, 300 << 16 | 40, false);
        r.write(reg::LINE_END, 303 << 16 | 40, true);
        assert_eq!((300..304).map(|x| px(&r, x, 40)).collect::<Vec<_>>(), [9, 0, 9, 0]);
    }

    /// What the IRIX desktop sends to shade an icon: fill mode 0x60 (line
    /// stipple, opaque), the foreground in RED, the background in BG_COLOR.
    /// Both colours land; nothing underneath shows through.
    #[test]
    fn opaque_stippled_line_draws_zero_bits_in_the_background_colour() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_LINE_STIPPLE | FILL_LINE_STIPPLE_OPAQUE, false);
        r.write(reg::RED, 0xF << 12, false);
        r.write(reg::BG_COLOR, 0x7, false);
        r.write(reg::LINE_STIPPLE, 0xAAAA_AAAA, false);
        r.write(reg::IR_ALIAS, 0x15, false);
        r.write(reg::LINE_START, 300 << 16 | 40, false);
        r.write(reg::LINE_END, 303 << 16 | 40, true);
        assert_eq!((300..304).map(|x| px(&r, x, 40)).collect::<Vec<_>>(), [0xF, 7, 0xF, 7]);
    }

    /// In RGB modes the background is three 12-bit components across two
    /// registers, as IRIX writes it for a 12-bit window's icon: white
    /// is 0xf00f00 (blue, green) and 0xf00 (red). These are the values from
    /// a trace of the desktop repainting its Icon Catalog.
    #[test]
    fn opaque_stipple_background_in_rgb_modes() {
        let mut r = x_server();
        r.write(reg::PP1FILLMODE, 0x0C00_4004, false); // RGB pixel type
        r.write(reg::COLORMASKLSBSA, 0xFFF, false);
        r.write(reg::FILLMODE, FILL_LINE_STIPPLE | FILL_LINE_STIPPLE_OPAQUE, false);
        r.write(reg::PACKEDCOLOR, 0xA0_A0A0, false);
        r.write(reg::BG_COLOR, 0xC00_700, false);
        r.write(reg::BG_COLOR_RED, 0x300, false);
        r.write(reg::LINE_STIPPLE, 0xAAAA_AAAA, false);
        r.write(reg::IR_ALIAS, 0x15, false);
        r.write(reg::LINE_START, 300 << 16 | 40, false);
        r.write(reg::LINE_END, 301 << 16 | 40, true);
        assert_eq!(px(&r, 300, 40), 0xAA_AAAA);
        assert_eq!(px(&r, 301, 40), pack_rgb(0x33, 0x77, 0xCC), "expanded RGB444 background");
    }

    /// Block type 0 is a fill in the iterated colour, drawn at once (the
    /// kernel's boot gradient: one block per line, no char data).
    #[test]
    fn block_type_0_fills_at_once() {
        let mut r = x_server();
        r.write(reg::FILLMODE, 0x18000, false);
        r.write(reg::RED, 0xDE << 12, false);
        block(&mut r, 10, 30, 12, 30);
        assert_eq!((10..14).map(|x| px(&r, x, 30)).collect::<Vec<_>>(), [0xDE, 0xDE, 0xDE, 0]);
    }

    /// An opaque char stipple draws its 0 bits in the background colour.
    #[test]
    fn opaque_char_stipple_draws_the_background() {
        let mut r = x_server();
        r.write(reg::FILLMODE, 1 << 22 | FILL_CHAR_STIPPLE | FILL_CHAR_STIPPLE_OPAQUE, false);
        r.write(reg::RED, 0x7 << 12, false);
        r.write(reg::BG_COLOR, 0x3, false);
        block(&mut r, 200, 10, 203, 10);
        r.write(reg::CHAR_H, 0xA000_0000, true);
        assert_eq!((200..205).map(|x| px(&r, x, 10)).collect::<Vec<_>>(), [7, 3, 7, 3, 0]);
    }

    /// X PolyPoint: IR opcode 4, then each execute of xline_xystarti draws
    /// one pixel there.
    #[test]
    fn points_draw_at_the_line_start_register() {
        let mut r = x_server();
        r.write(reg::RED, 0x5 << 12, false);
        r.write(reg::IR_ALIAS, 0x4, false);
        r.write(reg::LINE_START, 1205 << 16 | 174, true);
        r.write(reg::LINE_START, 1204 << 16 | 176, true);
        assert_eq!((px(&r, 1205, 174), px(&r, 1204, 176), px(&r, 1204, 174)), (5, 5, 0));
    }

    /// Fill mode bit 10 (X's CapNotLast) leaves the end point out.
    #[test]
    fn skip_last_leaves_the_end_point_out() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_LINE_SKIP_LAST, false);
        r.write(reg::RED, 0x9 << 12, false);
        r.write(reg::IR_ALIAS, 0x15, false);
        r.write(reg::LINE_START, 300 << 16 | 40, false);
        r.write(reg::LINE_END, 303 << 16 | 40, true);
        assert_eq!((300..305).map(|x| px(&r, x, 40)).collect::<Vec<_>>(), [9, 9, 9, 0, 0]);
    }

    /// Screen masks 1-4: each enabled one must pass (inside or outside, per
    /// its window mode bit).
    #[test]
    fn screen_masks_combine() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::FILL_COLOR_R, 0x7, false);
        // Mask 1: keep inside x 10..20; mask 3: keep outside x 14..15.
        r.write(reg::CLIP_X, 10 << 16 | 20, false);
        r.write(reg::CLIP_Y, 1023, false);
        r.write(reg::CLIP_X + 4, 14 << 16 | 15, false);
        r.write(reg::CLIP_Y + 4, 1023, false);
        r.write(reg::CLIP_MODE, 0x1 | 0x10 | 0x4, false);
        block(&mut r, 0, 5, 30, 5);
        let drawn: Vec<usize> = (0..31).filter(|&x| px(&r, x, 5) == 7).collect();
        assert_eq!(drawn, [10, 11, 12, 13, 16, 17, 18, 19, 20]);
    }

    /// Clip-ID drawing (draw field 0x50) writes the fill colour's low two
    /// bits under pp1winmode bits 11:10, not the colour planes; CIDmatch
    /// (bits 7:4, one bit per ID) limits other drawing to the IDs it names,
    /// and with no bit set nothing is checked.
    #[test]
    fn clipping_id_writes_masks_and_match() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::PP1FILLMODE, 0x142600, false);
        r.write(reg::PP1WINMODE, 0xC00, false);
        r.write(reg::FILL_COLOR_R, 7, false);
        block(&mut r, 10, 5, 12, 5);
        assert_eq!(px(&r, 10, 5), 0, "CID drawing keeps colour planes");
        assert_eq!(r.cid[1018 * WIDTH + 10], 3, "two clip-ID planes");
        r.write(reg::PP1WINMODE, 0x400, false);
        r.write(reg::FILL_COLOR_R, 0, false);
        block(&mut r, 10, 5, 10, 5);
        assert_eq!(r.cid[1018 * WIDTH + 10], 2, "plane 0 alone written");
        r.write(reg::PP1WINMODE, 0xC00, false);
        block(&mut r, 11, 5, 11, 5);

        r.write(reg::PP1FILLMODE, 0x0C00_4504, false);
        r.write(reg::COLORMASKLSBSA, 0xFF, false);
        // IDs now 2, 0, 3 at x 10, 11, 12.
        r.write(reg::PP1WINMODE, 1 << (4 + 2) | 1 << (4 + 3), false);
        r.write(reg::FILL_COLOR_R, 0x55, false);
        block(&mut r, 10, 5, 12, 5);
        assert_eq!((px(&r, 10, 5), px(&r, 11, 5), px(&r, 12, 5)), (0x55, 0, 0x55));
        r.write(reg::PP1WINMODE, 0xC00, false);
        block(&mut r, 11, 5, 11, 5);
        assert_eq!(px(&r, 11, 5), 0x55, "no CIDmatch bit: no check");
    }

    /// Without the opaque bit, BG_COLOR plays no part.
    #[test]
    fn transparent_stipple_ignores_the_background_colour() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_LINE_STIPPLE, false);
        r.write(reg::RED, 0xF << 12, false);
        r.write(reg::BG_COLOR, 0x7, false);
        r.write(reg::LINE_STIPPLE, 0x5555_5555, false);
        r.write(reg::IR_ALIAS, 0x15, false);
        r.write(reg::LINE_START, 300 << 16 | 40, false);
        r.write(reg::LINE_END, 303 << 16 | 40, true);
        assert_eq!((300..304).map(|x| px(&r, x, 40)).collect::<Vec<_>>(), [0, 0xF, 0, 0xF]);
    }
}

/// The armed transfer's shape, for the trace.
fn xfer_desc(r: &Rss) -> String {
    let mode = r.reg(reg::XFRMODE);
    format!("{}x? bpp={} fmt=({}, {}) begin={} stride_skip={}",
        r.reg(reg::XFRSIZE) & 0xFFFF, bytes_per_pixel(mode), (mode >> 4) & 0xF, mode & 0xF, (mode >> 8) & 7, (mode >> 14) & 0x1FF)
}
