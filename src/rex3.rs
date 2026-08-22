use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use crossbeam_utils::CachePadded;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device, Resettable, Saveable};
use crate::devlog::{LogModule, devlog_is_active, devlog};
use crate::snapshot::{get_field, u32_slice_to_toml, u16_slice_to_toml, u8_slice_to_toml, load_u32_slice, load_u16_slice, load_u8_slice, toml_u32, toml_u64, toml_u8, hex_u32, hex_u64, hex_u8};
use std::cell::{Cell, UnsafeCell};
use crate::vc2::Vc2;
use crate::xmap9::Xmap9;
use crate::cmap::Cmap;
use crate::bt445::Bt445;
use bitfield::bitfield;
use crate::disp::Rex3Screen;
use std::io::Write;

pub trait Renderer: Send {
    /// Composite and present one display frame.
    ///
    /// The renderer owns its compositor, debug overlay, and status bar texture.
    /// It drives the full pipeline: compose → overlay → status bar → swap.
    /// When `need_readback` is true the renderer must populate `screen.rgba`
    /// with the composited pixels (for screenshots) before returning.
    fn present(
        &mut self,
        screen:        &mut crate::disp::Rex3Screen,
        overlay:       &mut crate::debug_overlay::DebugOverlay,
        status:        &mut crate::disp::StatusBar,
        sbtex:         &mut crate::disp::StatusBarTexture,
        stats:         &crate::disp::BarStats,
        need_readback: bool,
        live_fb_rgb:   Option<&[u32]>,
        live_fb_aux:   Option<&[u32]>,
    );

    fn resize(&mut self, _width: usize, _height: usize) {}
    fn stop(&mut self) {}
    /// Switch to GL or SW compositor at runtime. Returns the active compositor name.
    fn switch_compositor(&mut self, _use_gl: bool) -> &'static str { "sw" }
    /// Return a short status string: "compositor=gl shader=integer" etc.
    fn compositor_status(&self) -> String { "compositor=sw shader=n/a".to_string() }
}

pub const REX3_SIZE: u32 = 0x2000; // 8KB
pub const REX3_BASE: u32 = 0x1F0F0000; // Physical base address of registers
pub const GFIFO_DEPTH: usize = 65536;
/// Real hardware GFIFO depth (32 entries). STATUS reports level=1 below this threshold.
pub const GFIFO_HW_DEPTH: usize = 32;
/// Special GFIFO command to trigger a GO without a register write.
pub const GFIFO_PURE_GO: u32 = 0xFFFF_0800;
/// GFIFO_PURE_GO with the GO bit stripped — the reg_offset seen by process_register.
pub const GFIFO_PURE_GO_REG: u32 = GFIFO_PURE_GO & !0x0800;
/// Special GFIFO command to signal the processor thread to exit.
pub const GFIFO_EXIT: u32 = 0xFFFF_0001;
/// Special GFIFO command: signals the display thread that all prior draws are done.
/// Consumer advances gfifo_fence to the payload value when it processes this entry.
/// No GO bit set, so process_register receives this value directly as reg_offset.
pub const GFIFO_DISP_SYNC: u32 = 0xFFFF_0002;
/// Internal-only offset (not a real REX3 register — top of the 8KB window, well past
/// any hardware register) used as a "pure GO, no register write" alias for `dma_read64`/
/// `dma_write64`. Consumed the same way as `GFIFO_PURE_GO`: triggers execute_go() without
/// touching any context field. Exists so MC's VDMA worker can prime/advance the HOSTRW
/// read-then-advance pipeline via the normal GFIFO path (correct ordering relative to
/// concurrent CPU-driven register writes) without needing its own bespoke sentinel.
/// GO bit (0x0800) is baked in, mirroring GFIFO_PURE_GO's convention.
pub const REX3_DMA_PURE_GO: u32 = 0x1FF0 | 0x0800;
/// REX3_DMA_PURE_GO with the GO bit stripped — the reg_offset seen by process_register.
pub const REX3_DMA_PURE_GO_REG: u32 = REX3_DMA_PURE_GO & !0x0800;
pub const REX3_COORD_BIAS: i32 = 4096; // Physical coordinate system offset.
pub const REX3_SCREEN_WIDTH: i32 = 1344; // 1280 displayable + 64 off-screen.
pub const REX3_SCREEN_HEIGHT: i32 = 1024; // Max displayable height.

// Register Offsets
pub const REX3_DRAWMODE1: u32 = 0x0000;
pub const REX3_DRAWMODE0: u32 = 0x0004;
pub const REX3_LSMODE: u32 = 0x0008;
pub const REX3_LSPATTERN: u32 = 0x000C;
pub const REX3_LSPATSAVE: u32 = 0x0010;
pub const REX3_ZPATTERN: u32 = 0x0014;
pub const REX3_COLORBACK: u32 = 0x0018;
pub const REX3_COLORVRAM: u32 = 0x001C;
pub const REX3_ALPHAREF: u32 = 0x0020;
pub const REX3_STALL0: u32 = 0x0024;
pub const REX3_SMASK0X: u32 = 0x0028;
pub const REX3_SMASK0Y: u32 = 0x002C;
pub const REX3_SETUP: u32 = 0x0030;
pub const REX3_STEPZ: u32 = 0x0034;
pub const REX3_LSRESTORE: u32 = 0x0038;
pub const REX3_LSSAVE: u32 = 0x003C;

pub const REX3_XSTART: u32 = 0x0100;
pub const REX3_YSTART: u32 = 0x0104;
pub const REX3_XEND: u32 = 0x0108;
pub const REX3_YEND: u32 = 0x010C;
pub const REX3_XSAVE: u32 = 0x0110;
pub const REX3_XYMOVE: u32 = 0x0114;
pub const REX3_BRESD: u32 = 0x0118;
pub const REX3_BRESS1: u32 = 0x011C;
pub const REX3_BRESOCTINC1: u32 = 0x0120;
pub const REX3_BRESRNDINC2: u32 = 0x0124;
pub const REX3_BRESE1: u32 = 0x0128;
pub const REX3_BRESS2: u32 = 0x012C;
pub const REX3_AWEIGHT0: u32 = 0x0130;
pub const REX3_AWEIGHT1: u32 = 0x0134;
pub const REX3_XSTARTF: u32 = 0x0138;
pub const REX3_YSTARTF: u32 = 0x013C;
pub const REX3_XENDF: u32 = 0x0140;
pub const REX3_YENDF: u32 = 0x0144;
pub const REX3_XSTARTI: u32 = 0x0148;
pub const REX3_XENDF1: u32 = 0x014C;
pub const REX3_XYSTARTI: u32 = 0x0150;
pub const REX3_XYENDI: u32 = 0x0154;
pub const REX3_XSTARTENDI: u32 = 0x0158;

pub const REX3_COLORRED: u32 = 0x0200;
pub const REX3_COLORALPHA: u32 = 0x0204;
pub const REX3_COLORGRN: u32 = 0x0208;
pub const REX3_COLORBLUE: u32 = 0x020C;
pub const REX3_SLOPERED: u32 = 0x0210;
pub const REX3_SLOPEALPHA: u32 = 0x0214;
pub const REX3_SLOPEGRN: u32 = 0x0218;
pub const REX3_SLOPEBLUE: u32 = 0x021C;
pub const REX3_WRMASK: u32 = 0x0220;
pub const REX3_COLORI: u32 = 0x0224;
pub const REX3_COLORX: u32 = 0x0228;
pub const REX3_SLOPERED1: u32 = 0x022C;
pub const REX3_HOSTRW0: u32 = 0x0230;
pub const REX3_HOSTRW1: u32 = 0x0234;
pub const REX3_HOSTRW64: u32 = 0x0231; // addr bit 0 = is_64bit flag
pub const REX3_DCBMODE: u32 = 0x0238;
pub const REX3_DCBDATA0: u32 = 0x0240;
pub const REX3_DCBDATA1: u32 = 0x0244;

pub const REX3_SMASK1X: u32 = 0x1300;
pub const REX3_SMASK1Y: u32 = 0x1304;
pub const REX3_SMASK2X: u32 = 0x1308;
pub const REX3_SMASK2Y: u32 = 0x130C;
pub const REX3_SMASK3X: u32 = 0x1310;
pub const REX3_SMASK3Y: u32 = 0x1314;
pub const REX3_SMASK4X: u32 = 0x1318;
pub const REX3_SMASK4Y: u32 = 0x131C;
pub const REX3_TOPSCAN: u32 = 0x1320;
pub const REX3_XYWIN: u32 = 0x1324;
pub const REX3_CLIPMODE: u32 = 0x1328;
pub const REX3_STALL1: u32 = 0x132C;
pub const REX3_CONFIG: u32 = 0x1330;
pub const REX3_STATUS: u32 = 0x1338;
pub const REX3_USER_STATUS: u32 = 0x133C;
pub const REX3_DCBRESET: u32 = 0x1340;

pub(crate) fn decode_dm0(v: u32) -> String {
    let dm = DrawMode0(v);
    let opcode = match dm.opcode() { 0=>"NOOP", 1=>"READ", 2=>"DRAW", 3=>"SCR2SCR", _=>"?" };
    let adrmode = match dm.adrmode() { 0=>"SPAN", 1=>"BLOCK", 2=>"ILINE", 3=>"FLINE", 4=>"ALINE", _=>"?" };
    let mut flags = String::new();
    if dm.dosetup()      { flags.push_str(" DOSETUP"); }
    if dm.colorhost()    { flags.push_str(" COLORHOST"); }
    if dm.alphahost()    { flags.push_str(" ALPHAHOST"); }
    if dm.stoponx()      { flags.push_str(" STOPONX"); }
    if dm.stopony()      { flags.push_str(" STOPONY"); }
    if dm.skipfirst()    { flags.push_str(" SKIPFIRST"); }
    if dm.skiplast()     { flags.push_str(" SKIPLAST"); }
    if dm.enzpattern()   { flags.push_str(" ENZPAT"); }
    if dm.enlspattern()  { flags.push_str(" ENLSPAT"); }
    if dm.lsadvlast()    { flags.push_str(" LSADVLAST"); }
    if dm.length32()     { flags.push_str(" LEN32"); }
    if dm.zpopaque()     { flags.push_str(" ZPOPAQUE"); }
    if dm.lsopaque()     { flags.push_str(" LSOPAQUE"); }
    if dm.shade()        { flags.push_str(" SHADE"); }
    if dm.lronly()       { flags.push_str(" LRONLY"); }
    if dm.xyoffset()     { flags.push_str(" XYOFFSET"); }
    if dm.ciclamp()      { flags.push_str(" CICLAMP"); }
    if dm.endptfilter()  { flags.push_str(" ENDPTFILT"); }
    if dm.ystride()      { flags.push_str(" YSTRIDE"); }
    format!("{} {}{}", opcode, adrmode, flags)
}

pub(crate) fn decode_dm1(v: u32) -> String {
    let dm = DrawMode1(v);
    let planes = match dm.planes() { 0=>"NONE", 1=>"RGB", 2=>"RGBA", 4=>"OLAY", 5=>"PUP", 6=>"CID", _=>"?" };
    let depth  = match dm.drawdepth()  { 0=>"4bpp", 1=>"8bpp", 2=>"12bpp", 3=>"24bpp", _=>"?" };
    let hdepth = match dm.hostdepth()  { 0=>"12bpp", 1=>"8bpp", 2=>"4bpp", 3=>"32bpp", _=>"?" };
    let logicop = match dm.logicop()   { 0=>"ZERO",1=>"AND",2=>"ANDR",3=>"SRC",4=>"ANDI",5=>"DST",
        6=>"XOR",7=>"OR",8=>"NOR",9=>"XNOR",10=>"NDST",11=>"ORR",12=>"NSRC",13=>"ORI",14=>"NAND",15=>"ONE", _=>"?" };
    let sfactor = ["ZERO","ONE","DCOL","1-DCOL","DALPHA","1-DALPHA","SALPHA","1-SALPHA"];
    let dfactor = sfactor;
    let sf = sfactor.get(dm.sfactor() as usize).copied().unwrap_or("?");
    let df = dfactor.get(dm.dfactor() as usize).copied().unwrap_or("?");
    let mut flags = String::new();
    if dm.dblsrc()      { flags.push_str(" DBLSRC"); }
    if dm.yflip()       { flags.push_str(" YFLIP"); }
    if dm.rwpacked()    { flags.push_str(" RWPACKED"); }
    if dm.rwdouble()    { flags.push_str(" RWDOUBLE"); }
    if dm.swapendian()  { flags.push_str(" SWAPEND"); }
    if dm.rgbmode()     { flags.push_str(" RGB"); } else { flags.push_str(" CI"); }
    if dm.dither()      { flags.push_str(" DITHER"); }
    if dm.fastclear()   { flags.push_str(" FASTCLR"); }
    if dm.blend()       { flags.push_str(format!(" BLEND({}+{})", sf, df).as_str()); }
    if dm.backblend()   { flags.push_str(" BACKBLEND"); }
    if dm.prefetch()    { flags.push_str(" PREFETCH"); }
    if dm.blendalpha()  { flags.push_str(" BLENDALPHA"); }
    format!("{} {} host:{} cmp:{} logicop:{}{}", planes, depth, hdepth, dm.compare(), logicop, flags)
}

fn rex3_reg_name(offset: u32) -> &'static str {
    match offset {
        REX3_DRAWMODE1 => "DRAWMODE1",
        REX3_DRAWMODE0 => "DRAWMODE0",
        REX3_LSMODE => "LSMODE",
        REX3_LSPATTERN => "LSPATTERN",
        REX3_LSPATSAVE => "LSPATSAVE",
        REX3_ZPATTERN => "ZPATTERN",
        REX3_COLORBACK => "COLORBACK",
        REX3_COLORVRAM => "COLORVRAM",
        REX3_ALPHAREF => "ALPHAREF",
        REX3_STALL0 => "STALL0",
        REX3_SMASK0X => "SMASK0X",
        REX3_SMASK0Y => "SMASK0Y",
        REX3_SETUP => "SETUP",
        REX3_STEPZ => "STEPZ",
        REX3_LSRESTORE => "LSRESTORE",
        REX3_LSSAVE => "LSSAVE",
        REX3_XSTART => "XSTART",
        REX3_YSTART => "YSTART",
        REX3_XEND => "XEND",
        REX3_YEND => "YEND",
        REX3_XSAVE => "XSAVE",
        REX3_XYMOVE => "XYMOVE",
        REX3_BRESD => "BRESD",
        REX3_BRESS1 => "BRESS1",
        REX3_BRESOCTINC1 => "BRESOCTINC1",
        REX3_BRESRNDINC2 => "BRESRNDINC2",
        REX3_BRESE1 => "BRESE1",
        REX3_BRESS2 => "BRESS2",
        REX3_AWEIGHT0 => "AWEIGHT0",
        REX3_AWEIGHT1 => "AWEIGHT1",
        REX3_XSTARTF => "XSTARTF",
        REX3_YSTARTF => "YSTARTF",
        REX3_XENDF => "XENDF",
        REX3_YENDF => "YENDF",
        REX3_XSTARTI => "XSTARTI",
        REX3_XENDF1 => "XENDF1",
        REX3_XYSTARTI => "XYSTARTI",
        REX3_XYENDI => "XYENDI",
        REX3_XSTARTENDI => "XSTARTENDI",
        REX3_COLORRED => "COLORRED",
        REX3_COLORALPHA => "COLORALPHA",
        REX3_COLORGRN => "COLORGRN",
        REX3_COLORBLUE => "COLORBLUE",
        REX3_SLOPERED => "SLOPERED",
        REX3_SLOPEALPHA => "SLOPEALPHA",
        REX3_SLOPEGRN => "SLOPEGRN",
        REX3_SLOPEBLUE => "SLOPEBLUE",
        REX3_WRMASK => "WRMASK",
        REX3_COLORI => "COLORI",
        REX3_COLORX => "COLORX",
        REX3_SLOPERED1 => "SLOPERED1",
        REX3_HOSTRW0 => "HOSTRW0",
        REX3_HOSTRW1 => "HOSTRW1",
        REX3_HOSTRW64 => "HOSTRW64",
        REX3_DCBMODE => "DCBMODE",
        REX3_DCBDATA0 => "DCBDATA0",
        REX3_DCBDATA1 => "DCBDATA1",
        REX3_SMASK1X => "SMASK1X",
        REX3_SMASK1Y => "SMASK1Y",
        REX3_SMASK2X => "SMASK2X",
        REX3_SMASK2Y => "SMASK2Y",
        REX3_SMASK3X => "SMASK3X",
        REX3_SMASK3Y => "SMASK3Y",
        REX3_SMASK4X => "SMASK4X",
        REX3_SMASK4Y => "SMASK4Y",
        REX3_TOPSCAN => "TOPSCAN",
        REX3_XYWIN => "XYWIN",
        REX3_CLIPMODE => "CLIPMODE",
        REX3_STALL1 => "STALL1",
        REX3_CONFIG => "CONFIG",
        REX3_STATUS => "STATUS",
        REX3_USER_STATUS => "USER_STATUS",
        REX3_DCBRESET => "DCBRESET",
        _ => "UNKNOWN",
    }
}

bitfield! {
    #[derive(Clone, Copy, Default)]
    #[repr(transparent)]
    pub struct DrawMode0(u32);
    impl Debug;
    pub opcode, _: 1, 0;
    pub adrmode, _: 4, 2;
    pub dosetup, _: 5;
    pub colorhost, _: 6;
    pub alphahost, _: 7;
    pub stoponx, _: 8;
    pub stopony, _: 9;
    pub skipfirst, _: 10;
    pub skiplast, _: 11;
    pub enzpattern, _: 12;
    pub enlspattern, _: 13;
    pub lsadvlast, _: 14;
    pub length32, _: 15;
    pub zpopaque, _: 16;
    pub lsopaque, _: 17;
    pub shade, _: 18;
    pub lronly, _: 19;
    pub xyoffset, _: 20;
    pub ciclamp, _: 21;
    pub endptfilter, _: 22;
    pub ystride, _: 23;
}

pub const DRAWMODE0_OPCODE_NOOP: u32 = 0x0;
pub const DRAWMODE0_OPCODE_READ: u32 = 0x1;
pub const DRAWMODE0_OPCODE_DRAW: u32 = 0x2;
pub const DRAWMODE0_OPCODE_SCR2SCR: u32 = 0x3;

pub const DRAWMODE0_ADRMODE_MASK: u32 = 0x1C;
pub const DRAWMODE0_ADRMODE_SHIFT: u32 = 2;
pub const DRAWMODE0_ADRMODE_SPAN: u32 = 0x0 << 2;
pub const DRAWMODE0_ADRMODE_BLOCK: u32 = 0x1 << 2;
pub const DRAWMODE0_ADRMODE_I_LINE: u32 = 0x2 << 2;
pub const DRAWMODE0_ADRMODE_F_LINE: u32 = 0x3 << 2;
pub const DRAWMODE0_ADRMODE_A_LINE: u32 = 0x4 << 2;

bitfield! {
    #[derive(Clone, Copy, Default)]
    #[repr(transparent)]
    pub struct DrawMode1(u32);
    impl Debug;
    pub planes, _: 2, 0;
    pub drawdepth, _: 4, 3;
    pub dblsrc, _: 5;
    pub yflip, _: 6;
    pub rwpacked, _: 7;
    pub hostdepth, _: 9, 8;
    pub rwdouble, _: 10;
    pub swapendian, _: 11;
    pub compare, _: 14, 12;
    pub rgbmode, _: 15;
    pub dither, _: 16;
    pub fastclear, _: 17;
    pub blend, _: 18;
    pub sfactor, _: 21, 19;
    pub dfactor, _: 24, 22;
    pub backblend, _: 25;
    pub prefetch, _: 26;
    pub blendalpha, _: 27;
    pub logicop, _: 31, 28;
}

/// Mask of dm0 bits that affect interpreter function-pointer selection.
/// opcode(1:0) | colorhost(6) | alphahost(7) | enzpattern(12) | enlspattern(13) |
/// zpopaque(16) | shade(18) | ciclamp(21)
pub const DRAWMODE0_INTERP_SETUP_MASK: u32 =
    0x3 | (1<<6) | (1<<7) | (1<<12) | (1<<13) | (1<<16) | (1<<18) | (1<<21);

/// Mask of dm1 bits that affect planes_setup / host_setup / proc selection.
/// planes(2:0) | drawdepth(4:3) | dblsrc(5) | rwpacked(7) | hostdepth(9:8) |
/// rwdouble(10) | compare(14:12) | rgbmode(15) | dither(16) | fastclear(17) | blend(18) |
/// sfactor(21:19) | dfactor(24:22) | backblend(25) | blendalpha(27) | logicop(31:28)
pub const DRAWMODE1_INTERP_SETUP_MASK: u32 =
    0x7 | (0x3<<3) | (1<<5) | (1<<7) | (0x3<<8) | (1<<10) | (0x7<<12) |
    (1<<15) | (1<<16) | (1<<17) | (1<<18) |
    (0x7<<19) | (0x7<<22) | (1<<25) | (1<<27) | (0xF<<28);

pub const DRAWMODE1_PLANES_NONE: u32 = 0;
pub const DRAWMODE1_PLANES_RGB: u32 = 1;
pub const DRAWMODE1_PLANES_RGBA: u32 = 2;
pub const DRAWMODE1_PLANES_OLAY: u32 = 4;
pub const DRAWMODE1_PLANES_PUP: u32 = 5;
pub const DRAWMODE1_PLANES_CID: u32 = 6;

/// COMPARE=0x7 (all three relations OR'ed) — afunction always passes, i.e. disabled.
/// Hardware reset default; real drawmode1 words always carry this explicitly.
pub const DRAWMODE1_COMPARE_DISABLE: u32 = 0x7 << 12;

pub const DRAWMODE1_LOGICOP_ZERO: u32 = 0 << 28;
pub const DRAWMODE1_LOGICOP_AND: u32 = 1 << 28;
pub const DRAWMODE1_LOGICOP_ANDR: u32 = 2 << 28;
pub const DRAWMODE1_LOGICOP_SRC: u32 = 3 << 28;
pub const DRAWMODE1_LOGICOP_ANDI: u32 = 4 << 28;
pub const DRAWMODE1_LOGICOP_DST: u32 = 5 << 28;
pub const DRAWMODE1_LOGICOP_XOR: u32 = 6 << 28;
pub const DRAWMODE1_LOGICOP_OR: u32 = 7 << 28;
pub const DRAWMODE1_LOGICOP_NOR: u32 = 8 << 28;
pub const DRAWMODE1_LOGICOP_XNOR: u32 = 9 << 28;
pub const DRAWMODE1_LOGICOP_NDST: u32 = 10 << 28;
pub const DRAWMODE1_LOGICOP_ORR: u32 = 11 << 28;
pub const DRAWMODE1_LOGICOP_NSRC: u32 = 12 << 28;
pub const DRAWMODE1_LOGICOP_ORI: u32 = 13 << 28;
pub const DRAWMODE1_LOGICOP_NAND: u32 = 14 << 28;
pub const DRAWMODE1_LOGICOP_ONE: u32 = 15 << 28;

bitfield! {
    #[derive(Clone, Copy, Default)]
    pub struct ModeEntry(u32);
    impl Debug;
    pub buf_sel, _: 0;
    pub ovl_buf_sel, _: 1;
    pub gamma_bypass, _: 2;
    pub msb_cmap, _: 7, 3;
    pub pix_mode, _: 9, 8;
    pub pix_size, _: 11, 10;
    pub video_mode, _: 13, 12;
    pub video_dither_bypass, _: 14;
    pub alpha_en, _: 15;
    pub aux_pix_mode, _: 18, 16;
    pub aux_msb_cmap, _: 23, 19;
}

// DCBMODE Register Bits
pub const DCBMODE_DATAWIDTH_MASK: u32 = 0x3;
pub const DCBMODE_DATAWIDTH_4: u32 = 0;
pub const DCBMODE_DATAWIDTH_1: u32 = 1;
pub const DCBMODE_DATAWIDTH_2: u32 = 2;
pub const DCBMODE_DATAWIDTH_3: u32 = 3;
pub const DCBMODE_ENDATAPACK: u32 = 1 << 2;
pub const DCBMODE_ENCRSINC: u32 = 1 << 3;
pub const DCBMODE_DCBCRS_MASK: u32 = 0x7 << 4;
pub const DCBMODE_DCBCRS_SHIFT: u32 = 4;
pub const DCBMODE_DCBADDR_MASK: u32 = 0xF << 7;
pub const DCBMODE_DCBADDR_SHIFT: u32 = 7;
pub const DCBMODE_SWAPENDIAN: u32 = 1 << 28;

// STATUS Register Bits
pub const STATUS_VERSION_MASK: u32 = 0x7;
pub const STATUS_VERSION_SHIFT: u32 = 0;
pub const STATUS_GFXBUSY: u32 = 1 << 3;
pub const STATUS_BACKBUSY: u32 = 1 << 4;
pub const STATUS_VRINT: u32 = 1 << 5;
pub const STATUS_VIDEOINT: u32 = 1 << 6;
pub const STATUS_GFIFOLEVEL_MASK: u32 = 0x3F << 7;
pub const STATUS_GFIFOLEVEL_SHIFT: u32 = 7;
pub const STATUS_BFIFOLEVEL_MASK: u32 = 0x1F << 13;
pub const STATUS_BFIFOLEVEL_SHIFT: u32 = 13;
pub const STATUS_BFIFO_INT: u32 = 1 << 18;
pub const STATUS_GFIFO_INT: u32 = 1 << 19;

// CONFIG Register Bits
pub const CONFIG_GIO32MODE: u32 = 1 << 0;
pub const CONFIG_BUSWIDTH: u32 = 1 << 1;
pub const CONFIG_EXTREGXCVR: u32 = 1 << 2;
pub const CONFIG_BFIFODEPTH_MASK: u32 = 0xF << 3;
pub const CONFIG_BFIFODEPTH_SHIFT: u32 = 3;
pub const CONFIG_BFIFOABOVEINT: u32 = 1 << 7;
pub const CONFIG_GFIFODEPTH_MASK: u32 = 0x1F << 8;
pub const CONFIG_GFIFODEPTH_SHIFT: u32 = 8;
pub const CONFIG_GFIFOABOVEINT: u32 = 1 << 13;
pub const CONFIG_TIMEOUT_MASK: u32 = 0x7 << 14;
pub const CONFIG_TIMEOUT_SHIFT: u32 = 14;
pub const CONFIG_VREFRESH_MASK: u32 = 0x7 << 17;
pub const CONFIG_VREFRESH_SHIFT: u32 = 17;
pub const CONFIG_FB_TYPE: u32 = 1 << 20;

// CLIPMODE Register Bits
pub const CLIPMODE_ENSMASK_MASK: u32 = 0x1F;
pub const CLIPMODE_CIDMATCH_MASK: u32 = 0xF << 9;
pub const CLIPMODE_CIDMATCH_SHIFT: u32 = 9;
/// Bits of clipmode that affect JIT shader code generation (ensmask + cidmatch).
pub const CLIPMODE_JIT_KEY_MASK: u32 = CLIPMODE_ENSMASK_MASK | CLIPMODE_CIDMATCH_MASK;

bitfield! {
    #[derive(Clone, Copy, Default)]
    #[repr(transparent)]
    pub struct LsMode(u32);
    impl Debug;
    pub lsrcount, set_lsrcount: 7, 0;
    pub lsrepeat, set_lsrepeat: 15, 8;
    pub lsrcntsave, set_lsrcntsave: 23, 16;
    pub lslength, set_lslength: 27, 24;
}

// Octant definitions
pub const OCTANT_YDEC: u32 = 1 << 0;
pub const OCTANT_XDEC: u32 = 1 << 1;
pub const OCTANT_XMAJOR: u32 = 1 << 2;

// Bresenham octant table (aped from MAME do_iline s_bresenham_infos).
// Fields: (incrx1, incrx2, incry1, incry2, y_major)
// MAME applies y as `y -= incry`, so positive incry moves y in the negative direction.
// Shared by setup() (fractional-endpoint correction) and draw_line_bresenham (the
// per-pixel walk) — both interpreter code paths, but setup() also gates what the
// JIT sees (see setup()'s doc comment on why the fractional correction lives there).
#[rustfmt::skip]
const REX3_BRES_OCTANTS: [(i32, i32, i32, i32, bool); 8] = [
    ( 0,  1, -1, -1, true ),  // octant 0
    ( 0,  1,  1,  1, true ),  // octant 1
    ( 0, -1, -1, -1, true ),  // octant 2
    ( 0, -1,  1,  1, true ),  // octant 3
    ( 1,  1,  0, -1, false),  // octant 4
    ( 1,  1,  0,  1, false),  // octant 5
    (-1, -1,  0, -1, false),  // octant 6
    (-1, -1,  0,  1, false),  // octant 7
];

bitfield! {
    #[derive(Clone, Copy, Default)]
    #[repr(transparent)]
    pub struct BresOctInc1(u32);
    impl Debug;
    pub incr1, set_incr1: 19, 0;
    pub octant, set_octant: 26, 24;
}

bitfield! {
    #[derive(Clone, Copy, Default)]
    #[repr(transparent)]
    pub struct BresRndInc2(u32);
    impl Debug;
    pub incr2, set_incr2: 20, 0;
    pub rnd, set_rnd: 31, 24;
}

/// Trait for handling REX3 MMIO register bit manipulation
///
/// REX3 has fixed-point registers that are accessed through MMIO with special semantics:
/// - When writing (rexset): incoming value is uuuuuuuuVVVVbbbb → stored as SSSSSSSVVVV0000
///   where u=unused top bits, V=value bits, b=bottom bits to mask, S=sign extension
/// - When reading (rexget): stored value is SSSSSSSVVVV0000 → returned as 00000000VVVV0000
///   where S=sign extended bits that get masked to zero
pub trait Rex3RegisterOps {
    /// Write to a REX3 register with sign extension and masking
    ///
    /// Takes a value and prepares it for writing to hardware by:
    /// 1. Masking bottom bits to zero
    /// 2. Sign-extending the value bits to fill the top bits
    ///
    /// # Arguments
    /// * `top_bits` - Number of top bits that are unused in input (will be sign-extended in output)
    /// * `bottom_bits` - Number of bottom bits to mask to zero
    ///
    /// # Example
    /// For a 12.4.7 format (12+4=16 value bits, top 9 bits unused, 7 bottom bits masked):
    /// ```
    /// use iris::rex3::Rex3RegisterOps;
    /// let val = 0x12345678u32.rexset(9, 7); // Input: uuuuuuuuuVVVVVVVVVVVVVVVVbbbbbbb
    ///                                        // Output: SSSSSSSSSVVVVVVVVVVVVVVVV0000000
    /// ```
    fn rexset(self, top_bits: u32, bottom_bits: u32) -> u32;

    /// Read from a REX3 register with masking
    ///
    /// Masks the sign-extended top bits to zero, keeping value and masked bottom bits.
    ///
    /// # Arguments
    /// * `top_bits` - Number of top bits to mask to zero (these were sign-extended)
    /// * `bottom_bits` - Number of bottom bits (already zero, kept as zero)
    ///
    /// # Example
    /// For reading a 12.4.7 format register (16 value bits, 9 top bits, 7 bottom bits):
    /// ```
    /// use iris::rex3::Rex3RegisterOps;
    /// let raw_register = 0u32;
    /// let val = raw_register.rexget(9, 7); // Input: SSSSSSSSSVVVVVVVVVVVVVVVV0000000
    ///                                       // Output: 000000000VVVVVVVVVVVVVVVV0000000
    /// ```
    fn rexget(self, top_bits: u32, bottom_bits: u32) -> u32;
}

impl Rex3RegisterOps for u32 {
    fn rexset(self, top_bits: u32, bottom_bits: u32) -> u32 {
        // Mask bottom bits to zero first
        let masked = self & !((1u32 << bottom_bits) - 1);
        // Sign extend by shifting left to position, then arithmetic shift right
        let shift = top_bits;
        ((masked << shift) as i32 >> shift) as u32
    }

    fn rexget(self, top_bits: u32, bottom_bits: u32) -> u32 {
        // Create mask that clears top_bits, keeps value bits and bottom zero bits
        let value_bits = 32 - top_bits - bottom_bits;
        let mask = ((1u32 << (value_bits + bottom_bits)) - 1) & !((1u32 << bottom_bits) - 1);
        self & mask
    }
}

// 16.4(7) format: 16 integer + 4 fractional + 7 masked = 27 bits, 5 top bits unused
// Coordinate registers store i32 in 21.11 fixed-point (11 fractional bits).
// Integer part: val >> 11.  Fractional part: val & 0x7FF.
// From integer: (x as i32) << 11.

// 16.4(7) format: 16 integer + 4 fractional + 7 masked = 27 bits
fn from16_4_7(val: u32) -> i32 {
    val.rexset(5, 7) as i32
}

fn to16_4_7(val: i32) -> u32 {
    (val as u32).rexget(5, 7)
}

// 12.4(7) / GL float format: IRIX writes IEEE 754 floats; hardware masks off bits 31:23
// (float exponent+sign), leaving only mantissa bits 22:7. Always non-negative.
// Stored in same 21.11 layout as from16_4_7 — no sign extension, top 9 bits zeroed.
fn from12_4_7(val: u32) -> i32 {
    (val & 0x007fff80) as i32
}

fn to12_4_7(val: i32) -> u32 {
    (val as u32).rexget(9, 7)
}

// COLORRED internal format: o12.11 (sign bit + 12 integer bits + 11 fractional bits = 24 bits).
// Write wire format depends on mode:
//   - 12-bit CI mode (rgbmode=0, drawdepth=2): o12.9 on the bus → shift left 2 into o12.11.
//   - All other modes: o12.11 on the bus → store raw low 24 bits.
// Read wire format: always o12.11, low 24 bits.
// get_colori() in CI mode: integer part = bits[22:11], i.e. (colorred >> 11) & 0xFFF.

fn from_color_red(val: u32, drawmode1: DrawMode1) -> u32 {
    if !drawmode1.rgbmode() && drawmode1.drawdepth() == 2 {
        // 12-bit CI mode: bus value is o12.9, shift left 2 to store as o12.11.
        (val << 2) & 0xFFFFFF
    } else {
        val & 0xFFFFFF
    }
}
fn to_color_red(val: u32, _drawmode1: DrawMode1) -> u32 {
    val & 0xFFFFFF
}
fn from_color(val: u32) -> u32 {
    val & 0xFFFFF
}
fn to_color(val: u32) -> u32 {
    val & 0xFFFFF
}

// Slope registers: sign-magnitude on the wire, two's-complement 24/20-bit stored internally.
// write decodes sign-magnitude → two's-complement, read returns raw stored bits.
//
// SLOPERED: s(7)12.11 write (bit31=sign, bits[22:0]=magnitude) → stored as 24-bit two's-complement.
// SLOPEALPHA/GRN/BLUE: s(11)8.11 write (bit31=sign, bits[18:0]=magnitude) → stored as 20-bit two's-complement.

fn from_slope_red(val: u32) -> i32 {
    let mag = val & 0x7FFFFF;
    if mag == 0 { return 0; } // negative-zero: sign bit set but magnitude zero → treat as 0
    let result = if val & 0x80000000 != 0 {
        // Negative: two's complement of magnitude, keep bit23 as sign
        (0x00800000u32.wrapping_sub(mag) | 0x00800000) as i32
    } else {
        mag as i32
    };
    // Sign-extend 24-bit → 32-bit
    (result << 8) >> 8
}
fn to_slope_red(val: i32) -> u32 {
    // Read back 24 bits (13.11 two's-complement)
    (val as u32) & 0xFFFFFF
}

fn from_slope(val: u32) -> i32 {
    let mag = val & 0x7FFFF;
    if mag == 0 { return 0; } // negative-zero → treat as 0
    let result = if val & 0x80000000 != 0 {
        (0x00080000u32.wrapping_sub(mag) | 0x00080000) as i32
    } else {
        mag as i32
    };
    // Sign-extend 20-bit → 32-bit
    (result << 12) >> 12
}
fn to_slope(val: i32) -> u32 {
    // Read back 20 bits (9.11 two's-complement)
    (val as u32) & 0xFFFFF
}

/// Write an 8-bit grayscale PNG (used by `rex fbdump`'s ci.png).
fn write_png_gray(path: &std::path::Path, rows: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    writer.write_image_data(rows).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

/// Write a 24-bit RGB PNG (used by `rex fbdump`'s rgb.png).
fn write_png_rgb(path: &std::path::Path, rows: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    writer.write_image_data(rows).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

/// Compact snapshot of one block/span draw for the draw-debug overlay.
#[derive(Clone, Copy, Debug, Default)]
pub struct DrawRecord {
    /// Destination rect in screen (display) coordinates.
    pub x0: i16, pub y0: i16, pub x1: i16, pub y1: i16,
    /// Source rect in screen coordinates (scr2scr only; zero otherwise).
    pub sx0: i16, pub sy0: i16, pub sx1: i16, pub sy1: i16,
    pub dm0: u32, pub dm1: u32,
    pub colori: u32, pub colorback: u32,
    pub wrmask: u32,
    pub lspat: u32, pub zpat: u32,
    /// Expected 32-bit HOSTRW word count for this draw (0 if colorhost=0).
    pub expected_words: u32,
    /// Expected 64-bit HOSTRW double count (0 if colorhost=0 or rwdouble=0).
    pub expected_doubles: u32,
    /// Actual HOSTRW writes received (32-bit words; each 64-bit counts as 1).
    pub hostrw_writes: u32,
    /// Writes that arrived on the HOSTRW path when colorhost=0 (unexpected).
    pub spurious_writes: u32,
    /// Actual HOSTRW reads performed (32-bit words; each 64-bit counts as 1).
    pub hostrw_reads: u32,
    /// Reads that arrived on the HOSTRW path when colorhost=0 (unexpected).
    pub spurious_reads: u32,
}

const DRAW_RING_SIZE: usize = 65536;

pub struct DrawRingBuf {
    pub entries: Vec<DrawRecord>,
    pub head: usize,   // next write slot
    pub count: usize,  // total valid entries (≤ DRAW_RING_SIZE)
    /// Index of the most recently pushed entry (for HOSTRW write attribution).
    pub pending: Option<usize>,
}

impl Default for DrawRingBuf {
    fn default() -> Self {
        Self {
            entries: vec![DrawRecord::default(); DRAW_RING_SIZE],
            head: 0,
            count: 0,
            pending: None,
        }
    }
}

impl DrawRingBuf {
    pub fn push(&mut self, r: DrawRecord) {
        let slot = self.head;
        self.entries[slot] = r;
        self.pending = Some(slot);
        self.head = (self.head + 1) % DRAW_RING_SIZE;
        if self.count < DRAW_RING_SIZE { self.count += 1; }
    }

    /// Called on every HOSTRW write (32-bit or 64-bit).
    /// Increments `hostrw_writes` on the pending draw, or `spurious_writes` if colorhost=0.
    pub fn on_hostrw_write(&mut self) {
        if let Some(idx) = self.pending {
            let r = &mut self.entries[idx];
            if r.expected_words > 0 {
                r.hostrw_writes += 1;
            } else {
                r.spurious_writes += 1;
            }
        }
    }

    /// Called on every HOSTRW read (32-bit or 64-bit).
    /// Increments `hostrw_reads` on the pending draw, or `spurious_reads` if colorhost=0.
    pub fn on_hostrw_read(&mut self) {
        if let Some(idx) = self.pending {
            let r = &mut self.entries[idx];
            if r.expected_words > 0 {
                r.hostrw_reads += 1;
            } else {
                r.spurious_reads += 1;
            }
        }
    }

    /// Iterate entries from newest to oldest.
    pub fn iter_newest_first(&self) -> impl Iterator<Item = &DrawRecord> {
        let n = self.count;
        let head = self.head;
        (0..n).map(move |i| {
            let idx = (head + DRAW_RING_SIZE - 1 - i) % DRAW_RING_SIZE;
            &self.entries[idx]
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Rex3Context {
    pub drawmode1: DrawMode1,
    pub drawmode0: DrawMode0,
    pub lsmode: LsMode,
    pub lspattern: u32,
    pub lspatsave: u32,
    pub zpattern: u32,
    pub colorback: u32,
    pub colorvram: u32,
    pub alpharef: u32,
    pub smask0x: u32,
    pub smask0y: u32,
    /// Coordinates stored as 21.11 fixed-point (plain i32, 11 fractional bits).
    /// Integer part: val >> 11. Write via from_coord_int() or from16_4_7/from12_4_7.
    pub xstart: i32,
    pub ystart: i32,
    pub xend: i32,
    pub yend: i32,
    pub xsave: i32,
    pub xymove: u32,
    pub bresd: u32,
    pub bress1: u32,
    pub bresoctinc1: BresOctInc1,
    pub bresrndinc2: BresRndInc2,
    pub brese1: u32,
    pub bress2: u32,
    pub aweight0: u32,
    pub aweight1: u32,
    /// o12.11 color DDA accumulator: bits[23:11]=integer, bits[10:0]=fraction, bit31=overflow/neg.
    pub colorred: u32,
    pub coloralpha: u32,
    pub colorgrn: u32,
    pub colorblue: u32,
    /// Signed 24-bit slope in two's-complement (after sign-magnitude decode on write).
    pub slopered: i32,
    pub slopealpha: i32,
    pub slopegrn: i32,
    pub slopeblue: i32,
    pub wrmask: u32,
    pub colorx: u32,
    pub smask1x: u32,
    pub smask1y: u32,
    pub smask2x: u32,
    pub smask2y: u32,
    pub smask3x: u32,
    pub smask3y: u32,
    pub smask4x: u32,
    pub smask4y: u32,
    pub topscan: u32,
    pub xywin: u32,
    pub clipmode: u32,
    pub hostrw: u64,
    pub host_shifter: u64,
    pub hostcnt: u32,
    /// Bit index (31..=0) for lspattern with lsmode repeat/length; reset to 31 at GO start and each new row.
    pub pat_bit: u8,
    /// Bit index (31..=0) for zpattern; always 32-bit repeating, reset to 31 at GO start and each new row.
    pub zpat_bit: u8,
    /// True while a multi-GO primitive is in progress (HOSTRW/READ continuation).
    /// Prevents setup() from being re-run on continuation GOs.
    pub mid_primitive: bool,
    pub lssave: u32,
    pub lsrestore: u32,
    pub stepz: u32,
    pub stall0: u32,
    pub stall1: u32,
}

impl Rex3Context {
    /// Power-on/reset state: like `Default::default()`, but with DRAWMODE1.COMPARE
    /// at its documented hardware reset value (0x7 = afunction disabled/always-pass).
    /// `#[derive(Default)]` zero-inits DrawMode1, which would leave COMPARE=0x0
    /// (always-kill) — real REX3 boots with afunction off until software sets it.
    pub fn power_on_default() -> Self {
        let mut ctx = Self::default();
        ctx.drawmode1 = DrawMode1(ctx.drawmode1.0 | (0x7 << 12));
        ctx
    }

    pub fn set_colori(&mut self, val: u32) {
        if self.drawmode1.rgbmode() {
            // CI-style integer write to RGB mode: each byte → component << 11
            let r = val & 0xFF;
            let g = (val >> 8) & 0xFF;
            let b = (val >> 16) & 0xFF;
            self.colorred   = r << 11;
            self.colorgrn   = g << 11;
            self.colorblue  = b << 11;
        } else {
            // CI mode: store index as o12.11 (integer part at bits[22:11])
            // so get_colori() can return colorred >> 11.
            self.colorred = val << 11;
        }
    }

    pub fn get_colori(&self) -> u32 {
        if self.drawmode1.rgbmode() {
            // Clamp each component on read-out
            (Self::clamp_color_component(self.colorblue)  << 16)
            | (Self::clamp_color_component(self.colorgrn) <<  8)
            |  Self::clamp_color_component(self.colorred)
        } else {
            self.colorred >> 11
        }
    }

    /// Extract and clamp one color component from its o12.11/o8.11 DDA register.
    /// integer = bits[22:11] & 0x1FF.
    /// negative (bit31) or int >= 0x180 → 0;  int > 0xFF → 0xFF.
    #[inline(always)]
    pub fn clamp_color_component(c: u32) -> u32 {
        let val = (c >> 11) & 0x1FF;
        if c & (1 << 31) != 0 || val >= 0x180 {
            0
        } else if val > 0xFF {
            0xFF
        } else {
            val
        }
    }

    pub fn set_xstart(&mut self, val: i32) {
        self.xstart = val;
        self.xsave = val;
    }
}

pub struct Rex3Config {
    pub config: AtomicU32,
    /// VRINT bit set by refresh thread on vblank assert, cleared when CPU reads STATUS.
    /// Interrupt line follows: cb(true) on assert, cb(false) on STATUS read.
    pub status: AtomicU32,
}

impl Default for Rex3Config {
    fn default() -> Self {
        Self {
            config: AtomicU32::new(0),
            status: AtomicU32::new(0),
        }
    }
}

#[derive(Debug)]
pub struct Rex3DcbState {
    pub dcbmode: u32,
    pub dcbdata0: u32,
    pub dcbdata1: u32,
    /// Set by DCB addr=12 write/read. Read during STATUS read (CPU thread only — same thread as DCB).
    pub backbusy_until: Option<std::time::Instant>,
}

impl Rex3DcbState {
    pub fn crs(&self) -> u8 {
        ((self.dcbmode & DCBMODE_DCBCRS_MASK) >> DCBMODE_DCBCRS_SHIFT) as u8
    }
    /// Returns CRS before incrementing. Increments by `n` if ENCRSINC is set.
    pub fn inc_crs(&mut self, n: u8) -> u8 {
        let old = self.crs();
        if (self.dcbmode & DCBMODE_ENCRSINC) != 0 {
            let new_crs = old.wrapping_add(n) & 0x7;
            self.dcbmode = (self.dcbmode & !DCBMODE_DCBCRS_MASK) | ((new_crs as u32) << DCBMODE_DCBCRS_SHIFT);
        }
        old
    }
}

impl Default for Rex3DcbState {
    fn default() -> Self {
        Self {
            dcbmode: 0xF << DCBMODE_DCBADDR_SHIFT, // DCBADDR init = 0xF per spec
            dcbdata0: 0,
            dcbdata1: 0,
            backbusy_until: None,
        }
    }
}

/// A GFIFO entry: plain addr + val, no synchronization fields.
/// Ordering is guaranteed by the push spinlock (producers) and the
/// tail Release store / head Acquire load (consumer).
#[derive(Clone, Copy)]
pub struct GFIFOEntry {
    pub addr: u32,
    pub val:  u64,
}

/// MPSC ring buffer for the GFIFO.
///
/// Multiple producers are serialized by `lock` (an atomic spinlock).
/// The lock ensures only one producer writes at a time, so tail can be
/// updated atomically after the write — no per-slot ready flag needed.
/// Single consumer (painter thread) advances `head`.
/// `head` and `tail` are on separate cache lines to avoid false sharing.
/// Capacity is `GFIFO_DEPTH` entries; holds at most `GFIFO_DEPTH - 1` live entries.
pub struct GFifo {
    lock: AtomicBool,
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
    shadow_head: CachePadded<Cell<usize>>,
    shadow_tail: CachePadded<Cell<usize>>,
    local_head: CachePadded<Cell<usize>>,
    buf:  [GFIFOEntry; GFIFO_DEPTH],
}

const GFIFO_MASK: usize = GFIFO_DEPTH - 1;

// SAFETY: GFifo is always heap-allocated. buf access is serialized:
// producers via lock, consumer via exclusive head ownership.
unsafe impl Send for GFifo {}
unsafe impl Sync for GFifo {}

impl GFifo {
    pub fn new() -> Self {
        // SAFETY: always heap-allocated; zeroed GFIFOEntry (u32+u64) is valid.
        unsafe { std::mem::zeroed() }
    }

    /// Returns the approximate number of entries currently in the queue.
    #[inline]
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail.wrapping_sub(head) & GFIFO_MASK
    }

    /// Push an entry. Spins if full. Safe to call from multiple producers concurrently.
    #[inline]
    pub fn push(&self, addr: u32, val: u64) {
        // Acquire the spinlock — uncontested in the common case (one active producer).
        while self.lock.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            while self.lock.load(Ordering::Relaxed) {
                std::hint::spin_loop();
            }
        }
        // Spin if full — consumer will drain it.
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = tail.wrapping_add(1) & GFIFO_MASK;
        let mut cached_head = self.shadow_head.get();
        if next_tail == cached_head {
            cached_head = self.head.load(Ordering::Acquire);
            self.shadow_head.set(cached_head);
            while next_tail == cached_head {
                std::hint::spin_loop();
                cached_head = self.head.load(Ordering::Acquire);
                self.shadow_head.set(cached_head);
            }
        }
        // SAFETY: we hold the lock; no other producer touches this slot.
        unsafe {
            let slot = self.buf.as_ptr().add(tail) as *mut GFIFOEntry;
            (*slot).addr = addr;
            (*slot).val  = val;
        }
        // Release: consumer's Acquire on tail sees the slot write above.
        self.tail.store(next_tail, Ordering::Release);
        self.lock.store(false, Ordering::Release);
    }

    /// Peek at the next entry without advancing head. Returns `None` if empty.
    /// Must be called from the consumer thread only.
    #[inline]
    pub fn peek(&self) -> Option<(u32, u64)> {
        let head = self.local_head.get();
        let mut cached_tail = self.shadow_tail.get();
        if head == cached_tail {
            cached_tail = self.tail.load(Ordering::Acquire);
            self.shadow_tail.set(cached_tail);
            if head == cached_tail {
                return None;
            }
        }
        // SAFETY: consumer owns head; no producer touches a slot before tail is published,
        // and our Acquire on tail pairs with the producer's Release store.
        let slot = unsafe { &*self.buf.as_ptr().add(head) };
        Some((slot.addr, slot.val))
    }

    /// Advance head past the current entry after it has been fully processed.
    /// Must follow a successful `peek()`.
    #[inline]
    pub fn consume(&self) {
        let head = self.local_head.get();
        let next_head = head.wrapping_add(1) & GFIFO_MASK;
        self.local_head.set(next_head);
        // Publish every 64 entries to massively reduce cache line invalidations
        // — but ALSO publish immediately whenever this consume just drained
        // the queue to empty (next_head == tail), regardless of the
        // batching boundary. External observers (is_empty/len — REX3's own
        // `busy_or_val!` register-read gate among them) read `head` directly
        // and have no way to know the consumer thread's private
        // `local_head` is already caught up; without this, consuming the
        // last entry of an otherwise-empty run (whenever that count isn't a
        // multiple of 64 — the overwhelmingly common case) leaves `head`
        // stale, so `is_empty()` keeps reporting "not empty" — and if the
        // consumer thread then exits (register_processor's own `else`
        // branch normally catches this on its *next* loop iteration via
        // `flush_head()`, but a `stop()`-driven `GFIFO_EXIT` can end the
        // loop before that next iteration ever runs) the stale `head` is
        // permanent: nothing ever re-derives it, and every future
        // busy_or_val!-gated register read reports busy forever even though
        // the queue is, and has been, genuinely empty. (Found live: `rex
        // status` showing `DRAW BUSY: no` with `GFIFO: 1/65536` — gfxbusy
        // correctly cleared, but `head` never got the memo.)
        let tail = self.tail.load(Ordering::Acquire);
        if next_head & 63 == 0 || next_head == tail {
            self.head.store(next_head, Ordering::Release);
        }
    }

    #[inline]
    pub fn flush_head(&self) {
        self.head.store(self.local_head.get(), Ordering::Release);
    }

    /// True when no entries are pending.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    /// Force the queue to the empty state, discarding any pending entries.
    /// Only safe to call with the consumer thread stopped (checkpoint
    /// restore's own contract — `restore_live_checkpoint` always calls
    /// `self.stop()` first) — this bypasses the normal producer lock/
    /// consumer head-ownership discipline entirely, so a live producer or
    /// consumer racing this call would corrupt the ring buffer's invariants.
    #[inline]
    pub fn reset(&self) {
        let tail = self.tail.load(Ordering::Relaxed);
        self.head.store(tail, Ordering::Release);
        self.local_head.set(tail);
        self.shadow_tail.set(tail);
        self.shadow_head.set(tail);
    }
}

#[derive(Default)]
struct DebugState {
    last_offset: Option<u32>,
    last_val: u32,
    count: u32,
}

pub struct Rex3 {
    pub config: Rex3Config,
    pub dcb: Mutex<Rex3DcbState>,
    pub context: UnsafeCell<Rex3Context>,
    // Framebuffer: 2048x1024 pixels, 32-bit per pixel.
    // Stored as a dense array for fast access.
    // Accessed by painter thread (write) and refresh thread (read).
    // No internal synchronization; external coordination or tolerance for tearing required.
    // Each bit of a pixel represents a plane.
    pub fb_rgb: UnsafeCell<Box<[u32]>>,
    pub fb_aux: UnsafeCell<Box<[u32]>>,
    pub px_rd: UnsafeCell<fn(&Rex3, u32) -> u32>,
    pub px_wr: UnsafeCell<fn(&Rex3, u32, u32)>,
    pub px_amp: UnsafeCell<fn(u32) -> u32>,
    pub px_logic: UnsafeCell<fn(u32, u32) -> u32>,
    /// Afunction test: (src_alpha, alpharef) -> pass. Selected in planes_setup from
    /// DRAWMODE1 COMPARE/RGBMODE; a no-op always-pass fn outside rgbmode or COMPARE==0x7.
    pub px_afunc: UnsafeCell<fn(u32, u32) -> bool>,
    /// Pack bayer index into bits [27:24] for dither compress functions (rgbmode=1 only).
    /// In CI mode this is a no-op (returns color unchanged) — CI values need no dithering.
    pub px_bayer: UnsafeCell<fn(u32, i32, i32) -> u32>,
    /// Compress 24-bit BGR → plane-depth pixel (rgbmode=1 only; identity otherwise).
    pub px_compress: UnsafeCell<fn(u32) -> u32>,
    /// Expand plane-depth pixel → 24-bit BGR (for blend dst; identity for CI/24bpp).
    pub px_expand: UnsafeCell<fn(u32) -> u32>,
    pub px_proc: UnsafeCell<fn(&Rex3, &mut Rex3Context, i32, i32)>,
    /// Per-pixel shade DDA step + optional clamp.  Called after every pixel (draw_block, draw_iline).
    pub px_shade: UnsafeCell<fn(&mut Rex3Context)>,
    /// Per-pixel pattern bit advance for lspattern and/or zpattern.
    /// Handles lsmode repeat/length for lspattern; simple rotate for zpattern.
    /// Called after every pixel (draw_block, draw_iline).
    pub px_pattern: UnsafeCell<fn(&mut Rex3Context)>,
    pub host_unpack: UnsafeCell<fn(u64) -> u32>,
    pub host_pack: UnsafeCell<fn(u64, u32) -> u64>,
    pub host_shift: UnsafeCell<u32>,
    pub host_count: UnsafeCell<u32>,
    pub gfifo: GFifo,

    pub vc2: Mutex<Vc2>,
    pub xmap0: Mutex<Xmap9>,
    pub xmap1: Mutex<Xmap9>,
    pub cmap0: Mutex<Cmap>,
    pub cmap1: Mutex<Cmap>,
    pub bt445: Mutex<Bt445>,
    clock: AtomicU64,
    running: AtomicBool,
    pub gfxbusy: Arc<AtomicBool>,
    pub processor_thread: Mutex<Option<thread::JoinHandle<()>>>,
    pub refresh_thread: Mutex<Option<thread::JoinHandle<()>>>,
    /// True while the REX3-Processor thread is parked on an empty gfifo. The
    /// producer (`gfifo_push`) checks this and unparks the consumer so a fresh
    /// command is picked up immediately instead of after the park timeout.
    #[cfg(feature = "idle-pause")]
    processor_parked: AtomicBool,
    /// Handle to the REX3-Processor thread, set once when it starts, used by
    /// `gfifo_push` to unpark it. OnceLock gives lock-free reads on the hot path.
    #[cfg(feature = "idle-pause")]
    processor_unparker: std::sync::OnceLock<thread::Thread>,
    /// Set by the gfifo consumer whenever it processes activity that may have
    /// changed the framebuffer. The refresh thread renders only when this (or a
    /// palette/cursor/mode change) is seen, rather than re-converting and
    /// re-uploading the whole framebuffer at 60 Hz on a static screen. Starts
    /// true so the first frame always renders.
    fb_dirty: AtomicBool,
    /// Rows filled via rex3_simd fastclear path (monitor `perf snapshot`).
    pub simd_fill_rows: AtomicU64,
    pub screen: Arc<Mutex<Rex3Screen>>,
    /// Drives the real REX3 `VV_INT_N` pin (vertical retrace / Kaleidoscope).
    pub vblank_cb: Mutex<Option<Arc<dyn Fn(bool) + Send + Sync>>>,
    /// Drives the real REX3 `FIFO_INT_N` pin (GFIFO/BFIFO above/below
    /// interrupt — one physical pin for both directions, selected by
    /// `CONFIG_GFIFOABOVEINT`; see `update_gfifo_irqs`).
    pub fifo_full_cb: Mutex<Option<Arc<dyn Fn(bool) + Send + Sync>>>,
    /// Incremented each time an XMAP mode table entry is written (buf_sel flip signal).
    /// Payload of GFIFO_DISP_SYNC pushed to the GFIFO on each such write.
    pub xmap_fence: AtomicU32,
    /// Written by the GFIFO consumer when it processes GFIFO_DISP_SYNC.
    /// Display thread spins until gfifo_fence >= sampled xmap_fence (wrapping).
    pub gfifo_fence: AtomicU32,
    pub debug: Arc<AtomicBool>,
    #[cfg(feature = "developer")]
    pub block_debug: Arc<AtomicBool>,
    #[cfg(feature = "developer")]
    pub draw_debug: Arc<AtomicBool>,
    #[cfg(feature = "developer")]
    pub draw_ring: Arc<Mutex<DrawRingBuf>>,
    #[cfg(feature = "developer")]
    pub gfifo_hwm: AtomicUsize,
    /// Number of execute_go() calls dispatched via the JIT (compiled shader hit).
    pub jit_go_count: AtomicU64,
    /// Number of execute_go() calls dispatched via the interpreter (JIT miss or disabled).
    pub interp_go_count: AtomicU64,
    /// Activity/lock diagnostic bits — set while holding a lock or inside a loop.
    /// Read at any time to see what the refresh/painter/processor threads are doing.
    pub diag: AtomicU64,
    debug_state: Mutex<DebugState>,
    pub renderer: Mutex<Option<Box<dyn Renderer>>>,
    #[cfg(feature = "rex-jit")]
    pub rex_jit: Option<std::sync::Arc<crate::rex3_jit::RexJit>>,
    /// Whether the JIT is enabled for dispatch (can be toggled at runtime via `rex jit on/off`).
    #[cfg(feature = "rex-jit")]
    pub jit_enabled: AtomicBool,
    /// Last-hit JIT shader cache: avoids HashMap lookup when (dm0, dm1, clipmode_key) is
    /// the same as the previous GO.  Only accessed from the GFIFO consumer thread — no sync needed.
    #[cfg(feature = "rex-jit")]
    pub jit_last: std::cell::Cell<(u32, u32, u32, Option<unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32)>)>,
    /// Cached interpreter setup key: (dm0 & DRAWMODE0_INTERP_SETUP_MASK,
    /// dm1_normalized & DRAWMODE1_INTERP_SETUP_MASK, clipmode_cidmatch).
    /// When these match the current GO, planes_setup/host_setup and function-pointer
    /// selection are skipped since nothing they depend on has changed.
    interp_setup_cache: std::cell::Cell<(u32, u32, u32)>,
    /// Shared activity heartbeat — set by all devices, polled+cleared by the refresh thread.
    /// bit 0 = enet TX, bit 1 = enet RX, bits 2-3 = red/green LED (persistent), bits 8-13 = SCSI IDs 0-5
    pub heartbeat: Arc<AtomicU64>,
    /// Pointer into the CPU's `MipsCore.hot.cycles` word (see
    /// `CyclesPtr`/`Hot::cycles`'s doc comments). Set exactly once, from
    /// `set_cpu_cycles`, during single-threaded `Machine::new` setup — after
    /// the CPU exists (`Rex3::new` runs before it does, so this can't be a
    /// constructor parameter; see `MipsCpu::cycles_ptr`) but strictly before
    /// any device thread (including this struct's own refresh thread) is
    /// spawned, so a plain `Cell` (same as `interp_setup_cache` above) is
    /// enough — no atomicity needed for the pointer variable itself.
    /// `CyclesPtr::dangling()` until `set_cpu_cycles` runs; its own `get()`
    /// treats that as "not wired up yet" and reports 0.
    pub cycles: std::cell::Cell<crate::mips_core::CyclesPtr>,
    /// CP0 Count==Compare match counter — incremented every fastick interrupt.
    pub fasttick_count: Arc<AtomicU64>,
    pub decoded_count: Arc<AtomicU64>,
    pub l1i_hit_count: Arc<AtomicU64>,
    pub l1i_fetch_count: Arc<AtomicU64>,
    pub uncached_fetch_count: Arc<AtomicU64>,
    /// Optional log file for block/span draws (set when block_debug is enabled).
    #[cfg(feature = "developer")]
    block_log: Mutex<Option<std::fs::File>>,
    /// GFIFO log (written by painter thread on process_register, one line per entry).
    #[cfg(feature = "developer")]
    rex3_log: Mutex<Option<std::fs::File>>,
    /// When true, the refresh thread overlays a 16x16 grid of 8x8 CMAP swatches.
    pub show_cmap: AtomicBool,
    /// When true, overlay decoded DID/XMAP mode info near the bottom of the screen.
    pub show_disp_debug: AtomicBool,
    /// Set by UI when RightCtrl+PrintScreen is pressed; cleared by refresh thread after saving.
    pub screenshot_pending: AtomicBool,
    /// Monotonically incrementing screenshot counter for unique filenames.
    pub screenshot_counter: AtomicU32,
    /// Atomic shadow of MipsCore::count_hz — updated by CPU thread, read by refresh thread.
    /// Wrapped in Mutex so machine.rs can swap in the real Arc from MipsCore after construction.
    #[cfg(feature = "developer")]
    pub count_hz_atomic: Mutex<Arc<AtomicU64>>,
}

unsafe impl Sync for Rex3 {}

impl Rex3 {
    pub fn new(heartbeat: Arc<AtomicU64>, fasttick_count: Arc<AtomicU64>, decoded_count: Arc<AtomicU64>, l1i_hit_count: Arc<AtomicU64>, l1i_fetch_count: Arc<AtomicU64>, uncached_fetch_count: Arc<AtomicU64>) -> Self {
        let config = Rex3Config::default();
        config.config.store(CONFIG_BUSWIDTH | CONFIG_EXTREGXCVR |
                        (8 << CONFIG_BFIFODEPTH_SHIFT) |
                        CONFIG_BFIFOABOVEINT |
                        (16 << CONFIG_GFIFODEPTH_SHIFT) |
                        CONFIG_GFIFOABOVEINT |
                        (1 << CONFIG_VREFRESH_SHIFT), Ordering::Relaxed);

        // Initialize with random data (noise)
        let mut rng_state = 0xDEADBEEFu32;
        let mut next_rand = || {
            let mut x = rng_state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            rng_state = x;
            x
        };

        let fb_rgb = (0..(2048 * 1024)).map(|_| next_rand()).collect::<Vec<u32>>().into_boxed_slice();
        let fb_aux = (0..(2048 * 1024)).map(|_| next_rand()).collect::<Vec<u32>>().into_boxed_slice();
        
        let screen = Arc::new(Mutex::new(crate::disp::Rex3Screen::new()));

        Self {
            config,
            dcb: Mutex::new(Rex3DcbState::default()),
            context: UnsafeCell::new(Rex3Context::power_on_default()),
            fb_rgb: UnsafeCell::new(fb_rgb),
            fb_aux: UnsafeCell::new(fb_aux),
            px_rd: UnsafeCell::new(Self::default_px_rd),
            px_wr: UnsafeCell::new(Self::default_px_wr),
            px_amp: UnsafeCell::new(Self::amplify_nop),
            px_logic: UnsafeCell::new(Self::logic_op_src),
            px_afunc: UnsafeCell::new(Self::afunc_always),
            px_bayer: UnsafeCell::new(Self::bayer_pack),
            px_compress: UnsafeCell::new(Self::identity),
            px_expand: UnsafeCell::new(Self::identity),
            px_proc: UnsafeCell::new(Self::process_pixel_noop),
            px_shade: UnsafeCell::new(Self::iterate_shade_noop),
            px_pattern: UnsafeCell::new(Self::iterate_pattern_noop),
            host_unpack: UnsafeCell::new(Self::host_unpack_nop),
            host_pack: UnsafeCell::new(Self::host_pack_nop),
            host_shift: UnsafeCell::new(0),
            host_count: UnsafeCell::new(0),
            gfifo: GFifo::new(),
            vc2: Mutex::new(Vc2::new()),
            xmap0: Mutex::new(Xmap9::new()),
            xmap1: Mutex::new(Xmap9::new()),
            cmap0: Mutex::new(Cmap::new(0)),
            cmap1: Mutex::new(Cmap::new(1)),
            bt445: Mutex::new(Bt445::new()),
            clock: AtomicU64::new(0),
            running: AtomicBool::new(false),
            gfxbusy: Arc::new(AtomicBool::new(false)),
            processor_thread: Mutex::new(None),
            refresh_thread: Mutex::new(None),
            #[cfg(feature = "idle-pause")]
            processor_parked: AtomicBool::new(false),
            #[cfg(feature = "idle-pause")]
            processor_unparker: std::sync::OnceLock::new(),
            fb_dirty: AtomicBool::new(true),
            simd_fill_rows: AtomicU64::new(0),
            screen,
            vblank_cb: Mutex::new(None),
            fifo_full_cb: Mutex::new(None),
            xmap_fence: AtomicU32::new(0),
            gfifo_fence: AtomicU32::new(0),
            debug: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "developer")]
            block_debug: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "developer")]
            draw_debug: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "developer")]
            draw_ring: Arc::new(Mutex::new(DrawRingBuf::default())),
            #[cfg(feature = "developer")]
            gfifo_hwm: AtomicUsize::new(0),
            jit_go_count: AtomicU64::new(0),
            interp_go_count: AtomicU64::new(0),
            debug_state: Mutex::new(DebugState::default()),
            diag: AtomicU64::new(0),
            renderer: Mutex::new(None),
            // IRIS_NO_JIT (set by the sandboxed Mac App Store GUI build) forces
            // the interpreter: Cranelift's mmap+mprotect executable pages aren't
            // MAP_JIT, so the App Sandbox kills the process with SIGKILL/
            // CODESIGNING the first time a compiled draw shader runs. Skip
            // *constructing* RexJit (not just dispatch) so its warm-up compiler
            // thread never allocates executable memory in the first place.
            #[cfg(feature = "rex-jit")]
            rex_jit: if std::env::var_os("IRIS_NO_JIT").is_some() {
                None
            } else {
                Some(std::sync::Arc::new(crate::rex3_jit::RexJit::new()))
            },
            #[cfg(feature = "rex-jit")]
            jit_enabled: AtomicBool::new(std::env::var_os("IRIS_NO_JIT").is_none()),
            #[cfg(feature = "rex-jit")]
            jit_last: std::cell::Cell::new((0, 0, 0, None)),
            interp_setup_cache: std::cell::Cell::new((u32::MAX, u32::MAX, u32::MAX)),
            heartbeat,
            cycles: std::cell::Cell::new(crate::mips_core::CyclesPtr::dangling()),
            fasttick_count,
            decoded_count,
            l1i_hit_count,
            l1i_fetch_count,
            uncached_fetch_count,
            #[cfg(feature = "developer")]
            block_log: Mutex::new(None),
            #[cfg(feature = "developer")]
            rex3_log: Mutex::new(None),
            show_cmap: AtomicBool::new(false),
            show_disp_debug: AtomicBool::new(false),
            screenshot_pending: AtomicBool::new(false),
            screenshot_counter: AtomicU32::new(0),
            #[cfg(feature = "developer")]
            count_hz_atomic: Mutex::new(Arc::new(AtomicU64::new(crate::mips_core::DEFAULT_COUNT_HZ))),
        }
    }

    /// Heartbeat bit definitions — all share the single heartbeat atomic
    pub const HB_ENET_TX:   u64 = 1 << 0;
    pub const HB_ENET_RX:   u64 = 1 << 1;
    pub const HB_LED_RED:   u64 = 1 << 2; // IOC front-panel red LED (persistent)
    pub const HB_LED_GREEN: u64 = 1 << 3; // IOC front-panel green LED (persistent)
    pub const HB_SCSI_BASE: u32 = 8; // bits 8-13 = SCSI IDs 0-5

    /// Mask of persistent bits that are NOT cleared by the per-frame fetch_and.
    const HB_PERSISTENT: u64 = Self::HB_LED_RED | Self::HB_LED_GREEN;

    // diag atomic bit assignments — set while holding a lock or inside a loop section.
    // Mutex bits (set between lock() and drop of guard)
    pub const DIAG_LOCK_CONFIG:       u64 = 1 << 0;
    pub const DIAG_LOCK_VC2:          u64 = 1 << 1;
    pub const DIAG_LOCK_CMAP0:        u64 = 1 << 2;
    pub const DIAG_LOCK_CMAP1:        u64 = 1 << 3;
    pub const DIAG_LOCK_XMAP0:        u64 = 1 << 4;
    pub const DIAG_LOCK_XMAP1:        u64 = 1 << 5;
    pub const DIAG_LOCK_SCREEN:       u64 = 1 << 6;
    pub const DIAG_LOCK_RENDERER:     u64 = 1 << 7;
    pub const DIAG_LOCK_VBLANK_CB:    u64 = 1 << 8;
    pub const DIAG_LOCK_DEBUG_STATE:  u64 = 1 << 11;
    pub const DIAG_LOCK_DCB:          u64 = 1 << 12;
    // Loop/section bits (set for the duration of a significant section)
    pub const DIAG_LOOP_FB_COPY:      u64 = 1 << 16;
    pub const DIAG_LOOP_VC2_COPY:     u64 = 1 << 17;
    pub const DIAG_LOOP_CMAP_COPY:    u64 = 1 << 18;
    pub const DIAG_LOOP_XMAP_COPY:    u64 = 1 << 19;
    pub const DIAG_LOOP_VID_TIMINGS:  u64 = 1 << 20;
    pub const DIAG_LOOP_DECODE_DID:   u64 = 1 << 21;
    pub const DIAG_LOOP_PIXEL_CONV:   u64 = 1 << 22;
    pub const DIAG_LOOP_GL_RENDER:    u64 = 1 << 23;
    pub const DIAG_LOOP_DRAW_BLOCK:   u64 = 1 << 24;
    pub const DIAG_LOOP_EXECUTE_GO:   u64 = 1 << 25;

    fn default_px_rd(rex: &Rex3, addr: u32) -> u32 {
        unsafe { (*rex.fb_rgb.get())[addr as usize] }
    }

    fn default_px_wr(rex: &Rex3, addr: u32, val: u32) {
        unsafe { (*rex.fb_rgb.get())[addr as usize] = val; }
    }

    pub fn set_vblank_callback(&self, cb: Arc<dyn Fn(bool) + Send + Sync>) {
        *self.vblank_cb.lock() = Some(cb);
    }

    pub fn set_fifo_full_callback(&self, cb: Arc<dyn Fn(bool) + Send + Sync>) {
        *self.fifo_full_cb.lock() = Some(cb);
    }

    /// Program VC2 with a host-side Newport timing preset (see `[graphics] resolution`).
    pub fn apply_display_resolution(&self, mode: crate::vc2_timings::NewportResolution) {
        if mode.is_guest() {
            return;
        }
        {
            let mut vc2 = self.vc2.lock();
            crate::vc2_timings::apply_newport_resolution(&mut vc2, mode);
        }
        // Idle background until the guest paints (compositor uses host-only
        // direct-colour fallback while xmap is still zero — see CaptureRenderer).
        if let Some((w, h)) = mode.visible_size() {
            const FILL: u32 = 0x0060_0000; // dark blue, Newport BGR (B<<16|G<<8|R)
            let w = w as usize;
            let h = h as usize;
            unsafe {
                let fb = &mut *self.fb_rgb.get();
                for y in 0..h.min(1024) {
                    let row = y * 2048;
                    fb[row..row + w.min(2048)].fill(FILL);
                }
            }
        }
        self.fb_dirty.store(true, Ordering::Relaxed);
    }

    #[inline]
    fn gfifo_hw_level(pending: usize) -> u32 {
        if pending == 0 {
            0
        } else {
            (pending.saturating_sub(GFIFO_DEPTH - GFIFO_HW_DEPTH) as u32).max(1)
        }
    }

    /// Recompute the GFIFO threshold condition and drive the real REX3
    /// `FIFO_INT_N` pin (one physical pin for both above- and
    /// below-threshold crossings — `CONFIG_GFIFOABOVEINT` just selects which
    /// direction triggers it; see `fifo_full_cb`'s doc comment).
    fn update_gfifo_irqs(&self) {
        let pending = self.gfifo.len();
        let level = Self::gfifo_hw_level(pending);
        let cfg = self.config.config.load(Ordering::Relaxed);
        let threshold = (cfg >> CONFIG_GFIFODEPTH_SHIFT) & 0x1F;
        let above_int = cfg & CONFIG_GFIFOABOVEINT != 0;

        let crossed = if above_int { level >= threshold } else { level < threshold };
        if crossed {
            self.config.status.fetch_or(STATUS_GFIFO_INT, Ordering::Relaxed);
        } else {
            self.config.status.fetch_and(!STATUS_GFIFO_INT, Ordering::Relaxed);
        }

        if let Some(cb) = self.fifo_full_cb.lock().clone() {
            cb(crossed);
        }
    }

    #[cfg(feature = "developer")]
    pub fn set_count_hz_atomic(&self, arc: Arc<AtomicU64>) {
        *self.count_hz_atomic.lock() = arc;
    }

    /// Wire up the CPU cycle counter — called from `Machine::new` once
    /// `MipsCpu` exists (`Rex3::new` runs before it does, so this can't be a
    /// constructor parameter; see `Hot::cycles`'s doc comment for why it's a
    /// raw pointer at all, not a shared `Arc<AtomicU64>`).
    pub fn set_cpu_cycles(&self, ptr: crate::mips_core::CyclesPtr) {
        self.cycles.set(ptr);
    }

    fn setup(&self, ctx: &mut Rex3Context) {
        let dx = ctx.xend - ctx.xstart;
        let dy = ctx.yend - ctx.ystart;

        let adx = dx.abs() >> 11;
        let ady = dy.abs() >> 11;

        let mut octant = 0u32;
        if dy < 0 { octant |= OCTANT_YDEC; }
        if dx < 0 { octant |= OCTANT_XDEC; }
        if adx > ady { octant |= OCTANT_XMAJOR; }

        let (major, minor) = if adx > ady { (adx, ady) } else { (ady, adx) };

        // Bresenham integer parameters (iline).
        //   incr1 = 2 * minor          (straight step increment, always >= 0)
        //   incr2 = 2 * (minor - major) (diagonal step increment, always <= 0)
        //   d     = incr1 - major       (initial decision variable, I_LINE formula)
        let incr1: i32 = 2 * minor;
        let incr2: i32 = 2 * (minor - major);
        let mut d: i32 = incr1 - major;

        // Store as plain two's-complement integers masked to register field widths.
        ctx.bresoctinc1.set_octant(octant);
        ctx.bresoctinc1.set_incr1((incr1 as u32) & 0xFFFFF);     // 20-bit, always positive
        ctx.bresrndinc2.set_incr2((incr2 as u32) & 0x1FFFFF);    // 21-bit signed

        // F_LINE/A_LINE: apply the fractional-endpoint correction HERE, not at draw
        // time. This must happen in setup() (not draw_line_bresenham/fline_apply_fract)
        // because setup() is the only Bresenham-state derivation shared by both the
        // interpreter and the JIT — the JIT-compiled entry point never calls back into
        // interpreter draw functions, it just replays whatever bresd/bresoctinc1/
        // bresrndinc2/xstart/ystart setup() already wrote into ctx. Applying the
        // correction only in draw_line_bresenham (the old approach) left the JIT
        // silently drawing plain-I_LINE trajectories for F_LINE/A_LINE, since the
        // JIT never reaches that interpreter-only code path.
        let adrmode = ctx.drawmode0.adrmode() << 2;
        let is_fractional = adrmode == DRAWMODE0_ADRMODE_F_LINE || adrmode == DRAWMODE0_ADRMODE_A_LINE;
        if is_fractional {
            let (_incrx1, incrx2, _incry1, incry2, y_major) = REX3_BRES_OCTANTS[(octant & 7) as usize];
            let mut x = ctx.xstart >> 11;
            let mut y = ctx.ystart >> 11;
            Self::fline_apply_fract(ctx, &mut d, &mut x, &mut y, incrx2, incry2, y_major);
            ctx.xstart = x << 11;
            ctx.ystart = y << 11;
        }

        ctx.bresd = (d as u32) & 0x7FF_FFFF;                     // 27-bit signed
    }

    /// Whether the draw-debug overlay/ring-buffer tracking is active. Always false
    /// in non-developer builds (the toggle command to enable it doesn't exist there),
    /// so callers can use this instead of touching the developer-gated fields directly.
    #[cfg(feature = "developer")]
    #[inline(always)]
    fn draw_debug_active(&self) -> bool { self.draw_debug.load(Ordering::Relaxed) }
    #[cfg(not(feature = "developer"))]
    #[inline(always)]
    fn draw_debug_active(&self) -> bool { false }

    /// Record a HOSTRW write/read against the pending draw-debug entry. No-op in
    /// non-developer builds.
    #[cfg(feature = "developer")]
    #[inline(always)]
    fn note_hostrw_write(&self) { if self.draw_debug_active() { self.draw_ring.lock().on_hostrw_write(); } }
    #[cfg(not(feature = "developer"))]
    #[inline(always)]
    fn note_hostrw_write(&self) {}

    #[cfg(feature = "developer")]
    #[inline(always)]
    fn note_hostrw_read(&self) { if self.draw_debug_active() { self.draw_ring.lock().on_hostrw_read(); } }
    #[cfg(not(feature = "developer"))]
    #[inline(always)]
    fn note_hostrw_read(&self) {}

    #[cfg(not(feature = "developer"))]
    #[inline(always)]
    fn log_block(&self, _ctx: &Rex3Context, _opcode: u32) {}

    #[cfg(feature = "developer")]
    fn log_block(&self, ctx: &Rex3Context, opcode: u32) {
        let need_block_log = self.block_log.lock().is_some();
        let need_draw_ring = self.draw_debug_active();
        if ctx.mid_primitive || (!need_block_log && !need_draw_ring) { return; }

        let is_scr2scr = opcode == DRAWMODE0_OPCODE_SCR2SCR;
        let is_span = ctx.drawmode0.adrmode() == 0;
        let x_win  = ((ctx.xywin  >> 16) & 0xFFFF) as i16 as i32;
        let y_win  = ( ctx.xywin         & 0xFFFF) as i16 as i32;
        let x_move = ((ctx.xymove >> 16) & 0xFFFF) as i16 as i32;
        let y_move = ( ctx.xymove        & 0xFFFF) as i16 as i32;
        let apply_xymove = is_scr2scr || ctx.drawmode0.xyoffset();
        let topscan = ctx.topscan as i32;

        let to_scr_x = |raw: i32| -> i16 {
            (raw + x_win + if apply_xymove { x_move } else { 0 } - REX3_COORD_BIAS) as i16
        };
        let to_scr_y = |raw: i32| -> i16 {
            let fb_y = raw + y_win + if apply_xymove { y_move } else { 0 } - REX3_COORD_BIAS;
            (fb_y - (topscan + 1)).rem_euclid(1024) as i16
        };
        let to_src_x = |raw: i32| -> i16 { (raw + x_win - REX3_COORD_BIAS) as i16 };
        let to_src_y = |raw: i32| -> i16 {
            let fb_y = raw + y_win - REX3_COORD_BIAS;
            (fb_y - (topscan + 1)).rem_euclid(1024) as i16
        };

        let x0r = ctx.xstart >> 11;
        let y0r = ctx.ystart >> 11;
        let x1r = ctx.xend >> 11;
        let y1r = ctx.yend >> 11;
        let dst_x0 = to_scr_x(x0r); let dst_y0 = to_scr_y(y0r);
        let dst_x1 = to_scr_x(x1r); let dst_y1 = to_scr_y(y1r);
        let (src_x0, src_y0, src_x1, src_y1) = if is_scr2scr {
            (to_src_x(x0r), to_src_y(y0r), to_src_x(x1r), to_src_y(y1r))
        } else { (0, 0, 0, 0) };

        if need_block_log {
            if let Some(f) = self.block_log.lock().as_mut() {
                let planes_str = match ctx.drawmode1.planes() {
                    DRAWMODE1_PLANES_RGB  => "RGB",  DRAWMODE1_PLANES_RGBA => "RGBA",
                    DRAWMODE1_PLANES_OLAY => "OLAY", DRAWMODE1_PLANES_PUP  => "PUP",
                    DRAWMODE1_PLANES_CID  => "CID",  _ => "?"
                };
                let bpp = match ctx.drawmode1.drawdepth() { 0=>4, 1=>8, 2=>12, 3=>24, _=>0 };
                let opcode_str = match opcode {
                    DRAWMODE0_OPCODE_READ    => "READ",
                    DRAWMODE0_OPCODE_DRAW    => "DRAW",
                    DRAWMODE0_OPCODE_SCR2SCR => "SCR2SCR",
                    _                        => "NOOP",
                };
                let adrmode_str = match ctx.drawmode0.adrmode() {
                    0=>"SPAN", 1=>"BLOCK", 2=>"I_LINE", 3=>"F_LINE", 4=>"A_LINE", _=>"?"
                };
                let logicop_str = match ctx.drawmode1.logicop() {
                    0=>"ZERO", 1=>"AND",  2=>"ANDR", 3=>"SRC",
                    4=>"ANDI", 5=>"DST",  6=>"XOR",  7=>"OR",
                    8=>"NOR",  9=>"XNOR",10=>"NDST",11=>"ORR",
                    12=>"NSRC",13=>"ORI",14=>"NAND",15=>"ONE", _=>"?"
                };
                let w = ((ctx.xend - ctx.xstart) >> 11).unsigned_abs() + 1;
                let h = if is_span { 1 } else { ((ctx.yend - ctx.ystart) >> 11).unsigned_abs() + 1 };
                let colorhost = ctx.drawmode0.colorhost();
                let alphahost = ctx.drawmode0.alphahost();
                let cidmatch = (ctx.clipmode >> CLIPMODE_CIDMATCH_SHIFT) & 0xF;
                let fastclear_active = ctx.drawmode1.fastclear() && cidmatch == 0xF;
                let op_label = if fastclear_active { "FASTCLEAR" } else { opcode_str };
                let src_info = if is_scr2scr {
                    format!(" src=({},{})-({},{})", src_x0, src_y0, src_x1, src_y1)
                } else { String::new() };
                let (log_x1, log_y1) = if is_span { (dst_x1, dst_y0) } else { (dst_x1, dst_y1) };
                let _ = writeln!(f, "{}: planes={} {} {}bpp logicop={} wrmask={:06x} colorhost={} alphahost={} dst=({},{})-({},{}) size={}x{}{}",
                    op_label, planes_str, adrmode_str, bpp, logicop_str,
                    ctx.wrmask, colorhost as u8, alphahost as u8,
                    dst_x0, dst_y0, log_x1, log_y1, w, h, src_info);
                let _ = writeln!(f, "  DM0={:08x} DM1={:08x}", ctx.drawmode0.0, ctx.drawmode1.0);
                if fastclear_active {
                    let _ = writeln!(f, "  colorvram={:08x} colorback={:08x}", ctx.colorvram, ctx.colorback);
                } else {
                    let _ = writeln!(f, "  color=(r:{:06x},g:{:05x},b:{:05x}) colori={:08x} colorback={:08x}",
                        ctx.colorred, ctx.colorgrn, ctx.colorblue,
                        ctx.get_colori(), ctx.colorback);
                }
                let _ = writeln!(f, "  enzpat={} zpat={:08x} zpopaque={} enlspat={} lspat={:08x} lsopaque={} dblsrc={}",
                    ctx.drawmode0.enzpattern() as u8, ctx.zpattern, ctx.drawmode0.zpopaque() as u8,
                    ctx.drawmode0.enlspattern() as u8, ctx.lspattern, ctx.drawmode0.lsopaque() as u8,
                    ctx.drawmode1.dblsrc() as u8);
                if colorhost || alphahost {
                    let hdepth = ctx.drawmode1.hostdepth();
                    let double = ctx.drawmode1.rwdouble();
                    let packed = ctx.drawmode1.rwpacked();
                    let hbpp = match hdepth { 0=>4, 1=>8, 2=>12, 3=>32, _=>0 };
                    let ppw = if packed { match hdepth { 0=>8, 1=>4, 2=>2, 3=>1, _=>1 } } else { 1 };
                    let row_align = if double { ppw * 2 } else { ppw };
                    let words_per_row = w.div_ceil(row_align) * (if double { 2 } else { 1 });
                    let total_32b = words_per_row * h;
                    let _ = writeln!(f, "  HOSTW: hbpp={} packed={} double={} ppw={} words_per_row={} expected_32b={} expected_64b={}",
                        hbpp, packed as u8, double as u8, ppw, words_per_row, total_32b, total_32b.div_ceil(2));
                }
            }
        }

        if need_draw_ring {
            let colorhost = ctx.drawmode0.colorhost();
            let alphahost = ctx.drawmode0.alphahost();
            let (expected_words, expected_doubles) = if colorhost || alphahost {
                let w = ((ctx.xend - ctx.xstart) >> 11).unsigned_abs() + 1;
                let h = if is_span { 1 } else { ((ctx.yend - ctx.ystart) >> 11).unsigned_abs() + 1 };
                let hdepth = ctx.drawmode1.hostdepth();
                let double = ctx.drawmode1.rwdouble();
                let packed = ctx.drawmode1.rwpacked();
                let ppw: u32 = if packed { match hdepth { 0=>8, 1=>4, 2=>2, _=>1 } } else { 1 };
                let row_align = if double { ppw * 2 } else { ppw };
                let words_per_row = w.div_ceil(row_align) * (if double { 2 } else { 1 });
                let total_32b = words_per_row * h;
                (total_32b, if double { total_32b / 2 } else { 0 })
            } else { (0, 0) };
            self.draw_ring.lock().push(DrawRecord {
                x0: dst_x0, y0: dst_y0, x1: dst_x1, y1: if is_span { dst_y0 } else { dst_y1 },
                sx0: src_x0, sy0: src_y0, sx1: src_x1, sy1: src_y1,
                dm0: ctx.drawmode0.0, dm1: ctx.drawmode1.0,
                colori: ctx.get_colori(), colorback: ctx.colorback,
                wrmask: ctx.wrmask, lspat: ctx.lspattern, zpat: ctx.zpattern,
                expected_words, expected_doubles,
                hostrw_writes: 0, spurious_writes: 0,
                hostrw_reads: 0, spurious_reads: 0,
            });
        }
    }

    fn draw_block(&self, ctx: &mut Rex3Context) {
        if crate::rex3_simd::try_fastclear_block(self, ctx) {
            return;
        }
        if crate::rex3_simd::try_src_block_rgb(self, ctx) {
            return;
        }

        let _w = (ctx.xend - ctx.xstart).abs();
        let _h = (ctx.yend - ctx.ystart).abs();

        let stopony = ctx.drawmode0.stopony();
        let length32 = ctx.drawmode0.length32();
        let ystride = ctx.drawmode0.ystride();
        let opcode = ctx.drawmode0.opcode();
        let colorhost = ctx.drawmode0.colorhost();
        let mut first = true;
        let skipfirst = ctx.drawmode0.skipfirst();
        let skiplast = ctx.drawmode0.skiplast();


        // In host mode (READ or DRAW+colorhost), each GO processes exactly one word's worth
        // of pixels (host_count pixels). stop_on_word causes the loop to exit after the
        // word boundary so the next GO picks up where we left off.
        let stop_on_word = opcode == DRAWMODE0_OPCODE_READ || colorhost;

        // stop_on_word takes priority over stoponx — host mode governs its own stop.
        let stoponx = ctx.drawmode0.stoponx() || stop_on_word;

        let octant = ctx.bresoctinc1.octant();
        let lronly = ctx.drawmode0.lronly();
        let x_dec = (octant & OCTANT_XDEC) != 0;
        let y_dec = (octant & OCTANT_YDEC) != 0;
        let lrskip = lronly && x_dec;
        // it is important to note that lrskip still performs y advance operations otherwise some triangles grow weird tails
        // Coordinate steps in 21.11 fixed-point: ±1 integer = ±2048
        let stepx: i32 = if x_dec { -(1 << 11) } else { 1 << 11 };
        let y_inc: i32 = if ystride { 2 } else { 1 };
        let stepy: i32 = if y_dec { -(y_inc << 11) } else { y_inc << 11 };

        // length32 only clamps if span is >= 32 pixels wide
        let span_len = ((ctx.xend - ctx.xstart).abs()) >> 11;
        let xstop = if length32 && span_len >= 32 { Some(ctx.xstart + stepx * 32) } else { None };

        let proc_fn    = unsafe { *self.px_proc.get() };
        let shade_fn   = unsafe { *self.px_shade.get() };
        let pattern_fn = unsafe { *self.px_pattern.get() };

        ctx.mid_primitive = true;
        self.diag.fetch_or(Self::DIAG_LOOP_DRAW_BLOCK, Ordering::Relaxed);
        loop {
            let x = ctx.xstart >> 11;
            let y = ctx.ystart >> 11;

            ctx.xstart += stepx;

            let x_end_reached = if x_dec { ctx.xstart < ctx.xend } else { ctx.xstart > ctx.xend };

            if !(first && skipfirst || x_end_reached && skiplast || lrskip) {
                proc_fn(self, ctx, x, y);
            }

            shade_fn(ctx);
            pattern_fn(ctx);

            if x_end_reached {
                // advance y, wrap x; reset pattern bits so each row starts at bit 31
                ctx.ystart += stepy;
                ctx.xstart = ctx.xsave;
                ctx.pat_bit  = 31;
                ctx.zpat_bit = 31;
                // lsrcount continues across rows — iterate_pattern_ls manages it per-pixel.

                if !stopony {
                    // Without STOPONY, hardware doesn't auto-advance rows — each row is
                    // its own complete primitive and the next GO starts a fresh one.
                    // Mirrors the JIT's emit_shader (!stopony branch in end_x_block),
                    // which already clears mid_primitive here. Leaving this true (as
                    // before) permanently wedged log_block()'s "primitive start" guard
                    // after the first non-stopony block ever ran, hiding every
                    // subsequent block/READ/DRAW header from block.log.
                    ctx.mid_primitive = false;
                    break;
                }

                let y_end_reached = if y_dec { ctx.ystart < ctx.yend } else { ctx.ystart > ctx.yend };

                if y_end_reached {
                    // All rows consumed — primitive done.
                    ctx.mid_primitive = false;
                    break;
                }

                // Host mode: a row boundary is always a forced word boundary too,
                // even if the row's width doesn't divide evenly into host_count
                // (e.g. a 17px-wide CI8 row's last word only has 1 of 4 slots
                // filled). Without this, a still-open partial word from this row
                // would keep accumulating pixels from the next row instead of
                // being flushed — hardware sends one word per GO in host mode,
                // full or not. Checked here (not just via the hostcnt==0 check
                // below) because that check alone never fires for a word that
                // never reaches host_count pixels.
                if stop_on_word && ctx.hostcnt > 0 {
                    break;
                }

                first = true; // pixel in next row will be first
            } else if let Some(limit) = xstop {
                let limit_reached = if x_dec { ctx.xstart <= limit } else { ctx.xstart >= limit };
                if limit_reached {
                    break;
                }
            }

            // Host mode: stop after one word (after y-advance so row boundary is handled first).
            // Primitive continues on next GO — mid_primitive stays true.
            if stop_on_word && ctx.hostcnt == 0 {
                break;
            }

            // stoponx/stopony: next GO advances to next step — mid_primitive stays true.
            if !stoponx {
                break;
            }
        }


        self.diag.fetch_and(!Self::DIAG_LOOP_DRAW_BLOCK, Ordering::Relaxed);

        if opcode == DRAWMODE0_OPCODE_READ {
            self.flush_host_pixel(ctx);
        }
    }

    fn draw_span(&self, ctx: &mut Rex3Context) {
        if crate::rex3_simd::try_src_span_rgb(self, ctx) {
            return;
        }
        if  ctx.drawmode0.lronly() && (ctx.bresoctinc1.octant() & OCTANT_XDEC) != 0{
            return;
        }
        let length32 = ctx.drawmode0.length32();
        let ystride = ctx.drawmode0.ystride();
        let opcode = ctx.drawmode0.opcode();
        let colorhost = ctx.drawmode0.colorhost();
        let mut first = true;
        let skipfirst = ctx.drawmode0.skipfirst();
        let skiplast = ctx.drawmode0.skiplast();

        // In host mode (READ or DRAW+colorhost), each GO processes exactly one word's worth
        // of pixels (host_count pixels). stop_on_word causes the loop to exit after the
        // word boundary so the next GO picks up where we left off.
        let stop_on_word = opcode == DRAWMODE0_OPCODE_READ || colorhost;

        // stop_on_word takes priority over stoponx — host mode governs its own stop.
        let stoponx = ctx.drawmode0.stoponx() || stop_on_word;

        // Spans always advance left-to-right (+1 in 21.11 fixed-point).
        // length32 only clamps if span is >= 32 pixels wide.
        let span_len = (ctx.xend - ctx.xstart) >> 11;
        let xstop = if length32 && span_len >= 32 { Some(ctx.xstart + (32 << 11)) } else { None };

        let proc_fn    = unsafe { *self.px_proc.get() };
        let shade_fn   = unsafe { *self.px_shade.get() };
        let pattern_fn = unsafe { *self.px_pattern.get() };

        ctx.mid_primitive = true;
        self.diag.fetch_or(Self::DIAG_LOOP_DRAW_BLOCK, Ordering::Relaxed);
        let x_end_reached = loop {
            let x = ctx.xstart >> 11;
            let y = ctx.ystart >> 11;

            ctx.xstart += 1 << 11;

            let x_end_reached = ctx.xstart > ctx.xend;

            if !(first && skipfirst || x_end_reached && skiplast) {
                proc_fn(self, ctx, x, y);
            }

            shade_fn(ctx);
            pattern_fn(ctx);

            if x_end_reached {
                break true;
            } else if let Some(limit) = xstop {
                if ctx.xstart >= limit {
                    break false;
                }
            }

            // Host mode: stop after one word.
            // Primitive continues on next GO — mid_primitive stays true.
            if stop_on_word && ctx.hostcnt == 0 {
                break false;
            }

            // stoponx: next GO advances to next step — mid_primitive stays true.
            if !stoponx {
                break false;
            }

            first = false;
        };

        if x_end_reached {
            // Span fully consumed — advance to next row and reset state.
            //let y_inc: i32 = if ystride { 2 } else { 1 };
            //ctx.ystart += y_inc << 11;
            //ctx.xstart = ctx.xsave;
            ctx.pat_bit  = 31;
            ctx.zpat_bit = 31;
            ctx.mid_primitive = false;
        }


        self.diag.fetch_and(!Self::DIAG_LOOP_DRAW_BLOCK, Ordering::Relaxed);

        if opcode == DRAWMODE0_OPCODE_READ {
            self.flush_host_pixel(ctx);
        }
    }

    /// Fractional d correction for F_LINE/A_LINE from 21.11 endpoint sub-pixel position.
    /// Fractional nibble is bits [10:7] of the 21.11 coordinate (16.4(7) layout).
    fn fline_apply_fract(
        ctx: &Rex3Context,
        d: &mut i32,
        x: &mut i32,
        y: &mut i32,
        incrx2: i32,
        incry2: i32,
        y_major: bool,
    ) {
        let octant = ctx.bresoctinc1.octant() & 7;
        let x1p = ctx.xstart >> 11;
        let y1p = ctx.ystart >> 11;
        let x2p = ctx.xend >> 11;
        let y2p = ctx.yend >> 11;
        let mut dx = (x1p - x2p).abs();
        let mut dy = (y1p - y2p).abs();
        let mut xf = ((ctx.xstart >> 7) & 0xF) as i32;
        let mut yf = ((ctx.ystart >> 7) & 0xF) as i32;

        match octant {
            1 => {
                std::mem::swap(&mut xf, &mut yf);
                std::mem::swap(&mut dx, &mut dy);
            }
            3 => {
                xf = 0x10 - xf;
                std::mem::swap(&mut xf, &mut yf);
                std::mem::swap(&mut dx, &mut dy);
            }
            7 => { xf = 0x10 - xf; }
            6 => {
                xf = 0x10 - xf;
                yf = 0x10 - yf;
            }
            2 => {
                let t = 0x10 - xf;
                xf = 0x10 - yf;
                yf = t;
                std::mem::swap(&mut dx, &mut dy);
            }
            0 => {
                let t = 0x10 - yf;
                yf = xf;
                xf = t;
                std::mem::swap(&mut dx, &mut dy);
            }
            4 => { yf = 0x10 - yf; }
            _ => {}
        }

        // `*d` arrives holding the I_LINE decision variable (2*minor - major,
        // computed by setup() and shared by all line adrmodes). The REX3 spec
        // (rex3_pdf.md 3.6.1.2/3.6.2.1) defines F_LINE/A_LINE's base d as
        // 3*minor - 2*major instead — confirmed against MAME's do_fline,
        // which independently derives the same 3dy-2dx formula. The two
        // formulas differ by exactly (minor - major); apply that correction
        // before adding the fractional term below. Without it, the fractional
        // term is added to the wrong baseline and can flip d's sign on the
        // very first step for near-degenerate (small minor-axis) lines,
        // producing a spurious extra step in the minor-axis direction that
        // the line never recovers from (confirmed via a real R4400/REX3
        // fractional-line test that undershot its endpoint by one row).
        *d += dy - dx;
        *d += 2 * (((dx * yf) >> 4) - ((dy * xf) >> 4));
        let major_delta = if y_major { dy } else { dx };
        let e = *d - 2 * major_delta;
        if e > 0 {
            *d = e;
            let x_major = !y_major;
            if x_major {
                *y -= incry2;
            } else {
                *x += incrx2;
            }
        }
    }

    fn draw_iline(&self, ctx: &mut Rex3Context) {
        self.draw_line_bresenham(ctx, false, false, false);
    }

    fn draw_fline(&self, ctx: &mut Rex3Context) {
        self.draw_line_bresenham(ctx, true, false, false);
    }

    fn draw_aline(&self, ctx: &mut Rex3Context) {
        let mut extra_skip_first = false;
        let mut extra_skip_last = false;
        if ctx.drawmode0.endptfilter() {
            // Basic endpoint filter: consult AWEIGHT LUT for sub-pixel coverage.
            let xsf = (ctx.xstart >> 7) & 0xF;
            let ysf = (ctx.ystart >> 7) & 0xF;
            let xef = (ctx.xend >> 7) & 0xF;
            let yef = (ctx.yend >> 7) & 0xF;
            if xsf != 0 || ysf != 0 {
                let wi = ((xsf + ysf) as usize).min(15);
                let w = (ctx.aweight0 >> (wi * 4)) & 0xF;
                if w == 0 {
                    extra_skip_first = true;
                }
            }
            if xef != 0 || yef != 0 {
                let wi = ((xef + yef) as usize).min(15);
                let w = (ctx.aweight1 >> (wi * 4)) & 0xF;
                if w == 0 {
                    extra_skip_last = true;
                }
            }
        }
        self.draw_line_bresenham(ctx, true, extra_skip_first, extra_skip_last);
    }

    fn draw_line_bresenham(
        &self,
        ctx: &mut Rex3Context,
        fract: bool,
        extra_skip_first: bool,
        extra_skip_last: bool,
    ) {
        let octant = (ctx.bresoctinc1.octant() & 7) as usize;
        let (incrx1, incrx2, incry1, incry2, y_major) = REX3_BRES_OCTANTS[octant];

        let x2 = ctx.xend >> 11;
        let y2 = ctx.yend >> 11;
        let mut x = ctx.xstart >> 11;
        let mut y = ctx.ystart >> 11;

        // All Bresenham state comes from registers — set by setup() or restored across GOs.
        // incr1: 20-bit, always positive (no sign extension needed).
        let incr1 = ctx.bresoctinc1.incr1() as i32;
        // incr2: 21-bit signed — sign-extend from bit 20.
        let incr2 = {
            let raw = ctx.bresrndinc2.incr2();
            if raw & (1 << 20) != 0 { (raw | 0xFFE0_0000) as i32 } else { raw as i32 }
        };
        // d: 27-bit signed — sign-extend from bit 26.  Persisted across step-mode GOs.
        // For F_LINE/A_LINE (fract=true), the fractional-endpoint correction was
        // already applied in setup() (see setup()'s doc comment) — bresd/xstart/ystart
        // read here are already correct, no further adjustment needed.
        let mut d = {
            let raw = ctx.bresd & 0x7FF_FFFF;
            if raw & (1 << 26) != 0 { (raw | 0xF800_0000) as i32 } else { raw as i32 }
        };

        // pixel_count = major_axis_length + 1 (both endpoints inclusive).
        // max(|dx|,|dy|): continuation GOs (dosetup clear) must still walk the full
        // segment when persisted octant y_major disagrees with start→end (e.g. a
        // degenerate setup GO followed by a horizontal stipple continuation).
        let adx = (x2 - x).abs();
        let ady = (y2 - y).abs();
        let major = adx.max(ady);
        let mut pixel_count = major + 1;
        if ctx.drawmode0.length32() && pixel_count > 32 {
            pixel_count = 32;
        }

        let iterate_one = !ctx.drawmode0.stoponx() && !ctx.drawmode0.stopony();
        let mut skip_first = ctx.drawmode0.skipfirst() || extra_skip_first;
        let mut skip_last = ctx.drawmode0.skiplast() || extra_skip_last;
        if iterate_one {
            pixel_count = 1;
            skip_first = false;
            skip_last = false;
        }

        let proc_fn    = unsafe { *self.px_proc.get() };
        let shade_fn   = unsafe { *self.px_shade.get() };
        let pattern_fn = unsafe { *self.px_pattern.get() };
        let lsadvlast  = ctx.drawmode0.lsadvlast();

        macro_rules! bres_step {
            () => {
                if d < 0 {
                    x += incrx1; y -= incry1; d += incr1;
                } else {
                    x += incrx2; y -= incry2; d += incr2;
                }
            };
        }

        for i in 0..pixel_count {
            let is_first = i == 0;
            let is_last  = i == pixel_count - 1;

            // Write pixel unless suppressed by skip_first/skip_last.
            // iterate_one overrides skip_first so single-step mode always draws.
            let draw = (!is_first || !skip_first) && (!is_last || !skip_last);
            if draw {
                proc_fn(self, ctx, x, y);
            }

            shade_fn(ctx);
            if !is_last || lsadvlast {
                pattern_fn(ctx);
            }

            // On the last pixel of a full I_LINE draw, verify Bresenham landed on x2,y2 —
            // integer Bresenham is exact, so this is a real invariant for I_LINE.
            // F_LINE/A_LINE do NOT get this check: a fractional start biases the initial
            // error term but the loop still steps by whole pixels along the major axis,
            // so the minor-axis position at the final major-axis step is the closest
            // integer approximation to the true (fractional) line, not necessarily the
            // literal requested endpoint — this is expected behavior for fractional
            // Bresenham (confirmed independently: real hardware/software fractional-DDA
            // implementations only guarantee landing in the endpoint's pixel *column/row*,
            // not its exact minor-axis coordinate).
            debug_assert!(
                fract || iterate_one || !is_last || (x == x2 && y == y2),
                "I_LINE bres mismatch: pos ({},{}) != end ({},{})", x, y, x2, y2
            );

            // In full-line mode: do NOT step after the last pixel — that would leave
            // xstart/ystart one position beyond the endpoint, breaking the next XYENDI GO
            // (dosetup re-derives Bresenham from xstart).
            // In step mode: always step so the next single-step GO starts at the next position.
            if !is_last || iterate_one {
                bres_step!();
            }
        }

        ctx.xstart = x << 11;
        ctx.ystart = y << 11;
        // Persist d so the next GO (step mode) picks up where we left off.
        ctx.bresd = (d as u32) & 0x7FF_FFFF;
    }

    fn identity(val: u32) -> u32 { val }

    fn amplify_rgb_4(val: u32) -> u32 { val | (val << 4) }
    fn amplify_rgb_8(val: u32) -> u32 { val | (val << 8) }
    fn amplify_rgb_12(val: u32) -> u32 { val | (val << 12) }
    fn amplify_rgb_24(val: u32) -> u32 { val }
    fn amplify_olay(val: u32) -> u32 { (val << 8) | (val << 16) }
    fn amplify_cid(val: u32) -> u32 { val | (val << 4) }
    fn amplify_pup(val: u32) -> u32 { (val << 2) | (val << 6) }
    fn amplify_nop(_val: u32) -> u32 { 0 }

    // ── Shade iterate functions ────────────────────────────────────────────────
    // Called once per pixel after drawing.  Advance color DDAs and, when
    // CICLAMP is set, clamp the result to legal range.
    //
    // The spec (§3.8 / DRAWMODE0 bit CICLAMP) says:
    //   • RGB mode: each component is in o12.11 format (integer part bits[22:11]).
    //     Clamp: if negative (bit 31 set) or integer >= 0x180 → 0; if > 0xFF → 0x7FFFF.
    //   • CI mode: only colorred clamped; depth-specific overflow bit check.
    //     8bpp: clamp if bit 19 set.  12bpp: clamp if bit 21 set.
    //     (4bpp: bit 15; 24bpp: no clamp per spec.)

    #[inline(always)]
    fn shade_add(ctx: &mut Rex3Context) {
        ctx.colorred   = ctx.colorred.wrapping_add(ctx.slopered   as u32);
        ctx.colorgrn   = ctx.colorgrn.wrapping_add(ctx.slopegrn   as u32);
        ctx.colorblue  = ctx.colorblue.wrapping_add(ctx.slopeblue  as u32);
        ctx.coloralpha = ctx.coloralpha.wrapping_add(ctx.slopealpha as u32);
    }

    fn iterate_shade_noop(_ctx: &mut Rex3Context) {}

    /// SHADE only, no clamping (CICLAMP=0).
    fn iterate_shade_unclamped(ctx: &mut Rex3Context) {
        Self::shade_add(ctx);
    }

    /// SHADE + CICLAMP, CI 8bpp: clamp colorred if bit 19 set.
    fn iterate_shade_ci8_clamp(ctx: &mut Rex3Context) {
        Self::shade_add(ctx);
        if ctx.colorred & (1 << 19) != 0 {
            ctx.colorred = 0x0007_FFFF;
        }
    }

    /// SHADE + CICLAMP, CI 12bpp: clamp colorred if bit 21 set.
    fn iterate_shade_ci12_clamp(ctx: &mut Rex3Context) {
        Self::shade_add(ctx);
        if ctx.colorred & (1 << 21) != 0 {
            ctx.colorred = 0x001F_FFFF;
        }
    }

    /// SHADE + RGB mode: clamp R,G,B,A each iteration, this happens even if we dont draw pixel
    /// integer = bits[22:11] & 0x1FF; negative (bit31) or int >= 0x180 → 0; int > 0xFF → 0x7FFFF.
    fn iterate_shade_rgb_clamp(ctx: &mut Rex3Context) {
        Self::shade_add(ctx);
        #[inline(always)]
        fn clamp(c: u32) -> u32 {
            let val = (c >> 11) & 0x1FF;
            if c & (1 << 31) != 0 || val >= 0x180 { 0 }
            else if val > 0xFF { 0x0007_FFFF }
            else { c }
        }
        ctx.colorred   = clamp(ctx.colorred);
        ctx.colorgrn   = clamp(ctx.colorgrn);
        ctx.colorblue  = clamp(ctx.colorblue);
        ctx.coloralpha = clamp(ctx.coloralpha);
    }

    // ── Pattern iterate functions ──────────────────────────────────────────────
    // Called once per pixel after drawing.  Advance lspattern and/or zpattern
    // bit counters.
    //
    // ZPATTERN: always 32-bit, simple rotate — zpat_bit = (zpat_bit - 1) & 31.
    //
    // LSPATTERN (via lsmode):
    //   LSRCOUNT  (down counter, 0..LSREPEAT-1): decremented each pixel.
    //   When LSRCOUNT == 0 after decrement → advance pat_bit, reload LSRCOUNT = LSREPEAT-1.
    //   LSLENGTH  (4 bits): pattern length = lslength + 17 (range 17..32).
    //   pat_bit wraps: when it would go below (32 - length), reset to 31.
    //   LSREPEAT==0 is treated as 1 (no-repeat is the degenerate case).
    //
    // Cursors start at 31 (MSB) on DOSETUP/row start (see execute_go) and walk
    // DOWN toward the wrap point — a contiguous MSB-to-LSB sweep. f2d0bff
    // briefly flipped this to increment-from-31 (+ an lspattern rotate-on-wrap
    // in place of the reset), but incrementing from 31 immediately wraps to 0
    // after a single step (`(31+1)&31 == 0`), producing a discontinuous,
    // backwards bit walk — confirmed as the cause of garbled/mirrored PROM
    // text. Reverted back to decrement/reset-to-31.

    fn iterate_pattern_noop(_ctx: &mut Rex3Context) {}

    #[inline(always)]
    fn advance_zpat(ctx: &mut Rex3Context) {
        ctx.zpat_bit = ctx.zpat_bit.wrapping_sub(1) & 31;
    }

    #[inline(always)]
    fn advance_lspat(ctx: &mut Rex3Context) {
        let lsrepeat = ctx.lsmode.lsrepeat() as u8;
        let repeat = if lsrepeat == 0 { 1 } else { lsrepeat };
        if ctx.lsmode.lsrcount() == 0 {
            // Reload counter, advance bit
            ctx.lsmode.set_lsrcount((repeat - 1) as u32);
            let length = ctx.lsmode.lslength() as u8 + 17; // 17..=32
            let wrap_point = 32u8.saturating_sub(length);  // bit index of pattern end
            if ctx.pat_bit == wrap_point {
                ctx.pat_bit = 31; // recirculate
            } else {
                ctx.pat_bit = ctx.pat_bit.wrapping_sub(1) & 31;
            }
        } else {
            ctx.lsmode.set_lsrcount(ctx.lsmode.lsrcount() - 1);
        }
    }

    fn iterate_pattern_z(ctx: &mut Rex3Context) {
        Self::advance_zpat(ctx);
    }

    fn iterate_pattern_ls(ctx: &mut Rex3Context) {
        Self::advance_lspat(ctx);
    }

    fn iterate_pattern_both(ctx: &mut Rex3Context) {
        Self::advance_zpat(ctx);
        Self::advance_lspat(ctx);
    }


    fn rgb4_to_rgb24(val: u32) -> u32 {
        let r = if (val & 1) != 0 { 0xFF } else { 0 };
        let g_raw = (val >> 1) & 3;
        let g = (g_raw << 6) | (g_raw << 4) | (g_raw << 2) | g_raw;
        let b = if (val & 8) != 0 { 0xFF } else { 0 };
        (b << 16) | (g << 8) | r
    }

    fn rgb24_to_rgb4(val: u32) -> u32 {
        let r = (val >> 7) & 1;
        let g = (val >> 14) & 3;
        let b = (val >> 23) & 1;
        (b << 3) | (g << 1) | r
    }

    fn rgb8_to_rgb24(val: u32) -> u32 {
        let r_raw = val & 7;
        let r = (r_raw << 5) | (r_raw << 2) | (r_raw >> 1);
        let g_raw = (val >> 3) & 7;
        let g = (g_raw << 5) | (g_raw << 2) | (g_raw >> 1);
        let b_raw = (val >> 6) & 3;
        let b = (b_raw << 6) | (b_raw << 4) | (b_raw << 2) | b_raw;
        (b << 16) | (g << 8) | r
    }

    fn rgb24_to_rgb8(val: u32) -> u32 {
        let r = (val >> 5) & 7;
        let g = (val >> 13) & 7;
        let b = (val >> 22) & 3;
        (b << 6) | (g << 3) | r
    }

    fn rgb12_to_rgb24(val: u32) -> u32 {
        let r_raw = val & 0xF;
        let r = (r_raw << 4) | r_raw;
        let g_raw = (val >> 4) & 0xF;
        let g = (g_raw << 4) | g_raw;
        let b_raw = (val >> 8) & 0xF;
        let b = (b_raw << 4) | b_raw;
        (b << 16) | (g << 8) | r
    }

    fn rgb24_to_rgb12(val: u32) -> u32 {
        let r = (val >> 4) & 0xF;
        let g = (val >> 12) & 0xF;
        let b = (val >> 20) & 0xF;
        (b << 8) | (g << 4) | r
    }

    // Bayer 4x4 dither matrix packed as 16 nibbles in a u64.
    // Indexed by (y&3)<<2|(x&3): threshold = (BAYER_PACKED >> (idx*4)) & 0xF.
    // Table: [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
    const BAYER_PACKED: u64 = 0x5D7F91B36E4CA280;

    /// Pack bayer index into bits 27:24 of color value (top byte unused by 24-bit BGR).
    /// Encoding: bits[3:2] = y&3, bits[1:0] = x&3 → index = (y&3)<<2|(x&3).
    /// Non-dither compress variants ignore these bits.
    #[inline(always)]
    fn bayer_pack(color: u32, x: i32, y: i32) -> u32 {
        (color & 0x00FFFFFF) | (((y as u32 & 3) << 2 | (x as u32 & 3)) << 24)
    }
    fn bayer_nop(color: u32, _x: i32, _y: i32) -> u32 { color }

    #[inline(always)]
    fn bayer_threshold(idx: u32) -> u32 {
        ((Self::BAYER_PACKED >> (idx * 4)) & 0xF) as u32
    }

    fn rgb24_to_rgb4_dither(val: u32) -> u32 {
        let bayer = Self::bayer_threshold(val >> 24);
        let r = (val & 0xFF) as u8;
        let g = ((val >> 8) & 0xFF) as u8;
        let b = ((val >> 16) & 0xFF) as u8;
        // 4bpp: 1-2-1 BGR. Each channel dithered down from 8-bit.
        let sr = (r >> 3).wrapping_sub(r >> 4);
        let sg = (g >> 2).wrapping_sub(g >> 4);
        let sb = (b >> 3).wrapping_sub(b >> 4);
        let mut dr = (sr >> 4) & 1;
        let mut dg = (sg >> 4) & 3;
        let mut db = (sb >> 4) & 1;
        if (sr & 0xf) as u32 > bayer { dr = (dr + 1).min(1); }
        if (sg & 0xf) as u32 > bayer { dg = (dg + 1).min(3); }
        if (sb & 0xf) as u32 > bayer { db = (db + 1).min(1); }
        ((db << 3) | (dg << 1) | dr) as u32
    }

    fn rgb24_to_rgb8_dither(val: u32) -> u32 {
        let bayer = Self::bayer_threshold(val >> 24);
        let r = (val & 0xFF) as u8;
        let g = ((val >> 8) & 0xFF) as u8;
        let b = ((val >> 16) & 0xFF) as u8;
        // 8bpp: 3-3-2 BGR.
        let sr = (r >> 1).wrapping_sub(r >> 4);
        let sg = (g >> 1).wrapping_sub(g >> 4);
        let sb = (b >> 2).wrapping_sub(b >> 4);
        let mut dr = (sr >> 4) & 7;
        let mut dg = (sg >> 4) & 7;
        let mut db = (sb >> 4) & 3;
        if (sr & 0xf) as u32 > bayer { dr = (dr + 1).min(7); }
        if (sg & 0xf) as u32 > bayer { dg = (dg + 1).min(7); }
        if (sb & 0xf) as u32 > bayer { db = (db + 1).min(3); }
        ((db << 6) | (dg << 3) | dr) as u32
    }

    fn rgb24_to_rgb12_dither(val: u32) -> u32 {
        let bayer = Self::bayer_threshold(val >> 24);
        let r = (val & 0xFF) as u32;
        let g = ((val >> 8) & 0xFF) as u32;
        let b = ((val >> 16) & 0xFF) as u32;
        // 12bpp: 4-4-4 BGR.
        let sr = r - (r >> 4);
        let sg = g - (g >> 4);
        let sb = b - (b >> 4);
        let mut dr = (sr >> 4) & 15;
        let mut dg = (sg >> 4) & 15;
        let mut db = (sb >> 4) & 15;
        if (sr & 0xf) > bayer { dr = (dr + 1).min(15); }
        if (sg & 0xf) > bayer { dg = (dg + 1).min(15); }
        if (sb & 0xf) > bayer { db = (db + 1).min(15); }
        (db << 8) | (dg << 4) | dr
    }

    fn host_unpack_nop(_val: u64) -> u32 { 0 }
    fn host_pack_nop(acc: u64, _val: u32) -> u64 { acc }

    // 4-bit (1-2-1 BGR) expansion
    pub fn expand_4_rgb(val: u32) -> u32 {
        let b = (val >> 3) & 1;
        let g1 = (val >> 2) & 1;
        let g0 = (val >> 1) & 1;
        let r = val & 1;
        // this is clever hack copied from MAME maps 0,1,2,3 to 0x00, 0x55, 0xAA, 0xFF
        let g = (0xAA * g1) | (0x55 * g0);
        (b * 255) << 16 | (g << 8) | (r * 255)
    }

    // 8-bit (3-3-2 BGR) expansion
    pub fn expand_8_rgb(val: u32) -> u32 {
        // Format: BBGGGRRR
        // Blue: bits 7,6 (2 bits) -> 0xAA, 0x55
        let b1 = (val >> 7) & 1;
        let b0 = (val >> 6) & 1;
        let b = (0xAA * b1) | (0x55 * b0);

        // Green: bits 5,4,3 (3 bits) -> 0x92, 0x49, 0x24
        let g2 = (val >> 5) & 1;
        let g1 = (val >> 4) & 1;
        let g0 = (val >> 3) & 1;
        let g = (0x92 * g2) | (0x49 * g1) | (0x24 * g0);

        // Red: bits 2,1,0 (3 bits) -> 0x92, 0x49, 0x24
        let r2 = (val >> 2) & 1;
        let r1 = (val >> 1) & 1;
        let r0 = val & 1;
        let r = (0x92 * r2) | (0x49 * r1) | (0x24 * r0);

        (b << 16) | (g << 8) | r
    }

    // 12-bit (4-4-4 BGR) expansion
    pub fn expand_12_rgb(val: u32) -> u32 {
        let b = (val >> 8) & 0xF;
        let g = (val >> 4) & 0xF;
        let r = val & 0xF;
        // 4 bits: 0..15 -> 0..255 (x17 or 0x11)
        (b * 0x11) << 16 | (g * 0x11) << 8 | (r * 0x11)
    }

    // 32-bit (ABGR) expansion
    fn expand_32_rgb(val: u32) -> u32 {
        // ARGB -> ARGB (internal format AABBGGRR)
        val
    }

    // Compression functions (Internal -> Host)
    fn compress_4_rgb(val: u32) -> u32 {
        let b = (val >> 16) & 0xFF;
        let g = (val >> 8) & 0xFF;
        let r = val & 0xFF;
        ((b >> 7) << 3) | ((g >> 6) << 1) | (r >> 7)
    }

    fn compress_8_rgb(val: u32) -> u32 {
        let b = (val >> 16) & 0xFF;
        let g = (val >> 8) & 0xFF;
        let r = val & 0xFF;
        // BBGGGRRR
        ((b >> 6) << 6) | ((g >> 5) << 3) | (r >> 5)
    }

    fn compress_12_rgb(val: u32) -> u32 {
        let b = (val >> 16) & 0xFF;
        let g = (val >> 8) & 0xFF;
        let r = val & 0xFF;
        ((b >> 4) << 8) | ((g >> 4) << 4) | (r >> 4)
    }

    fn compress_32_rgb(val: u32) -> u32 {
        val | 0xFF000000 // Set Alpha to 0xFF
    }

    // Host Unpack Functions
    // All extract from the top of the 64-bit shifter
    // We left-shift the shifter after each pixel so current pixel is always at the MSB.
    // 4bpp: 8-bit slot, pixel in low nibble → mask 0xf from bit 56
    // 8bpp: 8-bit slot → mask 0xff from bit 56
    // 12bpp: 16-bit slot → mask 0xfff from bit 48
    // 32bpp: 32-bit slot → bits [63:32]
    fn host_unpack_4_64(val: u64) -> u32 {
        Self::expand_4_rgb(((val >> 56) & 0xF) as u32)
    }
    fn host_unpack_8_64(val: u64) -> u32 {
        Self::expand_8_rgb(((val >> 56) & 0xFF) as u32)
    }
    fn host_unpack_12_64(val: u64) -> u32 {
        Self::expand_12_rgb(((val >> 48) & 0xFFF) as u32)
    }
    fn host_unpack_32_64(val: u64) -> u32 {
        Self::expand_32_rgb((val >> 32) as u32)
    }

    fn host_unpack_4_64_ci(val: u64) -> u32 {
        ((val >> 56) & 0xF) as u32
    }
    fn host_unpack_8_64_ci(val: u64) -> u32 {
        ((val >> 56) & 0xFF) as u32
    }
    fn host_unpack_12_64_ci(val: u64) -> u32 {
        ((val >> 48) & 0xFFF) as u32
    }
    fn host_unpack_32_64_ci(val: u64) -> u32 {
        (val >> 32) as u32
    }

    // Host Pack Functions — inverse of unpack.
    // Each pixel occupies the same slot size as unpack reads: shift acc left by slot, insert pixel in low bits.
    // 4bpp: 8-bit slot (shift=8), pixel in low nibble.
    // 12bpp: 16-bit slot (shift=16), pixel in low 12 bits.
    fn host_pack_4_rgb(acc: u64, pixel: u32) -> u64 {
        (acc << 8) | (Self::compress_4_rgb(pixel) as u64)
    }
    fn host_pack_8_rgb(acc: u64, pixel: u32) -> u64 {
        (acc << 8) | (Self::compress_8_rgb(pixel) as u64)
    }
    fn host_pack_12_rgb(acc: u64, pixel: u32) -> u64 {
        (acc << 16) | (Self::compress_12_rgb(pixel) as u64)
    }
    fn host_pack_32_rgb(acc: u64, pixel: u32) -> u64 {
        (acc << 32) | (Self::compress_32_rgb(pixel) as u64)
    }

    // CI Mode (Direct)
    fn host_pack_4_ci(acc: u64, pixel: u32) -> u64 {
        (acc << 8) | ((pixel & 0xF) as u64)
    }
    fn host_pack_8_ci(acc: u64, pixel: u32) -> u64 {
        (acc << 8) | ((pixel & 0xFF) as u64)
    }
    fn host_pack_12_ci(acc: u64, pixel: u32) -> u64 {
        (acc << 16) | ((pixel & 0xFFF) as u64)
    }
    fn host_pack_32_ci(acc: u64, pixel: u32) -> u64 {
        (acc << 32) | (pixel as u64)
    }

    fn host_setup(&self, drawmode1: DrawMode1) {
        let depth = drawmode1.hostdepth();
        let double = drawmode1.rwdouble();
        let packed = drawmode1.rwpacked();
        let rgb = drawmode1.rgbmode();

        // 4bpp pixels occupy an 8-bit slot (low nibble); shift is 8 not 4.
        // Non-packed: shift is 0 (count=1, shift never used).
        let shift = if packed {
            match depth { 0 | 1 => 8, 2 => 16, 3 => 32, _ => 0 }
        } else {
            0
        };

        // packed+double: 64/shift_bits pixels; packed+!double: 32/shift_bits pixels; non-packed: 1.
        // depth=0 (4bpp) uses 8-bit slots (shift=8), same counts as depth=1 (8bpp).
        let count = if packed {
            if double {
                match depth { 0 | 1 => 8, 2 => 4, 3 => 2, _ => 1 }
            } else {
                match depth { 0 | 1 => 4, 2 => 2, 3 => 1, _ => 1 }
            }
        } else {
            1
        };

        let unpack: fn(u64) -> u32 = if rgb {
            match depth {
                0 => Self::host_unpack_4_64,
                1 => Self::host_unpack_8_64,
                2 => Self::host_unpack_12_64,
                3 => Self::host_unpack_32_64,
                _ => Self::host_unpack_nop,
            }
        } else {
            match depth {
                0 => Self::host_unpack_4_64_ci,
                1 => Self::host_unpack_8_64_ci,
                2 => Self::host_unpack_12_64_ci,
                3 => Self::host_unpack_32_64_ci,
                _ => Self::host_unpack_nop,
            }
        };

        let pack: fn(u64, u32) -> u64 = if rgb {
            match depth {
                0 => Self::host_pack_4_rgb,
                1 => Self::host_pack_8_rgb,
                2 => Self::host_pack_12_rgb,
                3 => Self::host_pack_32_rgb,
                _ => Self::host_pack_nop,
            }
        } else {
            match depth {
                0 => Self::host_pack_4_ci,
                1 => Self::host_pack_8_ci,
                2 => Self::host_pack_12_ci,
                3 => Self::host_pack_32_ci,
                _ => Self::host_pack_nop,
            }
        };

        unsafe {
            *self.host_unpack.get() = unpack;
            *self.host_pack.get() = pack;
            *self.host_shift.get() = shift;
            *self.host_count.get() = count;
        }
    }

    fn fetch_host_pixel(&self, ctx: &mut Rex3Context) -> u32 {
        let rwdouble = ctx.drawmode1.rwdouble();
        let swapendian = ctx.drawmode1.swapendian();

        if ctx.hostcnt == 0 {
            // 32-bit writes land in the high half [63:32] via HOSTRW0, so data is already at MSB.
            ctx.host_shifter = ctx.hostrw;
            if swapendian {
                ctx.host_shifter = if rwdouble {
                    ctx.host_shifter.swap_bytes()
                } else {
                    // swap only the high 32-bit word
                    let hi = (ctx.host_shifter >> 32) as u32;
                    ((hi.swap_bytes() as u64) << 32) | (ctx.host_shifter & 0xFFFF_FFFF)
                };
            }
            ctx.hostcnt = unsafe { *self.host_count.get() };
        }

        let unpack = unsafe { *self.host_unpack.get() };
        let pixel = unpack(ctx.host_shifter);

        // Shift for next pixel
        let shift = unsafe { *self.host_shift.get() };
        
        ctx.host_shifter <<= shift;
        ctx.hostcnt -= 1;

        pixel
    }

    fn send_host_word(&self, ctx: &mut Rex3Context) {
        let rwdouble = ctx.drawmode1.rwdouble();
        let swapendian = ctx.drawmode1.swapendian();
        let mut val = ctx.host_shifter;

        if swapendian {
            val = val.swap_bytes();
        } else if !rwdouble {
            val <<= 32;
        }

        ctx.hostrw = val;
    }

    fn store_host_pixel(&self, ctx: &mut Rex3Context, pixel: u32) {
        if ctx.hostcnt == 0 {
            ctx.hostcnt = unsafe { *self.host_count.get() };
            ctx.host_shifter = 0;
        }

        let pack = unsafe { *self.host_pack.get() };
        ctx.host_shifter = pack(ctx.host_shifter, pixel);
        ctx.hostcnt -= 1;

        if ctx.hostcnt == 0 {
            self.send_host_word(ctx);
        }
    }

    fn flush_host_pixel(&self, ctx: &mut Rex3Context) {
        if ctx.hostcnt > 0 {
            // Shift remaining bits to align to MSB (since we pack LSB to MSB)
            let shift = unsafe { *self.host_shift.get() };
            ctx.host_shifter <<= ctx.hostcnt * shift;
            
            self.send_host_word(ctx);
            ctx.hostcnt = 0;
        }
    }

    fn combine_host_dda(ctx: &Rex3Context, host_pixel: u32) -> u32 {
        let color = if ctx.drawmode0.colorhost() { host_pixel } else { ctx.get_colori() };
        // Afunction's source alpha comes from "either DDA or host" (selected by
        // ALPHAHOST) independent of RGB/CI plane format (rex3 spec §3.8.1) — CI-mode
        // GL apps can stream a color index alongside a host alpha byte for ALPHAREF
        // comparison. Always overlay bits 31:24 with the resolved alpha so afunction
        // sees the right value in both modes; compress_fn/write paths for CI already
        // mask down to the plane-depth index and never look at these high bits.
        let a = if ctx.drawmode0.alphahost() {
            (host_pixel >> 24) & 0xFF
        } else {
            Rex3Context::clamp_color_component(ctx.coloralpha)
        };
        (color & 0x00FFFFFF) | (a << 24)
    }

    /// Replicate colorvram to fill plane-depth slots, matching MAME get_default_color() with fastclear=1.
    pub(crate) fn fastclear_color(ctx: &Rex3Context) -> u32 {
        let v = ctx.colorvram;
        match ctx.drawmode1.drawdepth() {
            0 => { let c = v & 0xf; c | (c << 4) | (c << 8) | (c << 16) }
            1 => { let c = v & 0xff; c | (c << 8) | (c << 16) }
            2 => {
                // 12bpp: in RGB mode use nibbles from colorvram, else lower 12 bits
                let c = if ctx.drawmode1.rgbmode() {
                    ((v & 0xf00000) >> 12) | ((v & 0xf000) >> 8) | ((v & 0xf0) >> 4)
                } else {
                    v & 0x000fff
                };
                c | (c << 12)
            }
            _ => v & 0xffffff, // 24bpp
        }
    }

    fn blend(&self, ctx: &Rex3Context, src: u32, dst: u32) -> u32 {
        let s_factor_sel = ctx.drawmode1.sfactor();
        let d_factor_sel = ctx.drawmode1.dfactor();

        let sa = (src >> 24) & 0xFF;
        
        let get_factor = |sel: u32, c: u32, a: u32| -> u32 {
            match sel {
                0 => 0,             // ZERO
                1 => 255,           // ONE
                2 => c,             // DC/SC
                3 => 255 - c,       // MDC/MSC
                4 => a,             // SA
                5 => 255 - a,       // MSA
                _ => 0,
            }
        };

        let mut res = 0;
        for i in 0..4 {
            let shift = i * 8;
            let s_c = (src >> shift) & 0xFF;
            let d_c = (dst >> shift) & 0xFF;
            
            let sf = get_factor(s_factor_sel, d_c, sa);
            let df = get_factor(d_factor_sel, s_c, sa);
            
            let val = (s_c * sf + d_c * df) / 255;
            let val_clamped = if val > 255 { 255 } else { val };
            
            res |= val_clamped << shift;
        }
        res
    }

    fn calculate_src_address(&self, x: i32, y: i32, ctx: &Rex3Context) -> Option<u32> {
        // xymove affects the destination, not the source.
        // Source is just (x, y) + xywin.
        let x_win = ((ctx.xywin >> 16) & 0xFFFF) as i16 as i32;
        let y_win = (ctx.xywin & 0xFFFF) as i16 as i32;

        let x_abs = x + x_win;
        let y_abs = y + y_win;

        let x_phys = x_abs - REX3_COORD_BIAS;
        let y_phys = if ctx.drawmode1.yflip() { 0x23FF - y_abs } else { y_abs - REX3_COORD_BIAS };

        if x_phys < 0 || x_phys >= REX3_SCREEN_WIDTH || y_phys < 0 || y_phys >= REX3_SCREEN_HEIGHT {
            return None;
        }

        Some((y_phys as u32) * 2048 + (x_phys as u32))
    }

    fn process_pixel_noop(&self, _ctx: &mut Rex3Context, _x: i32, _y: i32) {}

    fn process_pixel_fastclear(&self, ctx: &mut Rex3Context, x: i32, y: i32) {
        if let Some(addr) = self.calculate_fb_address(x, y, ctx, true) {
            let wr_fn = unsafe { *self.px_wr.get() };
            wr_fn(self, addr, Self::fastclear_color(ctx));
        }
    }

    fn process_pixel_read(&self, ctx: &mut Rex3Context, x: i32, y: i32) {
        let rd_fn = unsafe { *self.px_rd.get() };
        let expand_fn = unsafe { *self.px_expand.get() };
        let pixel = if let Some(addr) = self.calculate_fb_address(x, y, ctx, false) {
            // Expand plane-depth pixel to 24-bit BGR before packing into host buffer.
            // host_pack_N_rgb expects 24-bit BGR; expand is identity in CI mode.
            expand_fn(rd_fn(self, addr))
        } else {
            0
        };
        self.store_host_pixel(ctx, pixel);
    }

    fn process_pixel_draw(&self, ctx: &mut Rex3Context, x: i32, y: i32) {
        let colorhost = ctx.drawmode0.colorhost();
        let alphahost = ctx.drawmode0.alphahost();

        let mut use_bg = false;
        let mut check_ls = true;

        // Pattern checks first — host pixel only consumed when the pixel is actually drawn.
        // MAME: get_host_color() called only inside if(BIT(pattern, bit)), never on skip/opaque paths.
        if ctx.drawmode0.enzpattern() {
            let bit = (ctx.zpattern >> ctx.zpat_bit) & 1 != 0;
            if !bit {
                if ctx.drawmode0.zpopaque() {
                    use_bg = true;
                    check_ls = false;
                } else {
                    return;
                }
            }
        }

        if ctx.drawmode0.enlspattern() {
            let bit = (ctx.lspattern >> ctx.pat_bit) & 1 != 0;
            if check_ls && !bit {
                if ctx.drawmode0.lsopaque() {
                    use_bg = true;
                } else {
                    return;
                }
            }
        }

        // Fetch host pixel only when the pattern passes and we're drawing from host (not colorback).
        let host_pixel = if !use_bg && (alphahost || colorhost) {
            self.fetch_host_pixel(ctx)
        } else {
            0
        };

        if let Some(addr) = self.calculate_fb_address(x, y, ctx, true) {
            // CID Masking
            let cidmatch = (ctx.clipmode >> CLIPMODE_CIDMATCH_SHIFT) & 0xF;
            if cidmatch != 0xF {
                let aux_val = unsafe { (*self.fb_aux.get())[addr as usize] };
                // Compare against lower 4 bits of AUX (CID+PUP usually)
                if (aux_val & 0xF) != cidmatch {
                    return;
                }
            }

            // src color: 24-bit BGR in rgbmode, plane-depth index in CI mode
            let raw_src = if use_bg {
                ctx.colorback
            } else {
                Self::combine_host_dda(ctx, host_pixel)
            };

            // Afunction: inhibit the write unless source alpha (raw_src bits 31:24)
            // passes the DRAWMODE1 COMPARE relation against ALPHAREF. px_afunc is
            // selected once per mode change in planes_setup (see afunc_* below);
            // it's a no-op always-pass function outside rgbmode or when COMPARE==0x7.
            let afunc_fn = unsafe { *self.px_afunc.get() };
            if !afunc_fn(raw_src >> 24 & 0xFF, ctx.alpharef & 0xFF) {
                return;
            }

            let rd_fn = unsafe { *self.px_rd.get() };
            let wr_fn = unsafe { *self.px_wr.get() };
            let compress_fn = unsafe { *self.px_compress.get() };

            let res = if ctx.drawmode1.blend() {
                // Blend path: work in 24-bit BGR space throughout.
                // src is already 24-bit BGR (rgbmode) or CI index (expand=identity).
                // dst is read as plane-depth then expanded to 24-bit BGR.
                let expand_fn = unsafe { *self.px_expand.get() };
                let dst_raw = if ctx.drawmode1.backblend() {
                    ctx.colorback
                } else {
                    expand_fn(rd_fn(self, addr))
                };
                let blended = self.blend(ctx, raw_src, dst_raw);
                // Compress blended 24-bit result back to plane-depth before write.
                let bayer_fn = unsafe { *self.px_bayer.get() };
                compress_fn(bayer_fn(blended, x, y))
            } else {
                // Logic op path: compress src to plane-depth, amplify for dblsrc, then logic op.
                // dst also needs amplify: rd_fn shifts the value down (e.g. bits 15:8 → 7:0 for
                // dblsrc slot 1), so we must shift it back up to match the write mask position.
                let bayer_fn = unsafe { *self.px_bayer.get() };
                let amp_fn = unsafe { *self.px_amp.get() };
                let logic_fn = unsafe { *self.px_logic.get() };
                let src = amp_fn(compress_fn(bayer_fn(raw_src, x, y)));
                let dst = amp_fn(rd_fn(self, addr));
                logic_fn(src, dst)
            };
            wr_fn(self, addr, res);
        }
    }

    /// Specialization for character/glyph drawing: zpattern test only, SRC logicop,
    /// no blend, no colorhost, no CID check. Skips pixel on zpattern bit=0.
    /// Selected when: DRAW + enzpattern + !enlspattern + !zpopaque + !blend + !colorhost + !alphahost
    ///                + CID==0xF + logicop==SRC.
    fn process_pixel_zpattern(&self, ctx: &mut Rex3Context, x: i32, y: i32) {
        if (ctx.zpattern >> ctx.zpat_bit) & 1 == 0 {
            return;
        }
        if let Some(addr) = self.calculate_fb_address(x, y, ctx, true) {
            let wr_fn       = unsafe { *self.px_wr.get() };
            let amp_fn      = unsafe { *self.px_amp.get() };
            let bayer_fn    = unsafe { *self.px_bayer.get() };
            let compress_fn = unsafe { *self.px_compress.get() };
            // SRC logicop: result = src, no dst read needed.
            wr_fn(self, addr, amp_fn(compress_fn(bayer_fn(ctx.get_colori(), x, y))));
        }
    }

    /// Specialization for common solid draws: no blend, no colorhost, no alphahost, no CID check.
    /// Handles patterns (use_bg) and any logicop, but skips the blend and host-fetch paths.
    /// Selected when: DRAW + !blend + !colorhost + !alphahost + CID==0xF.
    fn process_pixel_noblend(&self, ctx: &mut Rex3Context, x: i32, y: i32) {
        let mut use_bg = false;

        if ctx.drawmode0.enzpattern() {
            let bit = (ctx.zpattern >> ctx.zpat_bit) & 1 != 0;
            if !bit {
                if ctx.drawmode0.zpopaque() { use_bg = true; } else { return; }
            }
        }

        if ctx.drawmode0.enlspattern() {
            let bit = (ctx.lspattern >> ctx.pat_bit) & 1 != 0;
            if !bit {
                if ctx.drawmode0.lsopaque() { use_bg = true; } else { return; }
            }
        }

        if let Some(addr) = self.calculate_fb_address(x, y, ctx, true) {
            let raw_src = if use_bg { ctx.colorback } else { ctx.get_colori() };
            let rd_fn       = unsafe { *self.px_rd.get() };
            let wr_fn       = unsafe { *self.px_wr.get() };
            let amp_fn      = unsafe { *self.px_amp.get() };
            let bayer_fn    = unsafe { *self.px_bayer.get() };
            let compress_fn = unsafe { *self.px_compress.get() };
            let logic_fn    = unsafe { *self.px_logic.get() };
            let src = amp_fn(compress_fn(bayer_fn(raw_src, x, y)));
            let dst = amp_fn(rd_fn(self, addr));
            wr_fn(self, addr, logic_fn(src, dst));
        }
    }

    fn process_pixel_scr2scr(&self, ctx: &mut Rex3Context, x: i32, y: i32) {
        let rd_fn = unsafe { *self.px_rd.get() };
        let amp_fn = unsafe { *self.px_amp.get() };
        let expand_fn = unsafe { *self.px_expand.get() };
        let compress_fn = unsafe { *self.px_compress.get() };

        // src: read plane-depth pixel, expand to 24-bit (rgbmode) or leave as-is (CI), amplify.
        let raw_src = if let Some(src_addr) = self.calculate_src_address(x, y, ctx) {
            expand_fn(rd_fn(self, src_addr))
        } else {
            0
        };

        if let Some(dst_addr) = self.calculate_fb_address(x, y, ctx, true) {
            let wr_fn = unsafe { *self.px_wr.get() };

            let res = if ctx.drawmode1.blend() {
                let dst_raw = if ctx.drawmode1.backblend() {
                    ctx.colorback
                } else {
                    expand_fn(rd_fn(self, dst_addr))
                };
                compress_fn(self.blend(ctx, raw_src, dst_raw))
            } else {
                let logic_fn = unsafe { *self.px_logic.get() };
                let src = amp_fn(compress_fn(raw_src));
                let dst = amp_fn(rd_fn(self, dst_addr));
                logic_fn(src, dst)
            };
            wr_fn(self, dst_addr, res);
        }
    }

    fn dcb_write(&self, mut val: u32) {
        self.diag.fetch_or(Self::DIAG_LOCK_DCB, Ordering::Relaxed);
        let mut dcb = self.dcb.lock();
        let addr = ((dcb.dcbmode & DCBMODE_DCBADDR_MASK) >> DCBMODE_DCBADDR_SHIFT) as u8;
        let data_width = dcb.dcbmode & DCBMODE_DATAWIDTH_MASK;

        dlog_dev!(LogModule::Dcb, "DCB Write: Val {:08x} Mode {:08x} (Addr {} CRS {} DW {})", val, dcb.dcbmode, addr, dcb.crs(), data_width);

        // DCBMODE bit 28 "Swap Byte Ordering": swap within the data width.
        if (dcb.dcbmode & DCBMODE_SWAPENDIAN) != 0 {
            val = match data_width {
                DCBMODE_DATAWIDTH_2 => ((val & 0x00FF00FF) << 8) | ((val >> 8) & 0x00FF00FF),
                DCBMODE_DATAWIDTH_3 => ((val & 0x0000FF) << 16) | (val & 0x00FF00) | ((val >> 16) & 0x0000FF),
                DCBMODE_DATAWIDTH_4 => val.swap_bytes(),
                _                   => val,
            };
        }

        match addr {
            0 => { // VC2
                let (width_bits, nbytes): (u8, u8) = match data_width {
                    DCBMODE_DATAWIDTH_1 => (8,  1),
                    DCBMODE_DATAWIDTH_2 => (16, 2),
                    DCBMODE_DATAWIDTH_3 => (24, 3),
                    _ =>                  (32, 4),
                };
                // DW_2: normalize to LSB-aligned — pick whichever half is nonzero.
                let vc2_val = if data_width == DCBMODE_DATAWIDTH_2 {
                    let hi = val >> 16;
                    if hi != 0 { hi } else { val & 0xFFFF }
                } else if data_width == DCBMODE_DATAWIDTH_1 {
                    val >> 24
                } else {
                    val
                };
                let crs = dcb.inc_crs(nbytes);
                self.vc2.lock().write(crs, vc2_val, width_bits);
            }
            1 | 2 | 3 => { // CMAP
                let write_cmap = |crs: u8, byte: u8| {
                    if addr == 1 || addr == 2 { self.cmap0.lock().write_crs(crs, byte); }
                    if addr == 1 || addr == 3 { self.cmap1.lock().write_crs(crs, byte); }
                };
                match data_width {
                    DCBMODE_DATAWIDTH_1 => { write_cmap(dcb.inc_crs(1), (val >> 24) as u8); }
                    DCBMODE_DATAWIDTH_2 => {
                        write_cmap(dcb.inc_crs(1), (val >> 24) as u8);
                        write_cmap(dcb.inc_crs(1), (val >> 16) as u8);
                    }
                    DCBMODE_DATAWIDTH_3 => { // write32 only — MSB-packed
                        write_cmap(dcb.inc_crs(1), (val >> 24) as u8);
                        write_cmap(dcb.inc_crs(1), (val >> 16) as u8);
                        write_cmap(dcb.inc_crs(1), (val >> 8) as u8);
                    }
                    _ => { // DCBMODE_DATAWIDTH_4
                        write_cmap(dcb.inc_crs(1), (val >> 24) as u8);
                        write_cmap(dcb.inc_crs(1), (val >> 16) as u8);
                        write_cmap(dcb.inc_crs(1), (val >> 8) as u8);
                        write_cmap(dcb.inc_crs(1), val as u8);
                    }
                }
            }
            4 | 5 | 6 => { // XMAP
                let write_xmap = |crs: u8, v: u32| {
                    if addr == 4 || addr == 5 { self.xmap0.lock().write_crs(crs, v); }
                    if addr == 4 || addr == 6 { self.xmap1.lock().write_crs(crs, v); }
                    // Mode table write (CRS 5) = buf_sel flip: push a DISP_SYNC fence
                    // so the display thread waits for all prior draws before snapshotting.
                    if crs == crate::xmap9::XMAP9_REG_MODE_TABLE_WRITE {
                        let fence = self.xmap_fence.fetch_add(1, Ordering::Relaxed) + 1;
                        self.gfifo_push(GFIFO_DISP_SYNC, fence as u64);
                    }
                };
                match data_width {
                    DCBMODE_DATAWIDTH_1 => { write_xmap(dcb.inc_crs(1), (val >> 24) & 0xFF); }
                    DCBMODE_DATAWIDTH_2 => {
                        write_xmap(dcb.inc_crs(1), (val >> 24) & 0xFF);
                        write_xmap(dcb.inc_crs(1), (val >> 16) & 0xFF);
                    }
                    DCBMODE_DATAWIDTH_3 => { // write32 only — MSB-packed
                        write_xmap(dcb.inc_crs(1), (val >> 24) & 0xFF);
                        write_xmap(dcb.inc_crs(1), (val >> 16) & 0xFF);
                        write_xmap(dcb.inc_crs(1), (val >> 8) & 0xFF);
                    }
                    _ => { // DCBMODE_DATAWIDTH_4 — single 32-bit write, CRS advances by 4
                        write_xmap(dcb.inc_crs(4), val);
                    }
                }
            }
            7 => { // RAMDAC (Bt445)
                match data_width {
                    DCBMODE_DATAWIDTH_1 => { self.bt445.lock().write_crs(dcb.inc_crs(1), (val >> 24) as u8); }
                    DCBMODE_DATAWIDTH_2 => {
                        self.bt445.lock().write_crs(dcb.inc_crs(1), (val >> 24) as u8);
                        self.bt445.lock().write_crs(dcb.inc_crs(1), (val >> 16) as u8);
                    }
                    DCBMODE_DATAWIDTH_3 => { // write32 only — MSB-packed
                        self.bt445.lock().write_crs(dcb.inc_crs(1), (val >> 24) as u8);
                        self.bt445.lock().write_crs(dcb.inc_crs(1), (val >> 16) as u8);
                        self.bt445.lock().write_crs(dcb.inc_crs(1), (val >> 8) as u8);
                    }
                    _ => { // DCBMODE_DATAWIDTH_4
                        self.bt445.lock().write_crs(dcb.inc_crs(1), (val >> 24) as u8);
                        self.bt445.lock().write_crs(dcb.inc_crs(1), (val >> 16) as u8);
                        self.bt445.lock().write_crs(dcb.inc_crs(1), (val >> 8) as u8);
                        self.bt445.lock().write_crs(dcb.inc_crs(1), val as u8);
                    }
                }
            }
            12 => {
                dcb.backbusy_until = Some(std::time::Instant::now() + std::time::Duration::from_millis(1));
                drop(dcb);
                self.diag.fetch_and(!Self::DIAG_LOCK_DCB, Ordering::Relaxed);
                return;
            }
            _ => {}
        }
        drop(dcb);
        self.diag.fetch_and(!Self::DIAG_LOCK_DCB, Ordering::Relaxed);
    }

    fn dcb_read(&self) -> u32 {
        self.diag.fetch_or(Self::DIAG_LOCK_DCB, Ordering::Relaxed);
        let mut dcb = self.dcb.lock();
        let addr = ((dcb.dcbmode & DCBMODE_DCBADDR_MASK) >> DCBMODE_DCBADDR_SHIFT) as u8;
        let data_width = dcb.dcbmode & DCBMODE_DATAWIDTH_MASK;

        let mut val = 0u32;

        match addr {
            0 => { // VC2
                let nbytes: u8 = match data_width {
                    DCBMODE_DATAWIDTH_1 => 1,
                    DCBMODE_DATAWIDTH_2 => 2,
                    DCBMODE_DATAWIDTH_3 => 3,
                    _ =>                   4,
                };
                let raw = self.vc2.lock().read(dcb.inc_crs(nbytes));
                // DW_2: replicate 16-bit value in both halves so drivers using
                // either >> 16 or & 0xFFFF both get the correct result.
                val = if data_width == DCBMODE_DATAWIDTH_2 {
                    let v = raw & 0xFFFF;
                    (v << 16) | v
                } else if data_width == DCBMODE_DATAWIDTH_1 {
                    raw << 24
                } else {
                    raw
                };
            }
            2 | 3 | 5 | 6 | 7 => {
                let read_byte = |crs: u8| -> u8 {
                    match addr {
                        2 => self.cmap0.lock().read_crs(crs),
                        3 => self.cmap1.lock().read_crs(crs),
                        5 => self.xmap0.lock().read_crs(crs),
                        6 => self.xmap1.lock().read_crs(crs),
                        7 => self.bt445.lock().read_crs(crs),
                        _ => 0,
                    }
                };
                match data_width {
                    DCBMODE_DATAWIDTH_1 => {
                        val = (read_byte(dcb.inc_crs(1)) as u32) << 24;
                    }
                    DCBMODE_DATAWIDTH_2 => {
                        let b0 = read_byte(dcb.inc_crs(1)) as u32;
                        let b1 = read_byte(dcb.inc_crs(1)) as u32;
                        val = (b0 << 24) | (b1 << 16);
                    }
                    DCBMODE_DATAWIDTH_3 => { // read32 only — MSB-packed
                        let b0 = read_byte(dcb.inc_crs(1)) as u32;
                        let b1 = read_byte(dcb.inc_crs(1)) as u32;
                        let b2 = read_byte(dcb.inc_crs(1)) as u32;
                        val = (b0 << 24) | (b1 << 16) | (b2 << 8);
                    }
                    _ => { // DCBMODE_DATAWIDTH_4
                        let b0 = read_byte(dcb.inc_crs(1)) as u32;
                        let b1 = read_byte(dcb.inc_crs(1)) as u32;
                        let b2 = read_byte(dcb.inc_crs(1)) as u32;
                        let b3 = read_byte(dcb.inc_crs(1)) as u32;
                        val = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
                    }
                }
            }
            12 => {
                dcb.backbusy_until = Some(std::time::Instant::now() + std::time::Duration::from_millis(1));
                drop(dcb);
                self.diag.fetch_and(!Self::DIAG_LOCK_DCB, Ordering::Relaxed);
                return 0;
            }
            _ => {}
        }

        // DCBMODE bit 28 "Swap Byte Ordering": swap within the data width.
        if (dcb.dcbmode & DCBMODE_SWAPENDIAN) != 0 {
            val = match data_width {
                DCBMODE_DATAWIDTH_2 => ((val & 0x00FF00FF) << 8) | ((val >> 8) & 0x00FF00FF),
                DCBMODE_DATAWIDTH_4 => val.swap_bytes(),
                _                   => val,
            };
        }

        dlog_dev!(LogModule::Dcb, "DCB Read: Mode {:08x} Addr {} CRS {} DW {} -> {:08x}", dcb.dcbmode, addr, dcb.crs(), data_width, val);
        dcb.dcbdata0 = val;
        drop(dcb);
        self.diag.fetch_and(!Self::DIAG_LOCK_DCB, Ordering::Relaxed);
        val
    }

    fn logic_op_zero(_src: u32, _dst: u32) -> u32 { 0 }
    fn logic_op_and(src: u32, dst: u32) -> u32 { src & dst }
    fn logic_op_andr(src: u32, dst: u32) -> u32 { src & !dst }
    fn logic_op_src(src: u32, _dst: u32) -> u32 { src }
    fn logic_op_andi(src: u32, dst: u32) -> u32 { !src & dst }
    fn logic_op_dst(_src: u32, dst: u32) -> u32 { dst }
    fn logic_op_xor(src: u32, dst: u32) -> u32 { src ^ dst }
    fn logic_op_or(src: u32, dst: u32) -> u32 { src | dst }
    fn logic_op_nor(src: u32, dst: u32) -> u32 { !(src | dst) }
    fn logic_op_xnor(src: u32, dst: u32) -> u32 { !(src ^ dst) }
    fn logic_op_ndst(_src: u32, dst: u32) -> u32 { !dst }
    fn logic_op_orr(src: u32, dst: u32) -> u32 { src | !dst }
    fn logic_op_nsrc(src: u32, _dst: u32) -> u32 { !src }
    fn logic_op_ori(src: u32, dst: u32) -> u32 { !src | dst }
    fn logic_op_nand(src: u32, dst: u32) -> u32 { !(src & dst) }
    fn logic_op_one(_src: u32, _dst: u32) -> u32 { !0 }

    // Afunction: (src_alpha, alpharef) -> pass.  COMPARE is three OR'ed enable
    // bits — src>ref(2), src=ref(1), src<ref(0) — selected once in planes_setup.
    fn afunc_always(_sa: u32, _aref: u32) -> bool { true }
    fn afunc_never(_sa: u32, _aref: u32) -> bool { false }
    fn afunc_lt(sa: u32, aref: u32) -> bool { sa < aref }
    fn afunc_eq(sa: u32, aref: u32) -> bool { sa == aref }
    fn afunc_le(sa: u32, aref: u32) -> bool { sa <= aref }
    fn afunc_gt(sa: u32, aref: u32) -> bool { sa > aref }
    fn afunc_ne(sa: u32, aref: u32) -> bool { sa != aref }
    fn afunc_ge(sa: u32, aref: u32) -> bool { sa >= aref }

    fn read_rgb_4_0(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_rgb.get())[addr as usize] };
        val & 0xF
    }

    fn read_rgb_4_1(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_rgb.get())[addr as usize] };
        (val >> 4) & 0xF
    }

    fn read_rgb_8_0(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_rgb.get())[addr as usize] };
        val & 0xFF
    }

    fn read_rgb_8_1(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_rgb.get())[addr as usize] };
        (val >> 8) & 0xFF
    }

    fn read_rgb_12_0(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_rgb.get())[addr as usize] };
        val & 0xFFF
    }

    fn read_rgb_12_1(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_rgb.get())[addr as usize] };
        (val >> 12) & 0xFFF
    }

    fn read_rgb_24(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_rgb.get())[addr as usize] };
        val & 0xFFFFFF
    }

    fn read_olay_0(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_aux.get())[addr as usize] };
        (val >> 8) & 0xFF
    }

    fn read_olay_1(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_aux.get())[addr as usize] };
        (val >> 16) & 0xFF
    }

    fn read_cid_0(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_aux.get())[addr as usize] };
        val & 0x3
    }

    fn read_cid_1(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_aux.get())[addr as usize] };
        (val >> 4) & 0x3
    }

    fn read_pup_0(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_aux.get())[addr as usize] };
        (val >> 2) & 0x3
    }

    fn read_pup_1(rex: &Rex3, addr: u32) -> u32 {
        let val = unsafe { (*rex.fb_aux.get())[addr as usize] };
        (val >> 6) & 0x3
    }

    fn read_zero(_rex: &Rex3, _addr: u32) -> u32 { 0 }

    fn write_rgb_4(rex: &Rex3, addr: u32, val: u32) {
        let mask = unsafe { (*rex.context.get()).wrmask };
        let fb = unsafe { &mut *rex.fb_rgb.get() };
        fb[addr as usize] = (fb[addr as usize] & !mask) | (val & mask);
    }

    fn write_rgb_8(rex: &Rex3, addr: u32, val: u32) {
        let mask = unsafe { (*rex.context.get()).wrmask };
        let fb = unsafe { &mut *rex.fb_rgb.get() };
        fb[addr as usize] = (fb[addr as usize] & !mask) | (val & mask);
    }

    fn write_rgb_12(rex: &Rex3, addr: u32, val: u32) {
        let mask = unsafe { (*rex.context.get()).wrmask };
        let fb = unsafe { &mut *rex.fb_rgb.get() };
        fb[addr as usize] = (fb[addr as usize] & !mask) | (val & mask);
    }

    fn write_rgb_24(rex: &Rex3, addr: u32, val: u32) {
        let mask = unsafe { (*rex.context.get()).wrmask };
        let fb = unsafe { &mut *rex.fb_rgb.get() };
        fb[addr as usize] = (fb[addr as usize] & !mask) | (val & mask);
    }

    fn write_olay(rex: &Rex3, addr: u32, val: u32) {
        let mask = unsafe { (*rex.context.get()).wrmask };
        let fb = unsafe { &mut *rex.fb_aux.get() };
        fb[addr as usize] = (fb[addr as usize] & !mask) | (val & mask);
    }

    fn write_cid(rex: &Rex3, addr: u32, val: u32) {
        let mask = unsafe { (*rex.context.get()).wrmask };
        let fb = unsafe { &mut *rex.fb_aux.get() };
        fb[addr as usize] = (fb[addr as usize] & !mask) | (val & mask);
    }

    fn write_pup(rex: &Rex3, addr: u32, val: u32) {
        let mask = unsafe { (*rex.context.get()).wrmask };
        let fb = unsafe { &mut *rex.fb_aux.get() };
        fb[addr as usize] = (fb[addr as usize] & !mask) | (val & mask);
    }

    fn write_nop(_rex: &Rex3, _addr: u32, _val: u32) {}

    fn gfifo_push(&self, addr: u32, val: u64) {
        #[cfg(feature = "developer")]
        {
            let len = self.gfifo.len() + 1;
            let _ = self.gfifo_hwm.try_update(Ordering::Relaxed, Ordering::Relaxed, |hwm| {
                if len > hwm { Some(len) } else { None }
            });
        }
        self.gfifo.push(addr, val);
        //self.update_gfifo_irqs();
        // Wake the consumer if it parked on an empty fifo (idle desktop). Cheap
        // on the hot path: a relaxed-ish load that is false whenever the
        // processor is actively draining.
        #[cfg(feature = "idle-pause")]
        if self.processor_parked.load(Ordering::Acquire) {
            if let Some(t) = self.processor_unparker.get() {
                t.unpark();
            }
        }
    }

    fn wait_idle(&self) {
        loop {
            // Acquire load: when gfxbusy goes false, all execute_go() writes become visible.
            let busy = self.gfxbusy.load(Ordering::Acquire);
            if !busy && self.gfifo.is_empty() { break; }
            if !self.running.load(Ordering::Relaxed) { break; }
            std::hint::spin_loop();
        }
    }

    pub fn calculate_fb_address(&self, x: i32, y: i32, ctx: &Rex3Context, is_write: bool) -> Option<u32> {
        // 1. XYOFFSET (Draw only, not SCR2SCR source)
        let opcode = ctx.drawmode0.opcode();
        let is_scr2scr = opcode == DRAWMODE0_OPCODE_SCR2SCR;
        
        let mut x_curr = x;
        let mut y_curr = y;

        // In scr2scr mode xymove is unconditional (it offsets the destination).
        // In regular paint mode it is conditional on the xyoffset flag.
        let apply_xymove = is_scr2scr || ctx.drawmode0.xyoffset();
        if apply_xymove {
            let x_move = (ctx.xymove >> 16) as i16 as i32;
            let y_move = (ctx.xymove & 0xFFFF) as i16 as i32;
            x_curr += x_move;
            y_curr += y_move;
        }

        // Apply XYWIN offset
        // XYWIN is 16,16. Assuming High=X, Low=Y.
        // Treated as signed 16-bit integers for coordinate biasing.
        let x_off = ((ctx.xywin >> 16) & 0xFFFF) as i16 as i32;
        let y_off = (ctx.xywin & 0xFFFF) as i16 as i32;

        let x_abs = x_curr + x_off;
        let y_abs = y_curr + y_off;

        if is_write {
            let clipmode = ctx.clipmode;
            let ensmask = clipmode & CLIPMODE_ENSMASK_MASK;

            // SMASK0 (Window Relative) — high16=min, low16=max
            if (ensmask & 1) != 0 {
                let min_x = ((ctx.smask0x >> 16) & 0xFFFF) as i16 as i32;
                let max_x = (ctx.smask0x & 0xFFFF) as i16 as i32;
                let min_y = ((ctx.smask0y >> 16) & 0xFFFF) as i16 as i32;
                let max_y = (ctx.smask0y & 0xFFFF) as i16 as i32;

                if x_curr < min_x || x_curr > max_x || y_curr < min_y || y_curr > max_y {
                    return None;
                }
            }

            // SMASK1-4 (Screen Absolute, unaffected by XYWIN per spec).
            // Host pre-biases smask values with the 4K,4K offset. The equivalent
            // pixel coordinate is x_phys + 0x1000 = (x_abs - 0x1000) + 0x1000 = x_abs.
            // So compare x_abs/y_abs directly against raw smask values. high16=min, low16=max.
            // Logic: pixel must be inside at least one enabled mask.
            let smask_enabled = (ensmask & 0x1E) != 0;
            if smask_enabled {
                let smasks = [
                    (ctx.smask1x, ctx.smask1y),
                    (ctx.smask2x, ctx.smask2y),
                    (ctx.smask3x, ctx.smask3y),
                    (ctx.smask4x, ctx.smask4y),
                ];
                let mut inside_any = false;
                for (bit, (sx, sy)) in smasks.iter().enumerate() {
                    if (ensmask & (1 << (bit + 1))) == 0 { continue; }
                    let min_x = (sx >> 16) as i16 as i32;
                    let max_x = (sx & 0xFFFF) as i16 as i32;
                    let min_y = (sy >> 16) as i16 as i32;
                    let max_y = (sy & 0xFFFF) as i16 as i32;
                    if x_abs >= min_x && x_abs <= max_x && y_abs >= min_y && y_abs <= max_y {
                        inside_any = true;
                        break;
                    }
                }
                if !inside_any {
                    return None;
                }
            }
        }
        // Physical Address Calculation
        let x_phys = x_abs - REX3_COORD_BIAS;
        let y_phys = if ctx.drawmode1.yflip() { 0x23FF - y_abs } else { y_abs - REX3_COORD_BIAS };

        // Sector Clipping (VRAM bounds)
        // Reads are not culled (but must be within VRAM allocation)
        let width_limit = if is_write { REX3_SCREEN_WIDTH } else { 2048 };
        if x_phys < 0 || x_phys >= width_limit || y_phys < 0 || y_phys >= REX3_SCREEN_HEIGHT {
            return None;
        }

        Some((y_phys as u32) * 2048 + (x_phys as u32))
    }

    fn register_processor(&self) {
        // Publish our thread handle so gfifo_push can unpark us when we park.
        #[cfg(feature = "idle-pause")]
        let _ = self.processor_unparker.set(thread::current());
        let backoff = crossbeam_utils::Backoff::new();
        let mut is_busy = false;
        loop {
            if let Some((addr, val)) = self.gfifo.peek() {
                backoff.reset();
                // Any consumed entry may have touched the framebuffer or display
                // state; flag it so the refresh thread re-renders this frame. An
                // idle screen leaves the fifo empty, so this stays clear and the
                // refresh thread skips its expensive full-frame work.
                self.fb_dirty.store(true, Ordering::Relaxed);

                if !is_busy {
                    self.gfxbusy.store(true, Ordering::Relaxed);
                    is_busy = true;
                }

                let is_go = addr & 0x0800 != 0;
                // Strip GO bit only; keep is_64bit (bit 0) for HOSTRW discrimination.
                // process_register handles GFIFO_EXIT/GFIFO_PURE_GO sentinels directly.
                let reg_offset = addr & !0x0800;
                let exit = self.process_register(reg_offset, val);

                // Log to rex3_log after processing.
                #[cfg(feature = "developer")]
                if let Some(f) = self.rex3_log.lock().as_mut() {
                    if addr == GFIFO_PURE_GO {
                        let _ = writeln!(f, "------- PURE_GO -------");
                    } else {
                        let val32 = val as u32;
                        let extra = match reg_offset {
                            REX3_DRAWMODE0 => format!("  ; {}", decode_dm0(val32)),
                            REX3_DRAWMODE1 => format!("  ; {}", decode_dm1(val32)),
                            _ => String::new(),
                        };
                        if is_go {
                            let _ = writeln!(f, "------- GO reg={:04x}({}) val={:016x}{} -------",
                                reg_offset, rex3_reg_name(reg_offset), val, extra);
                        } else {
                            let _ = writeln!(f, "reg={:04x}({}) val={:016x}{}",
                                reg_offset, rex3_reg_name(reg_offset), val, extra);
                        }
                    }
                }

                if is_go {
                    self.execute_go();
                }

                // Advance head and release busy atomically — CPU thread sees all writes
                // (context registers + FB) only after both process_register and execute_go
                // have completed.
                self.gfifo.consume();
                //self.update_gfifo_irqs();

                if exit {
                    self.gfxbusy.store(false, Ordering::Release);
                    self.gfifo.flush_head();
                    break; 
                } // GFIFO_EXIT
            } else {
                if is_busy {
                    self.gfxbusy.store(false, Ordering::Release);
                    is_busy = false;
                    //self.update_gfifo_irqs();
                }
                self.gfifo.flush_head();
                // Nothing in the ring — back off. Spin-hint/yield while a burst
                // might still be in flight; once emptiness is *sustained*
                // (crossbeam's backoff completes), stop burning a host core on
                // yield_now() and actually park. An idle IRIX desktop leaves this
                // fifo empty indefinitely, so without parking this thread pins a
                // CPU at ~100%.
                #[cfg(feature = "idle-pause")]
                if backoff.is_completed() {
                    // Set parked BEFORE the final emptiness re-check so a racing
                    // gfifo_push either (a) is seen by the peek below, or (b) sees
                    // parked=true and unparks us — the unpark token makes
                    // park_timeout return immediately, so no wakeup is lost. The
                    // 2ms timeout is only a backstop; a missed unpark costs latency,
                    // never correctness.
                    self.processor_parked.store(true, Ordering::Release);
                    if self.gfifo.peek().is_none() && self.running.load(Ordering::Relaxed) {
                        thread::park_timeout(std::time::Duration::from_millis(2));
                    }
                    self.processor_parked.store(false, Ordering::Release);
                    backoff.reset();
                } else { backoff.snooze(); }
                #[cfg(not(feature = "idle-pause"))]
                backoff.snooze();
            }
        }
    }

    pub(crate) fn execute_go(&self) {
        self.diag.fetch_or(Self::DIAG_LOOP_EXECUTE_GO, Ordering::Relaxed);
        let ctx = unsafe { &mut *self.context.get() };
        let opcode = ctx.drawmode0.opcode();

        // Pattern bit positions reset only on DOSETUP (new primitive).  Connected
        // stippled line segments keep pat_bit across GO via LSSAVE/LSRESTORE.
        // (This gating is a real, separate fix from f2d0bff, validated by
        // test_iline_lsadvlast_advances_on_last_pixel — do not confuse it with
        // f2d0bff's LSPATTERN/ZPATTERN direction+rotation change, which broke
        // PROM text rendering and was reverted separately in advance_zpat/
        // advance_lspat above.)
        if ctx.drawmode0.dosetup() {
            ctx.pat_bit  = 31;
            ctx.zpat_bit = 31;
        }
        // lsrcount is live state inside the lsmode register — do NOT reset it here.
        // The ARCS diag writes a pattern to lsmode, issues a GO, then reads it back and
        // expects the value unchanged.  Resetting lsrcount on GO would corrupt the readback.
        // GL manages lsrcount explicitly via LSSAVE/LSRESTORE for connected stippled lines.

        if ctx.drawmode0.dosetup() {
            self.setup(ctx);
        } else {
            // Continuation GO: re-derive Bresenham when the segment axis disagrees with
            // the persisted octant (e.g. degenerate setup point, then horizontal cont).
            let adrmode = ctx.drawmode0.adrmode() << 2;
            let is_line = adrmode == DRAWMODE0_ADRMODE_I_LINE
                || adrmode == DRAWMODE0_ADRMODE_F_LINE
                || adrmode == DRAWMODE0_ADRMODE_A_LINE;
            if is_line {
                let xs = ctx.xstart >> 11;
                let ys = ctx.ystart >> 11;
                let xe = ctx.xend >> 11;
                let ye = ctx.yend >> 11;
                let adx = (xe - xs).abs();
                let ady = (ye - ys).abs();
                if adx != ady {
                    let seg_x_major = adx > ady;
                    let oct_x_major = (ctx.bresoctinc1.octant() & OCTANT_XMAJOR) != 0;
                    if seg_x_major != oct_x_major {
                        self.setup(ctx);
                    }
                }
            }
        }

        if devlog_is_active(LogModule::Rex3) {
            let adrmode = ctx.drawmode0.adrmode();
            let prim_str = match adrmode {
                0 => "SPAN", 1 => "BLOCK", 2 => "I_LINE", 3 => "F_LINE", 4 => "A_LINE", _ => "UNKNOWN",
            };
            let op_str = match opcode {
                DRAWMODE0_OPCODE_NOOP    => "NOOP",
                DRAWMODE0_OPCODE_READ    => "READ",
                DRAWMODE0_OPCODE_DRAW    => "DRAW",
                DRAWMODE0_OPCODE_SCR2SCR => "SCR2SCR",
                _                        => "UNKNOWN",
            };
            dlog_dev!(LogModule::Rex3, "REX3 Draw: {} {} Mode0={:08x} Mode1={:08x}", prim_str, op_str, ctx.drawmode0.0, ctx.drawmode1.0);
            dlog_dev!(LogModule::Rex3, "  Coords: Start({:.2}, {:.2}) End({:.2}, {:.2})",
                ctx.xstart as f32 / 2048.0, ctx.ystart as f32 / 2048.0,
                ctx.xend as f32 / 2048.0, ctx.yend as f32 / 2048.0);
        }

        #[cfg(feature = "rex-jit")]
        {
            if self.jit_enabled.load(Ordering::Relaxed) {
            if let Some(ref jit) = self.rex_jit {
                let dm0 = ctx.drawmode0.0;
                // Normalize dm1 key to match compile_shader's normalization:
                // fastclear clears blend; scr2scr clears dither (copies quantized pixels).
                let dm1_raw = ctx.drawmode1.0;
                let dm1 = if dm1_raw & (1 << 17) != 0 { dm1_raw & !(1 << 18) }       // fastclear→no blend
                          else if opcode == DRAWMODE0_OPCODE_SCR2SCR { dm1_raw & !(1 << 16) } // scr2scr→no dither
                          else { dm1_raw };
                let adrmode = ctx.drawmode0.adrmode() << 2;
                let is_line = adrmode == DRAWMODE0_ADRMODE_I_LINE
                    || adrmode == DRAWMODE0_ADRMODE_F_LINE
                    || adrmode == DRAWMODE0_ADRMODE_A_LINE;
                let is_jittable = (opcode == DRAWMODE0_OPCODE_DRAW
                        || opcode == DRAWMODE0_OPCODE_SCR2SCR
                        || opcode == DRAWMODE0_OPCODE_READ)
                    && (adrmode == DRAWMODE0_ADRMODE_BLOCK || adrmode == DRAWMODE0_ADRMODE_SPAN || is_line);
                if is_jittable {
                    let cm = ctx.clipmode & CLIPMODE_JIT_KEY_MASK;
                    // Fast path: same (dm0, dm1, cm) as last GO — skip the HashMap lookup.
                    let last = self.jit_last.get();
                    let entry = if last.0 == dm0 && last.1 == dm1 && last.2 == cm && last.3.is_some() {
                        last.3
                    } else {
                        let e = jit.lookup(dm0, dm1, cm);
                        if let Some(_) = e { self.jit_last.set((dm0, dm1, cm, e)); }
                        else { jit.request_compile(dm0, dm1, cm); }
                        e
                    };
                    if let Some(entry) = entry {
                        // Mirror the interpreter's log_block() calls (see bottom of this
                        // function) so block/span primitives are traced identically whether
                        // dispatched via JIT or interpreter — block_debug/draw_debug were
                        // previously silent for JIT-compiled GOs since this path returns
                        // before ever reaching the interpreter's log_block() calls.
                        if adrmode == DRAWMODE0_ADRMODE_BLOCK || adrmode == DRAWMODE0_ADRMODE_SPAN {
                            self.log_block(ctx, opcode);
                        }
                        let fb_rgb = unsafe { (*self.fb_rgb.get()).as_mut_ptr() };
                        let fb_aux = unsafe { (*self.fb_aux.get()).as_mut_ptr() };
                        unsafe { entry(ctx as *mut Rex3Context, fb_rgb, fb_aux); }
                        self.jit_go_count.fetch_add(1, Ordering::Relaxed);
                        self.diag.fetch_and(!Self::DIAG_LOOP_EXECUTE_GO, Ordering::Relaxed);
                        return;
                    }
                    // fall through to interpreter
                }
            }
            } // jit_enabled
        }

        // Interpreter-only setup: function pointer selection and host/planes dispatch tables.
        // Skipped when JIT handles the draw (returned above) or when nothing affecting these
        // tables has changed since the last GO.
        let cidmatch_bits = (ctx.clipmode >> CLIPMODE_CIDMATCH_SHIFT) & 0xF;
        // SCR2SCR: normalize dm1 key the same way planes_setup does (clear dither bit).
        let dm1_norm = if opcode == DRAWMODE0_OPCODE_SCR2SCR {
            ctx.drawmode1.0 & !(1 << 16)
        } else {
            ctx.drawmode1.0
        };
        let setup_key = (
            ctx.drawmode0.0 & DRAWMODE0_INTERP_SETUP_MASK,
            dm1_norm        & DRAWMODE1_INTERP_SETUP_MASK,
            cidmatch_bits,
        );
        if self.interp_setup_cache.get() != setup_key {
            self.interp_setup_cache.set(setup_key);

            self.planes_setup(DrawMode1(dm1_norm));

            let proc = match opcode {
                DRAWMODE0_OPCODE_READ    => Self::process_pixel_read,
                DRAWMODE0_OPCODE_DRAW    => {
                    let no_cid      = cidmatch_bits == 0xF;
                    let no_host     = !ctx.drawmode0.colorhost() && !ctx.drawmode0.alphahost();
                    let no_blend    = !ctx.drawmode1.blend();
                    let en_z        = ctx.drawmode0.enzpattern();
                    let en_ls       = ctx.drawmode0.enlspattern();
                    let no_zpopaque = !ctx.drawmode0.zpopaque();
                    let is_src_op   = ctx.drawmode1.logicop() == DRAWMODE1_LOGICOP_SRC >> 28;
                    // Afunction (alpha-vs-ALPHAREF compare) applies regardless of rgbmode;
                    // only inhibits writes when COMPARE != 0x7 (always-pass/disabled).
                    let no_afunction = ctx.drawmode1.compare() == 0x7;

                    if ctx.drawmode1.fastclear() && no_cid && no_host {
                        Self::process_pixel_fastclear
                    } else if no_cid && no_host && no_blend && no_afunction && en_z && !en_ls && no_zpopaque && is_src_op {
                        // Character/glyph: zpattern kill only, SRC logicop — no dst read needed.
                        Self::process_pixel_zpattern
                    } else if no_cid && no_host && no_blend && no_afunction {
                        // Common solid fills, spans, lines: no blend, no host FIFO.
                        Self::process_pixel_noblend
                    } else {
                        Self::process_pixel_draw
                    }
                }
                DRAWMODE0_OPCODE_SCR2SCR => Self::process_pixel_scr2scr,
                _                        => Self::process_pixel_noop,
            };
            unsafe { *self.px_proc.get() = proc; }

            // Select shade iterate fn.
            // RGB mode: always clamp (spec §3.8: "DDA values of R,G,B,A are clamped each
            // iteration before sending down the pipeline").
            // CI mode: clamp only when CICLAMP is set (DRAWMODE0 bit 21 = ENCICLAMP).
            let shade_fn: fn(&mut Rex3Context) = if ctx.drawmode0.shade() {
                if ctx.drawmode1.rgbmode() {
                    Self::iterate_shade_rgb_clamp
                } else if ctx.drawmode0.ciclamp() {
                    match ctx.drawmode1.drawdepth() {
                        1 => Self::iterate_shade_ci8_clamp,
                        2 => Self::iterate_shade_ci12_clamp,
                        _ => Self::iterate_shade_unclamped, // 4bpp / 24bpp: no CI clamp per spec
                    }
                } else {
                    Self::iterate_shade_unclamped
                }
            } else {
                Self::iterate_shade_noop
            };
            unsafe { *self.px_shade.get() = shade_fn; }

            // Select pattern iterate fn.
            let en_z  = ctx.drawmode0.enzpattern();
            let en_ls = ctx.drawmode0.enlspattern();
            let pat_fn: fn(&mut Rex3Context) = match (en_z, en_ls) {
                (false, false) => Self::iterate_pattern_noop,
                (true,  false) => Self::iterate_pattern_z,
                (false, true)  => Self::iterate_pattern_ls,
                (true,  true)  => Self::iterate_pattern_both,
            };
            unsafe { *self.px_pattern.get() = pat_fn; }

        } // interp_setup_cache miss

        if opcode != DRAWMODE0_OPCODE_NOOP {
            let adrmode = ctx.drawmode0.adrmode() << 2;
            if adrmode == DRAWMODE0_ADRMODE_I_LINE {
                self.draw_iline(ctx);
            } else if adrmode == DRAWMODE0_ADRMODE_F_LINE {
                self.draw_fline(ctx);
            } else if adrmode == DRAWMODE0_ADRMODE_A_LINE {
                self.draw_aline(ctx);
            } else if adrmode == DRAWMODE0_ADRMODE_BLOCK {
                self.log_block(ctx, opcode);
                self.draw_block(ctx);
            } else if adrmode == DRAWMODE0_ADRMODE_SPAN {
                self.log_block(ctx, opcode);
                self.draw_span(ctx);
            }
        }
        self.interp_go_count.fetch_add(1, Ordering::Relaxed);
        self.diag.fetch_and(!Self::DIAG_LOOP_EXECUTE_GO, Ordering::Relaxed);
    }

    fn planes_setup(&self, drawmode1: DrawMode1) {
        let planes = drawmode1.planes();
        let depth = drawmode1.drawdepth();
        let dblsrc = drawmode1.dblsrc();

        let (rd, wr, amp): (fn(&Rex3, u32) -> u32, fn(&Rex3, u32, u32), fn(u32) -> u32) = match planes {
            DRAWMODE1_PLANES_RGB | DRAWMODE1_PLANES_RGBA => {
                match depth {
                    0 => ( // 4-bit
                        if dblsrc { Self::read_rgb_4_1 } else { Self::read_rgb_4_0 },
                        Self::write_rgb_4,
                        Self::amplify_rgb_4
                    ),
                    1 => ( // 8-bit
                        if dblsrc { Self::read_rgb_8_1 } else { Self::read_rgb_8_0 },
                        Self::write_rgb_8,
                        Self::amplify_rgb_8
                    ),
                    2 => ( // 12-bit
                        if dblsrc { Self::read_rgb_12_1 } else { Self::read_rgb_12_0 },
                        Self::write_rgb_12,
                        Self::amplify_rgb_12
                    ),
                    3 => ( // 24-bit
                        Self::read_rgb_24,
                        Self::write_rgb_24,
                        Self::amplify_rgb_24
                    ),
                    _ => (Self::read_zero, Self::write_nop, Self::amplify_nop)
                }
            },
            DRAWMODE1_PLANES_OLAY => (
                if dblsrc { Self::read_olay_1 } else { Self::read_olay_0 },
                Self::write_olay,
                Self::amplify_olay
            ),
            DRAWMODE1_PLANES_PUP => (
                if dblsrc { Self::read_pup_1 } else { Self::read_pup_0 },
                Self::write_pup,
                Self::amplify_pup
            ),
            DRAWMODE1_PLANES_CID => (
                if dblsrc { Self::read_cid_1 } else { Self::read_cid_0 },
                Self::write_cid,
                Self::amplify_cid
            ),
            _ => (Self::read_zero, Self::write_nop, Self::amplify_nop)
        };

        let logic_op = drawmode1.logicop();
        let logic_fn = match logic_op {
            0 => Self::logic_op_zero,
            1 => Self::logic_op_and,
            2 => Self::logic_op_andr,
            3 => Self::logic_op_src,
            4 => Self::logic_op_andi,
            5 => Self::logic_op_dst,
            6 => Self::logic_op_xor,
            7 => Self::logic_op_or,
            8 => Self::logic_op_nor,
            9 => Self::logic_op_xnor,
            10 => Self::logic_op_ndst,
            11 => Self::logic_op_orr,
            12 => Self::logic_op_nsrc,
            13 => Self::logic_op_ori,
            14 => Self::logic_op_nand,
            15 => Self::logic_op_one,
            _ => Self::logic_op_src,
        };

        let rgbmode = drawmode1.rgbmode();
        let dither = drawmode1.dither();
        // compress: 24-bit BGR → plane-depth pixel (only when rgbmode=1; CI src already plane-depth)
        // Dither variants expect bayer index packed in bits 27:24 via bayer_pack().
        let compress: fn(u32) -> u32 = if rgbmode {
            match depth {
                0 => if dither { Self::rgb24_to_rgb4_dither  } else { Self::rgb24_to_rgb4  },
                1 => if dither { Self::rgb24_to_rgb8_dither  } else { Self::rgb24_to_rgb8  },
                2 => if dither { Self::rgb24_to_rgb12_dither } else { Self::rgb24_to_rgb12 },
                _ => Self::identity, // 24bpp: no dithering needed
            }
        } else {
            Self::identity
        };
        // expand: plane-depth pixel → 24-bit BGR (for blend dst in RGB planes mode)
        let expand: fn(u32) -> u32 = if rgbmode {
            match depth {
                0 => Self::rgb4_to_rgb24,
                1 => Self::rgb8_to_rgb24,
                2 => Self::rgb12_to_rgb24,
                _ => Self::identity,
            }
        } else {
            Self::identity
        };

        let bayer_fn: fn(u32, i32, i32) -> u32 = if rgbmode { Self::bayer_pack } else { Self::bayer_nop };

        // Afunction: "source alpha (either from DDA or host)" vs ALPHAREF — applies in
        // both RGB and CI mode (spec doesn't scope it to RGB; combine_host_dda leaves
        // raw_src bits 31:24 as 0 in CI/DDA mode, so afunc_eq there degenerates to
        // "alpharef==0", which is the correct behavior, not a special case).
        let afunc_fn: fn(u32, u32) -> bool = match drawmode1.compare() {
            0b000 => Self::afunc_never,
            0b001 => Self::afunc_lt,
            0b010 => Self::afunc_eq,
            0b011 => Self::afunc_le,
            0b100 => Self::afunc_gt,
            0b101 => Self::afunc_ne,
            0b110 => Self::afunc_ge,
            _      => Self::afunc_always, // 0b111: disabled
        };

        unsafe {
            *self.px_rd.get() = rd;
            *self.px_wr.get() = wr;
            *self.px_amp.get() = amp;
            *self.px_logic.get() = logic_fn;
            *self.px_afunc.get() = afunc_fn;
            *self.px_bayer.get() = bayer_fn;
            *self.px_compress.get() = compress;
            *self.px_expand.get() = expand;
            self.host_setup(drawmode1);
        }
    }

    fn refresh_loop(&self) {
        let frame_duration = std::time::Duration::from_micros(16667); // ~60Hz
        let mut status_bar  = crate::disp::StatusBar::new();
        let mut overlay     = crate::debug_overlay::DebugOverlay::new();
        let mut sbtex       = crate::disp::StatusBarTexture::new();
        // Idle-skip bookkeeping (see the should_render gate below).
        let mut last_topscan: usize = usize::MAX;
        let mut frames_since_render: u32 = u32::MAX;
        let mut full_presents: u32 = 0;

        while self.running.load(Ordering::Relaxed) {
            let start = std::time::Instant::now();

            // Poll and clear activity bits; preserve persistent LED bits.
            let bar_stats = crate::disp::BarStats {
                now:          start,
                hb:           self.heartbeat.fetch_and(Self::HB_PERSISTENT, Ordering::Relaxed),
                cycles:       self.cycles.get().get(),
                fasttick:     self.fasttick_count.load(Ordering::Relaxed),
                #[cfg(feature = "developer")]
                decoded_delta: self.decoded_count.swap(0, Ordering::Relaxed),
                #[cfg(not(feature = "developer"))]
                decoded_delta: 0,
                #[cfg(feature = "developer")]
                l1i_hits:     self.l1i_hit_count.swap(0, Ordering::Relaxed),
                #[cfg(not(feature = "developer"))]
                l1i_hits:     0,
                #[cfg(feature = "developer")]
                l1i_fetches:  self.l1i_fetch_count.swap(0, Ordering::Relaxed),
                #[cfg(not(feature = "developer"))]
                l1i_fetches:  0,
                #[cfg(feature = "developer")]
                uncached:     self.uncached_fetch_count.swap(0, Ordering::Relaxed),
                #[cfg(not(feature = "developer"))]
                uncached:     0,
                #[cfg(feature = "developer")]
                count_hz:     self.count_hz_atomic.lock().load(Ordering::Relaxed),
                #[cfg(not(feature = "developer"))]
                count_hz:     0,
                gfifo_pending: self.gfifo.len(),
            };

            // Fence-based sync: wait until the GFIFO consumer has processed every
            // GFIFO_DISP_SYNC pushed since the last frame.  Each XMAP mode table
            // write increments xmap_fence and enqueues GFIFO_DISP_SYNC; the consumer
            // advances gfifo_fence to match.  If no mode table writes happened this
            // frame the fences are already equal and we fall through immediately —
            // no stall on static scenes.
            {
                let target = self.xmap_fence.load(Ordering::Acquire);
                let backoff = crossbeam_utils::Backoff::new();
                let fence_wait_start = std::time::Instant::now();
                loop {
                    let current = self.gfifo_fence.load(Ordering::Acquire);
                    // Wrapping comparison: current >= target
                    if current.wrapping_sub(target) < 0x8000_0000 {
                        break;
                    }
                    // Don't stall the refresh thread forever if the GFIFO consumer
                    // is behind (e.g. guest reprogramming XMAP while idle).
                    if fence_wait_start.elapsed() > std::time::Duration::from_millis(50) {
                        self.gfifo_fence.store(target, Ordering::Release);
                        break;
                    }
                    backoff.snooze();
                }
            }

            // Get unsafe access to framebuffers and context
            let fb_rgb = unsafe { &*self.fb_rgb.get() };
            let fb_aux = unsafe { &*self.fb_aux.get() };
            let topscan = unsafe { (*self.context.get()).topscan as usize };

            // Idle skip: only run the (expensive) full-frame refresh + GL upload
            // when something visible changed. `fb_dirty` covers all REX3 drawing
            // (set by the gfifo consumer); the palette/cursor/mode mutexes carry
            // their own dirty flags (peeked here, not cleared — refresh() clears
            // them when it runs). A periodic heartbeat keeps the live status bar
            // moving and bounds any missed-dirty staleness. VBLANK is still ticked
            // every frame (see the else branch), independent of host rendering.
            //
            // The dirty mutexes are locked one at a time (separate `let`s) so we
            // never hold two of them at once — avoids any lock-ordering hazard
            // against the consumer/CPU threads.
            const IDLE_HEARTBEAT_FRAMES: u32 = 6; // ≥10 Hz refresh floor when idle
            let palette_dirty = {
                let v = self.vc2.lock().dirty;
                let c = self.cmap0.lock().dirty;
                let b = self.bt445.lock().dirty;
                let x = self.xmap0.lock().dirty;
                v || c || b || x
            };
            let dbg_overlay = self.draw_debug_active()
                || self.show_cmap.load(Ordering::Relaxed)
                || self.show_disp_debug.load(Ordering::Relaxed);
            let fb_was_dirty = self.fb_dirty.swap(false, Ordering::Acquire);
            let should_render = fb_was_dirty
                || palette_dirty
                || dbg_overlay
                || topscan != last_topscan
                || self.screenshot_pending.load(Ordering::Relaxed)
                || frames_since_render >= IDLE_HEARTBEAT_FRAMES;

            if should_render {
                frames_since_render = 0;
                last_topscan = topscan;
                self.diag.fetch_or(Self::DIAG_LOCK_SCREEN, Ordering::Relaxed);
                let mut screen = self.screen.lock();
                screen.topscan = topscan;
                screen.status_bar_only = full_presents > 0
                    && !fb_was_dirty
                    && !palette_dirty
                    && !dbg_overlay
                    && !self.screenshot_pending.load(Ordering::Relaxed);

                // Push debug state into overlay
                overlay.show_cmap       = self.show_cmap.load(Ordering::Relaxed);
                overlay.show_disp_debug = self.show_disp_debug.load(Ordering::Relaxed);
                let dd = self.draw_debug_active();
                overlay.show_draw_debug = dd;
                #[cfg(feature = "developer")]
                if dd {
                    let ring = self.draw_ring.lock();
                    overlay.draw_snapshot.clear();
                    overlay.draw_snapshot.extend(ring.iter_newest_first().copied());
                }
                self.diag.fetch_or(Self::DIAG_LOCK_RENDERER, Ordering::Relaxed);
                let mut renderer = self.renderer.lock();

                let resized = screen.refresh(
                    &**fb_rgb,
                    &**fb_aux,
                    fb_was_dirty,
                    &self.vc2,
                    &self.xmap0,
                    &self.cmap0,
                    &self.bt445,
                    &self.diag,
                );
                if resized {
                    if let Some(ref mut r) = *renderer {
                        r.resize(screen.width, screen.height);
                    }
                }

                // Screenshot: grab pixels from the compositor before presenting.
                let take_screenshot = self.screenshot_pending.swap(false, Ordering::Relaxed);

                // Assert VBLANK: device state is now copied into screen caches, so
                // the CPU gets the maximum window to react before the next refresh.
                self.config.status.fetch_or(STATUS_VRINT, Ordering::Relaxed);
                {
                    self.diag.fetch_or(Self::DIAG_LOCK_VC2, Ordering::Relaxed);
                    let mut vc2 = self.vc2.lock();
                    vc2.regs[crate::vc2::VC2_REG_WORKING_CURSOR_Y as usize] = vc2.regs[crate::vc2::VC2_REG_CURSOR_Y_LOC as usize];
                    drop(vc2);
                    self.diag.fetch_and(!Self::DIAG_LOCK_VC2, Ordering::Relaxed);
                }
                {
                    self.diag.fetch_or(Self::DIAG_LOCK_VBLANK_CB, Ordering::Relaxed);
                    let cb = self.vblank_cb.lock().clone();
                    self.diag.fetch_and(!Self::DIAG_LOCK_VBLANK_CB, Ordering::Relaxed);
                    if let Some(cb) = cb { cb(true); }
                }

                // Present: compositor → overlay → status bar → swap
                if screen.width > 0 && screen.height > 0 {
                    if let Some(ref mut r) = *renderer {
                        self.diag.fetch_or(Self::DIAG_LOOP_GL_RENDER, Ordering::Relaxed);
                        let borrow = screen.fb_borrowed;
                        r.present(
                            &mut *screen,
                            &mut overlay,
                            &mut status_bar,
                            &mut sbtex,
                            &bar_stats,
                            take_screenshot,
                            if borrow { Some(&**fb_rgb) } else { None },
                            if borrow { Some(&**fb_aux) } else { None },
                        );
                        self.diag.fetch_and(!Self::DIAG_LOOP_GL_RENDER, Ordering::Relaxed);
                    }

                    if !screen.status_bar_only {
                        full_presents = full_presents.saturating_add(1);
                    }

                    if take_screenshot {
                        let width  = screen.width;
                        let height = screen.height;
                        let pixels = screen.rgba.clone();
                        let n      = self.screenshot_counter.fetch_add(1, Ordering::Relaxed);
                        let path   = format!("screenshot_{:04}.png", n);
                        thread::spawn(move || {
                            match crate::disp::save_screenshot(&path, &pixels, width, height) {
                                Ok(()) => println!("iris: screenshot saved to {}", path),
                                Err(e) => println!("iris: screenshot failed: {}", e),
                            }
                        });
                    }
                }

                self.diag.fetch_and(!(Self::DIAG_LOCK_SCREEN | Self::DIAG_LOCK_RENDERER), Ordering::Relaxed);
            } else {
                // Nothing visible changed — skip the full refresh + GL upload and
                // leave the already-presented front buffer on screen. Still tick
                // the hardware VBLANK and latch cursor-Y every frame: the guest's
                // vsync timing must not depend on whether the host re-rendered.
                frames_since_render = frames_since_render.saturating_add(1);
                self.config.status.fetch_or(STATUS_VRINT, Ordering::Relaxed);
                {
                    self.diag.fetch_or(Self::DIAG_LOCK_VC2, Ordering::Relaxed);
                    let mut vc2 = self.vc2.lock();
                    vc2.regs[crate::vc2::VC2_REG_WORKING_CURSOR_Y as usize] = vc2.regs[crate::vc2::VC2_REG_CURSOR_Y_LOC as usize];
                    drop(vc2);
                    self.diag.fetch_and(!Self::DIAG_LOCK_VC2, Ordering::Relaxed);
                }
                {
                    self.diag.fetch_or(Self::DIAG_LOCK_VBLANK_CB, Ordering::Relaxed);
                    let cb = self.vblank_cb.lock().clone();
                    self.diag.fetch_and(!Self::DIAG_LOCK_VBLANK_CB, Ordering::Relaxed);
                    if let Some(cb) = cb { cb(true); }
                }
            }

            // Timing & VBLANK
            let elapsed = start.elapsed();
            if elapsed < frame_duration {
                thread::sleep(frame_duration - elapsed);
            }

            // Sleep out the remainder of the frame. VBLANK stays asserted until
            // the CPU reads STATUS, which clears STATUS_VRINT and deasserts the
            // interrupt line (matching MAME newport behaviour).
            let elapsed = start.elapsed();
            if elapsed < frame_duration {
                thread::sleep(frame_duration - elapsed);
            }
        }

        if let Some(renderer) = self.renderer.lock().as_mut() {
            renderer.stop();
        }
    }

    /// Returns `true` if the register offset was recognized and updated state.
    /// Process a register write from the GFIFO consumer.
    /// `reg_offset` is `entry.addr & !0x0800` (GO bit stripped; is_64bit bit kept).
    /// Returns `true` if the consumer loop should exit (GFIFO_EXIT sentinel received).
    #[inline(always)]
    pub(crate) fn process_register(&self, reg_offset: u32, val64: u64) -> bool {
        let ctx = unsafe { &mut *self.context.get() };
        let val = val64 as u32;
        dlog_dev!(LogModule::Rex3, "REX3 Process: Offset {:04x} ({}) Val {:08x}", reg_offset, rex3_reg_name(reg_offset), val);

        let mut exit = false;
        match reg_offset {
            // EXIT sentinel: signal the consumer loop to stop.
            GFIFO_EXIT => { exit = true; }
            // DISP_SYNC sentinel: all prior draws are done — advance the fence.
            // Display thread spins on gfifo_fence >= sampled xmap_fence.
            GFIFO_DISP_SYNC => {
                self.gfifo_fence.store(val64 as u32, Ordering::Release);
            }
            // PURE_GO sentinel (GO-only, no register write): no-op here; execute_go() is
            // triggered by the GO bit in the loop.
            GFIFO_PURE_GO_REG => {}
            // dma_read64/dma_write64's priming alias — same no-op treatment as
            // GFIFO_PURE_GO_REG (see REX3_DMA_PURE_GO doc comment).
            REX3_DMA_PURE_GO_REG => {}
            // HOSTRW: store data port value; reset shift so the draw picks up pixels from MSB.
            // The actual draw/read is triggered by execute_go() when entry.go is set.
            // 64-bit write (REX3_HOSTRW0_64 = 0x0231): store full val64 directly.
            REX3_HOSTRW64 => {
                ctx.hostrw = val64;
                ctx.hostcnt = 0;
                self.note_hostrw_write();
                #[cfg(feature = "developer")]
                if let Some(f) = self.block_log.lock().as_mut() {
                    let _ = writeln!(f, "  HOSTRW64 {:016x}", val64);
                }
            }
            REX3_HOSTRW0 => {
                // 32-bit write to HOSTRW0: update high 32 bits [63:32].
                let new_val = (ctx.hostrw & 0x0000_0000_FFFF_FFFF) | ((val64 & 0xFFFF_FFFF) << 32);
                ctx.hostrw = new_val;
                ctx.hostcnt = 0;
                self.note_hostrw_write();
                #[cfg(feature = "developer")]
                if let Some(f) = self.block_log.lock().as_mut() {
                    let _ = writeln!(f, "  HOSTRW0 {:08x} -> {:016x}", val64 as u32, new_val);
                }
            }
            REX3_HOSTRW1 => {
                // 32-bit write to HOSTRW1: update low 32 bits [31:0].
                let new_val = (ctx.hostrw & 0xFFFF_FFFF_0000_0000) | (val64 & 0xFFFF_FFFF);
                ctx.hostrw = new_val;
                ctx.hostcnt = 0;
                self.note_hostrw_write();
                #[cfg(feature = "developer")]
                if let Some(f) = self.block_log.lock().as_mut() {
                    let _ = writeln!(f, "  HOSTRW1 {:08x} -> {:016x}", val64 as u32, new_val);
                }
            }
            REX3_DRAWMODE1 => ctx.drawmode1 = DrawMode1(val),
            REX3_DRAWMODE0 => ctx.drawmode0 = DrawMode0(val),
            REX3_LSMODE => ctx.lsmode = LsMode(val),
            REX3_LSPATTERN => ctx.lspattern = val,
            REX3_LSPATSAVE => ctx.lspatsave = val,
            REX3_ZPATTERN => ctx.zpattern = val,
            REX3_LSSAVE => {
                ctx.lssave = val;
                ctx.lspatsave = ctx.lspattern;
                ctx.lsmode.set_lsrcntsave(ctx.lsmode.lsrcount());
            },
            REX3_LSRESTORE => {
                ctx.lsrestore = val;
                ctx.lspattern = ctx.lspatsave;
                ctx.lsmode.set_lsrcount(ctx.lsmode.lsrcntsave());
            },
            REX3_STEPZ => ctx.stepz = val,
            REX3_STALL0 => ctx.stall0 = val,
            REX3_STALL1 => ctx.stall1 = val,
            REX3_COLORBACK => ctx.colorback = val,
            REX3_COLORVRAM => ctx.colorvram = val,
            REX3_ALPHAREF => ctx.alpharef = val,
            REX3_SMASK0X => ctx.smask0x = val,
            REX3_SMASK0Y => ctx.smask0y = val,
            REX3_SETUP => self.setup(ctx),
            
            // Coordinate registers - convert to 16.11 format (I21F11)
            REX3_XSTART => ctx.set_xstart(from16_4_7(val)),
            REX3_YSTART => ctx.ystart = from16_4_7(val),
            REX3_XEND => ctx.xend = from16_4_7(val),
            REX3_YEND => ctx.yend = from16_4_7(val),
            REX3_XSAVE => ctx.xsave = (val as i16 as i32) << 11,
            
            REX3_XYMOVE => ctx.xymove = val,
            REX3_BRESD => ctx.bresd = val & 0x7FFFFFF,
            REX3_BRESS1 => ctx.bress1 = val & 0x1FFFF,
            REX3_BRESOCTINC1 => ctx.bresoctinc1 = BresOctInc1(val & 0x07FFFFFF & !(0xF << 20)),
            REX3_BRESRNDINC2 => ctx.bresrndinc2 = BresRndInc2(val & !(0x7 << 21)),
            REX3_BRESE1 => ctx.brese1 = val & 0xFFFF,
            REX3_BRESS2 => ctx.bress2 = val & 0x3FFFFFF,
            REX3_AWEIGHT0 => ctx.aweight0 = val,
            REX3_AWEIGHT1 => ctx.aweight1 = val,
            
            REX3_XSTARTF => ctx.set_xstart(from12_4_7(val)),
            REX3_YSTARTF => ctx.ystart = from12_4_7(val),
            REX3_XENDF | REX3_XENDF1 => ctx.xend = from12_4_7(val),
            REX3_YENDF => ctx.yend = from12_4_7(val),
            
            REX3_XSTARTI => ctx.set_xstart((val as i16 as i32) << 11), // Integer format

            REX3_XYSTARTI => {
                ctx.set_xstart(((val >> 16) as i16 as i32) << 11);
                ctx.ystart = ((val & 0xFFFF) as i16 as i32) << 11;
            }
            REX3_XYENDI => {
                ctx.xend = ((val >> 16) as i16 as i32) << 11;
                ctx.yend = ((val & 0xFFFF) as i16 as i32) << 11;
            }
            REX3_XSTARTENDI => {
                ctx.set_xstart(((val >> 16) as i16 as i32) << 11);
                ctx.xend = ((val & 0xFFFF) as i16 as i32) << 11;
            }

            REX3_COLORRED => ctx.colorred = from_color_red(val, ctx.drawmode1),
            REX3_COLORALPHA => ctx.coloralpha = from_color(val),
            REX3_COLORGRN => ctx.colorgrn = from_color(val),
            REX3_COLORBLUE => ctx.colorblue = from_color(val),
            REX3_SLOPERED => ctx.slopered = from_slope_red(val),
            REX3_SLOPEALPHA => ctx.slopealpha = from_slope(val),
            REX3_SLOPEGRN => ctx.slopegrn = from_slope(val),
            REX3_SLOPEBLUE => ctx.slopeblue = from_slope(val),
            REX3_WRMASK => {
                //if val & 0xFFFFFF == 0x6db6db {
                //    eprintln!("REX3 process_register WRMASK write: val={:08x} -> forcing 0xffffff", val);
                //    ctx.wrmask = 0xFFFFFF;
                //} else {
                    ctx.wrmask = val;
                //}
            }
            REX3_COLORI => ctx.set_colori(val),
            REX3_COLORX => ctx.colorx = from_color_red(val, ctx.drawmode1),
            REX3_SLOPERED1 => ctx.slopered = from_slope_red(val),
            REX3_SMASK1X => ctx.smask1x = val,
            REX3_SMASK1Y => ctx.smask1y = val,
            REX3_SMASK2X => ctx.smask2x = val,
            REX3_SMASK2Y => ctx.smask2y = val,
            REX3_SMASK3X => ctx.smask3x = val,
            REX3_SMASK3Y => ctx.smask3y = val,
            REX3_SMASK4X => ctx.smask4x = val,
            REX3_SMASK4Y => ctx.smask4y = val,
            REX3_TOPSCAN => ctx.topscan = val,
            REX3_XYWIN => ctx.xywin = val,
            REX3_CLIPMODE => ctx.clipmode = val,
            // Handled on the CPU thread with immediate side effects; no draw-thread action needed.
            REX3_CONFIG | REX3_STATUS | REX3_USER_STATUS |
            REX3_DCBMODE | REX3_DCBDATA0 | REX3_DCBDATA1 | REX3_DCBRESET => {}
            _ => {
                eprintln!("REX3 Write: unhandled reg {:04x} ({}), val {:08x}", reg_offset, rex3_reg_name(reg_offset), val);
            }
        }
        exit
    }

    pub fn save_framebuffers(&self, dir: &std::path::Path) -> std::io::Result<()> {
        self.save_framebuffers_named(dir, "rex3")
    }

    /// Save RGB/aux planes with a custom filename prefix (e.g. `rex3_head1`).
    pub fn save_framebuffers_named(&self, dir: &std::path::Path, prefix: &str) -> std::io::Result<()> {
        let rgb = unsafe { &*self.fb_rgb.get() };
        let mut bytes = Vec::with_capacity(rgb.len() * 4);
        for &word in rgb.iter() {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        std::fs::write(dir.join(format!("{prefix}_rgb.bin")), &bytes)?;

        let aux = unsafe { &*self.fb_aux.get() };
        bytes.clear();
        for &word in aux.iter() {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        std::fs::write(dir.join(format!("{prefix}_aux.bin")), &bytes)?;
        Ok(())
    }

    /// Clone the framebuffers (RGB and aux) into native-endian Vec<u32>
    /// buffers. Pair with `restore_framebuffers_inmem` for the in-memory
    /// rollback checkpoint; bypasses the byte-shuffle the disk path needs.
    pub fn snapshot_framebuffers_inmem(&self) -> (Vec<u32>, Vec<u32>) {
        let rgb = unsafe { &*self.fb_rgb.get() };
        let aux = unsafe { &*self.fb_aux.get() };
        (rgb.to_vec(), aux.to_vec())
    }

    /// Restore framebuffers from buffers captured by
    /// `snapshot_framebuffers_inmem`. Lengths are clamped to the actual
    /// framebuffer size.
    pub fn restore_framebuffers_inmem(&self, rgb: &[u32], aux: &[u32]) {
        let dst_rgb = unsafe { &mut *self.fb_rgb.get() };
        let n = rgb.len().min(dst_rgb.len());
        dst_rgb[..n].copy_from_slice(&rgb[..n]);

        let dst_aux = unsafe { &mut *self.fb_aux.get() };
        let n = aux.len().min(dst_aux.len());
        dst_aux[..n].copy_from_slice(&aux[..n]);
    }

    pub fn load_framebuffers(&self, dir: &std::path::Path) -> std::io::Result<()> {
        self.load_framebuffers_named(dir, "rex3")
    }

    /// Dump the full 2048x1024 VRAM for offline inspection: raw rgb/aux/did
    /// planes plus ci.png (low 8 bits of rgb) and rgb.png (full 24-bit rgb).
    /// Unlike `save_framebuffers_named`, this also captures the decoded DID
    /// plane and renders PNG previews — for `rex fbdump`, not snapshotting.
    pub fn dump_framebuffer_debug(&self, dir: &std::path::Path) -> std::io::Result<()> {
        const W: usize = 2048;
        const H: usize = 1024;

        let rgb = unsafe { &*self.fb_rgb.get() };
        let aux = unsafe { &*self.fb_aux.get() };

        let mut bytes = Vec::with_capacity(rgb.len() * 4);
        for &word in rgb.iter() { bytes.extend_from_slice(&word.to_be_bytes()); }
        std::fs::write(dir.join("rgb.bin"), &bytes)?;

        bytes.clear();
        for &word in aux.iter() { bytes.extend_from_slice(&word.to_be_bytes()); }
        std::fs::write(dir.join("aux.bin"), &bytes)?;

        let did = self.screen.lock().did.clone();
        std::fs::write(dir.join("did.bin"), &did)?;

        // ci.png: low 8 bits of rgb (CI/plane-depth index) as grayscale.
        let mut ci_rows = Vec::with_capacity(W * H);
        for &word in rgb.iter() {
            ci_rows.push((word & 0xFF) as u8);
        }
        write_png_gray(&dir.join("ci.png"), &ci_rows, W, H)?;

        // rgb.png: full 24-bit rgb (word is 24-bit BGR: bits[7:0]=B [15:8]=G [23:16]=R).
        let mut rgb_rows = Vec::with_capacity(W * H * 3);
        for &word in rgb.iter() {
            rgb_rows.push(((word >> 16) & 0xFF) as u8); // R
            rgb_rows.push(((word >>  8) & 0xFF) as u8); // G
            rgb_rows.push(( word        & 0xFF) as u8); // B
        }
        write_png_rgb(&dir.join("rgb.png"), &rgb_rows, W, H)?;

        Ok(())
    }

    pub fn load_framebuffers_named(&self, dir: &std::path::Path, prefix: &str) -> std::io::Result<()> {
        let path_rgb = dir.join(format!("{prefix}_rgb.bin"));
        if path_rgb.exists() {
            let bytes = std::fs::read(path_rgb)?;
            let rgb = unsafe { &mut *self.fb_rgb.get() };
            let words = bytes.len() / 4;
            let count = words.min(rgb.len());
            for i in 0..count {
                let b = &bytes[i * 4..(i + 1) * 4];
                rgb[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
            }
        }

        let path_aux = dir.join(format!("{prefix}_aux.bin"));
        if path_aux.exists() {
            let bytes = std::fs::read(path_aux)?;
            let aux = unsafe { &mut *self.fb_aux.get() };
            let words = bytes.len() / 4;
            let count = words.min(aux.len());
            for i in 0..count {
                let b = &bytes[i * 4..(i + 1) * 4];
                aux[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        Ok(())
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_lock_fn;
        let r = self.clone();
        register_lock_fn("rex3::dcb",              move || r.dcb.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::vc2",              move || r.vc2.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::xmap0",            move || r.xmap0.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::xmap1",            move || r.xmap1.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::cmap0",            move || r.cmap0.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::cmap1",            move || r.cmap1.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::processor_thread", move || r.processor_thread.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::refresh_thread",   move || r.refresh_thread.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::vblank_cb",        move || r.vblank_cb.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::renderer",         move || r.renderer.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::debug_state",      move || r.debug_state.is_locked());
        crate::locks::register_mutex("rex3::screen", &self.screen);
    }
}

impl Device for Rex3 {
    fn step(&self, cycles: u64) {
        self.clock.fetch_add(cycles, Ordering::Relaxed);
    }

    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);

        // Send exit command to ensure processor thread wakes up and terminates.
        self.gfifo_push(GFIFO_EXIT, 0);

        if let Some(handle) = self.processor_thread.lock().take() {
            let _ = handle.join();
        }

        #[cfg(feature = "rex-jit")]
        if let Some(ref jit) = self.rex_jit {
            jit.save_profile();
        }

        if let Some(handle) = self.refresh_thread.lock().take() {
            let _ = handle.join();
        }

        #[cfg(feature = "developer")]
        eprintln!("REX3: GFIFO high-watermark = {} / {} entries",
            self.gfifo_hwm.load(Ordering::Relaxed), GFIFO_DEPTH);
    }

    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) { return; }

        let rex3 = unsafe { std::mem::transmute::<&Rex3, &'static Rex3>(self) };

        *self.processor_thread.lock() = Some(thread::Builder::new().name("REX3-Processor".to_string()).spawn(move || {
            crate::thread_affinity::pin_current(crate::thread_affinity::PerfRole::Rex3Processor);
            rex3.register_processor()
        }).unwrap());

        *self.refresh_thread.lock() = Some(thread::Builder::new().name("REX3-Refresh".to_string()).spawn(move || {
            crate::thread_affinity::pin_current(crate::thread_affinity::PerfRole::Rex3Refresh);
            rex3.refresh_loop();
        }).unwrap());
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    fn get_clock(&self) -> u64 {
        self.clock.load(Ordering::Relaxed)
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        #[allow(unused_mut)]
        let mut cmds = vec![
            ("rex".to_string(), "REX3 commands: rex status | rex jit <on|off|status|list> | rex jit <disable|enable> <dm0> <dm1> | rex debug <on|off> [DEV] | rex cmap <on|off> | rex buslog <on|off> [DEV] | rex fbdump [DIR]".to_string()),
            ("dcb".to_string(), "DCB commands: dcb debug <on|off> [DEV]".to_string()),
            ("vc2".to_string(), "VC2 commands: vc2 status | vc2 ramdump | vc2 debug <on|off> [DEV]".to_string()),
            ("xmap".to_string(), "XMAP commands: xmap status | xmap debug <on|off> [DEV]".to_string()),
            ("cmap".to_string(), "CMAP commands: cmap status | cmap debug <on|off> [DEV]".to_string()),
            ("bt445".to_string(), "BT445 RAMDAC: bt445 status | bt445 identity (reset palette to linear ramp) | bt445 debug <on|off> [DEV]".to_string()),
            ("disp".to_string(), "Display debug: disp status | disp debug <on|off> | disp compositor <gl|sw>".to_string()),
        ];
        #[cfg(feature = "developer")]
        cmds.extend([
            ("block".to_string(), "Block draw logging: block debug <on|off> [DEV]".to_string()),
            ("draw".to_string(), "Draw debug overlay: draw debug <on|off> [DEV]".to_string()),
        ]);
        cmds
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        if args.is_empty() {
            return Err(format!("Usage: {} debug <on|off> | {} status", cmd, cmd));
        }

        if cmd == "rex" && args[0] == "status" {
            let ctx = unsafe { &*self.context.get() };

            let x_win = ((ctx.xywin >> 16) & 0xFFFF) as i16;
            let y_win = (ctx.xywin & 0xFFFF) as i16;
            let x_move = (ctx.xymove >> 16) as i16;
            let y_move = (ctx.xymove & 0xFFFF) as i16;

            let dm0 = ctx.drawmode0;
            let dm1 = ctx.drawmode1;
            let octant = ctx.bresoctinc1.octant();

            writeln!(writer, "=== REX3 Drawing Register State ===").unwrap();
            writeln!(writer, "DRAWMODE0 : {:08x}  {}", dm0.0, decode_dm0(dm0.0)).unwrap();
            writeln!(writer, "DRAWMODE1 : {:08x}  {}", dm1.0, decode_dm1(dm1.0)).unwrap();
            writeln!(writer, "XSTART    : {:.3}  YSTART : {:.3}", ctx.xstart as f32 / 2048.0, ctx.ystart as f32 / 2048.0).unwrap();
            writeln!(writer, "XEND      : {:.3}  YEND   : {:.3}", ctx.xend as f32 / 2048.0, ctx.yend as f32 / 2048.0).unwrap();
            writeln!(writer, "XSAVE     : {:.3}  OCTANT : {:03b} (xdec={} ydec={})", ctx.xsave as f32 / 2048.0,
                octant, (octant & OCTANT_XDEC != 0) as u8, (octant & OCTANT_YDEC != 0) as u8).unwrap();
            writeln!(writer, "XYWIN     : {:08x}  (x={}, y={})", ctx.xywin, x_win, y_win).unwrap();
            writeln!(writer, "XYMOVE    : {:08x}  (dx={}, dy={})", ctx.xymove, x_move, y_move).unwrap();
            writeln!(writer, "COLORBACK : {:08x}", ctx.colorback).unwrap();
            writeln!(writer, "COLORVRAM : {:08x}", ctx.colorvram).unwrap();
            writeln!(writer, "COLORI    : {:08x}", ctx.get_colori()).unwrap();
            writeln!(writer, "COLORRED  : {:08x}  COLORGRN : {:08x}  COLORBLUE : {:08x}",
                ctx.colorred, ctx.colorgrn, ctx.colorblue).unwrap();
            writeln!(writer, "WRMASK    : {:08x}", ctx.wrmask).unwrap();
            writeln!(writer, "LSMODE    : {:08x}  LSPATTERN : {:08x}", ctx.lsmode.0, ctx.lspattern).unwrap();
            writeln!(writer, "ZPATTERN  : {:08x}", ctx.zpattern).unwrap();
            writeln!(writer, "CLIPMODE  : {:08x}  (ensmask={:05b} cidmatch={:04b})",
                ctx.clipmode,
                ctx.clipmode & CLIPMODE_ENSMASK_MASK,
                (ctx.clipmode >> CLIPMODE_CIDMATCH_SHIFT) & 0xF).unwrap();
            writeln!(writer, "SMASK0X   : {:08x}  SMASK0Y : {:08x}", ctx.smask0x, ctx.smask0y).unwrap();
            writeln!(writer, "SMASK1X   : {:08x}  SMASK1Y : {:08x}", ctx.smask1x, ctx.smask1y).unwrap();
            writeln!(writer, "SMASK2X   : {:08x}  SMASK2Y : {:08x}", ctx.smask2x, ctx.smask2y).unwrap();
            writeln!(writer, "SMASK3X   : {:08x}  SMASK3Y : {:08x}", ctx.smask3x, ctx.smask3y).unwrap();
            writeln!(writer, "SMASK4X   : {:08x}  SMASK4Y : {:08x}", ctx.smask4x, ctx.smask4y).unwrap();
            writeln!(writer, "TOPSCAN   : {:08x}", ctx.topscan).unwrap();
            writeln!(writer, "STATUS    : {:08x}  CONFIG : {:08x}",
                self.config.status.load(Ordering::Relaxed),
                self.config.config.load(Ordering::Relaxed)).unwrap();
            let gfxbusy = self.gfxbusy.load(Ordering::Relaxed);
            writeln!(writer, "DRAW BUSY : {}  GFIFO : {}/{} entries used",
                if gfxbusy { "YES" } else { "no" },
                self.gfifo.len(), GFIFO_DEPTH).unwrap();
            // `running` alone only proves start()/stop() were called in the
            // expected order — it says nothing about whether the processor
            // thread is actually alive right now (a panic inside
            // register_processor, or a start()/stop() mismatch specific to
            // some caller, would leave `running=true` with no thread ever
            // spawned, or with a dead one). `processor_thread`/
            // `refresh_thread`'s own `Option` presence is the more direct
            // signal: `Some` iff `start()` actually spawned and hasn't been
            // joined away by `stop()` yet. Surfaced here specifically
            // because gfxbusy=false with a non-empty GFIFO (should be
            // momentarily impossible while register_processor is alive and
            // looping — it always sets gfxbusy=true before processing any
            // entry it sees) is exactly the live signature of a GFIFO
            // sitting un-drained because nothing is actually consuming it.
            let processor_alive = self.processor_thread.lock().is_some();
            let refresh_alive = self.refresh_thread.lock().is_some();
            writeln!(writer, "THREADS   : running={} processor_thread={} refresh_thread={}",
                self.is_running(),
                if processor_alive { "alive" } else { "NOT RUNNING" },
                if refresh_alive { "alive" } else { "NOT RUNNING" }).unwrap();
            let jit_go   = self.jit_go_count.load(Ordering::Relaxed);
            let interp_go = self.interp_go_count.load(Ordering::Relaxed);
            let total_go  = jit_go + interp_go;
            let jit_pct   = if total_go > 0 { jit_go * 100 / total_go } else { 0 };
            writeln!(writer, "GO TOTAL  : {}  JIT : {} ({}%)  INTERP : {}",
                total_go, jit_go, jit_pct, interp_go).unwrap();
            #[cfg(feature = "rex-jit")]
            {
                if let Some(ref jit) = self.rex_jit {
                    let enabled = self.jit_enabled.load(Ordering::Relaxed);
                    writeln!(writer, "JIT       : {}  compiled={} queued={}  (rex jit status for details)",
                        if enabled { "enabled" } else { "DISABLED" },
                        jit.compiled_count(), jit.queued_count()).unwrap();
                } else {
                    writeln!(writer, "JIT       : not initialised").unwrap();
                }
            }
            #[cfg(not(feature = "rex-jit"))]
            writeln!(writer, "JIT       : not compiled in").unwrap();
            return Ok(());
        }

        if cmd == "rex" && args[0] == "fbdump" {
            let dir = std::path::Path::new(args.get(1).map(|s| *s).unwrap_or("."));
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            self.dump_framebuffer_debug(dir).map_err(|e| e.to_string())?;
            writeln!(writer, "REX3 framebuffer dump written to {}", dir.display()).unwrap();
            writeln!(writer, "  rgb.bin (2048x1024 x u32 BE, 24-bit BGR)").unwrap();
            writeln!(writer, "  aux.bin (2048x1024 x u32 BE, raw aux plane)").unwrap();
            writeln!(writer, "  did.bin (2048x1024 x u8, decoded DID)").unwrap();
            writeln!(writer, "  ci.png  (low 8 bits of rgb, grayscale)").unwrap();
            writeln!(writer, "  rgb.png (full 24-bit rgb)").unwrap();
            return Ok(());
        }

        if cmd == "rex" && args[0] == "diag" {
            let d = self.diag.load(Ordering::Relaxed);
            writeln!(writer, "=== REX3 Thread Activity (diag={:016x}) ===", d).unwrap();
            // Mutex bits
            let locks = [
                (Self::DIAG_LOCK_CONFIG,      "config"),
                (Self::DIAG_LOCK_VC2,         "vc2"),
                (Self::DIAG_LOCK_CMAP0,       "cmap0"),
                (Self::DIAG_LOCK_CMAP1,       "cmap1"),
                (Self::DIAG_LOCK_XMAP0,       "xmap0"),
                (Self::DIAG_LOCK_XMAP1,       "xmap1"),
                (Self::DIAG_LOCK_SCREEN,      "screen"),
                (Self::DIAG_LOCK_RENDERER,    "renderer"),
                (Self::DIAG_LOCK_VBLANK_CB,   "vblank_cb"),
                (Self::DIAG_LOCK_DEBUG_STATE, "debug_state"),
                (Self::DIAG_LOCK_DCB,         "dcb"),
            ];
            let loops = [
                (Self::DIAG_LOOP_FB_COPY,    "fb_copy"),
                (Self::DIAG_LOOP_VC2_COPY,   "vc2_copy"),
                (Self::DIAG_LOOP_CMAP_COPY,  "cmap_copy"),
                (Self::DIAG_LOOP_XMAP_COPY,  "xmap_copy"),
                (Self::DIAG_LOOP_VID_TIMINGS,"vid_timings"),
                (Self::DIAG_LOOP_DECODE_DID, "decode_did"),
                (Self::DIAG_LOOP_PIXEL_CONV, "pixel_conv"),
                (Self::DIAG_LOOP_GL_RENDER,  "gl_render"),
                (Self::DIAG_LOOP_DRAW_BLOCK,  "draw_block"),
                (Self::DIAG_LOOP_EXECUTE_GO,  "execute_go"),
            ];
            write!(writer, "  Locks held :").unwrap();
            let mut any = false;
            for (bit, name) in &locks { if d & bit != 0 { write!(writer, " {}", name).unwrap(); any = true; } }
            if !any { write!(writer, " (none)").unwrap(); }
            writeln!(writer).unwrap();
            write!(writer, "  Loops active:").unwrap();
            any = false;
            for (bit, name) in &loops { if d & bit != 0 { write!(writer, " {}", name).unwrap(); any = true; } }
            if !any { write!(writer, " (none)").unwrap(); }
            writeln!(writer).unwrap();
            writeln!(writer, "  gfxbusy={} gfifo_pending={}",
                self.gfxbusy.load(Ordering::Relaxed),
                self.gfifo.len()).unwrap();
            return Ok(());
        }

        if cmd == "rex" && args[0] == "cmap" {
            let val = match args.get(1).map(|s| *s) {
                Some("on")  => true,
                Some("off") => false,
                _ => return Err("Usage: rex cmap <on|off>".to_string()),
            };
            self.show_cmap.store(val, Ordering::Relaxed);
            writeln!(writer, "CMAP overlay {}", if val { "enabled" } else { "disabled" }).unwrap();
            return Ok(());
        }

        if cmd == "xmap" && (args[0] == "status" || args[0] == "dump") {
            self.xmap0.lock().print_status("XMAP0", &mut writer);
            self.xmap1.lock().print_status("XMAP1", &mut writer);
            return Ok(());
        }

        if cmd == "cmap" && (args[0] == "dump" || args[0] == "status") {
            self.cmap0.lock().print_status("CMAP0", &mut writer);
            self.cmap1.lock().print_status("CMAP1", &mut writer);
            return Ok(());
        }

        if cmd == "bt445" && (args[0] == "status" || args[0] == "dump") {
            self.bt445.lock().print_status(&mut writer);
            return Ok(());
        }

        if cmd == "bt445" && args[0] == "identity" {
            let mut dac = self.bt445.lock();
            for i in 0..crate::bt445::BT445_PALETTE_SIZE {
                dac.palette[i] = [i as u8, i as u8, i as u8];
            }
            dac.dirty = true;
            writeln!(writer, "BT445: palette set to identity ramp").unwrap();
            return Ok(());
        }

        #[cfg(feature = "developer")]
        if cmd == "rex" && args[0] == "buslog" {
            let val = match args.get(1).map(|s| *s) {
                Some("on")  => true,
                Some("off") => false,
                _ => return Err("Usage: rex buslog <on|off>".to_string()),
            };
            let mut log = self.rex3_log.lock();
            if val {
                match std::fs::File::create("rex3.log") {
                    Ok(f) => { *log = Some(f); writeln!(writer, "REX3 GFIFO logging enabled (rex3.log)").unwrap(); }
                    Err(e) => { writeln!(writer, "Failed to open rex3.log: {}", e).unwrap(); }
                }
            } else {
                *log = None;
                writeln!(writer, "REX3 GFIFO logging disabled").unwrap();
            }
            return Ok(());
        }
        #[cfg(not(feature = "developer"))]
        if cmd == "rex" && args[0] == "buslog" {
            return Err("rex buslog is only available in developer builds".to_string());
        }

        #[cfg(feature = "rex-jit")]
        if cmd == "rex" && args[0] == "jit" {
            match args.get(1).copied() {
                Some("on") | Some("off") => {
                    let val = args[1] == "on";
                    self.jit_enabled.store(val, Ordering::Relaxed);
                    self.jit_last.set((0, 0, 0, None));
                    writeln!(writer, "REX JIT dispatch: {}", if val { "enabled" } else { "disabled" }).unwrap();
                }
                Some("status") => {
                    if let Some(ref jit) = self.rex_jit {
                        let enabled = self.jit_enabled.load(Ordering::Relaxed);
                        let shaders = jit.shader_list();
                        let compiled: Vec<_> = shaders.iter().filter(|s| s.status == "compiled" || s.status == "disabled").collect();
                        let total_bytes: u32 = compiled.iter().map(|s| s.code_bytes).sum();
                        writeln!(writer, "REX3 JIT: {}  compiled={}  queued={}  failed={}  code={} bytes",
                            if enabled { "enabled" } else { "DISABLED" },
                            compiled.len(), jit.queued_count(),
                            shaders.iter().filter(|s| s.status == "failed").count(),
                            total_bytes).unwrap();
                        if !compiled.is_empty() {
                            #[cfg(feature = "developer")]
                            writeln!(writer, "{:>8}  {:>10}  {:>10}  {:>10}  {:>6}  {:>8}  {}",
                                "status", "dm0", "dm1", "cm", "bytes", "hits", "description").unwrap();
                            #[cfg(not(feature = "developer"))]
                            writeln!(writer, "{:>8}  {:>10}  {:>10}  {:>10}  {:>6}  {}",
                                "status", "dm0", "dm1", "cm", "bytes", "description").unwrap();
                            for s in &shaders {
                                if s.status == "compiled" || s.status == "disabled" {
                                    #[cfg(feature = "developer")]
                                    writeln!(writer, "{:>8}  {:#010x}  {:#010x}  {:#010x}  {:>6}  {:>8}  {}  |  {}",
                                        s.status, s.dm0, s.dm1, s.cm, s.code_bytes, s.hit_count,
                                        decode_dm0(s.dm0), decode_dm1(s.dm1)).unwrap();
                                    #[cfg(not(feature = "developer"))]
                                    writeln!(writer, "{:>8}  {:#010x}  {:#010x}  {:#010x}  {:>6}  {}  |  {}",
                                        s.status, s.dm0, s.dm1, s.cm, s.code_bytes,
                                        decode_dm0(s.dm0), decode_dm1(s.dm1)).unwrap();
                                }
                            }
                        }
                    } else {
                        writeln!(writer, "REX JIT: not initialised").unwrap();
                    }
                }
                Some("list") => {
                    if let Some(ref jit) = self.rex_jit {
                        let shaders = jit.shader_list();
                        if shaders.is_empty() {
                            writeln!(writer, "No shaders compiled yet.").unwrap();
                        } else {
                            writeln!(writer, "{:>8}  {:>10}  {:>10}  {:>10}  {:>6}  {}", "status", "dm0", "dm1", "cm", "bytes", "description").unwrap();
                            for s in &shaders {
                                writeln!(writer, "{:>8}  {:#010x}  {:#010x}  {:#010x}  {:>6}  {}  |  {}",
                                    s.status, s.dm0, s.dm1, s.cm, s.code_bytes,
                                    decode_dm0(s.dm0), decode_dm1(s.dm1)).unwrap();
                            }
                        }
                    }
                }
                Some("disable") | Some("enable") => {
                    let enable = args[1] == "enable";
                    let dm0_s = args.get(2).ok_or("Usage: rex jit <disable|enable> <dm0_hex> <dm1_hex> [cm_hex]")?;
                    let dm1_s = args.get(3).ok_or("Usage: rex jit <disable|enable> <dm0_hex> <dm1_hex> [cm_hex]")?;
                    let dm0 = u32::from_str_radix(dm0_s.trim_start_matches("0x"), 16)
                        .map_err(|_| format!("bad dm0: {dm0_s}"))?;
                    let dm1 = u32::from_str_radix(dm1_s.trim_start_matches("0x"), 16)
                        .map_err(|_| format!("bad dm1: {dm1_s}"))?;
                    let cm = if let Some(cm_s) = args.get(4) {
                        u32::from_str_radix(cm_s.trim_start_matches("0x"), 16)
                            .map_err(|_| format!("bad cm: {cm_s}"))?
                    } else { 0 };
                    if let Some(ref jit) = self.rex_jit {
                        if enable { jit.enable_shader(dm0, dm1, cm); } else { jit.disable_shader(dm0, dm1, cm); }
                        self.jit_last.set((0, 0, 0, None));
                        writeln!(writer, "Shader dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x}: {}",
                            if enable { "enabled" } else { "disabled" }).unwrap();
                    }
                }
                _ => return Err("Usage: rex jit <on|off|status|list> | rex jit <disable|enable> <dm0_hex> <dm1_hex> [cm_hex]".to_string()),
            }
            return Ok(());
        }

        if cmd == "disp" && args[0] == "debug" {
            let val = match args.get(1).map(|s| *s) {
                Some("on")  => true,
                Some("off") => false,
                _ => return Err("Usage: disp debug <on|off>".to_string()),
            };
            self.show_disp_debug.store(val, Ordering::Relaxed);
            writeln!(writer, "Display debug overlay {}", if val { "enabled" } else { "disabled" }).unwrap();
            return Ok(());
        }

        if cmd == "disp" && args[0] == "status" {
            let screen = self.screen.lock();
            writeln!(writer, "=== Display ===").unwrap();
            writeln!(writer, "  resolution: {}x{}", screen.width, screen.height).unwrap();
            writeln!(writer, "  topscan: {:03x}  (fb row {} maps to display row 0)", screen.topscan, (screen.topscan + 1) & 0x3FF).unwrap();
            writeln!(writer, "  show_disp_debug: {}", self.show_disp_debug.load(Ordering::Relaxed)).unwrap();
            let status = if let Some(r) = self.renderer.lock().as_ref() {
                r.compositor_status()
            } else {
                "no renderer".to_string()
            };
            writeln!(writer, "  {}", status).unwrap();
            return Ok(());
        }

        if cmd == "disp" && args[0] == "compositor" {
            let use_gl = match args.get(1).map(|s| *s) {
                Some("gl") => true,
                Some("sw") => false,
                _ => return Err("Usage: disp compositor <gl|sw>".to_string()),
            };
            let active = if let Some(renderer) = self.renderer.lock().as_mut() {
                renderer.switch_compositor(use_gl)
            } else {
                "no renderer"
            };
            writeln!(writer, "Compositor: {}", active).unwrap();
            return Ok(());
        }

        if cmd == "vc2" && args[0] == "status" {
            self.vc2.lock().print_status(&mut writer);
            return Ok(());
        }

        if cmd == "vc2" && args[0] == "ramdump" {
            self.vc2.lock().dump_ram(&mut writer);
            return Ok(());
        }

        if args[0] == "debug" {
            let val = match args.get(1).map(|s| *s) {
                Some("on") => true,
                Some("off") => false,
                _ => return Err(format!("Usage: {} debug <on|off>", cmd)),
            };

            match cmd {
                "rex" => {
                    self.debug.store(val, Ordering::Relaxed);
                    writeln!(writer, "REX3 debug {}", if val { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }
                "dcb" => {
                    if val { devlog().enable(LogModule::Dcb); } else { devlog().disable(LogModule::Dcb); }
                    writeln!(writer, "DCB debug {}", if val { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }
                #[cfg(feature = "developer")]
                "block" => {
                    self.block_debug.store(val, Ordering::Relaxed);
                    let mut log = self.block_log.lock();
                    if val {
                        match std::fs::File::create("block.log") {
                            Ok(f) => { *log = Some(f); writeln!(writer, "Block debug enabled, logging to block.log").unwrap(); }
                            Err(e) => { writeln!(writer, "Block debug enabled but failed to open log: {}", e).unwrap(); }
                        }
                    } else {
                        *log = None;
                        writeln!(writer, "Block debug disabled").unwrap();
                    }
                    return Ok(());
                }
                #[cfg(feature = "developer")]
                "draw" => {
                    self.draw_debug.store(val, Ordering::Relaxed);
                    if !val { self.draw_ring.lock().count = 0; }
                    writeln!(writer, "Draw debug overlay {}", if val { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }
                "vc2" => {
                    if val { devlog().enable(LogModule::Vc2); } else { devlog().disable(LogModule::Vc2); }
                    writeln!(writer, "VC2 debug {}", if val { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }
                "xmap" => {
                    if val { devlog().enable(LogModule::Xmap); } else { devlog().disable(LogModule::Xmap); }
                    writeln!(writer, "XMAP debug {}", if val { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }
                "cmap" => {
                    if val { devlog().enable(LogModule::Cmap); } else { devlog().disable(LogModule::Cmap); }
                    writeln!(writer, "CMAP debug {}", if val { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }
                "bt445" => {
                    self.bt445.lock().debug = val;
                    if val { devlog().enable(LogModule::Bt445); } else { devlog().disable(LogModule::Bt445); }
                    writeln!(writer, "BT445 debug {}", if val { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }

                _ => {}
            }
        }
        
        Err("Command not found".to_string())
    }
}

impl BusDevice for Rex3 {
    fn read32(&self, addr: u32) -> BusRead32 {
        let offset     = addr & (REX3_SIZE - 1);
        let reg_offset = offset & !0x0800;
        let is_go      = offset & 0x0800 != 0;

        // busy_or_val!(expr) — if pipeline is idle evaluate expr, else return busy.
        // Used for registers that require the GFIFO to be drained before reading.
        macro_rules! busy_or_val {
            ($val:expr) => {{
                if self.gfxbusy.load(Ordering::Acquire) || !self.gfifo.is_empty() {
                    return BusRead32::busy();
                }
                BusRead32::ok($val)
            }};
        }

        let result = match reg_offset {
            REX3_CONFIG => BusRead32::ok(self.config.config.load(Ordering::Relaxed) & 0x1FFFFF),

            REX3_STATUS | REX3_USER_STATUS => {
                let mut val = self.config.status.load(Ordering::Relaxed) & 0xFFFFF;
                val |= 3 << STATUS_VERSION_SHIFT;
                let pending = self.gfifo.len();
                if self.gfxbusy.load(Ordering::Acquire) || pending > 0 {
                    val |= STATUS_GFXBUSY;
                } else {
                    val &= !STATUS_GFXBUSY;
                }
                let level: u32 = if pending == 0 {
                    0
                } else {
                    (pending.saturating_sub(GFIFO_DEPTH - GFIFO_HW_DEPTH) as u32).max(1)
                };
                val = (val & !STATUS_GFIFOLEVEL_MASK) | ((level << STATUS_GFIFOLEVEL_SHIFT) & STATUS_GFIFOLEVEL_MASK);
                if reg_offset == REX3_STATUS {
                    let had_vrint = self.config.status.fetch_and(!STATUS_VRINT, Ordering::Relaxed) & STATUS_VRINT != 0;
                    if had_vrint {
                        let cb = self.vblank_cb.lock().clone();
                        if let Some(cb) = cb { cb(false); }
                    }
                    let had_gfifo = self.config.status.fetch_and(!STATUS_GFIFO_INT, Ordering::Relaxed) & STATUS_GFIFO_INT != 0;
                    if had_gfifo {
                        let cb = self.fifo_full_cb.lock().clone();
                        if let Some(cb) = cb { cb(false); }
                    }
                    let dcb = self.dcb.lock();
                    if let Some(until) = dcb.backbusy_until {
                        if std::time::Instant::now() < until { val |= STATUS_BACKBUSY; }
                    }
                }
                BusRead32::ok(val & 0xFFFFF)
            }

            REX3_DCBMODE => {
                self.diag.fetch_or(Self::DIAG_LOCK_DCB, Ordering::Relaxed);
                let val = self.dcb.lock().dcbmode & 0x1FFFFFFF;
                self.diag.fetch_and(!Self::DIAG_LOCK_DCB, Ordering::Relaxed);
                dlog_dev!(LogModule::Dcb, "DCB Mode Read -> {:08x}", val);
                BusRead32::ok(val)
            }
            REX3_DCBDATA0 | REX3_DCBDATA1 => BusRead32::ok(self.dcb_read()),

            // Context registers: stall until pipeline idle.
            _ => {
                let ctx = unsafe { &*self.context.get() };
                match reg_offset {
                    REX3_HOSTRW0      => busy_or_val!((ctx.hostrw >> 32) as u32),
                    REX3_HOSTRW1      => busy_or_val!(ctx.hostrw as u32),
                    REX3_DRAWMODE1    => busy_or_val!(ctx.drawmode1.0),
                    REX3_DRAWMODE0    => busy_or_val!(ctx.drawmode0.0 & 0xFFFFFF),
                    REX3_LSMODE       => busy_or_val!(ctx.lsmode.0 & 0x0FFFFFFF),
                    REX3_LSPATTERN    => busy_or_val!(ctx.lspattern),
                    REX3_LSPATSAVE    => busy_or_val!(ctx.lspatsave),
                    REX3_ZPATTERN     => busy_or_val!(ctx.zpattern),
                    REX3_LSSAVE       => busy_or_val!(ctx.lssave),
                    REX3_LSRESTORE    => busy_or_val!(ctx.lsrestore),
                    REX3_STEPZ        => busy_or_val!(ctx.stepz),
                    REX3_STALL0       => busy_or_val!(ctx.stall0),
                    REX3_STALL1       => busy_or_val!(ctx.stall1),
                    REX3_COLORBACK    => busy_or_val!(ctx.colorback),
                    REX3_COLORVRAM    => busy_or_val!(ctx.colorvram),
                    REX3_ALPHAREF     => busy_or_val!(ctx.alpharef & 0xFF),
                    REX3_SMASK0X      => busy_or_val!(ctx.smask0x),
                    REX3_SMASK0Y      => busy_or_val!(ctx.smask0y),
                    REX3_XSTART       => busy_or_val!(to16_4_7(ctx.xstart)),
                    REX3_YSTART       => busy_or_val!(to16_4_7(ctx.ystart)),
                    REX3_XEND         => busy_or_val!(to16_4_7(ctx.xend)),
                    REX3_YEND         => busy_or_val!(to16_4_7(ctx.yend)),
                    REX3_XSAVE        => busy_or_val!((ctx.xsave >> 11) as u32 & 0xFFFF),
                    REX3_XYMOVE       => busy_or_val!(ctx.xymove),
                    REX3_BRESD        => busy_or_val!(ctx.bresd & 0x7FFFFFF),
                    REX3_BRESS1       => busy_or_val!(ctx.bress1 & 0x1FFFF),
                    REX3_BRESOCTINC1  => busy_or_val!(ctx.bresoctinc1.0),
                    REX3_BRESRNDINC2  => busy_or_val!(ctx.bresrndinc2.0),
                    REX3_BRESE1       => busy_or_val!(ctx.brese1 & 0xFFFF),
                    REX3_BRESS2       => busy_or_val!(ctx.bress2 & 0x3FFFFFF),
                    REX3_AWEIGHT0     => busy_or_val!(ctx.aweight0),
                    REX3_AWEIGHT1     => busy_or_val!(ctx.aweight1),
                    REX3_XSTARTF      => busy_or_val!(to12_4_7(ctx.xstart)),
                    REX3_YSTARTF      => busy_or_val!(to12_4_7(ctx.ystart)),
                    REX3_XENDF        => busy_or_val!(to12_4_7(ctx.xend)),
                    REX3_YENDF        => busy_or_val!(to12_4_7(ctx.yend)),
                    REX3_XSTARTI      => busy_or_val!((ctx.xstart >> 11) as u32 & 0xFFFF),
                    REX3_XENDF1       => busy_or_val!(to12_4_7(ctx.xend)),
                    REX3_XYSTARTI     => busy_or_val!({
                        let x = (ctx.xstart >> 11) as u16 as u32;
                        let y = (ctx.ystart >> 11) as u16 as u32;
                        (x << 16) | (y & 0xFFFF)
                    }),
                    REX3_XYENDI       => busy_or_val!({
                        let x = (ctx.xend >> 11) as u16 as u32;
                        let y = (ctx.yend >> 11) as u16 as u32;
                        (x << 16) | (y & 0xFFFF)
                    }),
                    REX3_XSTARTENDI   => busy_or_val!({
                        let start = (ctx.xstart >> 11) as u16 as u32;
                        let end   = (ctx.xend   >> 11) as u16 as u32;
                        (start << 16) | (end & 0xFFFF)
                    }),
                    REX3_COLORRED     => busy_or_val!(to_color_red(ctx.colorred, ctx.drawmode1)),
                    REX3_COLORALPHA   => busy_or_val!(to_color(ctx.coloralpha)),
                    REX3_COLORGRN     => busy_or_val!(to_color(ctx.colorgrn)),
                    REX3_COLORBLUE    => busy_or_val!(to_color(ctx.colorblue)),
                    REX3_SLOPERED     => busy_or_val!(to_slope_red(ctx.slopered)),
                    REX3_SLOPEALPHA   => busy_or_val!(to_slope(ctx.slopealpha)),
                    REX3_SLOPEGRN     => busy_or_val!(to_slope(ctx.slopegrn)),
                    REX3_SLOPEBLUE    => busy_or_val!(to_slope(ctx.slopeblue)),
                    REX3_WRMASK       => busy_or_val!(ctx.wrmask & 0xFFFFFF),
                    REX3_COLORI       => busy_or_val!(ctx.get_colori() & 0xFFFFFF),
                    REX3_COLORX       => busy_or_val!(to_color_red(ctx.colorx, ctx.drawmode1) & 0xFFFFFF),
                    REX3_SLOPERED1    => busy_or_val!(to_slope_red(ctx.slopered)),
                    REX3_SMASK1X      => busy_or_val!(ctx.smask1x),
                    REX3_SMASK1Y      => busy_or_val!(ctx.smask1y),
                    REX3_SMASK2X      => busy_or_val!(ctx.smask2x),
                    REX3_SMASK2Y      => busy_or_val!(ctx.smask2y),
                    REX3_SMASK3X      => busy_or_val!(ctx.smask3x),
                    REX3_SMASK3Y      => busy_or_val!(ctx.smask3y),
                    REX3_SMASK4X      => busy_or_val!(ctx.smask4x),
                    REX3_SMASK4Y      => busy_or_val!(ctx.smask4y),
                    REX3_TOPSCAN      => busy_or_val!(ctx.topscan & 0x3FF),
                    REX3_XYWIN        => busy_or_val!(ctx.xywin),
                    REX3_CLIPMODE     => busy_or_val!(ctx.clipmode & 0x1FFF),
                    _ => {
                        eprintln!("REX3 Read32: unhandled reg {:04x} ({})", reg_offset, rex3_reg_name(reg_offset));
                        BusRead32::ok(0)
                    }
                }
            }
        };

        // STATUS is polled constantly — suppress from debug log.
        if reg_offset != REX3_STATUS && reg_offset != REX3_USER_STATUS {
            if result.is_ok() {
                let val = result.data;
                let mut dbg = self.debug_state.lock();
                if dbg.last_offset == Some(offset) && dbg.last_val == val {
                    dbg.count += 1;
                } else {
                    if dbg.count > 0 { dlog_dev!(LogModule::Rex3, "... repeated {} times", dbg.count); }
                    dbg.last_offset = Some(offset);
                    dbg.last_val = val;
                    dbg.count = 0;
                    dlog_dev!(LogModule::Rex3, "REX3 Read32: Offset {:04x} (Reg {:04x} {}) -> {:08x}", offset, reg_offset, rex3_reg_name(reg_offset), val);
                }
            } else {
                dlog_dev!(LogModule::Rex3, "REX3 Read32: Offset {:04x} (Reg {:04x} {}) -> err {:08x}", offset, reg_offset, rex3_reg_name(reg_offset), result.status);
            }
        }

        if reg_offset == REX3_HOSTRW0 || reg_offset == REX3_HOSTRW1 {
            self.note_hostrw_read();
            #[cfg(feature = "developer")]
            if let Some(f) = self.block_log.lock().as_mut() {
                if result.is_ok() {
                    let _ = writeln!(f, "  {} read -> {:08x}{}",
                        rex3_reg_name(reg_offset), result.data, if is_go { " (GO)" } else { "" });
                }
            }
        } else if is_go {
            // Any other register read with the GO bit also pushes a PURE_GO (see below),
            // which is indistinguishable from a HOSTRW GO-read in the generic PURE_GO
            // sentinel logged by register_processor. Surface it here so a HOSTRW word
            // count that comes up short of expected_64b/expected_words can be traced to
            // a non-HOSTRW GO-read stealing one of the PURE_GO slots, instead of looking
            // like a dropped/missing pixel word.
            #[cfg(feature = "developer")]
            if let Some(f) = self.block_log.lock().as_mut() {
                let _ = writeln!(f, "  {} read (GO, non-HOSTRW)", rex3_reg_name(reg_offset));
            }
        }

        if is_go { self.gfifo_push(GFIFO_PURE_GO, 0); }
        result
    }

    fn read8(&self, addr: u32) -> BusRead8 {
        let offset = addr & (REX3_SIZE - 1);
        let is_dcb = (offset & !7) == REX3_DCBDATA0;
        let res = if is_dcb {
            let val = (self.dcb_read() >> 24) as u8;
            dlog_dev!(LogModule::Dcb, "DCB Read8: Offset {:04x} -> {:02x}", offset, val);
            BusRead8::ok(val)
        } else {
            eprintln!("REX3 Read8: unhandled offset {:04x}", offset);
            BusRead8::err()
        };

        if res.is_ok() {
            dlog_dev!(LogModule::Rex3, "REX3 Read8: Offset {:04x} -> {:02x}", offset, res.data);
        } else {
            dlog_dev!(LogModule::Rex3, "REX3 Read8: Offset {:04x} -> err {:08x}", offset, res.status);
        }
        res
    }

    fn write8(&self, addr: u32, val: u8) -> u32 {
        let offset = addr & (REX3_SIZE - 1);
        let is_dcb = (offset & !7) == REX3_DCBDATA0;
        dlog_dev!(LogModule::Rex3, "REX3 Write8: Offset {:04x} Val {:02x}", offset, val);

        if is_dcb {
            dlog_dev!(LogModule::Dcb, "DCB Write8: Offset {:04x} Val {:02x} -> dcb_write({:08x})", offset, val, val as u32);
            self.dcb_write((val as u32) << 24);
            return BUS_OK;
        }
        eprintln!("REX3 Write8: unhandled offset {:04x} val {:02x}", offset, val);
        BUS_ERR
    }

    fn read16(&self, addr: u32) -> BusRead16 {
        let offset = addr & (REX3_SIZE - 1);
        let is_dcb = (offset & !7) == REX3_DCBDATA0;
        let res = if is_dcb {
            let val = (self.dcb_read() >> ((offset & 2) << 3)) as u16;
            dlog_dev!(LogModule::Dcb, "DCB Read16: Offset {:04x} -> {:04x}", offset, val);
            BusRead16::ok(val)
        } else {
            eprintln!("REX3 Read16: unhandled offset {:04x}", offset);
            BusRead16::err()
        };

        if res.is_ok() {
            dlog_dev!(LogModule::Rex3, "REX3 Read16: Offset {:04x} -> {:04x}", offset, res.data);
        } else {
            dlog_dev!(LogModule::Rex3, "REX3 Read16: Offset {:04x} -> err {:08x}", offset, res.status);
        }
        res
    }

    fn write16(&self, addr: u32, val: u16) -> u32 {
        let offset = addr & (REX3_SIZE - 1);
        let is_dcb = (offset & !7) == REX3_DCBDATA0;
        dlog_dev!(LogModule::Rex3, "REX3 Write16: Offset {:04x} Val {:04x}", offset, val);

        if is_dcb {
            dlog_dev!(LogModule::Dcb, "DCB Write16: Offset {:04x} Val {:04x} -> dcb_write({:08x})", offset, val, (val as u32) << ((offset & 2) << 3));
            self.dcb_write((val as u32) << ((offset & 2) << 3));
            return BUS_OK;
        }
        eprintln!("REX3 Write16: unhandled offset {:04x} val {:04x}", offset, val);
        BUS_ERR
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        let offset     = addr & (REX3_SIZE - 1);
        let reg_offset = offset & !0x0800;
        dlog_dev!(LogModule::Rex3, "REX3 Write32: Offset {:04x} (Reg {:04x} {}) Val {:08x}", offset, reg_offset, rex3_reg_name(reg_offset), val);

        // Additionally handle the few registers that need immediate CPU-thread side effects.
        match reg_offset {
            REX3_CONFIG => { self.config.config.store(val, Ordering::Relaxed); }
            REX3_DCBMODE => {
                self.diag.fetch_or(Self::DIAG_LOCK_DCB, Ordering::Relaxed);
                self.dcb.lock().dcbmode = val;
                self.diag.fetch_and(!Self::DIAG_LOCK_DCB, Ordering::Relaxed);
                dlog_dev!(LogModule::Dcb, "DCB Mode Write {:08x} (Addr {} CRS {} DW {} ENCRSINC={})",
                    val,
                    (val >> DCBMODE_DCBADDR_SHIFT) & 0xF,
                    (val >> DCBMODE_DCBCRS_SHIFT) & 0x7,
                    val & DCBMODE_DATAWIDTH_MASK,
                    (val & DCBMODE_ENCRSINC) != 0);
            }
            REX3_DCBDATA0 | REX3_DCBDATA1 => {
                dlog_dev!(LogModule::Dcb, "DCB Write32: Offset {:04x} Val {:08x} -> dcb_write({:08x})", offset, val, val);
                self.dcb_write(val);
            }
            REX3_DCBRESET => { *self.dcb.lock() = Rex3DcbState::default(); }
            _ => {
                self.gfifo_push(offset, val as u64);
                return BUS_OK;
            }
        }
        // if any of the matched registers was written with go
        if (offset & 0x0800) != 0 {
            self.gfifo_push(GFIFO_PURE_GO, 0);
        }
        BUS_OK
    }

    fn read64(&self, addr: u32) -> BusRead64 {
        let offset = addr & (REX3_SIZE - 1);
        let is_go64r = (offset & 0x0800) != 0;
        let reg_offset64r = offset & !0x0800;
        if reg_offset64r == REX3_HOSTRW0 {
            if self.gfxbusy.load(Ordering::Acquire) || !self.gfifo.is_empty() {
                return BusRead64::busy();
            }
            let val = unsafe { (*self.context.get()).hostrw };
            self.note_hostrw_read();
            #[cfg(feature = "developer")]
            if let Some(f) = self.block_log.lock().as_mut() {
                let _ = writeln!(f, "  HOSTRW0_64 read -> {:016x}{}", val, if is_go64r { " (GO)" } else { "" });
            }
            if is_go64r {
                // Enqueue a pure-go entry: no register update, just trigger next pixel batch.
                self.gfifo_push(GFIFO_PURE_GO, 0);
            }
            return BusRead64::ok(val);
        }

        // Two-register 64-bit read: read high word first WITHOUT go, then low word with go.
        let r_high = self.read32(addr & !0x0800);
        if !r_high.is_ok() { return BusRead64 { status: r_high.status, data: 0 }; }
        let r_low = self.read32(addr + 4);
        if !r_low.is_ok() { return BusRead64 { status: r_low.status, data: 0 }; }
        BusRead64::ok(((r_high.data as u64) << 32) | r_low.data as u64)
    }

    /// DMA-driven HOSTRW read (see BusDevice::dma_read64 doc comment).
    ///
    /// CPU-driven PIO gets REX3's read-then-advance protocol "for free": each
    /// GO-space read both returns the currently-latched word AND arms the next
    /// one, so software naturally discards its first (pre-primitive) read and
    /// trusts the rest — see xf86-video-newport's NewportXAAReadPixmap, which
    /// issues N-1 GO reads then 1 final non-GO read for N words, with an
    /// explicit "go has to be issued before we start reading" SETUP-GO primer.
    ///
    /// VDMA has no such software-side discard loop — every dma_read64() result
    /// must already be real data. So here we invert the ordering: push a
    /// GO-only entry, wait for it to execute (arms/advances the pipeline),
    /// THEN read the now-current ctx.hostrw — instead of read-then-advance.
    fn dma_read64(&self, addr: u32) -> BusRead64 {
        let offset = addr & (REX3_SIZE - 1);
        let is_go64r = (offset & 0x0800) != 0;
        let reg_offset64r = offset & !0x0800;
        if reg_offset64r != REX3_HOSTRW0 || !is_go64r {
            return self.read64(addr);
        }

        #[cfg(feature = "developer")]
        let before = unsafe { (*self.context.get()).hostrw };
        self.gfifo_push(REX3_DMA_PURE_GO, 0);
        self.wait_idle();

        let val = unsafe { (*self.context.get()).hostrw };
        self.note_hostrw_read();
        #[cfg(feature = "developer")]
        if let Some(f) = self.block_log.lock().as_mut() {
            let _ = writeln!(f, "  HOSTRW0_64 dma_read: before={:016x} after={:016x} (primed)", before, val);
        }
        BusRead64::ok(val)
    }

    fn write64(&self, addr: u32, val: u64) -> u32 {
        let offset = addr & (REX3_SIZE - 1);
        let is_go = (offset & 0x0800) != 0;
        let reg_offset64 = offset & !0x0800;
        if reg_offset64 == REX3_HOSTRW0 {
            // Encode as REX3_HOSTRW64 (0x0231) + GO bit if present.
            // addr bit 0 = is_64bit, bit 11 = GO.
            self.gfifo_push(REX3_HOSTRW64 | (offset & 0x0800), val);
            return BUS_OK;
        }

        // Two-register 64-bit write: write high word first WITHOUT go, then low word with go.
        let high = (val >> 32) as u32;
        let low = val as u32;
        match self.write32(addr & !0x0800, high) {
            BUS_OK => self.write32(addr + 4, low),
            status => status,
        }
    }
}

// ============================================================================
// Resettable + Saveable for Rex3 (including Vc2, Xmap9, Cmap)
// ============================================================================

impl Resettable for Rex3 {
    fn power_on(&self) {
        // Reset drawing context
        unsafe { *self.context.get() = Rex3Context::power_on_default(); }
        // Reset config registers
        self.config.config.store(0, Ordering::Relaxed);
        self.config.status.store(0, Ordering::Relaxed);
        *self.dcb.lock() = Rex3DcbState::default();
        // Clear framebuffers
        unsafe {
            (*self.fb_rgb.get()).fill(0);
            (*self.fb_aux.get()).fill(0);
        }
        // Reset Vc2, Xmap, Cmap
        *self.vc2.lock() = crate::vc2::Vc2::new();
        *self.xmap0.lock() = crate::xmap9::Xmap9::new();
        *self.xmap1.lock() = crate::xmap9::Xmap9::new();
        *self.cmap0.lock() = crate::cmap::Cmap::new(0);
        *self.cmap1.lock() = crate::cmap::Cmap::new(1);
        *self.bt445.lock() = crate::bt445::Bt445::new();
    }
}

/// Serialize Rex3Context to a flat TOML table.  All fixed-point fields are stored as raw u32 bits.
fn save_rex3_context(ctx: &Rex3Context) -> toml::Value {
    let mut tbl = toml::map::Map::new();
    macro_rules! u32f { ($f:ident) => { tbl.insert(stringify!($f).into(), hex_u32(ctx.$f)); } }
    u32f!(lspattern); u32f!(lspatsave); u32f!(zpattern); u32f!(colorback);
    u32f!(colorvram); u32f!(alpharef); u32f!(smask0x); u32f!(smask0y);
    u32f!(xymove); u32f!(bresd); u32f!(bress1); u32f!(brese1); u32f!(bress2);
    u32f!(aweight0); u32f!(aweight1); u32f!(wrmask);
    u32f!(smask1x); u32f!(smask1y); u32f!(smask2x); u32f!(smask2y);
    u32f!(smask3x); u32f!(smask3y); u32f!(smask4x); u32f!(smask4y);
    u32f!(topscan); u32f!(xywin); u32f!(clipmode); u32f!(hostcnt);
    u32f!(lssave); u32f!(lsrestore); u32f!(stepz); u32f!(stall0); u32f!(stall1);
    // Coordinate fields stored as raw i32 bits (21.11 fixed-point)
    tbl.insert("xstart".into(),     hex_u32(ctx.xstart as u32));
    tbl.insert("ystart".into(),     hex_u32(ctx.ystart as u32));
    tbl.insert("xend".into(),       hex_u32(ctx.xend   as u32));
    tbl.insert("yend".into(),       hex_u32(ctx.yend   as u32));
    tbl.insert("xsave".into(),      hex_u32(ctx.xsave  as u32));
    tbl.insert("colorred".into(),   hex_u32(ctx.colorred));
    tbl.insert("coloralpha".into(), hex_u32(ctx.coloralpha));
    tbl.insert("colorgrn".into(),   hex_u32(ctx.colorgrn));
    tbl.insert("colorblue".into(),  hex_u32(ctx.colorblue));
    tbl.insert("colorx".into(),     hex_u32(ctx.colorx));
    tbl.insert("slopered".into(),   hex_u32(ctx.slopered as u32));
    tbl.insert("slopealpha".into(), hex_u32(ctx.slopealpha as u32));
    tbl.insert("slopegrn".into(),   hex_u32(ctx.slopegrn as u32));
    tbl.insert("slopeblue".into(),  hex_u32(ctx.slopeblue as u32));
    tbl.insert("host_shifter".into(), hex_u64(ctx.host_shifter));
    tbl.insert("drawmode0".into(), hex_u32(ctx.drawmode0.0));
    tbl.insert("drawmode1".into(), hex_u32(ctx.drawmode1.0));
    tbl.insert("lsmode".into(),    hex_u32(ctx.lsmode.0));
    tbl.insert("bresoctinc1".into(), hex_u32(ctx.bresoctinc1.0));
    tbl.insert("bresrndinc2".into(), hex_u32(ctx.bresrndinc2.0));
    toml::Value::Table(tbl)
}

fn load_rex3_context(ctx: &mut Rex3Context, v: &toml::Value) {
    macro_rules! ldu32 { ($f:ident) => {
        if let Some(x) = get_field(v, stringify!($f)) { ctx.$f = toml_u32(x).unwrap_or(ctx.$f); }
    }}
    ldu32!(lspattern); ldu32!(lspatsave); ldu32!(zpattern); ldu32!(colorback);
    ldu32!(colorvram); ldu32!(alpharef); ldu32!(smask0x); ldu32!(smask0y);
    ldu32!(xymove); ldu32!(bresd); ldu32!(bress1); ldu32!(brese1); ldu32!(bress2);
    ldu32!(aweight0); ldu32!(aweight1); ldu32!(wrmask);
    ldu32!(smask1x); ldu32!(smask1y); ldu32!(smask2x); ldu32!(smask2y);
    ldu32!(smask3x); ldu32!(smask3y); ldu32!(smask4x); ldu32!(smask4y);
    ldu32!(topscan); ldu32!(xywin); ldu32!(clipmode); ldu32!(hostcnt);
    ldu32!(lssave); ldu32!(lsrestore); ldu32!(stepz); ldu32!(stall0); ldu32!(stall1);
    if let Some(x) = get_field(v, "xstart")     { ctx.xstart = toml_u32(x).unwrap_or(0) as i32; }
    if let Some(x) = get_field(v, "ystart")     { ctx.ystart = toml_u32(x).unwrap_or(0) as i32; }
    if let Some(x) = get_field(v, "xend")       { ctx.xend   = toml_u32(x).unwrap_or(0) as i32; }
    if let Some(x) = get_field(v, "yend")       { ctx.yend   = toml_u32(x).unwrap_or(0) as i32; }
    if let Some(x) = get_field(v, "xsave")      { ctx.xsave  = toml_u32(x).unwrap_or(0) as i32; }
    if let Some(x) = get_field(v, "colorred")   { ctx.colorred   = toml_u32(x).unwrap_or(0); }
    if let Some(x) = get_field(v, "coloralpha") { ctx.coloralpha = toml_u32(x).unwrap_or(0); }
    if let Some(x) = get_field(v, "colorgrn")   { ctx.colorgrn   = toml_u32(x).unwrap_or(0); }
    if let Some(x) = get_field(v, "colorblue")  { ctx.colorblue  = toml_u32(x).unwrap_or(0); }
    if let Some(x) = get_field(v, "colorx")     { ctx.colorx     = toml_u32(x).unwrap_or(0); }
    if let Some(x) = get_field(v, "slopered")   { ctx.slopered   = toml_u32(x).unwrap_or(0) as i32; }
    if let Some(x) = get_field(v, "slopealpha") { ctx.slopealpha = toml_u32(x).unwrap_or(0) as i32; }
    if let Some(x) = get_field(v, "slopegrn")   { ctx.slopegrn   = toml_u32(x).unwrap_or(0) as i32; }
    if let Some(x) = get_field(v, "slopeblue")  { ctx.slopeblue  = toml_u32(x).unwrap_or(0) as i32; }
    if let Some(x) = get_field(v, "host_shifter") { ctx.host_shifter = toml_u64(x).unwrap_or(0); }
    if let Some(x) = get_field(v, "drawmode0")    { ctx.drawmode0.0 = toml_u32(x).unwrap_or(0); }
    if let Some(x) = get_field(v, "drawmode1")    { ctx.drawmode1.0 = toml_u32(x).unwrap_or(0); }
    if let Some(x) = get_field(v, "lsmode")       { ctx.lsmode.0    = toml_u32(x).unwrap_or(0); }
    if let Some(x) = get_field(v, "bresoctinc1")  { ctx.bresoctinc1.0 = toml_u32(x).unwrap_or(0); }
    if let Some(x) = get_field(v, "bresrndinc2")  { ctx.bresrndinc2.0 = toml_u32(x).unwrap_or(0); }
}

impl Saveable for Rex3 {
    fn save_state(&self) -> toml::Value {
        let mut tbl = toml::map::Map::new();

        // Drawing context
        let ctx = unsafe { &*self.context.get() };
        tbl.insert("context".into(), save_rex3_context(ctx));

        // Config registers
        {
            let dcb = self.dcb.lock();
            let mut ctbl = toml::map::Map::new();
            ctbl.insert("config".into(),  hex_u32(self.config.config.load(Ordering::Relaxed)));
            ctbl.insert("status".into(),  hex_u32(self.config.status.load(Ordering::Relaxed)));
            ctbl.insert("dcbmode".into(),  hex_u32(dcb.dcbmode));
            ctbl.insert("dcbdata0".into(), hex_u32(dcb.dcbdata0));
            ctbl.insert("dcbdata1".into(), hex_u32(dcb.dcbdata1));
            tbl.insert("config_regs".into(), toml::Value::Table(ctbl));
        }

        // Vc2
        {
            let vc2 = self.vc2.lock();
            let mut vtbl = toml::map::Map::new();
            vtbl.insert("index".into(), hex_u32(vc2.index as u32));
            let regs16: Vec<u32> = vc2.regs.iter().map(|&x| x as u32).collect();
            vtbl.insert("regs".into(), u32_slice_to_toml(&regs16));
            vtbl.insert("ram".into(), u16_slice_to_toml(&vc2.ram));
            tbl.insert("vc2".into(), toml::Value::Table(vtbl));
        }

        // Xmap0
        {
            let xmap = self.xmap0.lock();
            tbl.insert("xmap0".into(), save_xmap9(&xmap));
        }

        // Xmap1
        {
            let xmap = self.xmap1.lock();
            tbl.insert("xmap1".into(), save_xmap9(&xmap));
        }

        // Cmap0
        {
            let cmap = self.cmap0.lock();
            tbl.insert("cmap0".into(), save_cmap(&cmap));
        }

        // Cmap1
        {
            let cmap = self.cmap1.lock();
            tbl.insert("cmap1".into(), save_cmap(&cmap));
        }

        // Bt445 RAMDAC (palette + registers) — missing this makes every
        // pixel decode to black after restore.
        {
            let dac = self.bt445.lock();
            tbl.insert("bt445".into(), save_bt445(&dac));
        }

        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        if let Some(ctx_v) = get_field(v, "context") {
            let ctx = unsafe { &mut *self.context.get() };
            load_rex3_context(ctx, ctx_v);
        }

        if let Some(cfg_v) = get_field(v, "config_regs") {
            let mut dcb = self.dcb.lock();
            if let Some(x) = get_field(cfg_v, "config") { self.config.config.store(toml_u32(x).unwrap_or(0), Ordering::Relaxed); }
            if let Some(x) = get_field(cfg_v, "status") { self.config.status.store(toml_u32(x).unwrap_or(0), Ordering::Relaxed); }
            if let Some(x) = get_field(cfg_v, "dcbmode")  { dcb.dcbmode  = toml_u32(x).unwrap_or(0); }
            if let Some(x) = get_field(cfg_v, "dcbdata0") { dcb.dcbdata0 = toml_u32(x).unwrap_or(0); }
            if let Some(x) = get_field(cfg_v, "dcbdata1") { dcb.dcbdata1 = toml_u32(x).unwrap_or(0); }
        }

        if let Some(vv) = get_field(v, "vc2") {
            let mut vc2 = self.vc2.lock();
            if let Some(x) = get_field(vv, "index") { vc2.index = toml_u32(x).unwrap_or(0) as u8; }
            if let Some(r) = get_field(vv, "regs") {
                let mut tmp = [0u32; 32];
                load_u32_slice(r, &mut tmp);
                for (i, &v) in tmp.iter().enumerate() { vc2.regs[i] = v as u16; }
            }
            if let Some(r) = get_field(vv, "ram") { load_u16_slice(r, &mut vc2.ram); }
            vc2.dirty = true;
        }

        if let Some(xv) = get_field(v, "xmap0") { load_xmap9(&mut self.xmap0.lock(), xv); }
        if let Some(xv) = get_field(v, "xmap1") { load_xmap9(&mut self.xmap1.lock(), xv); }
        if let Some(cv) = get_field(v, "cmap0") { load_cmap(&mut self.cmap0.lock(), cv); }
        if let Some(cv) = get_field(v, "cmap1") { load_cmap(&mut self.cmap1.lock(), cv); }
        if let Some(dv) = get_field(v, "bt445") { load_bt445(&mut self.bt445.lock(), dv); }

        // GFIFO is deliberately not part of the serialized snapshot at all
        // (a live, in-flight draw-command queue has no meaningful
        // "restore" — it's ephemeral producer/consumer state, not
        // architectural). But `load_state` is always called with the
        // processor thread stopped (every caller's contract —
        // `restore_live_checkpoint`/`load_snapshot_paused`), and the FIFO's
        // own head/tail are otherwise left exactly as whatever the pre-load
        // live state happened to be — reset it explicitly here rather than
        // relying on it having already drained to empty by coincidence.
        // See `GFifo::reset`'s own doc comment for why this is only safe
        // with the consumer stopped.
        self.gfifo.reset();
        self.gfxbusy.store(false, Ordering::Relaxed);

        Ok(())
    }
}

fn save_xmap9(xmap: &crate::xmap9::Xmap9) -> toml::Value {
    let mut tbl = toml::map::Map::new();
    tbl.insert("config".into(),           hex_u32(xmap.config          as u32));
    tbl.insert("cursor_cmap_msb".into(),  hex_u32(xmap.cursor_cmap_msb as u32));
    tbl.insert("popup_cmap_msb".into(),   hex_u32(xmap.popup_cmap_msb  as u32));
    tbl.insert("mode_addr".into(),        hex_u32(xmap.mode_addr       as u32));
    tbl.insert("mode_table".into(), u32_slice_to_toml(&xmap.mode_table));
    toml::Value::Table(tbl)
}

fn load_xmap9(xmap: &mut crate::xmap9::Xmap9, v: &toml::Value) {
    if let Some(x) = get_field(v, "config")          { xmap.config          = toml_u32(x).unwrap_or(0) as u8; }
    if let Some(x) = get_field(v, "cursor_cmap_msb") { xmap.cursor_cmap_msb = toml_u32(x).unwrap_or(0) as u8; }
    if let Some(x) = get_field(v, "popup_cmap_msb")  { xmap.popup_cmap_msb  = toml_u32(x).unwrap_or(0) as u8; }
    if let Some(x) = get_field(v, "mode_addr")       { xmap.mode_addr       = toml_u32(x).unwrap_or(0) as u8; }
    if let Some(r) = get_field(v, "mode_table")      { load_u32_slice(r, &mut xmap.mode_table); }
    xmap.dirty = true;
}

fn save_cmap(cmap: &crate::cmap::Cmap) -> toml::Value {
    let mut tbl = toml::map::Map::new();
    tbl.insert("addr_lo".into(),   hex_u32(cmap.addr_lo  as u32));
    tbl.insert("addr_hi".into(),   hex_u32(cmap.addr_hi  as u32));
    tbl.insert("command".into(),   hex_u32(cmap.command  as u32));
    tbl.insert("palette".into(), u32_slice_to_toml(&cmap.palette));
    toml::Value::Table(tbl)
}

fn load_cmap(cmap: &mut crate::cmap::Cmap, v: &toml::Value) {
    if let Some(x) = get_field(v, "addr_lo")  { cmap.addr_lo  = toml_u32(x).unwrap_or(0) as u8; }
    if let Some(x) = get_field(v, "addr_hi")  { cmap.addr_hi  = toml_u32(x).unwrap_or(0) as u8; }
    if let Some(x) = get_field(v, "command")  { cmap.command  = toml_u32(x).unwrap_or(0) as u8; }
    if let Some(r) = get_field(v, "palette")  { load_u32_slice(r, &mut cmap.palette); }
    cmap.dirty = true;
}

// Bt445 RAMDAC: palette + control registers. Critical for snapshot restore
// because `power_on` wipes the palette to all-zero, which makes every pixel
// decode to black after the gamma lookup in disp.rs::refresh.
fn save_bt445(dac: &crate::bt445::Bt445) -> toml::Value {
    let flatten = |rgb: &[[u8; 3]]| -> Vec<u8> {
        let mut v = Vec::with_capacity(rgb.len() * 3);
        for e in rgb { v.extend_from_slice(e); }
        v
    };
    let mut tbl = toml::map::Map::new();
    tbl.insert("palette".into(),      u8_slice_to_toml(&flatten(&dac.palette)));
    tbl.insert("overlay".into(),      u8_slice_to_toml(&flatten(&dac.overlay)));
    tbl.insert("cursor_color".into(), u8_slice_to_toml(&flatten(&dac.cursor_color)));
    tbl.insert("addr".into(),         hex_u8(dac.addr));
    tbl.insert("rgb_counter".into(),  hex_u8(dac.rgb_counter));
    tbl.insert("read_enable".into(),  hex_u8(dac.read_enable));
    tbl.insert("blink_enable".into(), hex_u8(dac.blink_enable));
    tbl.insert("cmd0".into(),         hex_u8(dac.cmd0));
    tbl.insert("rgb_ctrl".into(),     u8_slice_to_toml(&dac.rgb_ctrl));
    tbl.insert("setup".into(),        u8_slice_to_toml(&dac.setup));
    toml::Value::Table(tbl)
}

fn load_bt445(dac: &mut crate::bt445::Bt445, v: &toml::Value) {
    let unflatten = |bytes: &[u8], dest: &mut [[u8; 3]]| {
        for (i, chunk) in bytes.chunks(3).enumerate() {
            if i >= dest.len() { break; }
            if chunk.len() == 3 {
                dest[i] = [chunk[0], chunk[1], chunk[2]];
            }
        }
    };
    if let Some(r) = get_field(v, "palette") {
        let mut buf = vec![0u8; dac.palette.len() * 3];
        load_u8_slice(r, &mut buf);
        unflatten(&buf, &mut dac.palette);
    }
    if let Some(r) = get_field(v, "overlay") {
        let mut buf = vec![0u8; dac.overlay.len() * 3];
        load_u8_slice(r, &mut buf);
        unflatten(&buf, &mut dac.overlay);
    }
    if let Some(r) = get_field(v, "cursor_color") {
        let mut buf = vec![0u8; dac.cursor_color.len() * 3];
        load_u8_slice(r, &mut buf);
        unflatten(&buf, &mut dac.cursor_color);
    }
    if let Some(x) = get_field(v, "addr")         { if let Some(n) = toml_u8(x) { dac.addr = n; } }
    if let Some(x) = get_field(v, "rgb_counter")  { if let Some(n) = toml_u8(x) { dac.rgb_counter = n; } }
    if let Some(x) = get_field(v, "read_enable")  { if let Some(n) = toml_u8(x) { dac.read_enable = n; } }
    if let Some(x) = get_field(v, "blink_enable") { if let Some(n) = toml_u8(x) { dac.blink_enable = n; } }
    if let Some(x) = get_field(v, "cmd0")         { if let Some(n) = toml_u8(x) { dac.cmd0 = n; } }
    if let Some(r) = get_field(v, "rgb_ctrl")     { load_u8_slice(r, &mut dac.rgb_ctrl); }
    if let Some(r) = get_field(v, "setup")        { load_u8_slice(r, &mut dac.setup); }
    dac.dirty = true;
}

#[cfg(test)]
#[path = "rex3_tests.rs"]
mod tests;
