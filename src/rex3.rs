use std::sync::Arc;
use parking_lot::Mutex;
use spin::Mutex as SpinMutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use rtrb::{Consumer, Producer, RingBuffer};
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device, Resettable, Saveable};
use crate::devlog::{LogModule, devlog_is_active, devlog};
use crate::snapshot::{get_field, u32_slice_to_toml, u16_slice_to_toml, load_u32_slice, load_u16_slice, toml_u32, toml_u64, hex_u32, hex_u64};
use std::cell::UnsafeCell;
use crate::vc2::Vc2;
use crate::xmap9::Xmap9;
use crate::cmap::Cmap;
use crate::bt445::Bt445;
use bitfield::bitfield;
use crate::disp::Rex3Screen;
use std::io::Write;

pub trait Renderer: Send {
    /// Upload and draw the main emulator framebuffer (exactly `width × height` pixels,
    /// row stride 2048 in the buffer).
    fn render(&mut self, buffer: &[u32], width: usize, height: usize);
    /// Upload and alpha-blend the debug overlay on top of the main framebuffer.
    /// The overlay buffer covers `width × height` pixels, row stride 2048.
    /// Transparent pixels (alpha=0) leave the main framebuffer visible.
    fn render_overlay(&mut self, _buffer: &[u32], _width: usize, _height: usize, _overlay_extra_rows: usize) {}
    /// Upload and draw the status bar texture at the bottom of the window.
    /// Buffer is `width × STATUS_BAR_HEIGHT` pixels, row stride 2048.
    fn render_statusbar(&mut self, _buffer: &[u32], _width: usize) {}
    fn resize(&mut self, _width: usize, _height: usize) {}
    fn stop(&mut self) {}
}

pub const REX3_SIZE: u32 = 0x2000; // 8KB
pub const REX3_BASE: u32 = 0x1F0F0000; // Physical base address of registers
pub const GFIFO_DEPTH: usize = 65536;
/// Real hardware GFIFO depth (32 entries). STATUS reports level=1 below this threshold.
pub const GFIFO_HW_DEPTH: usize = 32;
/// Special GFIFO command to trigger a GO without a register write.
pub const GFIFO_PURE_GO: u32 = 0xFFFF_0800;
/// Special GFIFO command to signal the processor thread to exit.
pub const GFIFO_EXIT: u32 = 0xFFFF_0000;
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

pub const DRAWMODE1_PLANES_NONE: u32 = 0;
pub const DRAWMODE1_PLANES_RGB: u32 = 1;
pub const DRAWMODE1_PLANES_RGBA: u32 = 2;
pub const DRAWMODE1_PLANES_OLAY: u32 = 4;
pub const DRAWMODE1_PLANES_PUP: u32 = 5;
pub const DRAWMODE1_PLANES_CID: u32 = 6;

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

// COLORRED: o12.11 for CI mode, o8.15 for RGB mode
// Write format: o12.11 (9 top bits overflow, 12 integer, 11 fractional)
// Read format: o12.11 (same masking)
// COLORRED: o12.11 format in both CI and RGB modes.
// Wire format: bits[23:0] = o1 + 12 integer bits + 11 fractional bits.
// Color registers: plain u32.
// COLORRED:         o12.11 — bits[23:0] stored raw (24 bits). bit31 = overflow/neg after DDA.
// COLORALPHA/GRN/BLUE: o8.11 — bits[19:0] stored raw (20 bits).

fn from_color_red(val: u32, _drawmode1: DrawMode1) -> u32 {
    val & 0xFFFFFF
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
    /// VRINT bit written by refresh thread, cleared on STATUS read by CPU thread.
    /// Benign race: worst case a VRINT is missed or double-cleared.
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

/// A GFIFO entry encodes a register write as a raw bus address + 64-bit value.
///
/// Encoding of `addr`:
///   bit 11 (0x800): GO flag — set when the bus write used the GO address space
///   bit  0 (0x001): is_64bit flag — set for 64-bit HOSTRW writes (free bit: regs are 4-byte aligned)
///   bits 10:1 (with bit 0 cleared): `reg_offset` — the register being written
///
/// Special sentinel: `GFIFO_PURE_GO` encodes a "pure GO" — no register update,
/// just trigger `execute_go()`.  The reg_offset is not a real register so
/// `process_register` hits the `_ => {}` arm and is a no-op.

#[derive(Clone, Copy, Debug, Default)]
pub struct GFIFOEntry {
    /// Raw bus offset with GO bit (0x800) and is_64bit bit (0x001) encoded.
    pub addr: u32,
    /// Register value (low 32 bits for 32-bit writes; full 64 bits for 64-bit HOSTRW).
    pub val: u64,
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
    pub gfifo_producer: SpinMutex<Producer<GFIFOEntry>>,
    pub gfifo_consumer: Mutex<Option<Consumer<GFIFOEntry>>>,

    pub vc2: Mutex<Vc2>,
    pub xmap0: Mutex<Xmap9>,
    pub xmap1: Mutex<Xmap9>,
    pub cmap0: Mutex<Cmap>,
    pub cmap1: Mutex<Cmap>,
    pub bt445: Mutex<Bt445>,
    clock: AtomicU64,
    running: AtomicBool,
    pub gfxbusy: Arc<AtomicBool>,
    pub processor_thread: Mutex<Option<thread::JoinHandle<Consumer<GFIFOEntry>>>>,
    pub refresh_thread: Mutex<Option<thread::JoinHandle<()>>>,
    pub screen: Arc<Mutex<Rex3Screen>>,
    pub vblank_cb: Mutex<Option<Arc<dyn Fn(bool) + Send + Sync>>>,
    pub debug: Arc<AtomicBool>,
    pub block_debug: Arc<AtomicBool>,
    pub draw_debug: Arc<AtomicBool>,
    pub draw_ring: Arc<Mutex<DrawRingBuf>>,
    pub gfifo_pending: AtomicUsize,
    #[cfg(feature = "developer")]
    pub gfifo_hwm: AtomicUsize,
    pub hostrw: AtomicU64,
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
    /// Shared activity heartbeat — set by all devices, polled+cleared by the refresh thread.
    /// bit 0 = enet TX, bit 1 = enet RX, bits 2-3 = red/green LED (persistent), bits 8-13 = SCSI IDs 0-5
    pub heartbeat: Arc<AtomicU64>,
    /// CPU cycle counter, shared with the CPU thread.
    pub cycles: Arc<AtomicU64>,
    /// CP0 Count==Compare match counter — incremented every fastick interrupt.
    pub fasttick_count: Arc<AtomicU64>,
    pub decoded_count: Arc<AtomicU64>,
    pub l1i_hit_count: Arc<AtomicU64>,
    pub l1i_fetch_count: Arc<AtomicU64>,
    pub uncached_fetch_count: Arc<AtomicU64>,
    /// Optional log file for block/span draws (set when block_debug is enabled).
    block_log: Mutex<Option<std::fs::File>>,
    /// GFIFO log (written by painter thread on process_register, one line per entry).
    rex3_log: Mutex<Option<std::fs::File>>,
    /// When true, the refresh thread overlays a 16x16 grid of 8x8 CMAP swatches.
    pub show_cmap: AtomicBool,
    /// When true, overlay decoded DID/XMAP mode info near the bottom of the screen.
    pub show_disp_debug: AtomicBool,
    /// Atomic shadow of MipsCore::count_step — updated by CPU thread, read by refresh thread.
    /// Wrapped in Mutex so machine.rs can swap in the real Arc from MipsCore after construction.
    #[cfg(feature = "developer")]
    pub count_step_atomic: Mutex<Arc<AtomicU64>>,
}

unsafe impl Sync for Rex3 {}

impl Rex3 {
    pub fn new(heartbeat: Arc<AtomicU64>, cycles: Arc<AtomicU64>, fasttick_count: Arc<AtomicU64>, decoded_count: Arc<AtomicU64>, l1i_hit_count: Arc<AtomicU64>, l1i_fetch_count: Arc<AtomicU64>, uncached_fetch_count: Arc<AtomicU64>) -> Self {
        let (producer, consumer) = RingBuffer::new(GFIFO_DEPTH);

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
        
        let screen = Arc::new(Mutex::new(Rex3Screen {
            width: 2048,
            height: 1024,
            fb_rgb: vec![0; 2048 * 1024],
            fb_aux: vec![0; 2048 * 1024],
            did: vec![0; 2048 * 1024],
            rgba: vec![0; 2048 * 1024],
            overlay_rgba: vec![0; 2048 * 1024],
            statusbar_rgba: vec![0; 2048 * crate::disp::STATUS_BAR_HEIGHT],
            vc2_ram: vec![0; 32768],
            vc2_regs: [0; 32],
            cmap: [0; 8192],
            ramdac_palette: [0; 256],
            xmap_mode: [0; 32],
            xmap_config: 0,
            xmap_cursor_cmap: 0,
            xmap_popup_cmap: 0,
            topscan: 0,
            cursor_x_adjust: 0,
            show_cmap: false,
            show_disp_debug: false,
            seen_modes: [(0, 0, 0); 32],
            seen_modes_count: 0,
            show_draw_debug: false,
            draw_snapshot: Vec::new(),
            debug_font: crate::vga_font::VGA_8X16.to_vec(),
        }));

        Self {
            config,
            dcb: Mutex::new(Rex3DcbState::default()),
            context: UnsafeCell::new(Rex3Context::default()),
            fb_rgb: UnsafeCell::new(fb_rgb),
            fb_aux: UnsafeCell::new(fb_aux),
            px_rd: UnsafeCell::new(Self::default_px_rd),
            px_wr: UnsafeCell::new(Self::default_px_wr),
            px_amp: UnsafeCell::new(Self::amplify_nop),
            px_logic: UnsafeCell::new(Self::logic_op_src),
            px_compress: UnsafeCell::new(Self::identity),
            px_expand: UnsafeCell::new(Self::identity),
            px_proc: UnsafeCell::new(Self::process_pixel_noop),
            px_shade: UnsafeCell::new(Self::iterate_shade_noop),
            px_pattern: UnsafeCell::new(Self::iterate_pattern_noop),
            host_unpack: UnsafeCell::new(Self::host_unpack_nop),
            host_pack: UnsafeCell::new(Self::host_pack_nop),
            host_shift: UnsafeCell::new(0),
            host_count: UnsafeCell::new(0),
            gfifo_producer: SpinMutex::new(producer),
            gfifo_consumer: Mutex::new(Some(consumer)),
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
            screen,
            vblank_cb: Mutex::new(None),
            debug: Arc::new(AtomicBool::new(false)),
            block_debug: Arc::new(AtomicBool::new(false)),
            draw_debug: Arc::new(AtomicBool::new(false)),
            draw_ring: Arc::new(Mutex::new(DrawRingBuf::default())),
            gfifo_pending: AtomicUsize::new(0),
            #[cfg(feature = "developer")]
            gfifo_hwm: AtomicUsize::new(0),
            hostrw: AtomicU64::new(0),
            jit_go_count: AtomicU64::new(0),
            interp_go_count: AtomicU64::new(0),
            debug_state: Mutex::new(DebugState::default()),
            diag: AtomicU64::new(0),
            renderer: Mutex::new(None),
            #[cfg(feature = "rex-jit")]
            rex_jit: Some(std::sync::Arc::new(crate::rex3_jit::RexJit::new())),
            #[cfg(feature = "rex-jit")]
            jit_enabled: AtomicBool::new(true),
            heartbeat,
            cycles,
            fasttick_count,
            decoded_count,
            l1i_hit_count,
            l1i_fetch_count,
            uncached_fetch_count,
            block_log: Mutex::new(None),
            rex3_log: Mutex::new(None),
            show_cmap: AtomicBool::new(false),
            show_disp_debug: AtomicBool::new(false),
            #[cfg(feature = "developer")]
            count_step_atomic: Mutex::new(Arc::new(AtomicU64::new(1 << 15))),
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
    pub const DIAG_LOCK_GFIFO_PROD:   u64 = 1 << 9;
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

    #[cfg(feature = "developer")]
    pub fn set_count_step_atomic(&self, arc: Arc<AtomicU64>) {
        *self.count_step_atomic.lock() = arc;
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
        // fline will re-derive d with fractional adjustments at draw time, but incr1/incr2
        // remain the same — only d differs by a fractional correction term.
        //   incr1 = 2 * minor          (straight step increment, always >= 0)
        //   incr2 = 2 * (minor - major) (diagonal step increment, always <= 0)
        //   d     = incr1 - major       (initial decision variable)
        let incr1: i32 = 2 * minor;
        let incr2: i32 = 2 * (minor - major);
        let d:     i32 = incr1 - major;

        // Store as plain two's-complement integers masked to register field widths.
        // draw_iline/draw_fline read these back, update d each step, and write d back.
        ctx.bresoctinc1.set_octant(octant);
        ctx.bresoctinc1.set_incr1((incr1 as u32) & 0xFFFFF);     // 20-bit, always positive
        ctx.bresrndinc2.set_incr2((incr2 as u32) & 0x1FFFFF);    // 21-bit signed
        ctx.bresd = (d as u32) & 0x7FF_FFFF;                     // 27-bit signed
    }

    fn log_block(&self, ctx: &Rex3Context, opcode: u32) {
        let need_block_log = self.block_log.lock().is_some();
        let need_draw_ring = self.draw_debug.load(Ordering::Relaxed);
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
            });
        }
    }

    fn draw_block(&self, ctx: &mut Rex3Context) {
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
                    break;
                }

                let y_end_reached = if y_dec { ctx.ystart < ctx.yend } else { ctx.ystart > ctx.yend };

                if y_end_reached {
                    // All rows consumed — primitive done.
                    ctx.mid_primitive = false;
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

    fn draw_iline(&self, ctx: &mut Rex3Context) {
        // Bresenham octant table (aped from MAME do_iline s_bresenham_infos).
        // Fields: (incrx1, incrx2, incry1, incry2, y_major)
        // MAME applies y as `y -= incry`, so positive incry moves y in the negative direction.
        #[rustfmt::skip]
        const BRES: [(i32, i32, i32, i32, bool); 8] = [
            ( 0,  1, -1, -1, true ),  // octant 0
            ( 0,  1,  1,  1, true ),  // octant 1
            ( 0, -1, -1, -1, true ),  // octant 2
            ( 0, -1,  1,  1, true ),  // octant 3
            ( 1,  1,  0, -1, false),  // octant 4
            ( 1,  1,  0,  1, false),  // octant 5
            (-1, -1,  0, -1, false),  // octant 6
            (-1, -1,  0,  1, false),  // octant 7
        ];

        let octant = (ctx.bresoctinc1.octant() & 7) as usize;
        let (incrx1, incrx2, incry1, incry2, y_major) = BRES[octant];

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
        let mut d = {
            let raw = ctx.bresd & 0x7FF_FFFF;
            if raw & (1 << 26) != 0 { (raw | 0xF800_0000) as i32 } else { raw as i32 }
        };

        // pixel_count = major_axis_length + 1 (both endpoints inclusive).
        let major = if y_major { (y2 - y).abs() } else { (x2 - x).abs() };
        let mut pixel_count = major + 1;
        if ctx.drawmode0.length32() && pixel_count > 32 {
            pixel_count = 32;
        }

        let iterate_one = !ctx.drawmode0.stoponx() && !ctx.drawmode0.stopony();
        let mut skip_first = ctx.drawmode0.skipfirst();
        let mut skip_last = ctx.drawmode0.skiplast();
        if iterate_one {
            pixel_count = 1;
            skip_first = false;
            skip_last = false;
        }

        let proc_fn    = unsafe { *self.px_proc.get() };
        let shade_fn   = unsafe { *self.px_shade.get() };
        let pattern_fn = unsafe { *self.px_pattern.get() };

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
            pattern_fn(ctx);

            // On the last pixel of a full-line draw, verify Bresenham landed on x2,y2.
            // Skip in step mode (pixel_count==1) where we draw only one intermediate pixel.
            debug_assert!(
                iterate_one || !is_last || (x == x2 && y == y2),
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
        // Slope is sign-extended 24/20-bit two's-complement stored as i32.
        if ctx.slopered  & 0x7FFFFF != 0 { ctx.colorred   = ctx.colorred.wrapping_add(ctx.slopered  as u32); }
        if ctx.slopegrn  & 0x7FFFF  != 0 { ctx.colorgrn   = ctx.colorgrn.wrapping_add(ctx.slopegrn  as u32); }
        if ctx.slopeblue & 0x7FFFF  != 0 { ctx.colorblue  = ctx.colorblue.wrapping_add(ctx.slopeblue as u32); }
        if ctx.slopealpha & 0x7FFFF != 0 { ctx.coloralpha = ctx.coloralpha.wrapping_add(ctx.slopealpha as u32); }
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

    // Bayer 4x4 dither matrix, row-major [y][x], flat.
    // Indexed by (y&3)<<2|(x&3) which matches bayer_pack() encoding.
    const BAYER: [u8; 16] = [
         0,  8,  2, 10,
        12,  4, 14,  6,
         3, 11,  1,  9,
        15,  7, 13,  5,
    ];

    /// Pack bayer index into bits 27:24 of color value (top byte unused by 24-bit BGR).
    /// Encoding: bits[3:2] = y&3, bits[1:0] = x&3 → index = (y&3)<<2|(x&3).
    /// Non-dither compress variants ignore these bits.
    #[inline(always)]
    fn bayer_pack(color: u32, x: i32, y: i32) -> u32 {
        (color & 0x00FFFFFF) | (((y as u32 & 3) << 2 | (x as u32 & 3)) << 24)
    }

    fn rgb24_to_rgb4_dither(val: u32) -> u32 {
        let bayer = Self::BAYER[(val >> 24) as usize];
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
        if (sr & 0xf) > bayer { dr = (dr + 1).min(1); }
        if (sg & 0xf) > bayer { dg = (dg + 1).min(3); }
        if (sb & 0xf) > bayer { db = (db + 1).min(1); }
        ((db << 3) | (dg << 1) | dr) as u32
    }

    fn rgb24_to_rgb8_dither(val: u32) -> u32 {
        let bayer = Self::BAYER[(val >> 24) as usize];
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
        if (sr & 0xf) > bayer { dr = (dr + 1).min(7); }
        if (sg & 0xf) > bayer { dg = (dg + 1).min(7); }
        if (sb & 0xf) > bayer { db = (db + 1).min(3); }
        ((db << 6) | (dg << 3) | dr) as u32
    }

    fn rgb24_to_rgb12_dither(val: u32) -> u32 {
        let bayer = Self::BAYER[(val >> 24) as usize];
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
        if (sr & 0xf) > bayer as u32 { dr = (dr + 1).min(15); }
        if (sg & 0xf) > bayer as u32 { dg = (dg + 1).min(15); }
        if (sb & 0xf) > bayer as u32 { db = (db + 1).min(15); }
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
            ctx.host_shifter = self.hostrw.load(Ordering::Relaxed);
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

        self.hostrw.store(val, Ordering::Relaxed);
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
        if !ctx.drawmode1.rgbmode() {
            return color;
        }
        // RGB mode: overlay alpha channel
        let a = if ctx.drawmode0.alphahost() {
            (host_pixel >> 24) & 0xFF
        } else {
            Rex3Context::clamp_color_component(ctx.coloralpha)
        };
        (color & 0x00FFFFFF) | (a << 24)
    }

    /// Replicate colorvram to fill plane-depth slots, matching MAME get_default_color() with fastclear=1.
    fn fastclear_color(ctx: &Rex3Context) -> u32 {
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
        
        let host_pixel = if alphahost || colorhost {
            self.fetch_host_pixel(ctx)
        } else {
            0
        };

        let mut use_bg = false;
        let mut check_ls = true;

        // Pattern checks use independent bit indices; advance is done by px_pattern after the pixel.
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
                compress_fn(Self::bayer_pack(blended, x, y))
            } else {
                // Logic op path: compress src to plane-depth, amplify for dblsrc, then logic op.
                // dst also needs amplify: rd_fn shifts the value down (e.g. bits 15:8 → 7:0 for
                // dblsrc slot 1), so we must shift it back up to match the write mask position.
                let amp_fn = unsafe { *self.px_amp.get() };
                let logic_fn = unsafe { *self.px_logic.get() };
                let src = amp_fn(compress_fn(Self::bayer_pack(raw_src, x, y)));
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
            let compress_fn = unsafe { *self.px_compress.get() };
            // SRC logicop: result = src, no dst read needed.
            wr_fn(self, addr, amp_fn(compress_fn(Self::bayer_pack(ctx.get_colori(), x, y))));
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
            let compress_fn = unsafe { *self.px_compress.get() };
            let logic_fn    = unsafe { *self.px_logic.get() };
            let src = amp_fn(compress_fn(Self::bayer_pack(raw_src, x, y)));
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
        // DW_3 is ambiguous — skipped for now.
        if (dcb.dcbmode & DCBMODE_SWAPENDIAN) != 0 {
            val = match data_width {
                DCBMODE_DATAWIDTH_2 => ((val & 0xFF) << 8) | ((val >> 8) & 0xFF),
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
                    DCBMODE_DATAWIDTH_1 => { write_cmap(dcb.inc_crs(1), val as u8); }
                    DCBMODE_DATAWIDTH_2 => {
                        write_cmap(dcb.inc_crs(1), (val >> 8) as u8);
                        write_cmap(dcb.inc_crs(1), val as u8);
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
                };
                match data_width {
                    DCBMODE_DATAWIDTH_1 => { write_xmap(dcb.inc_crs(1), val & 0xFF); }
                    DCBMODE_DATAWIDTH_2 => {
                        write_xmap(dcb.inc_crs(1), (val >> 8) & 0xFF);
                        write_xmap(dcb.inc_crs(1), val & 0xFF);
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
                    DCBMODE_DATAWIDTH_1 => { self.bt445.lock().write_crs(dcb.inc_crs(1), val as u8); }
                    DCBMODE_DATAWIDTH_2 => {
                        self.bt445.lock().write_crs(dcb.inc_crs(1), (val >> 8) as u8);
                        self.bt445.lock().write_crs(dcb.inc_crs(1), val as u8);
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
                    _ =>                  4,
                };
                let raw = self.vc2.lock().read(dcb.inc_crs(nbytes));
                // DW_2: replicate 16-bit value in both halves so drivers using
                // either >> 16 or & 0xFFFF both get the correct result.
                val = if data_width == DCBMODE_DATAWIDTH_2 {
                    let v = raw & 0xFFFF;
                    (v << 16) | v
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
                        val = read_byte(dcb.inc_crs(1)) as u32;
                    }
                    DCBMODE_DATAWIDTH_2 => {
                        let b0 = read_byte(dcb.inc_crs(1)) as u32;
                        let b1 = read_byte(dcb.inc_crs(1)) as u32;
                        val = (b0 << 8) | b1;
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
                DCBMODE_DATAWIDTH_2 => ((val & 0xFF) << 8) | ((val >> 8) & 0xFF),
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

    fn gfifo_push(&self, entry: GFIFOEntry) {
        let pending = self.gfifo_pending.fetch_add(1, Ordering::SeqCst) + 1;
        #[cfg(feature = "developer")]
        let _ = self.gfifo_hwm.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |hwm| {
            if pending > hwm { Some(pending) } else { None }
        });
        #[cfg(not(feature = "developer"))]
        let _ = pending;
        // Hold the producer lock for the entire retry loop — dropping and
        // re-acquiring between retries lets another caller interleave its own
        // push and break write ordering for multi-entry sequences.
        let mut producer = self.gfifo_producer.lock();
        loop {
            if producer.push(entry).is_ok() {
                return;
            }
            // Buffer full — consumer is spinning and will drain it; just yield.
            std::hint::spin_loop();
        }
    }

    fn wait_idle(&self) {
        loop {
            // Acquire load: when gfxbusy goes false, all execute_go() writes become visible.
            let busy = self.gfxbusy.load(Ordering::Acquire);
            let pending = self.gfifo_pending.load(Ordering::Relaxed);
            if !busy && pending == 0 { break; }
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

    fn register_processor(&self, mut consumer: Consumer<GFIFOEntry>) -> Consumer<GFIFOEntry> {
        let backoff = crossbeam_utils::Backoff::new();
        loop {
            if let Ok(entry) = consumer.pop() {
                backoff.reset();
                if entry.addr == GFIFO_EXIT {
                    break;
                }

                let is_go      = entry.addr & 0x0800 != 0;
                let is_pure_go = entry.addr == GFIFO_PURE_GO;

                if !is_pure_go {
                    // Strip GO bit only; keep is_64bit (bit 0) for HOSTRW discrimination.
                    let reg_offset = entry.addr & !0x0800;
                    self.process_register(reg_offset, entry.val);
                }

                // Log to rex3_log after processing.
                if let Some(f) = self.rex3_log.lock().as_mut() {
                    if is_pure_go {
                        let _ = writeln!(f, "------- PURE_GO -------");
                    } else {
                        let reg_offset = entry.addr & !0x0800;
                        let val32 = entry.val as u32;
                        let extra = match reg_offset {
                            REX3_DRAWMODE0 => format!("  ; {}", decode_dm0(val32)),
                            REX3_DRAWMODE1 => format!("  ; {}", decode_dm1(val32)),
                            _ => String::new(),
                        };
                        if is_go {
                            let _ = writeln!(f, "------- GO reg={:04x}({}) val={:016x}{} -------",
                                reg_offset, rex3_reg_name(reg_offset), entry.val, extra);
                        } else {
                            let _ = writeln!(f, "reg={:04x}({}) val={:016x}{}",
                                reg_offset, rex3_reg_name(reg_offset), entry.val, extra);
                        }
                    }
                }

                if is_go {
                    self.gfxbusy.store(true, Ordering::Relaxed);
                    self.execute_go();
                    // Release: all context/FB writes from execute_go() are visible to any
                    // thread that subsequently does an Acquire load of gfxbusy (e.g. wait_idle
                    // and STATUS read on the CPU thread).
                    self.gfxbusy.store(false, Ordering::Release);
                }

                self.gfifo_pending.fetch_sub(1, Ordering::SeqCst);
            } else {
                // Nothing in the ring — back off. Starts with spin_loop() hints,
                // graduates to thread::yield_now() after sustained emptiness.
                backoff.snooze();
            }
        }
        consumer
    }

    pub(crate) fn execute_go(&self) {
        self.diag.fetch_or(Self::DIAG_LOOP_EXECUTE_GO, Ordering::Relaxed);
        let ctx = unsafe { &mut *self.context.get() };
        let opcode = ctx.drawmode0.opcode();
        // SCR2SCR copies already-quantized pixels — dithering would corrupt them.
        let dm1_for_setup = if opcode == DRAWMODE0_OPCODE_SCR2SCR {
            DrawMode1(ctx.drawmode1.0 & !(1 << 16)) // clear DITHER bit
        } else {
            ctx.drawmode1
        };
        self.planes_setup(dm1_for_setup);
        self.host_setup(ctx.drawmode1);
        // FIXME: unclear whether pat_bit/zpat_bit should reset to 31 on every GO or carry
        // over across primitives.  GL stippled line loops likely want continuity between
        // segments; resetting here would break that.  Need a real test app on hardware to
        // verify.  For now reset to 31
        ctx.pat_bit  = 31;
        ctx.zpat_bit = 31;
        // lsrcount is live state inside the lsmode register — do NOT reset it here.
        // The ARCS diag writes a pattern to lsmode, issues a GO, then reads it back and
        // expects the value unchanged.  Resetting lsrcount on GO would corrupt the readback.
        // GL manages lsrcount explicitly via LSSAVE/LSRESTORE for connected stippled lines.
        // FIXME: also unclear whether pattern advance is ROL (rotate-left, hardware
        // recirculation) or ROR as currently implemented.  Suspect ROL based on
        // "recirculating iterator" language in the spec.  Verify with test app on real Indy.

        let proc = match opcode {
            DRAWMODE0_OPCODE_READ    => Self::process_pixel_read,
            DRAWMODE0_OPCODE_DRAW    => {
                let cidmatch = (ctx.clipmode >> CLIPMODE_CIDMATCH_SHIFT) & 0xF;
                let no_cid      = cidmatch == 0xF;
                let no_host     = !ctx.drawmode0.colorhost() && !ctx.drawmode0.alphahost();
                let no_blend    = !ctx.drawmode1.blend();
                let en_z        = ctx.drawmode0.enzpattern();
                let en_ls       = ctx.drawmode0.enlspattern();
                let no_zpopaque = !ctx.drawmode0.zpopaque();
                let is_src_op   = ctx.drawmode1.logicop() == DRAWMODE1_LOGICOP_SRC >> 28;

                if ctx.drawmode1.fastclear() && no_cid {
                    Self::process_pixel_fastclear
                } else if no_cid && no_host && no_blend && en_z && !en_ls && no_zpopaque && is_src_op {
                    // Character/glyph: zpattern kill only, SRC logicop — no dst read needed.
                    Self::process_pixel_zpattern
                } else if no_cid && no_host && no_blend {
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

        if ctx.drawmode0.dosetup() {
            self.setup(ctx);
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
                let dm1 = ctx.drawmode1.0;
                let adrmode = ctx.drawmode0.adrmode() << 2;
                let is_jittable = opcode == DRAWMODE0_OPCODE_DRAW
                    && (adrmode == DRAWMODE0_ADRMODE_BLOCK || adrmode == DRAWMODE0_ADRMODE_SPAN)
                    && !ctx.drawmode0.colorhost()
                    && !ctx.drawmode0.alphahost();
                if is_jittable {
                    if let Some(entry) = jit.lookup(dm0, dm1) {
                        let fb_rgb = unsafe { (*self.fb_rgb.get()).as_mut_ptr() };
                        let fb_aux = unsafe { (*self.fb_aux.get()).as_mut_ptr() };
                        unsafe { entry(ctx as *mut Rex3Context, fb_rgb, fb_aux); }
                        self.jit_go_count.fetch_add(1, Ordering::Relaxed);
                        self.diag.fetch_and(!Self::DIAG_LOOP_EXECUTE_GO, Ordering::Relaxed);
                        return;
                    } else {
                        jit.request_compile(dm0, dm1);
                        // fall through to interpreter
                    }
                }
            }
            } // jit_enabled
        }

        if opcode != DRAWMODE0_OPCODE_NOOP {
            let adrmode = ctx.drawmode0.adrmode() << 2;
            if adrmode == DRAWMODE0_ADRMODE_I_LINE
                || adrmode == DRAWMODE0_ADRMODE_F_LINE
                || adrmode == DRAWMODE0_ADRMODE_A_LINE
            {
                self.draw_iline(ctx);
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

        unsafe {
            *self.px_rd.get() = rd;
            *self.px_wr.get() = wr;
            *self.px_amp.get() = amp;
            *self.px_logic.get() = logic_fn;
            *self.px_compress.get() = compress;
            *self.px_expand.get() = expand;
            self.host_setup(drawmode1);
        }
    }

    fn refresh_loop(&self) {
        let frame_duration = std::time::Duration::from_micros(16667); // ~60Hz
        let mut status_bar = crate::disp::StatusBar::new();

        while self.running.load(Ordering::Relaxed) {
            let start = std::time::Instant::now();

            // Poll and clear activity bits; preserve persistent LED bits.
            let bar_stats = crate::disp::BarStats {
                now:          start,
                hb:           self.heartbeat.fetch_and(Self::HB_PERSISTENT, Ordering::Relaxed),
                cycles:       self.cycles.load(Ordering::Relaxed),
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
                count_step:   self.count_step_atomic.lock().load(Ordering::Relaxed),
                #[cfg(not(feature = "developer"))]
                count_step:   0,
            };

            // Get unsafe access to framebuffers and context
            let fb_rgb = unsafe { &*self.fb_rgb.get() };
            let fb_aux = unsafe { &*self.fb_aux.get() };
            let topscan = unsafe { (*self.context.get()).topscan as usize };

            {
                self.diag.fetch_or(Self::DIAG_LOCK_SCREEN, Ordering::Relaxed);
                let mut screen = self.screen.lock();
                screen.topscan = topscan;
                screen.show_cmap = self.show_cmap.load(Ordering::Relaxed);
                screen.show_disp_debug = self.show_disp_debug.load(Ordering::Relaxed);
                let dd = self.draw_debug.load(Ordering::Relaxed);
                screen.show_draw_debug = dd;
                if dd {
                    let ring = self.draw_ring.lock();
                    screen.draw_snapshot.clear();
                    screen.draw_snapshot.extend(ring.iter_newest_first().copied());
                }
                self.diag.fetch_or(Self::DIAG_LOCK_RENDERER, Ordering::Relaxed);
                let mut renderer = self.renderer.lock();

                (*screen).refresh(
                    &**fb_rgb,
                    &**fb_aux,
                    &self.vc2,
                    &self.xmap0,
                    &self.cmap0,
                    &self.bt445,
                    &mut *renderer,
                    &self.diag,
                );

                let height = screen.height;
                let width  = screen.width;

                // Render main framebuffer (exact display size, no overlays)
                if let Some(ref mut r) = *renderer {
                    self.diag.fetch_or(Self::DIAG_LOOP_GL_RENDER, Ordering::Relaxed);
                    r.render(&screen.rgba, width, height);
                    self.diag.fetch_and(!Self::DIAG_LOOP_GL_RENDER, Ordering::Relaxed);
                }

                // Build debug overlay and composite it on top
                screen.render_overlay();
                if let Some(ref mut r) = *renderer {
                    self.diag.fetch_or(Self::DIAG_LOOP_GL_RENDER, Ordering::Relaxed);
                    r.render_overlay(&screen.overlay_rgba, width, height, 0);
                    self.diag.fetch_and(!Self::DIAG_LOOP_GL_RENDER, Ordering::Relaxed);
                }
                // Build and render status bar as a separate texture at the bottom
                screen.render_status_bar(&mut status_bar, &bar_stats);
                if let Some(ref mut r) = *renderer {
                    self.diag.fetch_or(Self::DIAG_LOOP_GL_RENDER, Ordering::Relaxed);
                    r.render_statusbar(&screen.statusbar_rgba, width);
                    self.diag.fetch_and(!Self::DIAG_LOOP_GL_RENDER, Ordering::Relaxed);
                }
                self.diag.fetch_and(!(Self::DIAG_LOCK_SCREEN | Self::DIAG_LOCK_RENDERER), Ordering::Relaxed);
            }

            // Timing & VBLANK
            let elapsed = start.elapsed();
            if elapsed < frame_duration {
                thread::sleep(frame_duration - elapsed);
            }

            // Assert VBLANK
            self.config.status.fetch_or(STATUS_VRINT, Ordering::Relaxed);

            // Update VC2 Working Cursor Y (at VBLANK)
            {
                self.diag.fetch_or(Self::DIAG_LOCK_VC2, Ordering::Relaxed);
                let mut vc2 = self.vc2.lock();
                vc2.regs[crate::vc2::VC2_REG_WORKING_CURSOR_Y as usize] = vc2.regs[crate::vc2::VC2_REG_CURSOR_Y_LOC as usize];
                drop(vc2);
                self.diag.fetch_and(!Self::DIAG_LOCK_VC2, Ordering::Relaxed);
            }

            // Trigger Interrupt
            {
                self.diag.fetch_or(Self::DIAG_LOCK_VBLANK_CB, Ordering::Relaxed);
                let cb = self.vblank_cb.lock().clone();
                self.diag.fetch_and(!Self::DIAG_LOCK_VBLANK_CB, Ordering::Relaxed);
                if let Some(cb) = cb { cb(true); }
            }

            thread::sleep(std::time::Duration::from_millis(1));

            self.config.status.fetch_and(!STATUS_VRINT, Ordering::Relaxed);
            {
                self.diag.fetch_or(Self::DIAG_LOCK_VBLANK_CB, Ordering::Relaxed);
                let cb = self.vblank_cb.lock().clone();
                self.diag.fetch_and(!Self::DIAG_LOCK_VBLANK_CB, Ordering::Relaxed);
                if let Some(cb) = cb { cb(false); }
            }
        }
    }

    /// Returns `true` if the register offset was recognized and updated state.
    pub(crate) fn process_register(&self, reg_offset: u32, val64: u64) -> bool {
        let ctx = unsafe { &mut *self.context.get() };
        let val = val64 as u32;
        dlog_dev!(LogModule::Rex3, "REX3 Process: Offset {:04x} ({}) Val {:08x}", reg_offset, rex3_reg_name(reg_offset), val);

        let mut matched = true;
        match reg_offset {
            // HOSTRW: store data port value; reset shift so the draw picks up pixels from MSB.
            // The actual draw/read is triggered by execute_go() when entry.go is set.
            // 64-bit write (REX3_HOSTRW0_64 = 0x0231): store full val64 directly.
            REX3_HOSTRW64 => {
                self.hostrw.store(val64, Ordering::Relaxed);
                ctx.hostcnt = 0;
                if self.draw_debug.load(Ordering::Relaxed) { self.draw_ring.lock().on_hostrw_write(); }
                if let Some(f) = self.block_log.lock().as_mut() {
                    let _ = writeln!(f, "  HOSTRW64 {:016x}", val64);
                }
            }
            REX3_HOSTRW0 => {
                // 32-bit write to HOSTRW0: update high 32 bits [63:32].
                let old = self.hostrw.load(Ordering::Relaxed);
                let new_val = (old & 0x0000_0000_FFFF_FFFF) | ((val64 & 0xFFFF_FFFF) << 32);
                self.hostrw.store(new_val, Ordering::Relaxed);
                ctx.hostcnt = 0;
                if self.draw_debug.load(Ordering::Relaxed) { self.draw_ring.lock().on_hostrw_write(); }
                if let Some(f) = self.block_log.lock().as_mut() {
                    let _ = writeln!(f, "  HOSTRW0 {:08x} -> {:016x}", val64 as u32, new_val);
                }
            }
            REX3_HOSTRW1 => {
                // 32-bit write to HOSTRW1: update low 32 bits [31:0].
                let old = self.hostrw.load(Ordering::Relaxed);
                let new_val = (old & 0xFFFF_FFFF_0000_0000) | (val64 & 0xFFFF_FFFF);
                self.hostrw.store(new_val, Ordering::Relaxed);
                ctx.hostcnt = 0;
                if self.draw_debug.load(Ordering::Relaxed) { self.draw_ring.lock().on_hostrw_write(); }
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
            REX3_WRMASK => ctx.wrmask = val,
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
            _ => {
                eprintln!("REX3 Write: unhandled reg {:04x} ({}), val {:08x}", reg_offset, rex3_reg_name(reg_offset), val);
                matched = false;
            }
        }
        matched
    }

    pub fn save_framebuffers(&self, dir: &std::path::Path) -> std::io::Result<()> {
        let rgb = unsafe { &*self.fb_rgb.get() };
        let mut bytes = Vec::with_capacity(rgb.len() * 4);
        for &word in rgb.iter() {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        std::fs::write(dir.join("rex3_rgb.bin"), &bytes)?;

        let aux = unsafe { &*self.fb_aux.get() };
        bytes.clear();
        for &word in aux.iter() {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        std::fs::write(dir.join("rex3_aux.bin"), &bytes)?;
        Ok(())
    }

    pub fn load_framebuffers(&self, dir: &std::path::Path) -> std::io::Result<()> {
        let path_rgb = dir.join("rex3_rgb.bin");
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

        let path_aux = dir.join("rex3_aux.bin");
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
        register_lock_fn("rex3::gfifo_producer",   move || r.gfifo_producer.is_locked());
        let r = self.clone();
        register_lock_fn("rex3::gfifo_consumer",   move || r.gfifo_consumer.is_locked());
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
        self.gfifo_push(GFIFOEntry { addr: GFIFO_EXIT, val: 0 });

        if let Some(handle) = self.processor_thread.lock().take() {
            if let Ok(consumer) = handle.join() {
                *self.gfifo_consumer.lock() = Some(consumer);
            }
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

        if let Some(renderer) = self.renderer.lock().as_mut() {
            renderer.stop();
        }
    }

    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) { return; }

        let mut cons_guard = self.gfifo_consumer.lock();
        let consumer = cons_guard.take().expect("GFIFO consumer missing");
        drop(cons_guard);

        let rex3 = unsafe { std::mem::transmute::<&Rex3, &'static Rex3>(self) };

        *self.processor_thread.lock() = Some(thread::Builder::new().name("REX3-Processor".to_string()).spawn(move || {
            rex3.register_processor(consumer)
        }).unwrap());

        *self.refresh_thread.lock() = Some(thread::Builder::new().name("REX3-Refresh".to_string()).spawn(move || {
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
        vec![
            ("rex".to_string(), "REX3 commands: rex status (includes JIT stats) | rex jit <on|off> | rex debug <on|off> [DEV] | rex cmap <on|off> | rex buslog <on|off> (logs to rex3.log) [DEV]".to_string()),
            ("dcb".to_string(), "DCB commands: dcb debug <on|off> [DEV]".to_string()),
            ("vc2".to_string(), "VC2 commands: vc2 status | vc2 debug <on|off> [DEV]".to_string()),
            ("block".to_string(), "Block draw logging: block debug <on|off> [DEV]".to_string()),
            ("draw".to_string(), "Draw debug overlay: draw debug <on|off>".to_string()),
            ("xmap".to_string(), "XMAP commands: xmap status | xmap debug <on|off> [DEV]".to_string()),
            ("cmap".to_string(), "CMAP commands: cmap status | cmap debug <on|off> [DEV]".to_string()),
            ("bt445".to_string(), "BT445 RAMDAC: bt445 status | bt445 identity (reset palette to linear ramp) | bt445 debug <on|off> [DEV]".to_string()),
            ("disp".to_string(), "Display debug: disp status | disp debug <on|off>".to_string()),
        ]
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
            let gfifo_pending = self.gfifo_pending.load(Ordering::Relaxed);
            writeln!(writer, "DRAW BUSY : {}  GFIFO : {}/{} entries used",
                if gfxbusy { "YES" } else { "no" },
                gfifo_pending, GFIFO_DEPTH).unwrap();
            let jit_go   = self.jit_go_count.load(Ordering::Relaxed);
            let interp_go = self.interp_go_count.load(Ordering::Relaxed);
            let total_go  = jit_go + interp_go;
            let jit_pct   = if total_go > 0 { jit_go * 100 / total_go } else { 0 };
            writeln!(writer, "GO TOTAL  : {}  JIT : {} ({}%)  INTERP : {}",
                total_go, jit_go, jit_pct, interp_go).unwrap();
            #[cfg(feature = "rex-jit")]
            {
                if let Some(ref jit) = self.rex_jit {
                    let pairs = jit.compiled_pairs();
                    let enabled = self.jit_enabled.load(Ordering::Relaxed);
                    writeln!(writer, "--- JIT : {} ---", if enabled { "enabled" } else { "DISABLED" }).unwrap();
                    writeln!(writer, "Compiled  : {}  Queued : {}",
                        pairs.len(), jit.queued_count()).unwrap();
                    if pairs.is_empty() {
                        writeln!(writer, "(no compiled shaders yet)").unwrap();
                    } else {
                        for (dm0, dm1) in &pairs {
                            writeln!(writer, "  dm0={:#010x}  dm1={:#010x}  {}  {}",
                                dm0, dm1,
                                decode_dm0(*dm0), decode_dm1(*dm1)).unwrap();
                        }
                    }
                } else {
                    writeln!(writer, "--- JIT : disabled (rex_jit not initialised) ---").unwrap();
                }
            }
            #[cfg(not(feature = "rex-jit"))]
            writeln!(writer, "--- JIT : not compiled in (build without --features rex-jit) ---").unwrap();
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
                (Self::DIAG_LOCK_GFIFO_PROD,  "gfifo_prod"),
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
                self.gfifo_pending.load(Ordering::Relaxed)).unwrap();
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

        #[cfg(feature = "rex-jit")]
        if cmd == "rex" && args[0] == "jit" {
            let val = match args.get(1).map(|s| *s) {
                Some("on")  => true,
                Some("off") => false,
                _ => return Err("Usage: rex jit <on|off>".to_string()),
            };
            self.jit_enabled.store(val, Ordering::Relaxed);
            writeln!(writer, "REX JIT dispatch: {}", if val { "enabled" } else { "disabled" }).unwrap();
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
            writeln!(writer, "  show_disp_debug: {}", screen.show_disp_debug).unwrap();
            return Ok(());
        }

        if cmd == "vc2" && args[0] == "status" {
            self.vc2.lock().print_status(&mut writer);
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
        // Check if address is within the 8KB register window
        if (addr & 0xFFFFE000) != REX3_BASE {
            return BusRead32::ok(0);
        }

        let offset = addr & (REX3_SIZE - 1);
        // Decode GO command (bit 11 set) - usually ignored for reads but address aliasing applies
        let reg_offset = offset & !0x0800;

        // HOSTRW reads: flush any pending GFIFO writes first (reads flush the pipeline).
        // SET read: return current hostrw register value (no draw triggered).
        // GO read:  return current hostrw register value, then enqueue an empty GO to
        //           trigger processing of the next pixel batch.
        if reg_offset == REX3_HOSTRW0 || reg_offset == REX3_HOSTRW1 {
            let is_go = (offset & 0x0800) != 0;
            if self.gfxbusy.load(Ordering::Acquire) || self.gfifo_pending.load(Ordering::Relaxed) > 0 {
                return BusRead32::busy();
            }
            let full = self.hostrw.load(Ordering::Relaxed);
            let val = if reg_offset == REX3_HOSTRW0 {
                (full >> 32) as u32
            } else {
                full as u32
            };
            if is_go {
                self.gfifo_push(GFIFOEntry { addr: GFIFO_PURE_GO, val: 0 });
            }
            return BusRead32::ok(val);
        }

        // Handle Config registers
        let result = match reg_offset {
            REX3_CONFIG => Some(BusRead32::ok(self.config.config.load(Ordering::Relaxed) & 0x1FFFFF)), // 21 bits
            REX3_STATUS | REX3_USER_STATUS => {
                let mut val = self.config.status.load(Ordering::Relaxed) & 0xFFFFF; // 20 bits
                val |= 3 << STATUS_VERSION_SHIFT; // REX3 version=3 (Newport XL24)
                // gfxbusy: Acquire load — pairs with Release store(false) in draw loop,
                // ensuring all context register writes by the draw thread are visible to us.
                let pending = self.gfifo_pending.load(Ordering::Relaxed);
                if self.gfxbusy.load(Ordering::Acquire) || pending > 0 {
                    val |= STATUS_GFXBUSY;
                } else {
                    val &= !STATUS_GFXBUSY;
                }

                // Map our large GFIFO onto the hardware 32-entry view:
                //   0              → 0  (empty)
                //   1 .. depth-32  → 1  (non-empty but plenty of room)
                //   depth-31 .. depth → 2..32  (last 31 entries, filling up)
                let level: u32 = if pending == 0 {
                    0
                } else {
                    let offset = pending.saturating_sub(GFIFO_DEPTH - GFIFO_HW_DEPTH);
                    (offset as u32).max(1)
                };
                val &= !STATUS_GFIFOLEVEL_MASK;
                val |= (level << STATUS_GFIFOLEVEL_SHIFT) & STATUS_GFIFOLEVEL_MASK;

                if reg_offset == REX3_STATUS {
                    // Clear VRINT on read (benign race with refresh thread's set).
                    self.config.status.fetch_and(!STATUS_VRINT, Ordering::Relaxed);
                    // backbusy_until is in dcb, CPU-thread-only — no lock needed.
                    let dcb = self.dcb.lock();
                    if let Some(until) = dcb.backbusy_until {
                        if std::time::Instant::now() < until {
                            val |= STATUS_BACKBUSY;
                        }
                        // Don't clear expired backbusy here — dcb is borrowed immutably.
                        // The next DCB addr=12 access will overwrite it anyway.
                    }
                    drop(dcb);
                }
                Some(BusRead32::ok(val & 0xFFFFF)) // Ensure 20 bits on return
            }
            // DCBMODE and DCBDATA use self.dcb mutex, handled below
            _ => None,
        };

        let result = if let Some(res) = result {
            res
        } else {
            match reg_offset {
                REX3_DCBMODE => {
                    self.diag.fetch_or(Self::DIAG_LOCK_DCB, Ordering::Relaxed);
                    let val = self.dcb.lock().dcbmode & 0x1FFFFFFF;
                    self.diag.fetch_and(!Self::DIAG_LOCK_DCB, Ordering::Relaxed);
                    dlog_dev!(LogModule::Dcb, "DCB Mode Read -> {:08x}", val);
                    BusRead32::ok(val)
                }
                REX3_DCBDATA0 => BusRead32::ok(self.dcb_read()),
                REX3_DCBDATA1 => BusRead32::ok(self.dcb_read()),
                _ => {
                    // Context registers: stall (GRXDLY) until pipeline idle, modelled as EXEC_RETRY.
                    // This lets cp0_count advance and interrupts fire between retries.
                    if self.gfxbusy.load(Ordering::Acquire) || self.gfifo_pending.load(Ordering::Relaxed) > 0 {
                        return BusRead32::busy();
                    }
                    let context = unsafe { &*self.context.get() };
                    match reg_offset {
                        REX3_DRAWMODE1 => BusRead32::ok(context.drawmode1.0), // 32 bits
                        REX3_DRAWMODE0 => BusRead32::ok(context.drawmode0.0 & 0xFFFFFF), // 24 bits
                        REX3_LSMODE => BusRead32::ok(context.lsmode.0 & 0x0FFFFFFF), // 28 bits
                        REX3_LSPATTERN => BusRead32::ok(context.lspattern), // 32 bits
                        REX3_LSPATSAVE => BusRead32::ok(context.lspatsave), // 32 bits
                        REX3_ZPATTERN => BusRead32::ok(context.zpattern), // 32 bits
                        REX3_LSSAVE => BusRead32::ok(context.lssave),
                        REX3_LSRESTORE => BusRead32::ok(context.lsrestore),
                        REX3_STEPZ => BusRead32::ok(context.stepz),
                        REX3_STALL0 => BusRead32::ok(context.stall0),
                        REX3_STALL1 => BusRead32::ok(context.stall1),
                        REX3_COLORBACK => BusRead32::ok(context.colorback), // 32 bits
                        REX3_COLORVRAM => BusRead32::ok(context.colorvram), // 32 bits
                        REX3_ALPHAREF => BusRead32::ok(context.alpharef & 0xFF), // 8 bits
                        REX3_SMASK0X => BusRead32::ok(context.smask0x), // 2c = two's complement, 16,16 = 32 bits
                        REX3_SMASK0Y => BusRead32::ok(context.smask0y), // 2c = two's complement, 16,16 = 32 bits
                        REX3_XSTART => BusRead32::ok(to16_4_7(context.xstart)),
                        REX3_YSTART => BusRead32::ok(to16_4_7(context.ystart)),
                        REX3_XEND => BusRead32::ok(to16_4_7(context.xend)),
                        REX3_YEND => BusRead32::ok(to16_4_7(context.yend)),
                        REX3_XSAVE => BusRead32::ok((context.xsave >> 11) as u32 & 0xFFFF), // 16 bits
                        REX3_XYMOVE => BusRead32::ok(context.xymove), // 16,16 = 32 bits
                        REX3_BRESD => BusRead32::ok(context.bresd & 0x7FFFFFF), // 27 bits
                        REX3_BRESS1 => BusRead32::ok(context.bress1 & 0x1FFFF), // 17 bits
                        REX3_BRESOCTINC1 => BusRead32::ok(context.bresoctinc1.0),
                        REX3_BRESRNDINC2 => BusRead32::ok(context.bresrndinc2.0),
                        REX3_BRESE1 => BusRead32::ok(context.brese1 & 0xFFFF), // 16 bits
                        REX3_BRESS2 => BusRead32::ok(context.bress2 & 0x3FFFFFF), // 26 bits
                        REX3_AWEIGHT0 => BusRead32::ok(context.aweight0), // 8 x 4 = 32 bits
                        REX3_AWEIGHT1 => BusRead32::ok(context.aweight1), // 8 x 4 = 32 bits
                        REX3_XSTARTF => BusRead32::ok(to12_4_7(context.xstart)),
                        REX3_YSTARTF => BusRead32::ok(to12_4_7(context.ystart)),
                        REX3_XENDF => BusRead32::ok(to12_4_7(context.xend)),
                        REX3_YENDF => BusRead32::ok(to12_4_7(context.yend)),
                        REX3_XSTARTI => BusRead32::ok((context.xstart >> 11) as u32 & 0xFFFF), // 16 bits integer
                        REX3_XENDF1 => BusRead32::ok(to12_4_7(context.xend)),
                        REX3_XYSTARTI => {
                            let x = (context.xstart >> 11) as u16 as u32;
                            let y = (context.ystart >> 11) as u16 as u32;
                            BusRead32::ok((x << 16) | (y & 0xFFFF))
                        }
                        REX3_XYENDI => {
                            let x = (context.xend >> 11) as u16 as u32;
                            let y = (context.yend >> 11) as u16 as u32;
                            BusRead32::ok((x << 16) | (y & 0xFFFF))
                        }
                        REX3_XSTARTENDI => {
                            let start = (context.xstart >> 11) as u16 as u32;
                            let end   = (context.xend   >> 11) as u16 as u32;
                            BusRead32::ok((start << 16) | (end & 0xFFFF))
                        }
                        REX3_COLORRED => BusRead32::ok(to_color_red(context.colorred, context.drawmode1)),
                        REX3_COLORALPHA => BusRead32::ok(to_color(context.coloralpha)),
                        REX3_COLORGRN => BusRead32::ok(to_color(context.colorgrn)),
                        REX3_COLORBLUE => BusRead32::ok(to_color(context.colorblue)),
                        REX3_SLOPERED => BusRead32::ok(to_slope_red(context.slopered)),
                        REX3_SLOPEALPHA => BusRead32::ok(to_slope(context.slopealpha)),
                        REX3_SLOPEGRN => BusRead32::ok(to_slope(context.slopegrn)),
                        REX3_SLOPEBLUE => BusRead32::ok(to_slope(context.slopeblue)),
                        REX3_WRMASK => BusRead32::ok(context.wrmask & 0xFFFFFF), // 24 bits
                        REX3_COLORI => BusRead32::ok(context.get_colori() & 0xFFFFFF), // 24 bits
                        REX3_COLORX => BusRead32::ok(to_color_red(context.colorx, context.drawmode1) & 0xFFFFFF), // 24 bits
                        REX3_SLOPERED1 => BusRead32::ok(to_slope_red(context.slopered)),
                        REX3_SMASK1X => BusRead32::ok(context.smask1x),
                        REX3_SMASK1Y => BusRead32::ok(context.smask1y),
                        REX3_SMASK2X => BusRead32::ok(context.smask2x),
                        REX3_SMASK2Y => BusRead32::ok(context.smask2y),
                        REX3_SMASK3X => BusRead32::ok(context.smask3x),
                        REX3_SMASK3Y => BusRead32::ok(context.smask3y),
                        REX3_SMASK4X => BusRead32::ok(context.smask4x),
                        REX3_SMASK4Y => BusRead32::ok(context.smask4y),
                        REX3_TOPSCAN => BusRead32::ok(context.topscan & 0x3FF), // 10 bits
                        REX3_XYWIN => BusRead32::ok(context.xywin), // 16,16 = 32 bits
                        REX3_CLIPMODE => BusRead32::ok(context.clipmode & 0x1FFF), // 13 bits
                        _ => {
                            eprintln!("REX3 Read32: unhandled reg {:04x} ({})", reg_offset, rex3_reg_name(reg_offset));
                            BusRead32::ok(0)
                        }
                    }
                }
            }
        };

        let is_dcb = reg_offset == REX3_DCBDATA0 || reg_offset == REX3_DCBDATA1 || reg_offset == REX3_DCBMODE || reg_offset == REX3_DCBRESET;
        let is_hostrw = reg_offset == REX3_HOSTRW0 || reg_offset == REX3_HOSTRW1;
        // STATUS/USER_STATUS are polled constantly; suppress to avoid spam.
        let is_status = reg_offset == REX3_STATUS || reg_offset == REX3_USER_STATUS;
        if !is_status {
            if result.is_ok() {
                let val = result.data;
                let mut dbg = self.debug_state.lock();
                if dbg.last_offset == Some(offset) && dbg.last_val == val {
                    dbg.count += 1;
                } else {
                    if dbg.count > 0 {
                        dlog_dev!(LogModule::Rex3, "... repeated {} times", dbg.count);
                    }
                    dbg.last_offset = Some(offset);
                    dbg.last_val = val;
                    dbg.count = 0;
                    dlog_dev!(LogModule::Rex3, "REX3 Read32: Offset {:04x} (Reg {:04x} {}) -> {:08x}", offset, reg_offset, rex3_reg_name(reg_offset), val);
                }
            } else {
                dlog_dev!(LogModule::Rex3, "REX3 Read32: Offset {:04x} (Reg {:04x} {}) -> err {:08x}", offset, reg_offset, rex3_reg_name(reg_offset), result.status);
            }
        }

        // GO bit on a non-HOSTRW read still triggers a draw (HOSTRW handled above).
        if (offset & 0x0800) != 0 {
            self.gfifo_push(GFIFOEntry { addr: GFIFO_PURE_GO, val: 0 });
        }

        result
    }

    fn read8(&self, addr: u32) -> BusRead8 {
        // Check if address is within the 8KB register window
        if (addr & 0xFFFFE000) != REX3_BASE {
            return BusRead8::ok(0);
        }

        let offset = addr & (REX3_SIZE - 1);
        let is_dcb = (offset & !7) == REX3_DCBDATA0;
        let res = if is_dcb {
            let val = self.dcb_read() as u8;
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
        // Check if address is within the 8KB register window
        if (addr & 0xFFFFE000) != REX3_BASE {
            return BUS_OK;
        }

        let offset = addr & (REX3_SIZE - 1);
        let is_dcb = (offset & !7) == REX3_DCBDATA0;
        dlog_dev!(LogModule::Rex3, "REX3 Write8: Offset {:04x} Val {:02x}", offset, val);

        if is_dcb {
            dlog_dev!(LogModule::Dcb, "DCB Write8: Offset {:04x} Val {:02x} -> dcb_write({:08x})", offset, val, val as u32);
            self.dcb_write(val as u32);
            return BUS_OK;
        }
        eprintln!("REX3 Write8: unhandled offset {:04x} val {:02x}", offset, val);
        BUS_ERR
    }

    fn read16(&self, addr: u32) -> BusRead16 {
        // Check if address is within the 8KB register window
        if (addr & 0xFFFFE000) != REX3_BASE {
            return BusRead16::ok(0);
        }

        let offset = addr & (REX3_SIZE - 1);
        let is_dcb = (offset & !7) == REX3_DCBDATA0;
        let res = if is_dcb {
            let val = self.dcb_read() as u16;
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
        // Check if address is within the 8KB register window
        if (addr & 0xFFFFE000) != REX3_BASE {
            return BUS_OK;
        }

        let offset = addr & (REX3_SIZE - 1);
        let is_dcb = (offset & !7) == REX3_DCBDATA0;
        dlog_dev!(LogModule::Rex3, "REX3 Write16: Offset {:04x} Val {:04x}", offset, val);

        if is_dcb {
            dlog_dev!(LogModule::Dcb, "DCB Write16: Offset {:04x} Val {:04x} -> dcb_write({:08x})", offset, val, val as u32);
            self.dcb_write(val as u32);
            return BUS_OK;
        }
        eprintln!("REX3 Write16: unhandled offset {:04x} val {:04x}", offset, val);
        BUS_ERR
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        // Check if address is within the 8KB register window
        if (addr & 0xFFFFE000) != REX3_BASE {
            return BUS_OK;
        }

        let offset = addr & (REX3_SIZE - 1);
        
        // Decode GO command (bit 11 set)
        let is_go = (offset & 0x0800) != 0;
        let reg_offset = offset & !0x0800;

        let is_dcb = reg_offset == REX3_DCBDATA0 || reg_offset == REX3_DCBDATA1 || reg_offset == REX3_DCBMODE || reg_offset == REX3_DCBRESET;
        let is_hostrw = reg_offset == REX3_HOSTRW0 || reg_offset == REX3_HOSTRW1;
        dlog_dev!(LogModule::Rex3, "REX3 Write32: Offset {:04x} (Reg {:04x} {}) Val {:08x}", offset, reg_offset, rex3_reg_name(reg_offset), val);

        // HOSTRW: route through GFIFO so the draw thread sees it in order.
        // 32-bit write: val64=val zero-extended. GO write triggers draw; SET write just loads.
        if is_hostrw {
            self.gfifo_push(GFIFOEntry { addr: offset, val: val as u64 });
            return BUS_OK;
        }

        // Handle registers that need early return before GFIFO
        let early_return = matches!(reg_offset,
            REX3_CONFIG | REX3_STATUS | REX3_USER_STATUS |
            REX3_DCBRESET | REX3_DCBMODE | REX3_DCBDATA0 | REX3_DCBDATA1);
        if early_return {
            match reg_offset {
                REX3_CONFIG => { self.config.config.store(val, Ordering::Relaxed); }
                REX3_STATUS | REX3_USER_STATUS => {} // writes ignored
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
                REX3_DCBDATA0 | REX3_DCBDATA1 => { dlog_dev!(LogModule::Dcb, "DCB Write32: Offset {:04x} Val {:08x} -> dcb_write({:08x})", offset, val, val); self.dcb_write(val); }
                REX3_DCBRESET => { *self.dcb.lock() = Rex3DcbState::default(); }
                _ => {}
            }
            if is_go { self.gfifo_push(GFIFOEntry { addr: GFIFO_PURE_GO, val: 0 }); }
            return BUS_OK;
        }

        // Pack into GFIFO
        self.gfifo_push(GFIFOEntry { addr: offset, val: val as u64 });

        BUS_OK
    }

    fn read64(&self, addr: u32) -> BusRead64 {
        // Check if address is within the 8KB register window
        if (addr & 0xFFFFE000) != REX3_BASE {
            return BusRead64::ok(0);
        }

        let offset = addr & (REX3_SIZE - 1);
        let is_go64r = (offset & 0x0800) != 0;
        let reg_offset64r = offset & !0x0800;
        if reg_offset64r == REX3_HOSTRW0 {
            if self.gfxbusy.load(Ordering::Acquire) || self.gfifo_pending.load(Ordering::Relaxed) > 0 {
                return BusRead64::busy();
            }
            let val = self.hostrw.load(Ordering::Relaxed);
            if is_go64r {
                // Enqueue a pure-go entry: no register update, just trigger next pixel batch.
                self.gfifo_push(GFIFOEntry { addr: GFIFO_PURE_GO, val: 0 });
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

    fn write64(&self, addr: u32, val: u64) -> u32 {
        // Check if address is within the 8KB register window
        if (addr & 0xFFFFE000) != REX3_BASE {
            return BUS_OK;
        }

        let offset = addr & (REX3_SIZE - 1);
        let is_go = (offset & 0x0800) != 0;
        let reg_offset64 = offset & !0x0800;
        if reg_offset64 == REX3_HOSTRW0 {
            // Encode as REX3_HOSTRW64 (0x0231) + GO bit if present.
            // addr bit 0 = is_64bit, bit 11 = GO.
            self.gfifo_push(GFIFOEntry { addr: REX3_HOSTRW64 | (offset & 0x0800), val });
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
        unsafe { *self.context.get() = Rex3Context::default(); }
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

#[cfg(test)]
#[path = "rex3_tests.rs"]
mod tests;
