use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use crossbeam_utils::CachePadded;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BUS_BUSY, BusDevice, Device, Resettable, Saveable};
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
/// The GO bit (0x0800) is already set in the value itself — 0x1FF0 has bit 11
/// — mirroring GFIFO_PURE_GO's convention. Do not re-OR it in; that reads as
/// if it were setting the bit when it is merely a no-op.
pub const REX3_DMA_PURE_GO: u32 = 0x1FF0;
/// REX3_DMA_PURE_GO with the GO bit stripped — the reg_offset seen by process_register.
pub const REX3_DMA_PURE_GO_REG: u32 = REX3_DMA_PURE_GO & !0x0800;

/// Capacity of the HOSTRW data port array, in 64-bit words (1 MiB of u64).
///
/// A transfer larger than this is chunked by the caller — which it would have
/// to be at any size, so the size trades round trips against footprint rather
/// than capping anything.
pub const HOSTRW_BUF_QWORDS: usize = (1024 * 1024) / 8;

/// Internal-only offset (as REX3_DMA_PURE_GO, one slot below it): a **batched**
/// HOSTRW write. `val` carries the qword count; the payload itself has already
/// carried by the GFIFO entries that follow the token, which the register
/// pushed.
///
/// One token stands in for what used to be N separate `dma_write64` pushes, so
/// a full-line blit costs one GFIFO round trip instead of one per 8 bytes. The
/// register processor sets `host_cursor = 0`, `host_len = count`, and the GO
/// bit runs the shader once over the whole run.
pub const REX3_DMA_BATCH_W: u32 = 0x1FE0; // bit 11 (GO) already set
/// REX3_DMA_BATCH_W with the GO bit stripped — the reg_offset seen by process_register.
pub const REX3_DMA_BATCH_W_REG: u32 = REX3_DMA_BATCH_W & !0x0800;

/// Internal-only offset: a **batched** HOSTRW read. `val` carries the qword
/// count the shader should produce into `Rex3Context::hostrw`.
///
/// Replaces one `wait_idle()` full-pipeline drain per qword with one per batch:
/// the consumer sets `host_cursor = 0`, `host_len = count`, runs the shader
/// once, and `dma_read64_bulk` then copies the filled buffer out.
pub const REX3_DMA_BATCH_R: u32 = 0x1FD0; // bit 11 (GO) already set
/// REX3_DMA_BATCH_R with the GO bit stripped — the reg_offset seen by process_register.
pub const REX3_DMA_BATCH_R_REG: u32 = REX3_DMA_BATCH_R & !0x0800;

/// Address marking a GFIFO entry as batch *payload* rather than a register
/// write. Entries carrying these follow a `REX3_DMA_BATCH_W` token and are
/// consumed by it; encountering one on its own is inert.
///
/// No GO bit, so `process_register` sees this value directly as reg_offset.
pub const GFIFO_PAYLOAD: u32 = 0xFFFF_0003;
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
    let adrmode = match dm.adrmode() {
        DRAWMODE0_ADRMODE_SPAN => "SPAN",
        DRAWMODE0_ADRMODE_BLOCK => "BLOCK",
        DRAWMODE0_ADRMODE_I_LINE => "ILINE",
        DRAWMODE0_ADRMODE_F_LINE => "FLINE",
        DRAWMODE0_ADRMODE_A_LINE => "ALINE",
        _ => "?",
    };
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
    let depth  = match dm.drawdepth() {
        DRAWMODE1_DRAWDEPTH_4 => "4bpp",
        DRAWMODE1_DRAWDEPTH_8 => "8bpp",
        DRAWMODE1_DRAWDEPTH_12 => "12bpp",
        DRAWMODE1_DRAWDEPTH_24 => "24bpp",
        _ => "?",
    };
    let hdepth = match dm.hostdepth() {
        DRAWMODE1_HOSTDEPTH_12 => "12bpp",
        DRAWMODE1_HOSTDEPTH_8 => "8bpp",
        DRAWMODE1_HOSTDEPTH_4 => "4bpp",
        DRAWMODE1_HOSTDEPTH_32 => "32bpp",
        _ => "?",
    };
    let logicop = match dm.logicop()   { 0=>"ZERO",1=>"AND",2=>"ANDR",3=>"SRC",4=>"ANDI",5=>"DST",
        6=>"XOR",7=>"OR",8=>"NOR",9=>"XNOR",10=>"NDST",11=>"ORR",12=>"NSRC",13=>"ORI",14=>"NAND",15=>"ONE", _=>"?" };
    // rex3 spec Tables 13/14: SFACTOR 010/011 select the *destination* colour while
    // DFACTOR 010/011 select the *source* colour; 100/101 are source alpha in both.
    // 110/111 are not defined by the hardware.
    let sfactor = ["ZERO","ONE","DC","MDC","SA","MSA","?6","?7"];
    let dfactor = ["ZERO","ONE","SC","MSC","SA","MSA","?6","?7"];
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

pub(crate) fn rex3_reg_name(offset: u32) -> &'static str {
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
        // Internal GFIFO sentinels. Not hardware registers, but they travel the
        // same path and reach the bus log with the GO bit already stripped, so
        // without these they show up as a bare "UNKNOWN" — which is exactly
        // what a real unmapped register looks like.
        REX3_DMA_PURE_GO_REG => "DMA_PURE_GO",
        REX3_DMA_BATCH_W_REG => "DMA_BATCH_W",
        REX3_DMA_BATCH_R_REG => "DMA_BATCH_R",
        GFIFO_PURE_GO_REG => "GFIFO_PURE_GO",
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
// ADRMODE field values — what `dm0_adrmode` returns. Compare against these.
pub const DRAWMODE0_ADRMODE_SPAN: u32 = 0x0;
pub const DRAWMODE0_ADRMODE_BLOCK: u32 = 0x1;
pub const DRAWMODE0_ADRMODE_I_LINE: u32 = 0x2;
pub const DRAWMODE0_ADRMODE_F_LINE: u32 = 0x3;
pub const DRAWMODE0_ADRMODE_A_LINE: u32 = 0x4;

// Register-position forms (`_SH`), for OR-ing into a DRAWMODE0 word.
pub const DRAWMODE0_ADRMODE_SPAN_SH: u32 = DRAWMODE0_ADRMODE_SPAN << DRAWMODE0_ADRMODE_SHIFT;
pub const DRAWMODE0_ADRMODE_BLOCK_SH: u32 = DRAWMODE0_ADRMODE_BLOCK << DRAWMODE0_ADRMODE_SHIFT;
pub const DRAWMODE0_ADRMODE_I_LINE_SH: u32 = DRAWMODE0_ADRMODE_I_LINE << DRAWMODE0_ADRMODE_SHIFT;
pub const DRAWMODE0_ADRMODE_F_LINE_SH: u32 = DRAWMODE0_ADRMODE_F_LINE << DRAWMODE0_ADRMODE_SHIFT;
pub const DRAWMODE0_ADRMODE_A_LINE_SH: u32 = DRAWMODE0_ADRMODE_A_LINE << DRAWMODE0_ADRMODE_SHIFT;

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

/// SFACTOR/DFACTOR selector values (spec Tables 13/14). BF_DC/BF_MDC name the
/// destination colour when used as SFACTOR; the same encodings name the source
/// colour as DFACTOR (BF_SC/BF_MSC). BF_SA/BF_MSA mean source alpha in both.
pub const DRAWMODE1_BF_ZERO: u32 = 0;
pub const DRAWMODE1_BF_ONE:  u32 = 1;
/// The *other* operand's colour: destination when used as SFACTOR, source when
/// used as DFACTOR. Named BF_DC/BF_SC respectively in the spec tables.
pub const DRAWMODE1_BF_OC:   u32 = 2;
/// 255 minus [`DRAWMODE1_BF_OC`] (BF_MDC / BF_MSC).
pub const DRAWMODE1_BF_MOC:  u32 = 3;
pub const DRAWMODE1_BF_SA:   u32 = 4;
pub const DRAWMODE1_BF_MSA:  u32 = 5;

pub const DRAWMODE1_PLANES_NONE: u32 = 0;
pub const DRAWMODE1_PLANES_RGB: u32 = 1;
pub const DRAWMODE1_PLANES_RGBA: u32 = 2;
pub const DRAWMODE1_PLANES_OLAY: u32 = 4;
pub const DRAWMODE1_PLANES_PUP: u32 = 5;
pub const DRAWMODE1_PLANES_CID: u32 = 6;

/// COMPARE=0x7 (all three relations OR'ed) — afunction always passes, i.e. disabled.
/// Hardware reset default; real drawmode1 words always carry this explicitly.
/// COMPARE field position in DRAWMODE1.
pub const DRAWMODE1_COMPARE_SHIFT: u32 = 12;
/// COMPARE=0x7 (all three relations OR'ed) — afunction always passes.
// DRAWDEPTH: bits per pixel in the framebuffer plane.
pub const DRAWMODE1_DRAWDEPTH_4: u32 = 0;
pub const DRAWMODE1_DRAWDEPTH_8: u32 = 1;
pub const DRAWMODE1_DRAWDEPTH_12: u32 = 2;
pub const DRAWMODE1_DRAWDEPTH_24: u32 = 3;

// HOSTDEPTH: bits per pixel in a host transfer. **Not the same encoding as
// DRAWDEPTH** — the order is 12/8/4/32, not 4/8/12/24. Mixing the two silently
// picks the wrong slot width, which is why both have named constants.
pub const DRAWMODE1_HOSTDEPTH_12: u32 = 0;
pub const DRAWMODE1_HOSTDEPTH_8: u32 = 1;
pub const DRAWMODE1_HOSTDEPTH_4: u32 = 2;
pub const DRAWMODE1_HOSTDEPTH_32: u32 = 3;

// COMPARE is three OR-able relation bits (spec §3.8.1): LT | EQ | GT. All three
// set (0x7) means every comparison passes, i.e. afunction disabled.
pub const DRAWMODE1_COMPARE_NEVER: u32 = 0x0;
pub const DRAWMODE1_COMPARE_LT:    u32 = 0x1;
pub const DRAWMODE1_COMPARE_EQ:    u32 = 0x2;
pub const DRAWMODE1_COMPARE_LE:    u32 = 0x3;
pub const DRAWMODE1_COMPARE_GT:    u32 = 0x4;
pub const DRAWMODE1_COMPARE_NE:    u32 = 0x5;
pub const DRAWMODE1_COMPARE_GE:    u32 = 0x6;
pub const DRAWMODE1_COMPARE_DISABLE: u32 = 0x7;
pub const DRAWMODE1_COMPARE_DISABLE_SH: u32 =
    DRAWMODE1_COMPARE_DISABLE << DRAWMODE1_COMPARE_SHIFT;

/// LOGICOP field position in DRAWMODE1.
pub const DRAWMODE1_LOGICOP_SHIFT: u32 = 28;
pub const DRAWMODE1_LOGICOP_MASK: u32 = 0xF << DRAWMODE1_LOGICOP_SHIFT;

// LOGICOP field values — what `dm1_logicop` returns. Compare against these;
// the `_SH` forms below are for OR-ing into a DRAWMODE1 word.
pub const DRAWMODE1_LOGICOP_ZERO: u32 = 0;
pub const DRAWMODE1_LOGICOP_AND: u32 = 1;
pub const DRAWMODE1_LOGICOP_ANDR: u32 = 2;
pub const DRAWMODE1_LOGICOP_SRC: u32 = 3;
pub const DRAWMODE1_LOGICOP_ANDI: u32 = 4;
pub const DRAWMODE1_LOGICOP_DST: u32 = 5;
pub const DRAWMODE1_LOGICOP_XOR: u32 = 6;
pub const DRAWMODE1_LOGICOP_OR: u32 = 7;
pub const DRAWMODE1_LOGICOP_NOR: u32 = 8;
pub const DRAWMODE1_LOGICOP_XNOR: u32 = 9;
pub const DRAWMODE1_LOGICOP_NDST: u32 = 10;
pub const DRAWMODE1_LOGICOP_ORR: u32 = 11;
pub const DRAWMODE1_LOGICOP_NSRC: u32 = 12;
pub const DRAWMODE1_LOGICOP_ORI: u32 = 13;
pub const DRAWMODE1_LOGICOP_NAND: u32 = 14;
pub const DRAWMODE1_LOGICOP_ONE: u32 = 15;

pub const DRAWMODE1_LOGICOP_ZERO_SH: u32 = DRAWMODE1_LOGICOP_ZERO << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_AND_SH: u32 = DRAWMODE1_LOGICOP_AND << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_ANDR_SH: u32 = DRAWMODE1_LOGICOP_ANDR << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_SRC_SH: u32 = DRAWMODE1_LOGICOP_SRC << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_ANDI_SH: u32 = DRAWMODE1_LOGICOP_ANDI << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_DST_SH: u32 = DRAWMODE1_LOGICOP_DST << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_XOR_SH: u32 = DRAWMODE1_LOGICOP_XOR << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_OR_SH: u32 = DRAWMODE1_LOGICOP_OR << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_NOR_SH: u32 = DRAWMODE1_LOGICOP_NOR << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_XNOR_SH: u32 = DRAWMODE1_LOGICOP_XNOR << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_NDST_SH: u32 = DRAWMODE1_LOGICOP_NDST << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_ORR_SH: u32 = DRAWMODE1_LOGICOP_ORR << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_NSRC_SH: u32 = DRAWMODE1_LOGICOP_NSRC << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_ORI_SH: u32 = DRAWMODE1_LOGICOP_ORI << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_NAND_SH: u32 = DRAWMODE1_LOGICOP_NAND << DRAWMODE1_LOGICOP_SHIFT;
pub const DRAWMODE1_LOGICOP_ONE_SH: u32 = DRAWMODE1_LOGICOP_ONE << DRAWMODE1_LOGICOP_SHIFT;

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
/// SMASK0 enable — the window-relative scissor (bit 0 of ENSMASK).
pub const CLIPMODE_ENSMASK_SMASK0: u32 = 0x01;
/// SMASK1-4 enables — the screen-absolute scissors (bits 1-4). A pixel passes
/// if it is inside *any* enabled one.
pub const CLIPMODE_ENSMASK_SMASK1_4: u32 = 0x1E;
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
pub(crate) const REX3_BRES_OCTANTS: [(i32, i32, i32, i32, bool); 8] = [
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
    if !drawmode1.rgbmode() && drawmode1.drawdepth() == DRAWMODE1_DRAWDEPTH_12 {
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

    /// Batched form of `on_hostrw_write`: one DMA token delivers `n` words in a
    /// single call, so counting it as one write makes the overlay report
    /// `0/4802` (or `1/4802`) for a transfer that actually carried everything.
    pub fn on_hostrw_writes(&mut self, n: u32) {
        if let Some(idx) = self.pending {
            let r = &mut self.entries[idx];
            if r.expected_words > 0 {
                r.hostrw_writes += n;
            } else {
                r.spurious_writes += n;
            }
        }
    }

    /// Batched form of `on_hostrw_read`.
    pub fn on_hostrw_reads(&mut self, n: u32) {
        if let Some(idx) = self.pending {
            let r = &mut self.entries[idx];
            if r.expected_words > 0 {
                r.hostrw_reads += n;
            } else {
                r.spurious_reads += n;
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

/// The HOSTRW data port's backing array.
///
/// A newtype purely so `Rex3Context` can keep `#[derive(Default, Debug)]`:
/// Rust implements neither for arrays longer than 32. Transparent repr, so the
/// field offset the JIT computes is the array's own address.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct HostRwArray(pub [u64; HOSTRW_BUF_QWORDS]);

impl Default for HostRwArray {
    fn default() -> Self { Self([0; HOSTRW_BUF_QWORDS]) }
}

impl std::fmt::Debug for HostRwArray {
    /// Prints the length, not a megabyte of zeroes.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HostRwArray[{}]", HOSTRW_BUF_QWORDS)
    }
}

impl std::ops::Index<usize> for HostRwArray {
    type Output = u64;
    #[inline(always)]
    fn index(&self, i: usize) -> &u64 { &self.0[i] }
}

impl std::ops::IndexMut<usize> for HostRwArray {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut u64 { &mut self.0[i] }
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
    /// Index of the next qword in `hostrw`. Always <= `host_len`.
    ///
    /// Not snapshotted, and correctly so: a transfer lives entirely inside one
    /// `execute_go`, which runs on the GFIFO consumer thread with the queue
    /// drained, so no snapshot can observe one in flight.
    pub host_cursor: u32,
    /// Number of valid qwords in `hostrw` for the current transfer.
    ///
    /// A PIO register write sets this to 1; a DMA batch sets it to the batch
    /// size. Zero means nothing is loaded.
    pub host_len: u32,
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

    /// Back-pointer to the owning `Rex3`, for host services the draw path needs
    /// but cannot compute from the context alone (block logging, the host FIFO).
    ///
    /// Lets the draw functions take the same arguments the compiled shader does
    /// — `(ctx, fb_rgb, fb_aux)` — instead of threading a `&Rex3` through every
    /// call. Null until `Rex3::new` wires it up; the accessors below treat null
    /// as "no host attached" so a bare `Rex3Context` (tests, snapshots) is still
    /// usable.
    ///
    /// Not part of the register state: excluded from snapshots and never
    /// compared. `Rex3Context` is `Copy`, and copies share the pointer, which is
    /// correct — they all refer to the same device.
    pub host: *const Rex3,

    /// The HOSTRW data port — an **array**, not a scalar.
    ///
    /// Every access is `hostrw[host_cursor]`: a PIO register write is just
    /// `host_len = 1` with the cursor at 0, and a DMA transfer is the same
    /// thing with a larger length. That is the whole point of making this an
    /// array — there is no batched-vs-single branch anywhere downstream, only
    /// a count. It is inline storage, so it always exists and can never be
    /// absent.
    ///
    /// **Must stay the last field.** `repr(C)` lays fields out in declaration
    /// order, and the JIT addresses every scalar by `offset_of!` as a Cranelift
    /// `Offset32` immediate. With a megabyte sitting mid-struct, everything
    /// after it lands at a ~1 MB offset and the generated loads/stores go
    /// wrong — which showed up as smask/zpattern/cid JIT tests failing while
    /// the interpreter stayed correct. Keeping the array last leaves every
    /// scalar at a small offset and the array's own base is reached by a
    /// register anyway.
    pub hostrw: HostRwArray,
}

// Safety: `host` is only ever set to the owning Rex3, which outlives every
// context that points at it, and is only dereferenced from the draw path (the
// GFIFO consumer thread) for services that take their own locks.
unsafe impl Send for Rex3Context {}
unsafe impl Sync for Rex3Context {}

impl Rex3Context {
    /// Power-on/reset state: like `Default::default()`, but with DRAWMODE1.COMPARE
    /// at its documented hardware reset value (0x7 = afunction disabled/always-pass).
    /// `#[derive(Default)]` zero-inits DrawMode1, which would leave COMPARE=0x0
    /// (always-kill) — real REX3 boots with afunction off until software sets it.
    pub fn power_on_default() -> Self {
        let mut ctx = Self::default();
        ctx.drawmode1 = DrawMode1(ctx.drawmode1.0 | (0x7 << 12));
        // The data port always holds at least the one word PIO uses. A zero
        // count would make the shader treat a plain HOSTRW register write as an
        // empty transfer; 1 is the resting state, and only a batch token
        // raises it.
        ctx.host_len = 1;
        ctx
    }

    // ── HOSTRW array access ─────────────────────────────────────────────────
    //
    // Every read/write of the data port goes through these. `host_cursor` picks
    // the element; PIO leaves it at 0 with `host_len == 1`, a DMA batch walks it
    // across the run. Nothing downstream distinguishes the two cases.

    /// The current HOSTRW word — `hostrw[host_cursor]`.
    ///
    /// For the resting `host_len == 1` the cursor is always 0, so this is
    /// element 0: the same slot the compiled shaders address by fixed offset.
    #[inline(always)]
    pub fn hostrw_get(&self) -> u64 {
        self.hostrw[self.hostrw_index()]
    }

    /// Store to the current HOSTRW word.
    #[inline(always)]
    pub fn hostrw_set(&mut self, val: u64) {
        let i = self.hostrw_index();
        self.hostrw[i] = val;
    }

    /// The element the port currently addresses.
    ///
    /// Clamped to the last valid word: the cursor may sit one past the end
    /// after the final `hostrw_advance`, and a PIO read after a transfer must
    /// still see the word that was written.
    #[inline(always)]
    fn hostrw_index(&self) -> usize {
        let last = self.host_len.saturating_sub(1);
        (self.host_cursor.min(last) as usize).min(HOSTRW_BUF_QWORDS - 1)
    }

    /// Arm the port for a single-word PIO access: cursor 0, length 1.
    ///
    /// This is what makes a register write indistinguishable from a one-word
    /// DMA batch downstream.
    #[inline(always)]
    pub fn hostrw_arm_single(&mut self) {
        self.host_cursor = 0;
        self.host_len = 1;
    }

    /// Has the whole transfer been consumed?
    ///
    /// The cursor counts words *taken*, so it runs from 0 up to `host_len`.
    /// The walkers call this at a word boundary, after `fetch_host_pixel` has
    /// loaded a word and stepped past it, so `cursor == host_len` means the
    /// last word has been used.
    ///
    /// With the resting `host_len == 1` this is true after one word, which
    /// reproduces the old one-word-per-GO rule exactly.
    #[inline(always)]
    pub fn hostrw_drained(&self) -> bool {
        self.host_cursor >= self.host_len
    }


    /// Step past the word just taken.
    ///
    /// The cursor may reach `host_len` (one past the end); `hostrw_index`
    /// clamps, so an over-long primitive re-reads the last word instead of
    /// running off the array — the same thing the scalar port did when the CPU
    /// under-fed HOSTRW. Saturates there rather than wrapping.
    #[inline(always)]
    pub fn hostrw_advance(&mut self) {
        if self.host_cursor < self.host_len {
            self.host_cursor += 1;
        }
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

    /// Try to push an entry without blocking. Returns `false` if another
    /// producer holds the lock or the queue is full — the caller should report
    /// back-pressure and retry rather than spin here.
    ///
    /// Spinning inside the CPU's store path is what this exists to avoid. That
    /// spin runs with no interrupt servicing, so a sustained full queue starves
    /// IP7 delivery — and because the guest's own clock is driven by IP7, it
    /// also *dilates guest time*: wall-clock advances while guest-visible time
    /// does not. Any guest-side benchmark then reports inflated throughput, and
    /// inflated most for whatever configuration spins most. Returning `false`
    /// lets the bus write report `BUS_BUSY` (== `EXEC_RETRY`), so the CPU leaves
    /// the store, re-enters `step()` — sampling interrupts in `step_preamble!`
    /// — and re-dispatches the same instruction. Nothing is lost by not making
    /// progress here: if the queue is full the CPU cannot retire this store
    /// anyway.
    #[inline]
    pub fn try_push(&self, addr: u32, val: u64) -> bool {
        // Acquire the spinlock — uncontested in the common case (one active
        // producer: IRIX only drives DMA for pixmap blits, never while the CPU
        // is writing REX3 registers), so a failure here is rare.
        if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return false;
        }
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = tail.wrapping_add(1) & GFIFO_MASK;
        let mut cached_head = self.shadow_head.get();
        if next_tail == cached_head {
            cached_head = self.head.load(Ordering::Acquire);
            self.shadow_head.set(cached_head);
            if next_tail == cached_head {
                // Full — release the lock and let the caller retry once the
                // consumer has drained something.
                self.lock.store(false, Ordering::Release);
                return false;
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
        true
    }

    /// Push two consecutive register writes as one atomic unit.
    ///
    /// A 64-bit store to REX3 is two register writes, and IRIX/GL issues them
    /// constantly — coordinate pairs, colour pairs, Bresenham terms. Pushing
    /// both under one lock acquisition rather than two costs three atomics
    /// instead of six and skips a second trip through the `write32` register
    /// match.
    ///
    /// **It also fixes a real bug.** The old path called `write32` twice and
    /// returned the second one's status, so a queue that filled between them
    /// left the first word pushed and still reported `BUS_BUSY` — and the CPU
    /// re-executes the *whole* store on retry, pushing that first word a second
    /// time. Duplicated register writes into the GFIFO, exactly what
    /// `try_push`'s "commit no other state first" rule exists to prevent. Here
    /// the capacity check covers both slots before either is written, so the
    /// pair either lands completely or not at all.
    #[inline]
    pub fn try_push2(&self, addr0: u32, val0: u64, addr1: u32, val1: u64) -> bool {
        if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return false;
        }
        let tail = self.tail.load(Ordering::Relaxed);
        let next = tail.wrapping_add(1) & GFIFO_MASK;
        let next2 = tail.wrapping_add(2) & GFIFO_MASK;
        // Room for BOTH before writing either: a partial push is what the old
        // two-`write32` path got wrong.
        let mut cached_head = self.shadow_head.get();
        if next == cached_head || next2 == cached_head {
            cached_head = self.head.load(Ordering::Acquire);
            self.shadow_head.set(cached_head);
            if next == cached_head || next2 == cached_head {
                self.lock.store(false, Ordering::Release);
                return false;
            }
        }
        // SAFETY: we hold the lock; no other producer touches these slots.
        unsafe {
            let slot0 = self.buf.as_ptr().add(tail) as *mut GFIFOEntry;
            (*slot0).addr = addr0;
            (*slot0).val  = val0;
            let slot1 = self.buf.as_ptr().add(next) as *mut GFIFOEntry;
            (*slot1).addr = addr1;
            (*slot1).val  = val1;
        }
        // One Release publishes both slots: the consumer's Acquire on tail
        // orders it after every write above.
        self.tail.store(next2, Ordering::Release);
        self.lock.store(false, Ordering::Release);
        true
    }

    /// Push a batch token followed by its payload words, as one atomic unit.
    ///
    /// The token carries the count; the `vals.len()` entries after it carry the
    /// data, and the consumer streams them into `ctx.hostrw[]`. Payload entries
    /// reuse `GFIFO_PAYLOAD` as their address so a stray read of one outside a
    /// batch is inert rather than being mistaken for a register write.
    ///
    /// Capacity for token + payload is checked before a single slot is written,
    /// so this is all-or-nothing in the same way `try_push2` is: the DMA worker
    /// has no EXEC_RETRY, and a partial commit followed by a retry would
    /// duplicate the prefix.
    ///
    /// Blocks rather than reporting busy. The batch may be larger than the
    /// queue, so "wait for room" is the only workable contract; callers are the
    /// DMA worker, which has nothing better to do, never the CPU store path.
    pub fn push_batch(&self, token: u32, count_val: u64, vals: &[u64]) {
        let need = vals.len() + 1;
        assert!(need < GFIFO_DEPTH, "batch of {} exceeds GFIFO capacity", vals.len());
        loop {
            if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
                std::hint::spin_loop();
                continue;
            }
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.head.load(Ordering::Acquire);
            self.shadow_head.set(head);
            // Free slots, keeping the one-empty-slot invariant the ring uses to
            // distinguish full from empty.
            let used = tail.wrapping_sub(head) & GFIFO_MASK;
            let free = GFIFO_MASK - used;
            if free < need {
                // Not enough room yet — drop the lock so the consumer can drain.
                self.lock.store(false, Ordering::Release);
                std::hint::spin_loop();
                continue;
            }
            // SAFETY: we hold the lock and have verified capacity for all
            // `need` slots; no other producer touches them.
            unsafe {
                let slot = self.buf.as_ptr().add(tail) as *mut GFIFOEntry;
                (*slot).addr = token;
                (*slot).val  = count_val;
                let mut idx = tail;
                for &v in vals {
                    idx = idx.wrapping_add(1) & GFIFO_MASK;
                    let slot = self.buf.as_ptr().add(idx) as *mut GFIFOEntry;
                    (*slot).addr = GFIFO_PAYLOAD;
                    (*slot).val  = v;
                }
            }
            let new_tail = tail.wrapping_add(need) & GFIFO_MASK;
            // One Release publishes the token and every payload slot.
            self.tail.store(new_tail, Ordering::Release);
            self.lock.store(false, Ordering::Release);
            return;
        }
    }

    /// Push an entry, spinning until it fits. Safe to call from multiple
    /// producers concurrently.
    ///
    /// For callers with no way to report back-pressure: shutdown sentinels, and
    /// MC's VDMA worker thread, which has no EXEC_RETRY mechanism of its own.
    /// The CPU store path uses `try_push` instead — see its doc comment.
    #[inline]
    pub fn push(&self, addr: u32, val: u64) {
        while !self.try_push(addr, val) {
            std::hint::spin_loop();
        }
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
    /// Consume up to `n` payload entries that follow a batch token, copying
    /// their values into `dst`. Returns how many were taken.
    ///
    /// The producer wrote the token and all of its payload under one lock with
    /// a single Release on tail, so once the token is visible every payload
    /// entry behind it is too — this never has to wait.
    /// Test hook: pretend only `t` entries have been published yet.
    ///
    /// `push_batch` publishes a whole batch with one Release, so a consumer
    /// that sees the token always sees the payload too — which makes it
    /// impossible to test the drain loop's mid-batch behaviour honestly
    /// without this. Not compiled into a normal build.
    #[cfg(test)]
    pub fn rewind_tail_for_test(&self, t: usize) {
        self.tail.store(t & GFIFO_MASK, Ordering::Release);
        self.shadow_tail.set(t & GFIFO_MASK);
    }

    /// Test hook: publish up to `t` entries, undoing `rewind_tail_for_test`.
    #[cfg(test)]
    pub fn restore_tail_for_test(&self, t: usize) {
        self.tail.store(t & GFIFO_MASK, Ordering::Release);
    }

    pub fn drain_payload(&self, n: usize, dst: &mut HostRwArray) -> usize {
        // `local_head` still points at the token: the consumer loop peeks and
        // only calls `consume()` after process_register returns. Step over it
        // so we start at the first payload slot, and leave it stepped — the
        // caller's `consume()` then retires the last payload entry instead of
        // the token, keeping the head advance exactly `1 + taken` overall.
        let mut head = self.local_head.get().wrapping_add(1) & GFIFO_MASK;
        let mut tail = self.tail.load(Ordering::Acquire);
        let mut taken = 0;
        // Drain EXACTLY the promised count. The token says N words follow, and
        // `push_batch` publishes the token and all N under one Release, so they
        // are guaranteed to be there — but `tail` is sampled once and a batch
        // is far bigger than the 64-entry head-publish interval, so a single
        // sample can sit mid-batch. Returning short there would silently drop
        // the rest of an image: the caller has no way to tell a truncated
        // drain from a complete one, and the pixels are simply gone.
        //
        // Re-sample `tail` instead of stopping. A payload entry that is not yet
        // visible is only a matter of waiting for the producer's Release store,
        // which has already happened logically — this cannot deadlock, because
        // `push_batch` publishes the whole batch before it ever returns.
        while taken < n {
            if head == tail {
                tail = self.tail.load(Ordering::Acquire);
                if head == tail {
                    std::hint::spin_loop();
                    continue;
                }
            }
            // SAFETY: consumer owns head; the slot was published with the token.
            let slot = unsafe { &*self.buf.as_ptr().add(head) };
            // There is no legitimate non-payload slot inside a batch:
            // `push_batch` writes the token and all N payload entries under
            // one lock and publishes them with a single Release. Anything else
            // here means the token/payload pairing is broken and pixel data is
            // already being lost, so say so loudly rather than absorbing it.
            // Still consumes the slot: stopping short would desync the head
            // from the count the caller was promised.
            #[cfg(feature = "developer")]
            if slot.addr != GFIFO_PAYLOAD {
                eprintln!("!!!!!!!!!! GFIFO BATCH CORRUPTION !!!!!!!!!! slot {taken} of {n} \
has addr={:#010x}, expected GFIFO_PAYLOAD ({:#010x}) — token/payload pairing is broken \
and pixel data is being lost", slot.addr, GFIFO_PAYLOAD);
            }
            dst[taken] = slot.val;
            taken += 1;
            head = head.wrapping_add(1) & GFIFO_MASK;
        }
        self.shadow_tail.set(tail);
        // Rewind by one: the caller's consume() advances past the final entry.
        self.local_head.set(head.wrapping_sub(1) & GFIFO_MASK);
        taken
    }

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

    /// Reconcile the published `tail` from a reader thread, under the producer
    /// lock.
    ///
    /// `try_push` publishes `tail` on every push today, so this is a no-op in
    /// the current topology — but it is the hook a deferred/batched tail needs,
    /// and taking the lock is what makes it safe to call from a thread that is
    /// neither the producer nor the consumer. See
    /// `rules/rex3/gfifo-batching-constraints.md`: a batched producer must
    /// publish before any `busy_or_val!` or STATUS read, or the reader sees an
    /// emptier queue than reality and skips the retry it owed.
    ///
    /// Returns `false` if the lock was contended — the caller should treat that
    /// as "busy, retry" rather than spin, for the same reason `try_push` does:
    /// spinning inside the CPU's load path starves IP7 and dilates guest time.
    #[inline]
    pub fn publish_tail(&self) -> bool {
        if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return false;
        }
        // A producer-local tail would be published here. With eager publication
        // the authoritative `tail` is already current; re-storing it under the
        // lock is harmless and keeps this the single place that changes when
        // batching lands.
        let tail = self.tail.load(Ordering::Relaxed);
        self.tail.store(tail, Ordering::Release);
        self.lock.store(false, Ordering::Release);
        true
    }

    /// True when no entries are pending.
    ///
    /// Reads the *published* `head`, which the consumer advances in batches of
    /// 64 (plus immediately on drain-to-empty, see `consume`). So a `true` here
    /// is authoritative — the consumer publishes the moment it empties the ring
    /// — while a `false` may be up to 63 entries pessimistic mid-drain. That
    /// direction is the safe one for every caller: it over-reports busy, never
    /// under-reports.
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
    /// Compiled shaders, keyed by `(dm0, dm1_normalized, clipmode_key)`.
    ///
    /// Always present, not gated on `rex-jit`: the LLVM-compiled shaders in
    /// `rex3_shaders` are built into every binary, and they use the same ABI as
    /// a Cranelift one. Seeded from the static table at construction; with
    /// `rex-jit` the compiler thread adds to it as new shapes appear.
    pub shaders: Arc<RwLock<crate::rex3_shape::ShapeMap<crate::rex3_shaders::ShaderFn>>>,
    /// One-entry memo in front of `shaders`, on the GFIFO consumer thread only.
    pub shader_last: std::cell::Cell<(u32, u32, u32, Option<crate::rex3_shaders::ShaderFn>)>,
    /// Every draw shape this run has dispatched — the corpus the shader
    /// generator consumes.
    ///
    /// Lives on `Rex3` rather than `RexJit` because it must be recorded in every
    /// build: without `rex-jit` the generated table serves draws and Cranelift
    /// never runs, so a JIT-owned corpus would record nothing and could never
    /// grow to cover new shapes. Written on the GFIFO consumer thread only when
    /// a shape is new, which the `shader_last` memo makes rare.
    pub seen_shapes: Mutex<crate::rex3_shape::ShapeSet>,
    #[cfg(feature = "rex-jit")]
    pub rex_jit: Option<std::sync::Arc<crate::rex3_jit::RexJit>>,
    /// Whether the JIT is enabled for dispatch (can be toggled at runtime via `rex jit on/off`).
    #[cfg(feature = "rex-jit")]
    pub jit_enabled: AtomicBool,
    /// Last-hit JIT shader cache: avoids HashMap lookup when (dm0, dm1, clipmode_key) is
    /// the same as the previous GO.  Only accessed from the GFIFO consumer thread — no sync needed.
    #[cfg(feature = "rex-jit")]
    pub jit_last: std::cell::Cell<(u32, u32, u32, Option<unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32)>)>,
    /// Shared activity heartbeat — set by all devices, polled+cleared by the refresh thread.
    /// bit 0 = enet TX, bit 1 = enet RX, bits 2-3 = red/green LED (persistent), bits 8-13 = SCSI IDs 0-5
    pub heartbeat: Arc<AtomicU64>,
    /// Pointer into the CPU's `MipsCore.hot.cycles` word (see
    /// `CyclesPtr`/`Hot::cycles`'s doc comments). Set exactly once, from
    /// `set_cpu_cycles`, during single-threaded `Machine::new` setup — after
    /// the CPU exists (`Rex3::new` runs before it does, so this can't be a
    /// constructor parameter; see `MipsCpu::cycles_ptr`) but strictly before
    /// any device thread (including this struct's own refresh thread) is
    /// spawned, so a plain `Cell` is
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

/// One row of `rex jit list`: a draw shape and what serves it.
pub struct ShaderRow {
    pub dm0: u32,
    pub dm1: u32,
    pub cm: u32,
    /// `precompiled` (generated Rust), `jit` (Cranelift), `queued`, `failed`,
    /// `disabled`, or `generic` (no specialised shader — the runtime `DynMode`
    /// path serves it).
    pub origin: &'static str,
    /// Native code size, for Cranelift shaders. 0 for the others: precompiled
    /// shaders are inlined into the binary and have no separately tracked size.
    pub bytes: u32,
}

impl Rex3 {
    pub fn new(heartbeat: Arc<AtomicU64>, fasttick_count: Arc<AtomicU64>, decoded_count: Arc<AtomicU64>, l1i_hit_count: Arc<AtomicU64>, l1i_fetch_count: Arc<AtomicU64>, uncached_fetch_count: Arc<AtomicU64>) -> Self {
        // The dispatch map, shared with the Cranelift compiler thread so both
        // shader sources publish into one place.
        //
        // Seeded with the generated LLVM shaders — except under cfg(test), where
        // the JIT-vs-generic comparison tests need Cranelift to actually run:
        // a pre-seeded LLVM shader would serve those shapes first and the test
        // would compare the generic path against itself.
        #[cfg(not(test))]
        let shaders_shared: Arc<RwLock<crate::rex3_shape::ShapeMap<crate::rex3_shaders::ShaderFn>>> =
            Arc::new(RwLock::new(crate::rex3_shaders::SHADERS.iter().copied().collect()));
        #[cfg(test)]
        let shaders_shared: Arc<RwLock<crate::rex3_shape::ShapeMap<crate::rex3_shaders::ShaderFn>>> =
            Arc::new(RwLock::new(crate::rex3_shape::ShapeMap::default()));

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
                Some(std::sync::Arc::new(crate::rex3_jit::RexJit::new(
                    Arc::clone(&shaders_shared),
                )))
            },
            #[cfg(feature = "rex-jit")]
            jit_enabled: AtomicBool::new(std::env::var_os("IRIS_NO_JIT").is_none()),
            // Seed with every LLVM-compiled shader. These are available in all
            // builds; Cranelift only ever adds to this map.
            //
            // Not seeded under `cfg(test)`: the JIT-vs-interpreter comparison
            // tests exist to check that *Cranelift's* output matches the generic
            // path, and a pre-seeded LLVM shader would serve those shapes first,
            // so the test would compare the generic path against itself and pass
            // vacuously. Tests that want the generated table exercise
            // `rex3_shaders::lookup` directly.
            shaders: Arc::clone(&shaders_shared),
            shader_last: std::cell::Cell::new((0, 0, 0, None)),
            seen_shapes: Mutex::new(crate::rex3_shape::ShapeSet::default()),
            #[cfg(feature = "rex-jit")]
            jit_last: std::cell::Cell::new((0, 0, 0, None)),
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
        let adrmode = ctx.drawmode0.adrmode();
        let is_fractional = adrmode == DRAWMODE0_ADRMODE_F_LINE || adrmode == DRAWMODE0_ADRMODE_A_LINE;
        if is_fractional {
            let (_incrx1, incrx2, _incry1, incry2, y_major) = REX3_BRES_OCTANTS[(octant & 7) as usize];
            let mut x = ctx.xstart >> 11;
            let mut y = ctx.ystart >> 11;
            crate::rex3_generic::fline_apply_fract(ctx, &mut d, &mut x, &mut y, incrx2, incry2, y_major);
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
    /// Count `n` HOSTRW words at once — a DMA batch delivers them in one call.
    #[cfg(feature = "developer")]
    fn note_hostrw_writes(&self, n: u32) { if self.draw_debug_active() { self.draw_ring.lock().on_hostrw_writes(n); } }
    #[cfg(not(feature = "developer"))]
    fn note_hostrw_writes(&self, _n: u32) {}
    #[cfg(not(feature = "developer"))]
    #[inline(always)]
    fn note_hostrw_write(&self) {}

    #[cfg(feature = "developer")]
    #[inline(always)]
    fn note_hostrw_read(&self) { if self.draw_debug_active() { self.draw_ring.lock().on_hostrw_read(); } }
    /// Count `n` HOSTRW words at once — see `note_hostrw_writes`.
    #[cfg(feature = "developer")]
    fn note_hostrw_reads(&self, n: u32) { if self.draw_debug_active() { self.draw_ring.lock().on_hostrw_reads(n); } }
    #[cfg(not(feature = "developer"))]
    fn note_hostrw_reads(&self, _n: u32) {}
    #[cfg(not(feature = "developer"))]
    #[inline(always)]
    fn note_hostrw_read(&self) {}

    #[cfg(not(feature = "developer"))]
    #[inline(always)]
    pub(crate) fn log_block(&self, _ctx: &Rex3Context, _opcode: u32) {}

    #[cfg(feature = "developer")]
    pub(crate) fn log_block(&self, ctx: &Rex3Context, opcode: u32) {
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
                let bpp = match ctx.drawmode1.drawdepth() {
                    DRAWMODE1_DRAWDEPTH_4 => 4,
                    DRAWMODE1_DRAWDEPTH_8 => 8,
                    DRAWMODE1_DRAWDEPTH_12 => 12,
                    DRAWMODE1_DRAWDEPTH_24 => 24,
                    _ => 0,
                };
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
    pub(crate) fn expand_32_rgb(val: u32) -> u32 {
        // ARGB -> ARGB (internal format AABBGGRR)
        val
    }

    // Compression functions (Internal -> Host)
    pub(crate) fn compress_4_rgb(val: u32) -> u32 {
        let b = (val >> 16) & 0xFF;
        let g = (val >> 8) & 0xFF;
        let r = val & 0xFF;
        ((b >> 7) << 3) | ((g >> 6) << 1) | (r >> 7)
    }

    pub(crate) fn compress_8_rgb(val: u32) -> u32 {
        let b = (val >> 16) & 0xFF;
        let g = (val >> 8) & 0xFF;
        let r = val & 0xFF;
        // BBGGGRRR
        ((b >> 6) << 6) | ((g >> 5) << 3) | (r >> 5)
    }

    pub(crate) fn compress_12_rgb(val: u32) -> u32 {
        let b = (val >> 16) & 0xFF;
        let g = (val >> 8) & 0xFF;
        let r = val & 0xFF;
        ((b >> 4) << 8) | ((g >> 4) << 4) | (r >> 4)
    }

    pub(crate) fn compress_32_rgb(val: u32) -> u32 {
        val | 0xFF000000 // Set Alpha to 0xFF
    }

    /// Summary counts for `rex jit status` / `rex jit list`.
    fn write_shader_summary(&self, writer: &mut Box<dyn Write + Send>) {
        let rows = self.shader_report();
        let count = |o: &str| rows.iter().filter(|r| r.origin == o).count();
        let jit_bytes: u32 = rows.iter().filter(|r| r.origin == "jit").map(|r| r.bytes).sum();

        #[cfg(feature = "rex-jit")]
        let engine = match self.rex_jit {
            Some(_) if self.jit_enabled.load(Ordering::Relaxed) => "cranelift enabled",
            Some(_) => "cranelift DISABLED",
            None => "cranelift not initialised",
        };
        #[cfg(not(feature = "rex-jit"))]
        let engine = "cranelift not compiled in";

        writeln!(writer, "REX3 shaders: {engine}").unwrap();
        writeln!(
            writer,
            "  precompiled={}  jit={} ({} bytes)  generic={}  queued={}  failed={}",
            count("precompiled"),
            count("jit"),
            jit_bytes,
            count("generic"),
            count("queued"),
            count("failed"),
        ).unwrap();
        writeln!(
            writer,
            "  {} draw shapes seen this session, {} shapes known in total",
            self.seen_shapes.lock().len(),
            rows.len(),
        ).unwrap();
    }

    /// One row of the shader report.
    pub fn shader_report(&self) -> Vec<ShaderRow> {
        use std::collections::BTreeSet;

        let precompiled: std::collections::HashSet<(u32, u32, u32)> =
            crate::rex3_shaders::SHADERS.iter().map(|(k, _)| *k).collect();

        // Every shape worth reporting: what was drawn, what is precompiled, and
        // (with rex-jit) what Cranelift knows about.
        let mut keys: BTreeSet<(u32, u32, u32)> =
            self.seen_shapes.lock().iter().copied().collect();
        keys.extend(precompiled.iter().copied());

        #[cfg(feature = "rex-jit")]
        let jit_info: std::collections::HashMap<(u32, u32, u32), (&'static str, u32)> =
            match self.rex_jit {
                Some(ref jit) => {
                    let list = jit.shader_list();
                    keys.extend(list.iter().map(|s| (s.dm0, s.dm1, s.cm)));
                    list.iter()
                        .map(|s| ((s.dm0, s.dm1, s.cm), (s.status, s.code_bytes)))
                        .collect()
                }
                None => std::collections::HashMap::new(),
            };

        keys.into_iter()
            .map(|(dm0, dm1, cm)| {
                // Precompiled wins: the dispatch map is seeded with those, so a
                // shape covered by Rust is never sent to Cranelift.
                if precompiled.contains(&(dm0, dm1, cm)) {
                    return ShaderRow { dm0, dm1, cm, origin: "precompiled", bytes: 0 };
                }
                #[cfg(feature = "rex-jit")]
                if let Some((status, bytes)) = jit_info.get(&(dm0, dm1, cm)) {
                    let origin = match *status {
                        "compiled" => "jit",
                        other => other, // queued / failed / disabled
                    };
                    return ShaderRow { dm0, dm1, cm, origin, bytes: *bytes };
                }
                ShaderRow { dm0, dm1, cm, origin: "generic", bytes: 0 }
            })
            .collect()
    }

    /// Write the draw-shape corpus to disk.
    ///
    /// Unions this run's shapes with whatever the profile already held, so a run
    /// can only ever add. The generated shader table's keys go in too: those
    /// shapes are served by compiled-in Rust and may never reach Cranelift, and
    /// without them a regeneration would emit a smaller table than the one it
    /// replaced.
    fn save_shape_corpus(&self) {
        let drawn: std::collections::HashSet<(u32, u32, u32)> =
            self.seen_shapes.lock().iter().copied().collect();
        let on_disk = crate::rex3_profile::load_profile_quiet();

        // Union of three sources. A run can only ever add: a session that drew
        // two shapes must not shrink a corpus collected over many.
        let mut all: std::collections::HashSet<(u32, u32, u32)> = drawn.clone();
        all.extend(on_disk.iter().copied());
        // The generated table's keys: those shapes are served by compiled-in
        // Rust and may never reach Cranelift, so without this a regeneration
        // would emit a smaller table than the one it replaced.
        all.extend(crate::rex3_shaders::SHADERS.iter().map(|(k, _)| *k));

        if all.is_empty() {
            return;
        }

        // How much this session actually contributed, which is the number worth
        // seeing: a run that discovers nothing new should say so.
        let known: std::collections::HashSet<(u32, u32, u32)> = on_disk
            .iter()
            .copied()
            .chain(crate::rex3_shaders::SHADERS.iter().map(|(k, _)| *k))
            .collect();
        let new_this_run = drawn.difference(&known).count();

        let mut triples: Vec<(u32, u32, u32)> = all.into_iter().collect();
        triples.sort_unstable();

        eprintln!(
            "REX3 corpus: {} drawn this session ({} new), {} on disk, {} precompiled -> saving {}",
            drawn.len(),
            new_this_run,
            on_disk.len(),
            crate::rex3_shaders::SHADERS.len(),
            triples.len(),
        );
        if let Err(e) = crate::rex3_profile::save_profile(&triples) {
            eprintln!("REX3: failed to save shape corpus: {e}");
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




    /// The `(shift, mask)` a plane read applies, derived from
    /// `(planes, drawdepth, dblsrc)`.
    ///
    /// `None` means the read yields zero (unmapped plane).
    #[inline(always)]
    pub(crate) const fn plane_read_shift_mask(
        planes: u32,
        drawdepth: u32,
        dblsrc: bool,
    ) -> Option<(u32, u32)> {
        match planes {
            DRAWMODE1_PLANES_RGB | DRAWMODE1_PLANES_RGBA => match drawdepth {
                0 => Some((if dblsrc { 4 } else { 0 }, 0xF)),
                1 => Some((if dblsrc { 8 } else { 0 }, 0xFF)),
                2 => Some((if dblsrc { 12 } else { 0 }, 0xFFF)),
                3 => Some((0, 0xFFFFFF)),
                _ => None,
            },
            DRAWMODE1_PLANES_OLAY => Some((if dblsrc { 16 } else { 8 }, 0xFF)),
            DRAWMODE1_PLANES_PUP => Some((if dblsrc { 6 } else { 2 }, 0x3)),
            DRAWMODE1_PLANES_CID => Some((if dblsrc { 4 } else { 0 }, 0x3)),
            _ => None,
        }
    }

    /// True when this plane selection reads from `fb_aux` rather than `fb_rgb`.
    #[inline(always)]
    pub(crate) const fn plane_is_aux(planes: u32) -> bool {
        matches!(
            planes,
            DRAWMODE1_PLANES_OLAY | DRAWMODE1_PLANES_PUP | DRAWMODE1_PLANES_CID
        )
    }

    /// Masked read-modify-write of one framebuffer word.
    ///
    /// The single implementation behind every plane write: the variants differed
    /// only in which framebuffer they touched, and each re-derived `wrmask` from
    /// `rex.context` even though every caller already held the context. Taking
    /// the mask as an argument removes that second route to the same data, which
    /// is what lets the generic draw path hold `&mut Rex3Context` across a pixel
    /// write.
    #[inline(always)]
    pub(crate) fn write_masked(fb: &mut [u32], addr: u32, val: u32, mask: u32) {
        let slot = &mut fb[addr as usize];
        *slot = (*slot & !mask) | (val & mask);
    }

    // ── Shims for the generic draw path (src/rex3_generic.rs) ────────────────
    // The generic path selects among these bodies by decoded shape instead of
    // through the px_* function pointers. They are re-exported rather than
    // reimplemented: the colour packings are irregular (1-2-1 at 4bpp, 3-3-2 at
    // 8bpp), the dither variants carry a specific error-diffusion form, and
    // blend has spec-cited factor semantics — all of which a second copy would
    // get subtly wrong.

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

    /// Non-blocking `gfifo_push`: returns `false` when the queue is full or a
    /// producer holds the lock, so a bus write can report `BUS_BUSY`
    /// (== `EXEC_RETRY`) instead of spinning with interrupts unserviced.
    ///
    /// Only safe for callers that commit no other state first: the CPU
    /// re-executes the entire store on retry, so anything done beforehand would
    /// be applied twice.
    #[must_use]
    fn gfifo_try_push(&self, addr: u32, val: u64) -> bool {
        #[cfg(feature = "developer")]
        {
            let len = self.gfifo.len() + 1;
            let _ = self.gfifo_hwm.try_update(Ordering::Relaxed, Ordering::Relaxed, |hwm| {
                if len > hwm { Some(len) } else { None }
            });
        }
        if !self.gfifo.try_push(addr, val) {
            return false;
        }
        #[cfg(feature = "idle-pause")]
        if self.processor_parked.load(Ordering::Acquire) {
            if let Some(t) = self.processor_unparker.get() {
                t.unpark();
            }
        }
        true
    }

    /// Does this register offset need handling on the CPU thread rather than
    /// through the queue? Mirrors `write32`'s match arms.
    #[inline(always)]
    fn reg_needs_cpu_side_effect(reg_offset: u32) -> bool {
        matches!(
            reg_offset & !0x0800,
            REX3_CONFIG | REX3_DCBMODE | REX3_DCBDATA0 | REX3_DCBDATA1 | REX3_DCBRESET
        )
    }

    /// Two-entry `gfifo_try_push`, for a 64-bit store's register pair.
    #[must_use]
    fn gfifo_try_push2(&self, addr0: u32, val0: u64, addr1: u32, val1: u64) -> bool {
        #[cfg(feature = "developer")]
        {
            let len = self.gfifo.len() + 2;
            let _ = self.gfifo_hwm.try_update(Ordering::Relaxed, Ordering::Relaxed, |hwm| {
                if len > hwm { Some(len) } else { None }
            });
        }
        if !self.gfifo.try_push2(addr0, val0, addr1, val1) {
            return false;
        }
        #[cfg(feature = "idle-pause")]
        if self.processor_parked.load(Ordering::Acquire) {
            if let Some(t) = self.processor_unparker.get() {
                t.unpark();
            }
        }
        true
    }

    /// Push a batch token plus its payload words through the GFIFO.
    ///
    /// See `GFifo::push_batch`. Wakes a parked consumer the same way
    /// `gfifo_push` does.
    fn gfifo_push_batch(&self, token: u32, vals: &[u64]) {
        #[cfg(feature = "developer")]
        {
            let len = self.gfifo.len() + vals.len() + 1;
            let _ = self.gfifo_hwm.try_update(Ordering::Relaxed, Ordering::Relaxed, |hwm| {
                if len > hwm { Some(len) } else { None }
            });
        }
        self.gfifo.push_batch(token, vals.len() as u64, vals);
        #[cfg(feature = "idle-pause")]
        if self.processor_parked.load(Ordering::Acquire) {
            if let Some(t) = self.processor_unparker.get() {
                t.unpark();
            }
        }
    }

    pub(crate) fn wait_idle(&self) {
        loop {
            // Publish before testing: unlike the bus read paths this one has no
            // retry escape — it spins here until the queue reports empty — so a
            // queue that looks emptier or fuller than it is turns into a hang
            // rather than an extra round trip.
            self.gfifo.publish_tail();
            // Acquire load: when gfxbusy goes false, all execute_go() writes become visible.
            let busy = self.gfxbusy.load(Ordering::Acquire);
            if !busy && self.gfifo.is_empty() { break; }
            if !self.running.load(Ordering::Relaxed) { break; }
            std::hint::spin_loop();
        }
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
                            // A batch token stands in for N HOSTRW pushes that
                            // never appear individually in this log — the
                            // payload is consumed inside drain_payload. Without
                            // the word count and a sample here, a bulk transfer
                            // is a single opaque line, while the scalar path it
                            // replaced printed every word. Show enough to tell
                            // "the data arrived" from "the data was wrong".
                            let batch = match reg_offset {
                                REX3_DMA_BATCH_W_REG => {
                                    let ctxr = unsafe { &*self.context.get() };
                                    format!("  ; batch write {} qwords, first={:016x}",
                                        val, ctxr.hostrw[0])
                                }
                                REX3_DMA_BATCH_R_REG =>
                                    format!("  ; batch read {} qwords (filled after GO)", val),
                                _ => String::new(),
                            };
                            let _ = writeln!(f, "------- GO reg={:04x}({}) val={:016x}{}{} -------",
                                reg_offset, rex3_reg_name(reg_offset), val, extra, batch);
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
        #[cfg(feature = "rexdiag")]
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
            let adrmode = ctx.drawmode0.adrmode();
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

        // ── Shader dispatch ──────────────────────────────────────────────
        //
        // One map, two producers. LLVM-compiled shaders (rex3_shaders) are built
        // into every binary and seeded at construction; Cranelift adds to the
        // same map when `rex-jit` is on. Both use the identical ABI, so the call
        // below does not care which compiled the entry it found.
        //
        // Only the *compile request* is gated on `rex-jit` — the lookup, the
        // memo and the call are unconditional, which is what lets the generated
        // table work in the default build.
        {
            let dm0 = ctx.drawmode0.0;
            // Shared with compile_shader and the interpreter setup key below —
            // all three must agree or shaders get filed under a key nobody looks up.
            let dm1 = crate::rex3_shape::normalize_dm1(ctx.drawmode1.0, opcode);
            let adrmode = ctx.drawmode0.adrmode();
            let is_line = adrmode == DRAWMODE0_ADRMODE_I_LINE
                || adrmode == DRAWMODE0_ADRMODE_F_LINE
                || adrmode == DRAWMODE0_ADRMODE_A_LINE;
            let is_shadeable = (opcode == DRAWMODE0_OPCODE_DRAW
                    || opcode == DRAWMODE0_OPCODE_SCR2SCR
                    || opcode == DRAWMODE0_OPCODE_READ)
                && (adrmode == DRAWMODE0_ADRMODE_BLOCK || adrmode == DRAWMODE0_ADRMODE_SPAN || is_line);

            #[cfg(feature = "rex-jit")]
            let dispatch_on = self.jit_enabled.load(Ordering::Relaxed);
            #[cfg(not(feature = "rex-jit"))]
            let dispatch_on = true;

            if is_shadeable {
                let cm = ctx.clipmode & CLIPMODE_JIT_KEY_MASK;
                // Record the shape: this is the corpus, and it has to reflect
                // what the guest draws regardless of which engine serves it.
                // Skipped when the memo already holds this key, so a run of
                // same-shape GOs costs one compare rather than a lock.
                let last = self.shader_last.get();
                if (last.0, last.1, last.2) != (dm0, dm1, cm) {
                    self.seen_shapes.lock().insert((dm0, dm1, cm));
                }
            }

            if is_shadeable && dispatch_on {
                let cm = ctx.clipmode & CLIPMODE_JIT_KEY_MASK;
                // Fast path: same key as the last GO — skip the map lookup.
                let last = self.shader_last.get();
                let entry = if last.0 == dm0 && last.1 == dm1 && last.2 == cm && last.3.is_some() {
                    last.3
                } else {
                    let e = self.shaders.read().get(&(dm0, dm1, cm)).copied();
                    if e.is_some() {
                        self.shader_last.set((dm0, dm1, cm, e));
                    } else {
                        // Nothing precompiled for this shape. Ask Cranelift to
                        // build one (if it is compiled in) and run the generic
                        // path meanwhile; without rex-jit the generic path is
                        // simply what always runs for uncovered shapes.
                        #[cfg(feature = "rex-jit")]
                        if let Some(ref jit) = self.rex_jit {
                            jit.request_compile(dm0, dm1, cm);
                        }
                    }
                    e
                };
                // Batched HOSTRW transfers used to be excluded here, because
                // the generated code addressed hostrw[0] by fixed offset and
                // never stepped the cursor — a multi-word transfer would
                // consume one word and repeat it for the whole run.
                //
                // The shader now addresses hostrw[hostrw_index()] and advances
                // the cursor on store (emit_hostrw_slot_ptr /
                // emit_hostrw_advance in rex3_jit::compiler), mirroring the
                // interpreter's accessors, so the bypass is gone: both engines
                // walk the array identically. `batch_jit_equivalence` pins
                // that, and the general rule it serves — JIT and interpreter
                // must be indistinguishable — is why bypassing was never an
                // acceptable long-term answer.
                if let Some(entry) = entry {
                    // Mirror the interpreter's log_block() calls so block/span
                    // primitives trace identically whichever engine ran them.
                    if adrmode == DRAWMODE0_ADRMODE_BLOCK || adrmode == DRAWMODE0_ADRMODE_SPAN {
                        self.log_block(ctx, opcode);
                    }
                    let fb_rgb = unsafe { (*self.fb_rgb.get()).as_mut_ptr() };
                    let fb_aux = unsafe { (*self.fb_aux.get()).as_mut_ptr() };
                    unsafe { entry(ctx as *mut Rex3Context, fb_rgb, fb_aux); }
                    #[cfg(feature = "rexdiag")]
                    self.jit_go_count.fetch_add(1, Ordering::Relaxed);
                    if ctx.host_len > 1 && (ctx.drawmode0.colorhost() || ctx.drawmode0.alphahost()) {
                        self.note_hostrw_writes(ctx.host_len);
                    }
                    // Same batch resume loop the interpreter path runs below: a
                    // shader consumes exactly one host word per call (its
                    // host_xstop stops it at a word boundary), so a batch of N
                    // words needs N calls. Without this the JIT would paint the
                    // first word and silently drop the rest — which is why
                    // batches used to bypass the JIT entirely instead.
                    //
                    // Not gated on `mid_primitive`: without STOPONY the walker
                    // clears it at every row end, and each row is its own
                    // primitive. `host_cursor` is what terminates the loop.
                    while ctx.host_len > 1 && ctx.host_cursor < ctx.host_len {
                        let before = ctx.host_cursor;
                        ctx.hostcnt = 0;
                        unsafe { entry(ctx as *mut Rex3Context, fb_rgb, fb_aux); }
                        #[cfg(feature = "rexdiag")]
                        self.jit_go_count.fetch_add(1, Ordering::Relaxed);
                        // A shape that cannot make progress must not spin here
                        // holding the GFIFO.
                        if ctx.host_cursor == before { break; }
                    }
                    if ctx.host_len > 1 {
                        ctx.host_len = 1;
                        ctx.host_cursor = 0;
                    }
                    #[cfg(feature = "rexdiag")]
                    self.diag.fetch_and(!Self::DIAG_LOOP_EXECUTE_GO, Ordering::Relaxed);
                    return;
                }
                // fall through to the generic draw path
            }
        }

        // Interpreter-only setup: function pointer selection and host/planes dispatch tables.
        // Skipped when JIT handles the draw (returned above) or when nothing affecting these
        // tables has changed since the last GO.
        let cidmatch_bits = (ctx.clipmode >> CLIPMODE_CIDMATCH_SHIFT) & 0xF;
        // Same normalization as the JIT dispatch key above and compile_shader.
        // Previously this did only the SCR2SCR half, so a fastclear draw that
        // toggled BLEND re-ran planes_setup for a bit planes_setup ignores.
        // Folding it out here is safe — see planes_setup: BLEND is not among the
        // fields it reads — and it keeps the three keys identical, which the
        // generated draw table will depend on.
        let dm1_norm = crate::rex3_shape::normalize_dm1(ctx.drawmode1.0, opcode);
        let setup_key = (
            ctx.drawmode0.0 & DRAWMODE0_INTERP_SETUP_MASK,
            dm1_norm        & DRAWMODE1_INTERP_SETUP_MASK,
            cidmatch_bits,
        );
        // One decode, one entry point. rex3_generic::draw fans out to the
        // per-adrmode walkers; every shape-selecting field reaches it as its own
        // argument, which is the list stage 5 promotes to const generics.
        crate::rex3_generic::draw_primitive(ctx);
        // Attribute this batch's words to the record `draw_primitive` just
        // created (via log_block). Doing it any earlier counts against the
        // wrong record — see the batch-token arm in process_register.
        //
        // Only for a draw that actually consumes host data. `host_len` stays
        // raised until this function retires it, so a following non-host
        // primitive would otherwise be credited with the previous batch's
        // words — and because it has colorhost=0 they land in
        // `spurious_writes`, making the overlay report phantom host traffic on
        // an ordinary DRAW.
        if ctx.host_len > 1 && (ctx.drawmode0.colorhost() || ctx.drawmode0.alphahost()) {
            self.note_hostrw_writes(ctx.host_len);
        }
        // A host-mode primitive stops at a word or row boundary and expects the
        // next GO to resume it — that is how the CPU's one-word-per-GO feed
        // works. A batch carries N words behind a *single* GO, so anything the
        // primitive did not consume this pass has to be driven round again
        // here, or it is silently dropped: a multi-row pixmap upload would
        // paint its first row and discard the rest, and a tiled fill would
        // leave bands of untouched framebuffer.
        // NOTE: deliberately not gated on `mid_primitive`. Without STOPONY the
        // walker clears that flag and breaks at the end of *every row* — each
        // row is its own primitive, resumed by the next GO. The CPU's
        // one-word-per-GO feed supplies those GOs; a batch has exactly one, so
        // gating the resume on `mid_primitive` drops every row after the first.
        // That is what left holes in the tiled wallpaper and the login icons.
        // The `host_cursor` bound is what actually terminates this loop.
        while ctx.host_len > 1 && ctx.host_cursor < ctx.host_len {
            let before = ctx.host_cursor;
            // Each round must start on a word boundary, exactly as the CPU's
            // one-word-per-GO feed does: that feed hands the walker a fresh
            // HOSTRW word per GO, so a partial word left open when the walker
            // broke at a row boundary is *discarded*, not carried into the next
            // row. Leaving `hostcnt` set here instead shifts every row after
            // the first — the corruption that made pixmaps and tiled fills come
            // out wrong.
            ctx.hostcnt = 0;
            crate::rex3_generic::draw_primitive(ctx);
            // Guard against a primitive that consumes nothing: without this a
            // shape that cannot make progress (a degenerate block, a mode the
            // walker declines) would spin here forever holding the GFIFO.
            if ctx.host_cursor == before {
                break;
            }
        }
        // Retire the transfer: back to the resting single-word state. Leaving
        // a batch count set would make the next PIO HOSTRW access walk a stale
        // run instead of the one word it wrote. Element 0 keeps the last word,
        // which is what a PIO read after a transfer expects to see.
        ctx.host_len = 1;
        ctx.host_cursor = 0;
        #[cfg(feature = "rexdiag")]
        self.interp_go_count.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "rexdiag")]
        self.diag.fetch_and(!Self::DIAG_LOOP_EXECUTE_GO, Ordering::Relaxed);
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
            // Batch payload consumed by its token (see REX3_DMA_BATCH_W_REG).
            // Reaching one here means it was left behind by a truncated batch;
            // ignoring it is inert, which is the point of the distinct address.
            GFIFO_PAYLOAD => {}
            // Batched HOSTRW write: the payload is already in host_buf (the
            // producer filled it before pushing this token). Arm the cursor and
            // let the GO bit run the shader once over the whole run.
            REX3_DMA_BATCH_W_REG => {
                // The token's payload is the next `count` entries in the queue.
                // Stream them straight into the data-port array, then set the
                // count the shader walks. The GO bit on this same entry runs
                // the primitive once over the whole run.
                let count = (val64 as usize).min(HOSTRW_BUF_QWORDS);
                let got = self.gfifo.drain_payload(count, &mut ctx.hostrw);
                ctx.host_cursor = 0;
                ctx.host_len = got as u32;
                ctx.hostcnt = 0;
                // Deliberately NOT counted here. `log_block` — which creates
                // the draw record this would be attributed to — does not run
                // until `execute_go` dispatches the primitive, so a count
                // issued now lands on the *previous* record or is dropped
                // entirely (`pending == None`). That is why the overlay showed
                // `0/4802` for a transfer that carried every word. The words
                // are counted in `execute_go` instead, once the record exists.
                #[cfg(feature = "developer")]
                if let Some(f) = self.block_log.lock().as_mut() {
                    let _ = writeln!(f, "  HOSTRW batch write: {} qwords (asked {})", got, count);
                }
            }
            // Batched HOSTRW read: arm the cursor so the shader scatters its
            // output across host_buf instead of latching one word.
            REX3_DMA_BATCH_R_REG => {
                // Nothing to stream in: the shader produces the words. Just set
                // the count it should fill.
                ctx.host_cursor = 0;
                ctx.host_len = (val64 as u32).min(HOSTRW_BUF_QWORDS as u32);
                ctx.hostcnt = 0;
                #[cfg(feature = "developer")]
                if let Some(f) = self.block_log.lock().as_mut() {
                    let _ = writeln!(f, "  HOSTRW batch read: {} qwords", ctx.host_len);
                }
            }
            // HOSTRW: store data port value; reset shift so the draw picks up pixels from MSB.
            // The actual draw/read is triggered by execute_go() when entry.go is set.
            // 64-bit write (REX3_HOSTRW0_64 = 0x0231): store full val64 directly.
            REX3_HOSTRW64 => {
                ctx.hostrw_arm_single();
                ctx.hostrw_set(val64);
                ctx.hostcnt = 0;
                self.note_hostrw_write();
                #[cfg(feature = "developer")]
                if let Some(f) = self.block_log.lock().as_mut() {
                    let _ = writeln!(f, "  HOSTRW64 {:016x}", val64);
                }
            }
            REX3_HOSTRW0 => {
                // 32-bit write to HOSTRW0: update high 32 bits [63:32].
                ctx.hostrw_arm_single();
                let new_val = (ctx.hostrw_get() & 0x0000_0000_FFFF_FFFF) | ((val64 & 0xFFFF_FFFF) << 32);
                ctx.hostrw_set(new_val);
                ctx.hostcnt = 0;
                self.note_hostrw_write();
                #[cfg(feature = "developer")]
                if let Some(f) = self.block_log.lock().as_mut() {
                    let _ = writeln!(f, "  HOSTRW0 {:08x} -> {:016x}", val64 as u32, new_val);
                }
            }
            REX3_HOSTRW1 => {
                // 32-bit write to HOSTRW1: update low 32 bits [31:0].
                ctx.hostrw_arm_single();
                let new_val = (ctx.hostrw_get() & 0xFFFF_FFFF_0000_0000) | (val64 & 0xFFFF_FFFF);
                ctx.hostrw_set(new_val);
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

        // Save the corpus in every build. `seen_shapes` is device state, so this
        // works with or without `rex-jit` and with IRIS_NO_JIT set — the cases
        // where the old JIT-owned save wrote nothing at all.
        self.save_shape_corpus();

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

        // Wire the context's back-pointer now that `self` is at its final
        // address. The draw path reaches host services (block logging, the host
        // FIFO) through this, which is what lets its functions take the same
        // arguments the compiled shader does instead of a `&Rex3`.
        unsafe { (*self.context.get()).host = rex3 as *const Rex3; }

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

        // Shader reporting works in every build: the precompiled table serves
        // draws with no Cranelift present, so `rex jit status` being gated on
        // rex-jit hid exactly the case worth inspecting.
        if cmd == "rex" && (args[0] == "shaders" || args[0] == "shader") {
            match args.get(1).copied() {
                Some("list") => {
                    self.write_shader_summary(&mut writer);
                    let rows = self.shader_report();
                    if rows.is_empty() {
                        writeln!(writer, "No draw shapes seen yet.").unwrap();
                    } else {
                        writeln!(writer, "{:>11}  {:>10}  {:>10}  {:>10}  {:>6}  {}",
                            "origin", "dm0", "dm1", "cm", "bytes", "description").unwrap();
                        for r in &rows {
                            writeln!(writer, "{:>11}  {:#010x}  {:#010x}  {:#010x}  {:>6}  {}  |  {}",
                                r.origin, r.dm0, r.dm1, r.cm, r.bytes,
                                decode_dm0(r.dm0), decode_dm1(r.dm1)).unwrap();
                        }
                    }
                }
                _ => self.write_shader_summary(&mut writer),
            }
            return Ok(());
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
                    self.write_shader_summary(&mut writer);
                }
                Some("list") => {
                    self.write_shader_summary(&mut writer);
                    let rows = self.shader_report();
                    if rows.is_empty() {
                        writeln!(writer, "No draw shapes seen yet.").unwrap();
                    } else {
                        writeln!(writer, "{:>11}  {:>10}  {:>10}  {:>10}  {:>6}  {}",
                            "origin", "dm0", "dm1", "cm", "bytes", "description").unwrap();
                        for r in &rows {
                            writeln!(writer, "{:>11}  {:#010x}  {:#010x}  {:#010x}  {:>6}  {}  |  {}",
                                r.origin, r.dm0, r.dm1, r.cm, r.bytes,
                                decode_dm0(r.dm0), decode_dm1(r.dm1)).unwrap();
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
                // Publish any producer-side pending entries before testing
                // emptiness: a read must see the queue as it really is, or it
                // skips the retry it owed and returns pre-write register state.
                // Contention here means a producer is mid-push, which is itself
                // "not empty" — report busy and let the CPU retry.
                if !self.gfifo.publish_tail() {
                    return BusRead32::busy();
                }
                if self.gfxbusy.load(Ordering::Acquire) || !self.gfifo.is_empty() {
                    return BusRead32::busy();
                }
                BusRead32::ok($val)
            }};
        }

        let result = match reg_offset {
            REX3_CONFIG => BusRead32::ok(self.config.config.load(Ordering::Relaxed) & 0x1FFFFF),

            REX3_STATUS | REX3_USER_STATUS => {
                // STATUS is the guest's flow control — it reports queue depth and
                // GFXBUSY, and the driver throttles on it. An under-reported depth
                // is worse than a stale register read: the guest concludes the
                // engine is idle and keeps writing. Publish before sampling.
                self.gfifo.publish_tail();
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
                    REX3_HOSTRW0      => busy_or_val!((ctx.hostrw_get() >> 32) as u32),
                    REX3_HOSTRW1      => busy_or_val!(ctx.hostrw_get() as u32),
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

        // Blocking push, deliberately: `result` is already computed above and a
        // HOSTRW read has already called note_hostrw_read(), advancing the
        // read-then-advance pipeline. Returning BUS_BUSY here would re-run that
        // on retry. Unlike write32's default arm, this path cannot be made
        // retryable without moving the push ahead of the read side effects.
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
                // The push is this write's ONLY effect, so a full queue can
                // safely report BUS_BUSY (== EXEC_RETRY): the CPU re-executes
                // the store from scratch, having sampled interrupts in
                // step_preamble!, and nothing was half-applied. Spinning here
                // would starve IP7 — and with it the guest clock — for as long
                // as the queue stays full.
                if !self.gfifo_try_push(offset, val as u64) {
                    return BUS_BUSY;
                }
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
            // Same rule as busy_or_val! on the 32-bit path: publish, then test.
            if !self.gfifo.publish_tail() {
                return BusRead64::busy();
            }
            if self.gfxbusy.load(Ordering::Acquire) || !self.gfifo.is_empty() {
                return BusRead64::busy();
            }
            let val = unsafe { (*self.context.get()).hostrw_get() };
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
        let before = unsafe { (*self.context.get()).hostrw_get() };
        self.gfifo_push(REX3_DMA_PURE_GO, 0);
        self.wait_idle();

        let val = unsafe { (*self.context.get()).hostrw_get() };
        self.note_hostrw_read();
        #[cfg(feature = "developer")]
        if let Some(f) = self.block_log.lock().as_mut() {
            let _ = writeln!(f, "  HOSTRW0_64 dma_read: before={:016x} after={:016x} (primed)", before, val);
        }
        BusRead64::ok(val)
    }

    /// Batched HOSTRW write: stream `vals` into the host buffer and run the
    /// shader once over the whole run.
    ///
    /// Replaces N `gfifo_push(REX3_HOSTRW64)` round trips with one token. The
    /// payload is copied into `host_buf` **before** the token is pushed, so the
    /// consumer never observes a token pointing at a half-written buffer.
    ///
    /// All-or-nothing, as the trait requires: the capacity check and the single
    /// token push both happen before anything is consumed, so a refusal leaves
    /// no state behind for the caller's retry to duplicate.
    fn dma_write64_bulk(&self, addr: u32, vals: &[u64]) -> u32 {
        let offset = addr & (REX3_SIZE - 1);
        let is_go = (offset & 0x0800) != 0;
        if (offset & !0x0800) != REX3_HOSTRW0 || !is_go {
            // Not the batchable port — let the caller fall back to scalar.
            return BUS_ERR;
        }
        if vals.is_empty() { return BUS_OK; }
        if vals.len() > HOSTRW_BUF_QWORDS {
            // Caller must chunk. Reporting ERR (not BUSY) so a retry loop
            // cannot spin forever on a request that can never fit.
            return BUS_ERR;
        }

        // Token first, then the payload words, all as GFIFO entries: the token
        // tells the processor how many words follow, and it streams them into
        // ctx.hostrw[] as it consumes them. Keeping the payload in the queue
        // (rather than writing the array behind the queue's back) is what makes
        // the transfer ordered correctly against concurrent CPU register writes
        // — the words arrive in the stream at the point the producer put them.
        self.gfifo_push_batch(REX3_DMA_BATCH_W, vals);
        BUS_OK
    }

    /// Batched HOSTRW read: one token, one pipeline drain, then copy the
    /// filled array out.
    ///
    /// This is where the big win is. The scalar path pays a `wait_idle()` —
    /// a full pipeline drain — for every 8 bytes; here one drain covers the
    /// entire transfer.
    ///
    /// Keeps `dma_read64`'s ordering inversion at transfer granularity: the
    /// token is pushed and waited on *before* anything is read out, so the
    /// array is already filled when we copy from it.
    fn dma_read64_bulk(&self, addr: u32, out: &mut [u64]) -> u32 {
        let offset = addr & (REX3_SIZE - 1);
        let is_go64r = (offset & 0x0800) != 0;
        if (offset & !0x0800) != REX3_HOSTRW0 || !is_go64r {
            return BUS_ERR;
        }
        if out.is_empty() { return BUS_OK; }
        if out.len() > HOSTRW_BUF_QWORDS {
            return BUS_ERR;
        }

        self.gfifo_push(REX3_DMA_BATCH_R, out.len() as u64);
        self.wait_idle();

        // SAFETY: wait_idle() means the consumer has finished and its writes to
        // ctx.hostrw are visible (gfxbusy is loaded Acquire).
        unsafe {
            let ctx = &*self.context.get();
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = ctx.hostrw[i];
            }
        }
        // `out.len()` words came back in this one call — count them all, or the
        // overlay under-reports a bulk readback the same way it did writes.
        self.note_hostrw_reads(out.len() as u32);
        // The token line in the bus log is written before the shader runs, so
        // it cannot show what came back. Close the loop here, where the data
        // exists: otherwise a bulk readback is the only transfer in the log
        // whose result is never visible.
        #[cfg(feature = "developer")]
        if let Some(f) = self.rex3_log.lock().as_mut() {
            let _ = writeln!(f, "------- BATCH_R done: {} qwords, first={:016x} last={:016x} -------",
                out.len(), out[0], out[out.len() - 1]);
        }
        BUS_OK
    }

    fn write64(&self, addr: u32, val: u64) -> u32 {
        let offset = addr & (REX3_SIZE - 1);
        let is_go = (offset & 0x0800) != 0;
        let reg_offset64 = offset & !0x0800;
        if reg_offset64 == REX3_HOSTRW0 {
            // Encode as REX3_HOSTRW64 (0x0231) + GO bit if present.
            // addr bit 0 = is_64bit, bit 11 = GO.
            // Sole effect of this write, so a full queue reports BUS_BUSY and
            // the CPU retries the store (see write32's default arm).
            if !self.gfifo_try_push(REX3_HOSTRW64 | (offset & 0x0800), val) {
                return BUS_BUSY;
            }
            return BUS_OK;
        }

        // Two-register 64-bit write: high word first WITHOUT go, then low word
        // with go. Pushed as one atomic pair.
        //
        // This used to be two `write32` calls returning the second's status,
        // which was wrong twice over: six atomics where three suffice, and a
        // queue that filled between the two left the high word pushed while
        // still reporting BUS_BUSY — so the CPU's retry re-pushed it. See
        // `GFifo::try_push2`.
        let high = (val >> 32) as u32;
        let low = val as u32;
        let off_hi = offset & !0x0800;
        let off_lo = off_hi + 4;

        // Registers with CPU-thread side effects cannot take the queue path;
        // they are rare (CONFIG, DCB*) and never part of a hot pair.
        if Self::reg_needs_cpu_side_effect(off_hi) || Self::reg_needs_cpu_side_effect(off_lo) {
            return match self.write32(addr & !0x0800, high) {
                BUS_OK => self.write32(addr + 4, low),
                status => status,
            };
        }

        // `is_go` puts the GO bit on the low word, matching the two-write order.
        let go_lo = if is_go { off_lo | 0x0800 } else { off_lo };
        if !self.gfifo_try_push2(off_hi, high as u64, go_lo, low as u64) {
            return BUS_BUSY;
        }
        BUS_OK
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
