// Bt445 RAMDAC
//
// Brooktree Bt445 — 150 MHz Monolithic CMOS Triple 256×8 RAMDAC
// Newport NG1 uses it for gamma correction (main palette acts as a gamma ramp).
//
// DCB device 7 (DCB_ADDR_RAMDAC = 7).
//
// C(2:0) in the DCB protocol maps directly to CRS[2:0] (the register-select field).
// The 8-bit internal address register selects which sub-register within each CRS group.
//
// CRS layout (Table 1 of datasheet):
//   CRS 0 / RDAC_CRS_ADDR_REG  — address register (auto-increments modulo 3 on palette R/W)
//   CRS 1 / RDAC_CRS_PAL_RAM   — primary color palette RAM  (256 × RGB, modulo-3 load)
//   CRS 2 / RDAC_CRS_CTRL      — control registers (ID, revision, cmd0, blink, read-enable…)
//   CRS 3 / RDAC_CRS_OVL_RAM   — overlay color palette (16 entries × RGB)
//   CRS 4 / RDAC_CRS_RESERVED  — reserved
//   CRS 5 / RDAC_CRS_RGB_CTRL  — pixel format registers (RGB field positions / widths, etc.)
//   CRS 6 / RDAC_CRS_SETUP     — setup / clock / PLL registers (incl. Command Register 1)
//   CRS 7 / RDAC_CRS_CURSOR    — cursor color registers (4 entries × RGB)
//
// We implement the full register map so that firmware reads don't return garbage.

use std::io::Write;
use crate::devlog::{LogModule, devlog_is_active};

// ── CRS selects ────────────────────────────────────────────────────────────────────────────────
pub const RDAC_CRS_ADDR_REG: u8 = 0; // address register
pub const RDAC_CRS_PAL_RAM:  u8 = 1; // primary (gamma) palette RAM
pub const RDAC_CRS_CTRL:     u8 = 2; // control registers
pub const RDAC_CRS_OVL_RAM:  u8 = 3; // overlay palette RAM
pub const RDAC_CRS_RESERVED: u8 = 4; // reserved
pub const RDAC_CRS_RGB_CTRL: u8 = 5; // pixel format / RGB field control registers
pub const RDAC_CRS_SETUP:    u8 = 6; // setup / PLL / clock registers
pub const RDAC_CRS_CURSOR:   u8 = 7; // cursor color registers

// ── CRS 2 (control registers) sub-addresses ───────────────────────────────────────────────────
pub const RDAC_CTRL_ID:           u8 = 0x00; // ID register          reset=0x3A  (read-only)
pub const RDAC_CTRL_REVISION:     u8 = 0x01; // Revision register    reset=0xA0  (read-only)
pub const RDAC_CTRL_RESERVED0:    u8 = 0x02; // Reserved             reset=0xFF
pub const RDAC_CTRL_RESERVED1:    u8 = 0x03; // Reserved             reset=0x00
pub const RDAC_CTRL_READ_ENABLE:  u8 = 0x04; // Read Enable register reset=0x00
pub const RDAC_CTRL_BLINK_ENABLE: u8 = 0x05; // Blink Enable         reset=0x00
pub const RDAC_CTRL_CMD0:         u8 = 0x06; // Command Register 0   reset=0x43
pub const RDAC_CTRL_TEST0:        u8 = 0x07; // Test Register 0      reset=0x00  (read-only)

// ── Command Register 0 bits (CRS 2, addr 0x06) ────────────────────────────────────────────────
// Reset value = 0x43 = 0b01000011
//   bit 7  — reserved (was Bt458 4:1 mux select; ignored in Bt445)  reset=0
//   bit 6  — Overlay color 0 disable: 0=use overlay color 0, 1=use color palette RAM  reset=1
//   bit 5  — Blink rate [1]  \  00=16on/48off(25%), 01=16on/16off(50%)
//   bit 4  — Blink rate [0]  /  10=32on/32off(50%), 11=64on/64off(50%)  reset=00
//   bit 3  — Overlay plane 1 blink enable: 1=enable  reset=0
//   bit 2  — Overlay plane 0 blink enable: 1=enable  reset=0
//   bit 1  — Overlay plane 1 display enable: 0=force to 0, 1=pass through  reset=1
//   bit 0  — Overlay plane 0 display enable: 0=force to 0, 1=pass through  reset=1
pub const CMD0_OVL_COLOR0_DISABLE: u8 = 1 << 6; // 1 → use color palette RAM instead of overlay color 0
pub const CMD0_BLINK_RATE_MASK:    u8 = 3 << 4; // blink rate field [5:4]
pub const CMD0_OVL1_BLINK_EN:     u8 = 1 << 3; // overlay plane 1 blink enable
pub const CMD0_OVL0_BLINK_EN:     u8 = 1 << 2; // overlay plane 0 blink enable
pub const CMD0_OVL1_DISP_EN:      u8 = 1 << 1; // overlay plane 1 display enable
pub const CMD0_OVL0_DISP_EN:      u8 = 1 << 0; // overlay plane 0 display enable

// ── CRS 5 (RGB pixel format control) sub-addresses ────────────────────────────────────────────
// Red   (0x00–0x03), Green (0x08–0x0B), Blue (0x10–0x13),
// Overlay (0x18–0x1B), Cursor (0x20–0x23)
pub const RDAC_RGB_RED_MSB_POS: u8 = 0x00; // Red MSB position         reset=0x07
pub const RDAC_RGB_RED_WIDTH:   u8 = 0x01; // Red width control         reset=0x08
pub const RDAC_RGB_RED_DISP_EN: u8 = 0x02; // Red display enable        reset=0xFF
pub const RDAC_RGB_RED_BLINK:   u8 = 0x03; // Red blink enable          reset=0x00
pub const RDAC_RGB_GRN_MSB_POS: u8 = 0x08; // Green MSB position        reset=0x07
pub const RDAC_RGB_GRN_WIDTH:   u8 = 0x09; // Green width control       reset=0x08
pub const RDAC_RGB_GRN_DISP_EN: u8 = 0x0A; // Green display enable      reset=0xFF
pub const RDAC_RGB_GRN_BLINK:   u8 = 0x0B; // Green blink enable        reset=0x00
pub const RDAC_RGB_BLU_MSB_POS: u8 = 0x10; // Blue MSB position         reset=0x07
pub const RDAC_RGB_BLU_WIDTH:   u8 = 0x11; // Blue width control        reset=0x08
pub const RDAC_RGB_BLU_DISP_EN: u8 = 0x12; // Blue display enable       reset=0xFF
pub const RDAC_RGB_BLU_BLINK:   u8 = 0x13; // Blue blink enable         reset=0x00
pub const RDAC_RGB_OVL_MSB_POS: u8 = 0x18; // Overlay MSB position      reset=0x09
pub const RDAC_RGB_OVL_WIDTH:   u8 = 0x19; // Overlay width control     reset=0x02
pub const RDAC_RGB_OVL_DISP_EN: u8 = 0x1A; // Overlay display enable    reset=0x03
pub const RDAC_RGB_OVL_BLINK:   u8 = 0x1B; // Overlay blink enable      reset=0x00
pub const RDAC_RGB_CUR_MSB_POS: u8 = 0x20; // Cursor MSB position       reset=0x00
pub const RDAC_RGB_CUR_WIDTH:   u8 = 0x21; // Cursor width control      reset=0x02
pub const RDAC_RGB_CUR_DISP_EN: u8 = 0x22; // Cursor display enable     reset=0x03
pub const RDAC_RGB_CUR_BLINK:   u8 = 0x23; // Cursor blink enable       reset=0x00

// ── CRS 6 (setup / PLL) sub-addresses ─────────────────────────────────────────────────────────
pub const RDAC_SETUP_TEST1:       u8 = 0x00; // Test Register 1           reset=0x00
pub const RDAC_SETUP_CMD1:        u8 = 0x01; // Command Register 1        reset=0x40
pub const RDAC_SETUP_DIGOUT_CTRL: u8 = 0x02; // Digital Output Control    reset=0x00
pub const RDAC_SETUP_VIDCLK_CTRL: u8 = 0x03; // VIDCLK* Cycle Control     reset=0x03
pub const RDAC_SETUP_PLL_RATE0:   u8 = 0x05; // Pixel PLL Rate Register 0 reset=0x19
pub const RDAC_SETUP_PLL_RATE1:   u8 = 0x06; // Pixel PLL Rate Register 1 reset=0x04
pub const RDAC_SETUP_PLL_CTRL:    u8 = 0x07; // PLL Control Register      reset=0x??
pub const RDAC_SETUP_PIX_LOAD:    u8 = 0x08; // Pixel Load Control        reset=0x04
pub const RDAC_SETUP_PIX_START:   u8 = 0x09; // Pixel Port Start Position reset=0x28
pub const RDAC_SETUP_PIX_FMT:     u8 = 0x0A; // Pixel Format Control      reset=0x08
pub const RDAC_SETUP_MPX_RATE:    u8 = 0x0B; // MPX Rate Register         reset=0x03
pub const RDAC_SETUP_SIG_ANLYS:   u8 = 0x0C; // Signature Analysis
pub const RDAC_SETUP_PIX_DEPTH:   u8 = 0x0D; // Pixel Depth Control       reset=0x0A
pub const RDAC_SETUP_PAL_BYPASS:  u8 = 0x0E; // Palette Bypass Position   reset=0x00
pub const RDAC_SETUP_PAL_BYPASSW: u8 = 0x0F; // Palette Bypass Width      reset=0x01

// ── Command Register 1 bits (CRS 6, addr 0x01) ────────────────────────────────────────────────
// Reset value = 0x40 = 0b01000000
//   bit 7  — Green sync enable (SoG): 0=disabled, 1=sync on IOG output  reset=0
//   bit 6  — Pedestal enable: 0=0 IRE blank level, 1=7.5 IRE pedestal   reset=1
//   bit 5  — Reserved                                                    reset=0
//   bit 4  — Power Down [1]  \  00=normal, 01=DACs off (LUT→TTL),
//   bit 3  — Power Down [0]  /  10=DACs+RAM off, 11=disable clocking    reset=00
//   bit 2  — Palette addressing mode: 0=sparse (MSB replicate), 1=contiguous (zero-pad)  reset=0
//   bit 1  — Signature Analysis enable: 0=disabled, 1=enabled           reset=0
//   bit 0  — Reset Pipelined Depth: 0→1 transition resets pixel pipeline reset=0
//
// SoG is NOT enabled at power-on reset (bit 7 = 0, pedestal enabled by default via bit 6 = 1).
// PROM and X server write this register to configure it at runtime.
pub const CMD1_SOG_ENABLE:    u8 = 1 << 7; // Sync on Green (IOG) enable
pub const CMD1_PEDESTAL_EN:   u8 = 1 << 6; // 7.5 IRE blanking pedestal (0=0 IRE, 1=7.5 IRE)
pub const CMD1_PWRDOWN_MASK:  u8 = 3 << 3; // power-down field [4:3]
pub const CMD1_PWRDOWN_SHIFT: u8 = 3;
pub const CMD1_PAL_ADDR_MODE: u8 = 1 << 2; // palette address mode (0=sparse, 1=contiguous)
pub const CMD1_SAR_ENABLE:    u8 = 1 << 1; // signature analysis enable
pub const CMD1_RESET_PIPE:    u8 = 1 << 0; // reset pixel pipeline depth

// Power-down field decode values
pub const CMD1_PWR_NORMAL:       u8 = 0b00; // normal operation
pub const CMD1_PWR_DACS_OFF:     u8 = 0b01; // DACs off, LUT→TTL outputs
pub const CMD1_PWR_DACS_RAM_OFF: u8 = 0b10; // DACs and RAM off
pub const CMD1_PWR_CLK_OFF:      u8 = 0b11; // disable internal clocking

// ── Pixel Format Control bits (CRS 6, addr 0x0A) ──────────────────────────────────────────────
//   bit 7  — Pixel Unpacking Order: 0=MSB first, 1=LSB first
//   bit 5  — Cursor Enable
//   bit 4  — Cursor Color 0 Enable
//   bit 3  — Overlay Enable
//   bit 1:0 — Palette Bypass Control: 00=always use palette, 01=always bypass, 10=use pixel field
pub const PIX_FMT_LSB_UNPACK:  u8 = 1 << 7;
pub const PIX_FMT_CURSOR_EN:   u8 = 1 << 5;
pub const PIX_FMT_CURSOR_C0EN: u8 = 1 << 4;
pub const PIX_FMT_OVERLAY_EN:  u8 = 1 << 3;
pub const PIX_FMT_PAL_BYPASS_MASK: u8 = 0x03;

// ── Sizes ─────────────────────────────────────────────────────────────────────────────────────
pub const BT445_PALETTE_SIZE: usize = 256; // primary gamma palette entries
pub const BT445_OVERLAY_SIZE: usize = 16;  // overlay palette entries
pub const BT445_CURSOR_SIZE:  usize = 4;   // cursor color entries

// ── Device struct ─────────────────────────────────────────────────────────────────────────────
pub struct Bt445 {
    // Primary color palette RAM (256 × RGB24) — used as gamma ramp by Newport
    pub palette: [[u8; 3]; BT445_PALETTE_SIZE],
    // Overlay palette (16 × RGB24)
    pub overlay: [[u8; 3]; BT445_OVERLAY_SIZE],
    // Cursor color registers (4 × RGB24)
    pub cursor_color: [[u8; 3]; BT445_CURSOR_SIZE],

    // Address register (8 bits, shared across CRS 1/3/7 accesses)
    pub addr: u8,
    // RGB sub-cycle counter (0→R, 1→G, 2→B; resets on addr write)
    pub rgb_counter: u8,

    // CRS 2 control registers
    pub read_enable:  u8,  // Read Enable     reset=0x00
    pub blink_enable: u8,  // Blink Enable    reset=0x00
    pub cmd0:         u8,  // Command Reg 0   reset=0x43

    // CRS 5 pixel format registers (indexed by sub-addr, 0x00–0x27)
    pub rgb_ctrl: [u8; 0x28],

    // CRS 6 setup/PLL registers (indexed by sub-addr, 0x00–0x0F)
    pub setup: [u8; 0x10],

    pub debug: bool,

    // Set when palette/overlay/cursor change — disp.rs copies on dirty
    pub dirty: bool,
}

impl Bt445 {
    pub fn new() -> Self {
        let mut s = Self {
            palette:      [[0; 3]; BT445_PALETTE_SIZE],
            overlay:      [[0; 3]; BT445_OVERLAY_SIZE],
            cursor_color: [[0; 3]; BT445_CURSOR_SIZE],
            addr:         0,
            rgb_counter:  0,
            read_enable:  0,
            blink_enable: 0,
            cmd0:         0x43, // bit6=1 (ovl color0 disable), bit1=1, bit0=1 (both ovl planes enabled)
            rgb_ctrl:     [0; 0x28],
            setup:        [0; 0x10],
            debug:        false,
            dirty:        true,
        };
        // Apply datasheet reset values to rgb_ctrl (CRS 5)
        s.rgb_ctrl[RDAC_RGB_RED_MSB_POS as usize] = 0x07;
        s.rgb_ctrl[RDAC_RGB_RED_WIDTH   as usize] = 0x08;
        s.rgb_ctrl[RDAC_RGB_RED_DISP_EN as usize] = 0xFF;
        s.rgb_ctrl[RDAC_RGB_GRN_MSB_POS as usize] = 0x07;
        s.rgb_ctrl[RDAC_RGB_GRN_WIDTH   as usize] = 0x08;
        s.rgb_ctrl[RDAC_RGB_GRN_DISP_EN as usize] = 0xFF;
        s.rgb_ctrl[RDAC_RGB_BLU_MSB_POS as usize] = 0x07;
        s.rgb_ctrl[RDAC_RGB_BLU_WIDTH   as usize] = 0x08;
        s.rgb_ctrl[RDAC_RGB_BLU_DISP_EN as usize] = 0xFF;
        s.rgb_ctrl[RDAC_RGB_OVL_MSB_POS as usize] = 0x09;
        s.rgb_ctrl[RDAC_RGB_OVL_WIDTH   as usize] = 0x02;
        s.rgb_ctrl[RDAC_RGB_OVL_DISP_EN as usize] = 0x03;
        s.rgb_ctrl[RDAC_RGB_CUR_WIDTH   as usize] = 0x02;
        s.rgb_ctrl[RDAC_RGB_CUR_DISP_EN as usize] = 0x03;
        // Apply datasheet reset values to setup (CRS 6)
        s.setup[RDAC_SETUP_CMD1        as usize] = 0x40; // bit6=1 (7.5 IRE pedestal); SoG=0
        s.setup[RDAC_SETUP_VIDCLK_CTRL as usize] = 0x03;
        s.setup[RDAC_SETUP_PLL_RATE0   as usize] = 0x19;
        s.setup[RDAC_SETUP_PLL_RATE1   as usize] = 0x04;
        s.setup[RDAC_SETUP_PIX_LOAD    as usize] = 0x04;
        s.setup[RDAC_SETUP_PIX_START   as usize] = 0x28;
        s.setup[RDAC_SETUP_PIX_FMT     as usize] = 0x08; // bit3=1 (overlay enable)
        s.setup[RDAC_SETUP_MPX_RATE    as usize] = 0x03;
        s.setup[RDAC_SETUP_PIX_DEPTH   as usize] = 0x0A;
        s.setup[RDAC_SETUP_PAL_BYPASSW as usize] = 0x01;
        s
    }

    // ── Internal helpers ──────────────────────────────────────────────────────────────────────

    fn cmd1(&self) -> u8 { self.setup[RDAC_SETUP_CMD1 as usize] }

    fn set_addr(&mut self, val: u8) {
        self.addr = val;
        self.rgb_counter = 0;
    }

    fn inc_addr(&mut self) {
        self.addr = self.addr.wrapping_add(1);
        self.rgb_counter = 0;
    }

    // ── SoG accessors (used by monitor status) ────────────────────────────────────────────────

    pub fn sog_enabled(&self) -> bool {
        self.cmd1() & CMD1_SOG_ENABLE != 0
    }

    // ── CRS write ─────────────────────────────────────────────────────────────────────────────

    pub fn write_crs(&mut self, crs: u8, val: u8) {
        if devlog_is_active(LogModule::Bt445) {
            dlog!(LogModule::Bt445, "BT445 Write CRS {} addr={:02x} rgb={} val={:02x}",
                crs, self.addr, self.rgb_counter, val);
        }
        match crs {
            RDAC_CRS_ADDR_REG => {
                self.set_addr(val);
            }

            RDAC_CRS_PAL_RAM => {
                let idx = self.addr as usize % BT445_PALETTE_SIZE;
                match self.rgb_counter {
                    0 => { self.palette[idx][0] = val; self.rgb_counter = 1; }
                    1 => { self.palette[idx][1] = val; self.rgb_counter = 2; }
                    2 => {
                        self.palette[idx][2] = val;
                        self.dirty = true;
                        // After blue: addr auto-increments (wraps naturally within u8/256)
                        self.addr = self.addr.wrapping_add(1);
                        self.rgb_counter = 0;
                    }
                    _ => {}
                }
            }

            RDAC_CRS_CTRL => {
                // Control register accesses do NOT auto-increment the address
                match self.addr {
                    RDAC_CTRL_READ_ENABLE  => self.read_enable  = val,
                    RDAC_CTRL_BLINK_ENABLE => self.blink_enable = val,
                    RDAC_CTRL_CMD0 => {
                        self.cmd0 = val;
                        dlog!(LogModule::Bt445, "BT445 CMD0 = {:02x} ({})", val,
                            Self::decode_cmd0(val));
                    }
                    _ => {} // ID/revision/test/reserved — read-only or ignored
                }
            }

            RDAC_CRS_OVL_RAM => {
                let idx = (self.addr as usize) & 0x0F;
                match self.rgb_counter {
                    0 => { self.overlay[idx][0] = val; self.rgb_counter = 1; }
                    1 => { self.overlay[idx][1] = val; self.rgb_counter = 2; }
                    2 => {
                        self.overlay[idx][2] = val;
                        self.dirty = true;
                        self.inc_addr();
                    }
                    _ => {}
                }
            }

            RDAC_CRS_RGB_CTRL => {
                let idx = self.addr as usize;
                if idx < self.rgb_ctrl.len() {
                    self.rgb_ctrl[idx] = val;
                }
            }

            RDAC_CRS_SETUP => {
                let idx = self.addr as usize;
                if idx < self.setup.len() {
                    let prev = self.setup[idx];
                    self.setup[idx] = val;
                    if idx == RDAC_SETUP_CMD1 as usize {
                        dlog!(LogModule::Bt445, "BT445 CMD1 = {:02x} ({})", val,
                            Self::decode_cmd1(val));
                        // Report SoG transitions explicitly
                        let prev_sog = prev & CMD1_SOG_ENABLE != 0;
                        let new_sog  = val  & CMD1_SOG_ENABLE != 0;
                        if prev_sog != new_sog {
                            eprintln!("BT445: Sync-on-Green (IOG) {}",
                                if new_sog { "ENABLED" } else { "DISABLED" });
                        }
                    }
                }
            }

            RDAC_CRS_CURSOR => {
                let idx = (self.addr as usize) & 0x03;
                match self.rgb_counter {
                    0 => { self.cursor_color[idx][0] = val; self.rgb_counter = 1; }
                    1 => { self.cursor_color[idx][1] = val; self.rgb_counter = 2; }
                    2 => {
                        self.cursor_color[idx][2] = val;
                        self.dirty = true;
                        self.inc_addr();
                    }
                    _ => {}
                }
            }

            _ => {}
        }
    }

    // ── Bulk write (32-bit DCB path) ───────────────────────────────────────────────
    //
    // ramdac_write():
    //   CRS 0: addr = data[7:0]
    //   CRS 1: R = data[15:8], G = data[23:16], B = data[31:24]; addr++
    //
    // The rex3.rs DCB path calls write_crs byte-by-byte for multi-byte transfers.
    // This provides an alternative for direct 32-bit gamma-load writes.
    pub fn write32(&mut self, crs: u8, data: u32) {
        match crs {
            RDAC_CRS_ADDR_REG => {
                self.set_addr(data as u8);
            }
            RDAC_CRS_PAL_RAM => {
                let idx = self.addr as usize % BT445_PALETTE_SIZE;
                self.palette[idx][0] = (data >>  8) as u8; // R in bits [15:8]
                self.palette[idx][1] = (data >> 16) as u8; // G in bits [23:16]
                self.palette[idx][2] = (data >> 24) as u8; // B in bits [31:24]
                self.dirty = true;
                self.addr = self.addr.wrapping_add(1);
                self.rgb_counter = 0;
            }
            _ => {
                self.write_crs(crs, data as u8);
            }
        }
    }

    // ── CRS read ──────────────────────────────────────────────────────────────────────────────

    pub fn read_crs(&mut self, crs: u8) -> u8 {
        let val = match crs {
            RDAC_CRS_ADDR_REG => self.addr,

            RDAC_CRS_PAL_RAM => {
                let idx = self.addr as usize % BT445_PALETTE_SIZE;
                match self.rgb_counter {
                    0 => { self.rgb_counter = 1; self.palette[idx][0] }
                    1 => { self.rgb_counter = 2; self.palette[idx][1] }
                    2 => {
                        let b = self.palette[idx][2];
                        self.addr = self.addr.wrapping_add(1);
                        self.rgb_counter = 0;
                        b
                    }
                    _ => 0,
                }
            }

            RDAC_CRS_CTRL => {
                match self.addr {
                    RDAC_CTRL_ID           => 0x3A,
                    RDAC_CTRL_REVISION     => 0xA0,
                    RDAC_CTRL_RESERVED0    => 0xFF,
                    RDAC_CTRL_READ_ENABLE  => self.read_enable,
                    RDAC_CTRL_BLINK_ENABLE => self.blink_enable,
                    RDAC_CTRL_CMD0         => self.cmd0,
                    RDAC_CTRL_TEST0        => 0,
                    _                      => 0,
                }
            }

            RDAC_CRS_OVL_RAM => {
                let idx = (self.addr as usize) & 0x0F;
                match self.rgb_counter {
                    0 => { self.rgb_counter = 1; self.overlay[idx][0] }
                    1 => { self.rgb_counter = 2; self.overlay[idx][1] }
                    2 => {
                        let b = self.overlay[idx][2];
                        self.inc_addr();
                        b
                    }
                    _ => 0,
                }
            }

            RDAC_CRS_RGB_CTRL => {
                let idx = self.addr as usize;
                if idx < self.rgb_ctrl.len() { self.rgb_ctrl[idx] } else { 0 }
            }

            RDAC_CRS_SETUP => {
                let idx = self.addr as usize;
                if idx < self.setup.len() { self.setup[idx] } else { 0 }
            }

            RDAC_CRS_CURSOR => {
                let idx = (self.addr as usize) & 0x03;
                match self.rgb_counter {
                    0 => { self.rgb_counter = 1; self.cursor_color[idx][0] }
                    1 => { self.rgb_counter = 2; self.cursor_color[idx][1] }
                    2 => {
                        let b = self.cursor_color[idx][2];
                        self.inc_addr();
                        b
                    }
                    _ => 0,
                }
            }

            _ => 0,
        };
        if devlog_is_active(LogModule::Bt445) {
            dlog!(LogModule::Bt445, "BT445 Read  CRS {} addr={:02x} rgb={} -> {:02x}",
                crs, self.addr, self.rgb_counter, val);
        }
        val
    }

    // ── Command register decoders ─────────────────────────────────────────────────────────────

    pub fn decode_cmd0(v: u8) -> String {
        let mut parts = Vec::new();
        if v & CMD0_OVL_COLOR0_DISABLE != 0 { parts.push("OVL0→CMAP"); } else { parts.push("OVL0→OVL_COLOR"); }
        let blink = match (v & CMD0_BLINK_RATE_MASK) >> 4 {
            0 => "blink=16/48(25%)",
            1 => "blink=16/16(50%)",
            2 => "blink=32/32(50%)",
            3 => "blink=64/64(50%)",
            _ => "blink=?",
        };
        parts.push(blink);
        if v & CMD0_OVL1_BLINK_EN != 0 { parts.push("OVL1_BLINK"); }
        if v & CMD0_OVL0_BLINK_EN != 0 { parts.push("OVL0_BLINK"); }
        if v & CMD0_OVL1_DISP_EN  != 0 { parts.push("OVL1_EN"); } else { parts.push("OVL1_DIS"); }
        if v & CMD0_OVL0_DISP_EN  != 0 { parts.push("OVL0_EN"); } else { parts.push("OVL0_DIS"); }
        parts.join(" ")
    }

    pub fn decode_cmd1(v: u8) -> String {
        let mut parts = Vec::new();
        if v & CMD1_SOG_ENABLE   != 0 { parts.push("SOG=ON"); }  else { parts.push("SOG=off"); }
        if v & CMD1_PEDESTAL_EN  != 0 { parts.push("pedestal=7.5IRE"); } else { parts.push("pedestal=0IRE"); }
        let pwr = match (v & CMD1_PWRDOWN_MASK) >> CMD1_PWRDOWN_SHIFT {
            CMD1_PWR_NORMAL       => "pwr=normal",
            CMD1_PWR_DACS_OFF     => "pwr=DACs_off",
            CMD1_PWR_DACS_RAM_OFF => "pwr=DACs+RAM_off",
            CMD1_PWR_CLK_OFF      => "pwr=clk_off",
            _                     => "pwr=?",
        };
        parts.push(pwr);
        if v & CMD1_PAL_ADDR_MODE != 0 { parts.push("pal=contiguous"); } else { parts.push("pal=sparse"); }
        if v & CMD1_SAR_ENABLE    != 0 { parts.push("SAR"); }
        if v & CMD1_RESET_PIPE    != 0 { parts.push("RESET_PIPE"); }
        parts.join(" ")
    }

    // ── Palette snapshot for disp.rs ──────────────────────────────────────────────────────────

    pub fn palette_as_rgb(&self) -> [u32; BT445_PALETTE_SIZE] {
        let mut out = [0u32; BT445_PALETTE_SIZE];
        for (i, rgb) in self.palette.iter().enumerate() {
            out[i] = ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | (rgb[2] as u32);
        }
        out
    }

    // ── Status dump ───────────────────────────────────────────────────────────────────────────

    pub fn print_status(&self, writer: &mut dyn Write) {
        let cmd0 = self.cmd0;
        let cmd1 = self.cmd1();
        let pix_fmt = self.setup[RDAC_SETUP_PIX_FMT as usize];

        writeln!(writer, "BT445 RAMDAC status:").unwrap();
        writeln!(writer, "  ID=0x3A  Revision=0xA0").unwrap();
        writeln!(writer, "  addr={:02x}  rgb_counter={}", self.addr, self.rgb_counter).unwrap();

        writeln!(writer, "  CMD0={:02x}  {}", cmd0, Self::decode_cmd0(cmd0)).unwrap();
        writeln!(writer, "  CMD1={:02x}  {}", cmd1, Self::decode_cmd1(cmd1)).unwrap();

        // SoG is the most operationally interesting bit — call it out clearly
        writeln!(writer, "  Sync-on-Green (IOG): {}",
            if cmd1 & CMD1_SOG_ENABLE != 0 { "ENABLED" } else { "disabled (power-on default)" }).unwrap();
        writeln!(writer, "  Blanking pedestal: {}",
            if cmd1 & CMD1_PEDESTAL_EN != 0 { "7.5 IRE (power-on default)" } else { "0 IRE" }).unwrap();

        let pwr = (cmd1 & CMD1_PWRDOWN_MASK) >> CMD1_PWRDOWN_SHIFT;
        writeln!(writer, "  Power mode: {}", match pwr {
            CMD1_PWR_NORMAL       => "normal (power-on default)",
            CMD1_PWR_DACS_OFF     => "DACs off (LUT→TTL outputs)",
            CMD1_PWR_DACS_RAM_OFF => "DACs and RAM off",
            CMD1_PWR_CLK_OFF      => "internal clocking disabled",
            _                     => "?",
        }).unwrap();

        writeln!(writer, "  Pixel format ctrl={:02x}: unpack={} cursor={} ovl={} pal_bypass={}",
            pix_fmt,
            if pix_fmt & PIX_FMT_LSB_UNPACK != 0 { "LSB" } else { "MSB" },
            if pix_fmt & PIX_FMT_CURSOR_EN  != 0 { "on" } else { "off" },
            if pix_fmt & PIX_FMT_OVERLAY_EN != 0 { "on" } else { "off" },
            match pix_fmt & PIX_FMT_PAL_BYPASS_MASK {
                0 => "always-use",
                1 => "always-bypass",
                2 => "pixel-field",
                _ => "reserved",
            },
        ).unwrap();

        writeln!(writer, "  read_enable={:02x}  blink_enable={:02x}", self.read_enable, self.blink_enable).unwrap();

        writeln!(writer, "  PLL: rate0={:02x} rate1={:02x} ctrl={:02x}",
            self.setup[RDAC_SETUP_PLL_RATE0 as usize],
            self.setup[RDAC_SETUP_PLL_RATE1 as usize],
            self.setup[RDAC_SETUP_PLL_CTRL  as usize]).unwrap();
        writeln!(writer, "  Pix: start={:02x} fmt={:02x} mpx_rate={:02x} depth={:02x}",
            self.setup[RDAC_SETUP_PIX_START as usize],
            self.setup[RDAC_SETUP_PIX_FMT   as usize],
            self.setup[RDAC_SETUP_MPX_RATE  as usize],
            self.setup[RDAC_SETUP_PIX_DEPTH as usize]).unwrap();
        writeln!(writer, "  VIDCLK_ctrl={:02x}  pal_bypass_pos={:02x}  pal_bypass_width={:02x}",
            self.setup[RDAC_SETUP_VIDCLK_CTRL as usize],
            self.setup[RDAC_SETUP_PAL_BYPASS  as usize],
            self.setup[RDAC_SETUP_PAL_BYPASSW as usize]).unwrap();

        // Palette: full dump, 8 entries per row, skip all-zero rows
        {
            let nz_count = (0..BT445_PALETTE_SIZE).filter(|&i| self.palette[i] != [0,0,0]).count();
            if nz_count == 0 {
                writeln!(writer, "  Palette: all zeros").unwrap();
            } else {
                writeln!(writer, "  Palette ({} non-zero entries):", nz_count).unwrap();
                for row in 0..(BT445_PALETTE_SIZE / 8) {
                    let base = row * 8;
                    if self.palette[base..base+8].iter().all(|e| e == &[0,0,0]) { continue; }
                    let mut line = format!("  {:02x}:", base);
                    for i in 0..8 {
                        let e = &self.palette[base + i];
                        line.push_str(&format!("  {:02x}{:02x}{:02x}", e[0], e[1], e[2]));
                    }
                    writeln!(writer, "{}", line).unwrap();
                }
            }
        }

        // Overlay palette
        let ovl_nz: Vec<usize> = (0..BT445_OVERLAY_SIZE)
            .filter(|&i| self.overlay[i] != [0, 0, 0])
            .collect();
        if ovl_nz.is_empty() {
            writeln!(writer, "  Overlay palette: all zeros").unwrap();
        } else {
            let show: Vec<String> = ovl_nz.iter()
                .map(|&i| format!("[{:x}]={:02x}{:02x}{:02x}",
                    i, self.overlay[i][0], self.overlay[i][1], self.overlay[i][2]))
                .collect();
            writeln!(writer, "  Overlay palette: {}", show.join("  ")).unwrap();
        }

        // Cursor colors
        let curs: Vec<String> = (0..BT445_CURSOR_SIZE)
            .map(|i| format!("[{}]={:02x}{:02x}{:02x}",
                i, self.cursor_color[i][0], self.cursor_color[i][1], self.cursor_color[i][2]))
            .collect();
        writeln!(writer, "  Cursor colors: {}", curs.join("  ")).unwrap();
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for Bt445 {
    fn default() -> Self { Self::new() }
}
