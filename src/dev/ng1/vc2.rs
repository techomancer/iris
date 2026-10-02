// VC2 (Video Controller 2)
use bitfield::bitfield;
use crate::devlog::{LogModule, devlog_is_active};

// Register Indices
pub const VC2_REG_VIDEO_ENTRY_PTR: u8 = 0x00;
pub const VC2_REG_CURSOR_ENTRY_PTR: u8 = 0x01;
pub const VC2_REG_CURSOR_X_LOC: u8 = 0x02;
pub const VC2_REG_CURSOR_Y_LOC: u8 = 0x03;
pub const VC2_REG_CURRENT_CURSOR_X: u8 = 0x04;
pub const VC2_REG_DID_ENTRY_PTR: u8 = 0x05;
pub const VC2_REG_SCANLINE_LEN: u8 = 0x06;
pub const VC2_REG_RAM_ADDR: u8 = 0x07;
pub const VC2_REG_VT_FRAME_PTR: u8 = 0x08;
pub const VC2_REG_VT_LINE_SEQ_PTR: u8 = 0x09;
pub const VC2_REG_VT_LINES_IN_RUN: u8 = 0x0A;
pub const VC2_REG_VERT_LINE_CTR: u8 = 0x0B;
pub const VC2_REG_CURSOR_TABLE_PTR: u8 = 0x0C;
pub const VC2_REG_WORKING_CURSOR_Y: u8 = 0x0D;
pub const VC2_REG_DID_FRAME_PTR: u8 = 0x0E;
pub const VC2_REG_DID_LINE_PTR: u8 = 0x0F;
pub const VC2_REG_DISPLAY_CONTROL: u8 = 0x10;
pub const VC2_REG_CONFIG: u8 = 0x1F;

// Configuration Register Bits (0x1F)
pub const VC2_CONFIG_SOFT_RESET: u16 = 1 << 0;
pub const VC2_CONFIG_SLOW_CLOCK: u16 = 1 << 1;
pub const VC2_CONFIG_CURSOR_ERR: u16 = 1 << 2;
pub const VC2_CONFIG_DID_ERR: u16 = 1 << 3;
pub const VC2_CONFIG_VTG_ERR: u16 = 1 << 4;
pub const VC2_CONFIG_REV_MASK: u16 = 0x7 << 5;
pub const VC2_CONFIG_REV_SHIFT: u8 = 5;
pub const VC2_CONFIG_TEMP_HI_MASK: u16 = 0xFF << 8;
pub const VC2_CONFIG_TEMP_HI_SHIFT: u8 = 8;

// Display Control Register Bits (0x10)
pub const VC2_CTRL_VINTR_EN: u16 = 1 << 0;
pub const VC2_CTRL_BLACKOUT: u16 = 1 << 1;
pub const VC2_CTRL_VIDEO_TIMING_EN: u16 = 1 << 2;
pub const VC2_CTRL_DID_EN: u16 = 1 << 3;
pub const VC2_CTRL_CURSOR_FUNC_EN: u16 = 1 << 4;
pub const VC2_CTRL_GENSYNC_EN: u16 = 1 << 5;
pub const VC2_CTRL_INTERLACE: u16 = 1 << 6;
pub const VC2_CTRL_CURSOR_EN: u16 = 1 << 7;
pub const VC2_CTRL_CURSOR_MODE: u16 = 1 << 8;
pub const VC2_CTRL_CURSOR_SIZE: u16 = 1 << 9;
pub const VC2_CTRL_GENLOCK_SEL: u16 = 1 << 10;

// Video Timing state bits — signals driven by the VC2 VTG.
// Each line-sequence entry carries three 7-bit state bytes (A, B, C).
// These constants are bit positions *within* each state byte (bits 0..6).
// The packed 21-bit word order is: state_c[6:0] | state_b[6:0]<<7 | state_a[6:0]<<14
//
// State C (bits 0..6 of packed word — lowest in RAM encoding)
pub const VT_VERT_INT_REX_N:   u8 = 1 << 0; // C[0]  Vertical interrupt to REX3
pub const VT_VSYNC_ARC_N:      u8 = 1 << 1; // C[1]  Vertical sync to ARC monitor
pub const VT_HSYNC_ARC_N:      u8 = 1 << 2; // C[2]  Horizontal sync to ARC monitor
pub const VT_CSYNC_ARC_N:      u8 = 1 << 3; // C[3]  Composite sync to ARC monitor
pub const VT_VERT_STAT_GIO_N:  u8 = 1 << 4; // C[4]  Vertical status to GIO
pub const VT_SPARE:            u8 = 1 << 5; // C[5]  Spare
pub const VT_CBLANK_XMAP_N:    u8 = 1 << 6; // C[6]  Composite blanking to XMAP
// State B (bits 7..13 of packed word)
pub const VT_HBLANK_AB_N:      u8 = 1 << 0; // B[0]  Horizontal blanking to AB1
pub const VT_EOF_AB_N:         u8 = 1 << 1; // B[1]  End of frame to AB1
pub const VT_CBLANK_CMAP_N:    u8 = 1 << 2; // B[2]  Composite blanking to CMAP
pub const VT_SET_TSC_REX_N:    u8 = 1 << 3; // B[3]  Vertical sync to REX3 (clears line counter)
pub const VT_ODDFIELD_VC_N:    u8 = 1 << 4; // B[4]  Odd field (interlaced/stereo)
pub const VT_EOF_VC_N:         u8 = 1 << 5; // B[5]  End of field/frame; flushes DID FIFO
pub const VT_VPOS_VC_N:        u8 = 1 << 6; // B[6]  End of field; resets vertical line counter and positions cursor vertically
// State A (bits 14..20 of packed word — highest in RAM encoding)
pub const VT_VIS_LN_VC_N:      u8 = 1 << 0; // A[0]  Visible line — asserted during active display; used for DID generation
pub const VT_HPOS_VC_N:        u8 = 1 << 1; // A[1]  Horizontal position — leading edge positions cursor horizontally; falling edge increments vertical line counter and triggers cursor EOL
pub const VT_DSPLY_EN_RO_N:    u8 = 1 << 2; // A[2]  Display enable to RO1 (pixel data output)
pub const VT_SER_EN_RO_N:      u8 = 1 << 3; // A[3]  Serial enable to RO1 (VRAM clock enable)
pub const VT_TX_REQ_REX_N:     u8 = 1 << 4; // A[4]  VRAM transfer request to REX3
pub const VT_CSYNC_DAC_N:      u8 = 1 << 5; // A[5]  Composite sync to RAMDAC
pub const VT_CBLANK_DAC_N:     u8 = 1 << 6; // A[6]  Composite blanking to RAMDAC

bitfield! {
    #[derive(Clone, Copy, Default)]
    pub struct VideoTiming(u32);
    impl Debug;
    pub vert_int_rex_n, _: 0;
    pub vsync_arc_n, _: 1;
    pub hsync_arc_n, _: 2;
    pub csync_arc_n, _: 3;
    pub vert_stat_gio_n, _: 4;
    pub spare, _: 5;
    pub cblank_xmap_n, _: 6;
    pub hblank_ab_n, _: 7;
    pub eof_ab_n, _: 8;
    pub cblank_cmap_n, _: 9;
    pub set_tsc_rex_n, _: 10;
    pub oddfield_vc_n, _: 11;
    pub eof_vc_n, _: 12;
    pub vpos_vc_n, _: 13;
    pub vis_ln_vc_n, _: 14;
    pub hpos_vc_n, _: 15;
    pub dsply_en_ro_n, _: 16;
    pub ser_en_ro_n, _: 17;
    pub tx_req_rex_n, _: 18;
    pub csync_dac_n, _: 19;
    pub cblank_dac_n, _: 20;
}

fn vc2_reg_name(idx: usize) -> &'static str {
    match idx as u8 {
        VC2_REG_VIDEO_ENTRY_PTR => "Video Entry Ptr",
        VC2_REG_CURSOR_ENTRY_PTR => "Cursor Entry Ptr",
        VC2_REG_CURSOR_X_LOC => "Cursor X Loc",
        VC2_REG_CURSOR_Y_LOC => "Cursor Y Loc",
        VC2_REG_CURRENT_CURSOR_X => "Current Cursor X",
        VC2_REG_DID_ENTRY_PTR => "DID Entry Ptr",
        VC2_REG_SCANLINE_LEN => "Scanline Len",
        VC2_REG_RAM_ADDR => "RAM Addr",
        VC2_REG_VT_FRAME_PTR => "VT Frame Ptr",
        VC2_REG_VT_LINE_SEQ_PTR => "VT Line Seq Ptr",
        VC2_REG_VT_LINES_IN_RUN => "VT Lines In Run",
        VC2_REG_VERT_LINE_CTR => "Vert Line Ctr",
        VC2_REG_CURSOR_TABLE_PTR => "Cursor Table Ptr",
        VC2_REG_WORKING_CURSOR_Y => "Working Cursor Y",
        VC2_REG_DID_FRAME_PTR => "DID Frame Ptr",
        VC2_REG_DID_LINE_PTR => "DID Line Ptr",
        VC2_REG_DISPLAY_CONTROL => "Display Control",
        VC2_REG_CONFIG => "Config",
        _ => "Unknown",
    }
}

pub struct Vc2 {
    pub index: u8,
    pub regs: [u16; 32],
    pub ram: Vec<u16>,
    pub dirty: bool,
}

impl Vc2 {
    pub fn new() -> Self {
        Self {
            index: 0,
            regs: [0; 32],
            ram: vec![0; 32768],
            dirty: true,
        }
    }

    fn update_temp_hi(&mut self, reg_idx: usize, val: u16) {
        // Config[15:8] (TEMP_HI) captures the high byte of the last CRS=1
        // access for registers 0x00..=0x10 (read-only, updated on every access).
        if reg_idx <= VC2_REG_DISPLAY_CONTROL as usize {
            let hi = (val >> 8) as u16;
            let cfg = &mut self.regs[VC2_REG_CONFIG as usize];
            *cfg = (*cfg & !VC2_CONFIG_TEMP_HI_MASK) | (hi << VC2_CONFIG_TEMP_HI_SHIFT);
        }
    }

    pub fn read(&mut self, crs: u8) -> u32 {
        let val = match crs {
            1 => { // Register Read
                let idx = (self.index & 0x1F) as usize;
                let v = self.regs[idx];
                self.update_temp_hi(idx, v);
                v as u32
            }
            3 => { // RAM Read
                let addr = self.regs[VC2_REG_RAM_ADDR as usize] as usize;
                if addr < self.ram.len() {
                    let val = self.ram[addr];
                    self.regs[VC2_REG_RAM_ADDR as usize] = (addr.wrapping_add(1) & 0x7FFF) as u16;
                    val as u32
                } else {
                    0
                }
            }
            _ => 0,
        };
        dlog!(LogModule::Vc2, "VC2 Read  CRS {} [{}] -> {:04x}", crs, vc2_reg_name((self.index & 0x1F) as usize), val);
        val
    }

    fn write_reg_crs1(&mut self, idx: usize, val: u16) {
        self.update_temp_hi(idx, val);
        self.write_reg(idx, val);
    }

    fn write_reg(&mut self, idx: usize, val: u16) {
        if devlog_is_active(LogModule::Vc2) {
            let reg_name = vc2_reg_name(idx);
            let extra = if idx == VC2_REG_DISPLAY_CONTROL as usize {
                format!(" (Blackout={}, Timing={})",
                    (val & VC2_CTRL_BLACKOUT) != 0,
                    (val & VC2_CTRL_VIDEO_TIMING_EN) != 0)
            } else if idx == VC2_REG_RAM_ADDR as usize {
                format!(" (RAM Addr={:04x})", val)
            } else if idx == VC2_REG_SCANLINE_LEN as usize {
                format!(" (Len={})", val >> 5)
            } else {
                String::new()
            };
            dlog!(LogModule::Vc2, "VC2 Write Reg {:02x} [{}] <- {:04x}{}", idx, reg_name, val, extra);
        }
        self.regs[idx] = val;
        if idx == VC2_REG_CURSOR_Y_LOC as usize {
            self.regs[VC2_REG_CURRENT_CURSOR_X as usize] = self.regs[VC2_REG_CURSOR_X_LOC as usize];
        }
        self.dirty = true;
    }

    pub fn write(&mut self, crs: u8, val: u32, width: u8) {
        dlog!(LogModule::Vc2, "VC2 Write CRS {} <- {:08x} (w{})", crs, val, width);
        match width {
            8 => {
                match crs {
                    0 => self.index = (val as u8) & 0x1F,
                    1 => {
                        let idx = (self.index & 0x1F) as usize;
                        let new_val = (self.regs[idx] & 0x00FF) | ((val as u16) << 8);
                        self.write_reg_crs1(idx, new_val);
                    }
                    2 => {
                        let idx = (self.index & 0x1F) as usize;
                        let new_val = (self.regs[idx] & 0xFF00) | ((val as u8) as u16);
                        self.write_reg(idx, new_val);
                    }
                    _ => {}
                }
            }
            16 => {
                // 16-bit DCB transfer: data is LSB-aligned in bits[15:0]
                let w = val as u16;
                if crs == 3 {
                    // RAM Write
                    let addr = self.regs[VC2_REG_RAM_ADDR as usize] as usize;
                    if addr < self.ram.len() {
                        dlog!(LogModule::Vc2, "VC2 RAM Write [{:04x}] = {:04x}", addr, w);
                        self.ram[addr] = w;
                        self.regs[VC2_REG_RAM_ADDR as usize] = (addr.wrapping_add(1) & 0x7FFF) as u16;
                        self.dirty = true;
                    }
                } else if crs == 0 {
                    self.index = (w as u8) & 0x1F;
                } else {
                    // Register Write (16-bit) via CRS=1
                    let idx = (self.index & 0x1F) as usize;
                    self.write_reg_crs1(idx, w);
                }
            }
            24 | 32 => {
                if crs == 0 {
                    // Register Write: Index in bits 24-31, Data in 8-23
                    let idx = ((val >> 24) & 0x1F) as usize;
                    let data = ((val >> 8) & 0xFFFF) as u16;
                    self.index = idx as u8;
                    self.write_reg(idx, data);
                }
            }
            _ => {}
        }
    }

    /// Dump the VC2 RAM as hex words, 16 per line, for offline decoding.
    pub fn dump_ram(&self, writer: &mut dyn std::io::Write) {
        let regs = &self.regs;
        // Print key register pointers first so the decoder knows where to start.
        writeln!(writer, "# VC2 RAM dump").unwrap();
        writeln!(writer, "# VIDEO_ENTRY_PTR  = {:04x}", regs[VC2_REG_VIDEO_ENTRY_PTR as usize]).unwrap();
        writeln!(writer, "# VT_FRAME_PTR     = {:04x}", regs[VC2_REG_VT_FRAME_PTR as usize]).unwrap();
        writeln!(writer, "# DID_ENTRY_PTR    = {:04x}", regs[VC2_REG_DID_ENTRY_PTR as usize]).unwrap();
        writeln!(writer, "# DID_FRAME_PTR    = {:04x}", regs[VC2_REG_DID_FRAME_PTR as usize]).unwrap();
        writeln!(writer, "# SCANLINE_LEN     = {:04x}  (scan_len={})", regs[VC2_REG_SCANLINE_LEN as usize], (regs[VC2_REG_SCANLINE_LEN as usize] >> 5) as usize).unwrap();
        writeln!(writer, "# CURSOR_ENTRY_PTR = {:04x}", regs[VC2_REG_CURSOR_ENTRY_PTR as usize]).unwrap();
        writeln!(writer, "# DISPLAY_CONTROL  = {:04x}", regs[VC2_REG_DISPLAY_CONTROL as usize]).unwrap();
        // Only dump through the last non-zero word to keep output compact.
        let last_nonzero = self.ram.iter().rposition(|&w| w != 0).unwrap_or(0);
        let dump_len = (last_nonzero + 16) & !15; // round up to 16-word boundary
        writeln!(writer, "# words: {} (dumping 0..{:04x})", self.ram.len(), dump_len).unwrap();
        for (i, chunk) in self.ram[..dump_len].chunks(16).enumerate() {
            let addr = i * 16;
            let hex: Vec<String> = chunk.iter().map(|w| format!("{:04x}", w)).collect();
            writeln!(writer, "{:04x}: {}", addr, hex.join(" ")).unwrap();
        }
    }

    pub fn print_status(&self, writer: &mut dyn std::io::Write) {
        let regs: &[(u8, &str)] = &[
            (VC2_REG_VIDEO_ENTRY_PTR,  "VideoEntryPtr "),
            (VC2_REG_CURSOR_ENTRY_PTR, "CursorEntryPtr"),
            (VC2_REG_CURSOR_X_LOC,     "CursorXLoc    "),
            (VC2_REG_CURSOR_Y_LOC,     "CursorYLoc    "),
            (VC2_REG_CURRENT_CURSOR_X, "CurrentCursorX"),
            (VC2_REG_DID_ENTRY_PTR,    "DIDEntryPtr   "),
            (VC2_REG_SCANLINE_LEN,     "ScanlineLen   "),
            (VC2_REG_RAM_ADDR,         "RAMAddr       "),
            (VC2_REG_VT_FRAME_PTR,     "VTFramePtr    "),
            (VC2_REG_VT_LINE_SEQ_PTR,  "VTLineSeqPtr  "),
            (VC2_REG_VT_LINES_IN_RUN,  "VTLinesInRun  "),
            (VC2_REG_VERT_LINE_CTR,    "VertLineCtr   "),
            (VC2_REG_CURSOR_TABLE_PTR, "CursorTablePtr"),
            (VC2_REG_WORKING_CURSOR_Y, "WorkingCursorY"),
            (VC2_REG_DID_FRAME_PTR,    "DIDFramePtr   "),
            (VC2_REG_DID_LINE_PTR,     "DIDLinePtr    "),
            (VC2_REG_DISPLAY_CONTROL,  "DisplayControl"),
            (VC2_REG_CONFIG,           "Config        "),
        ];
        writeln!(writer, "=== VC2 Registers ===").unwrap();
        for &(idx, name) in regs {
            writeln!(writer, "  [{:02x}] {} = {:04x}", idx, name, self.regs[idx as usize]).unwrap();
        }
        let dc = self.regs[VC2_REG_DISPLAY_CONTROL as usize];
        writeln!(writer, "  DisplayControl: VINTR_EN={} BLACKOUT={} TIMING_EN={} DID_EN={} CURSOR_FUNC_EN={} GENSYNC_EN={} INTERLACE={} CURSOR_EN={} CURSOR_MODE={} CURSOR_SIZE={} GENLOCK_SEL={}",
            (dc & VC2_CTRL_VINTR_EN)        != 0,
            (dc & VC2_CTRL_BLACKOUT)        != 0,
            (dc & VC2_CTRL_VIDEO_TIMING_EN) != 0,
            (dc & VC2_CTRL_DID_EN)          != 0,
            (dc & VC2_CTRL_CURSOR_FUNC_EN)  != 0,
            (dc & VC2_CTRL_GENSYNC_EN)      != 0,
            (dc & VC2_CTRL_INTERLACE)       != 0,
            (dc & VC2_CTRL_CURSOR_EN)       != 0,
            (dc & VC2_CTRL_CURSOR_MODE)     != 0,
            (dc & VC2_CTRL_CURSOR_SIZE)     != 0,
            (dc & VC2_CTRL_GENLOCK_SEL)     != 0,
        ).unwrap();
        let cfg = self.regs[VC2_REG_CONFIG as usize];
        writeln!(writer, "  Config: SOFT_RESET={} SLOW_CLOCK={} CURSOR_ERR={} DID_ERR={} VTG_ERR={} REV={} TEMP_HI={:#04x}",
            (cfg & VC2_CONFIG_SOFT_RESET)   != 0,
            (cfg & VC2_CONFIG_SLOW_CLOCK)   != 0,
            (cfg & VC2_CONFIG_CURSOR_ERR)   != 0,
            (cfg & VC2_CONFIG_DID_ERR)      != 0,
            (cfg & VC2_CONFIG_VTG_ERR)      != 0,
            (cfg & VC2_CONFIG_REV_MASK)     >> VC2_CONFIG_REV_SHIFT,
            (cfg & VC2_CONFIG_TEMP_HI_MASK) >> VC2_CONFIG_TEMP_HI_SHIFT,
        ).unwrap();
        writeln!(writer, "  index={:02x}  dirty={}", self.index, self.dirty).unwrap();
    }
}

impl Default for Vc2 {
    fn default() -> Self {
        Self::new()
    }
}
