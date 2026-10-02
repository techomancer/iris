// CMAP (Color Map)
use crate::devlog::{LogModule, devlog_is_active};

pub const CMAP_REG_ADDR_LO: u8 = 0;
pub const CMAP_REG_ADDR_HI: u8 = 1;
pub const CMAP_REG_PALETTE: u8 = 2;
pub const CMAP_REG_COMMAND: u8 = 3;
pub const CMAP_REG_STATUS: u8 = 4;
pub const CMAP_REG_COLOR_BUF: u8 = 5;
pub const CMAP_REG_REVISION: u8 = 6;
pub const CMAP_REG_READ_INIT: u8 = 7;

pub struct Cmap {
    pub id: usize,
    pub palette: [u32; 8192], // 13-bit address space
    pub addr_lo: u8,
    pub addr_hi: u8,
    pub rgb_counter: u8, // 0..2
    pub red_temp: u8,
    pub green_temp: u8,
    pub command: u8,
    pub revision: u8,
    pub dirty: bool,
}

impl Cmap {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            palette: [0; 8192],
            addr_lo: 0,
            addr_hi: 0,
            rgb_counter: 0,
            red_temp: 0,
            green_temp: 0,
            command: 0,
            revision: if id == 0 {0x02 } else {0xA2}, // XL24 (Indy 24-bit): cmap_revision=0x02, cmap1 has monitor id (10) in upper nibble
            dirty: true,
        }
    }

    fn get_address(&self) -> usize {
        ((self.addr_hi as usize) << 8) | (self.addr_lo as usize)
    }

    fn inc_address(&mut self) {
        let mut addr = self.get_address();
        addr = (addr + 1) & 0x1FFF;
        self.addr_lo = (addr & 0xFF) as u8;
        self.addr_hi = ((addr >> 8) & 0xFF) as u8;
    }

    pub fn read_crs(&mut self, crs: u8) -> u8 {
        let val = match crs {
            CMAP_REG_ADDR_LO => self.addr_lo,
            CMAP_REG_ADDR_HI => self.addr_hi,
            CMAP_REG_PALETTE => {
                let addr = self.get_address();
                let color = self.palette[addr];
                match self.rgb_counter {
                    0 => { // Red
                        self.rgb_counter = 1;
                        (color & 0xFF) as u8
                    }
                    1 => { // Green
                        self.rgb_counter = 2;
                        ((color >> 8) & 0xFF) as u8
                    }
                    2 => { // Blue
                        self.rgb_counter = 0;
                        self.inc_address();
                        ((color >> 16) & 0xFF) as u8
                    }
                    _ => 0,
                }
            }
            CMAP_REG_COMMAND => self.command,
            CMAP_REG_STATUS => {
                // SB0: RGB0
                // SB1: RGB1
                // SB2: EFB (Empty Flag) - 0=Empty
                // SB3: HFB/AFB - 0=Half Full (Active Low) -> 1=Not Half Full
                // SB4: FFB - 0=Full (Active Low) -> 1=Not Full
                let mut status = 0;
                if (self.rgb_counter & 1) != 0 { status |= 1; }
                if (self.rgb_counter & 2) != 0 { status |= 2; }
                // FIFO is always empty in this emulation
                status |= 0 << 2; // Empty (Low)
                status |= 1 << 3; // Not Half Full (High)
                status |= 1 << 4; // Not Full (High)
                status
            }
            CMAP_REG_COLOR_BUF => {
                match self.rgb_counter {
                    0 => {
                        self.rgb_counter = 1;
                        self.red_temp
                    }
                    1 => {
                        self.green_temp
                    }
                    _ => 0,
                }
            }
            CMAP_REG_REVISION => self.revision,
            CMAP_REG_READ_INIT => 0,
            _ => 0,
        };
        if devlog_is_active(LogModule::Cmap) {
            dlog_dev!(LogModule::Cmap, "CMAP({}) Read  CRS {} -> {:02x}", self.id, crs, val);
        }
        val
    }

    pub fn write_crs(&mut self, crs: u8, val: u8) {
        if devlog_is_active(LogModule::Cmap) {
            dlog_dev!(LogModule::Cmap, "CMAP({}) Write CRS {} <- {:02x}", self.id, crs, val);
        }
        self.dirty = true;
        match crs {
            CMAP_REG_ADDR_LO => {
                self.addr_lo = val;
                self.rgb_counter = 0;
            }
            CMAP_REG_ADDR_HI => {
                self.addr_hi = val;
                self.rgb_counter = 0;
            }
            CMAP_REG_PALETTE => {
                match self.rgb_counter {
                    0 => { // Red
                        self.red_temp = val;
                        self.rgb_counter = 1;
                    }
                    1 => { // Green
                        self.green_temp = val;
                        self.rgb_counter = 2;
                    }
                    2 => { // Blue
                        let addr = self.get_address();
                        // 0x00BBGGRR
                        let color = (val as u32) << 16 | (self.green_temp as u32) << 8 | (self.red_temp as u32);
                        if devlog_is_active(LogModule::Cmap) {
                            dlog_dev!(LogModule::Cmap, "CMAP({}) Palette[{:04x}] = {:06x}", self.id, addr, color);
                        }
                        if addr < self.palette.len() {
                            self.palette[addr] = color;
                        }
                        self.rgb_counter = 0;
                        self.inc_address();
                    }
                    _ => {}
                }
            }
            CMAP_REG_COMMAND => self.command = val,
            CMAP_REG_READ_INIT => {
                self.rgb_counter = 0;
            }
            _ => {}
        }
    }

    pub fn print_status(&self, label: &str, writer: &mut dyn std::io::Write) {
        writeln!(writer, "=== {} ===", label).unwrap();
        writeln!(writer, "  addr={:04x} addr_lo={:02x} addr_hi={:02x}", self.get_address(), self.addr_lo, self.addr_hi).unwrap();
        writeln!(writer, "  rgb_counter={} red_temp={:02x} green_temp={:02x}", self.rgb_counter, self.red_temp, self.green_temp).unwrap();
        writeln!(writer, "  command={:02x} revision={:02x}", self.command, self.revision).unwrap();
        writeln!(writer, "  Palette (non-zero rows):").unwrap();
        for row in 0..(self.palette.len() / 16) {
            let base = row * 16;
            if self.palette[base..base+16].iter().all(|&v| v == 0) { continue; }
            let mut line = format!("  {:04x}:", base);
            for i in 0..16 {
                line.push_str(&format!(" {:06x}", self.palette[base + i] & 0xFFFFFF));
            }
            writeln!(writer, "{}", line).unwrap();
        }
    }
}

impl Default for Cmap {
    fn default() -> Self {
        Self::new(0)
    }
}