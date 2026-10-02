// XMAP9 (Cross-Map 9)
use crate::devlog::{LogModule, devlog_is_active};

pub const XMAP9_REG_CONFIG: u8 = 0x00;
pub const XMAP9_REG_REVISION: u8 = 0x01;
pub const XMAP9_REG_FIFO_AVAIL: u8 = 0x02;
pub const XMAP9_REG_CURSOR_CMAP_MSB: u8 = 0x03;
pub const XMAP9_REG_POPUP_CMAP_MSB: u8 = 0x04;
pub const XMAP9_REG_MODE_TABLE_WRITE: u8 = 0x05;
pub const XMAP9_REG_MODE_TABLE_READ: u8 = 0x05;
pub const XMAP9_REG_MODE_ADDR: u8 = 0x07;

pub struct Xmap9 {
    pub config: u8,
    pub cursor_cmap_msb: u8,
    pub popup_cmap_msb: u8,
    pub mode_addr: u8,
    pub mode_table: [u32; 32], // 24-bit entries
    pub fifo_avail: u8, // 3 bits
    pub dirty: bool,
}

impl Xmap9 {
    pub fn new() -> Self {
        Self {
            config: 0,
            cursor_cmap_msb: 0,
            popup_cmap_msb: 0,
            mode_addr: 0,
            mode_table: [0; 32],
            fifo_avail: 2, // 010 = 3 entries available (reset value)
            dirty: true,
        }
    }

    pub fn read_crs(&mut self, crs: u8) -> u8 {
        let val = match crs {
            XMAP9_REG_CONFIG => self.config,
            XMAP9_REG_REVISION => 3, // XL24 (Indy 24-bit): revision 3 (MAME: xmap_revision=3); XL8 would be 1
            XMAP9_REG_FIFO_AVAIL => self.fifo_avail,
            XMAP9_REG_CURSOR_CMAP_MSB => self.cursor_cmap_msb,
            XMAP9_REG_POPUP_CMAP_MSB => self.popup_cmap_msb,
            XMAP9_REG_MODE_TABLE_READ => {
                // Read from Mode Register Table based on CRS 7
                // Bits 6:2 = Entry Index
                let entry_idx = (self.mode_addr >> 2) & 0x1F;
                // Bits 1:0 = Byte Select
                let byte_sel = self.mode_addr & 0x3;
                let entry = self.mode_table[entry_idx as usize];
                match byte_sel {
                    0 => (entry & 0xFF) as u8,         // Byte 2 (7:0)
                    1 => ((entry >> 8) & 0xFF) as u8,  // Byte 1 (15:8)
                    2 => ((entry >> 16) & 0xFF) as u8, // Byte 0 (23:16)
                    _ => 0, // Reserved
                }
            }
            XMAP9_REG_MODE_ADDR => self.mode_addr,
            _ => 0,
        };
        dlog!(LogModule::Xmap, "XMAP9 Read  CRS {} -> {:02x}", crs, val);
        val
    }

    pub fn write_crs(&mut self, crs: u8, val: u32) {
        dlog!(LogModule::Xmap, "XMAP9 Write CRS {} <- {:06x}", crs, val);
        self.dirty = true;
        match crs {
            XMAP9_REG_CONFIG => self.config = val as u8,
            XMAP9_REG_CURSOR_CMAP_MSB => self.cursor_cmap_msb = val as u8,
            XMAP9_REG_POPUP_CMAP_MSB => self.popup_cmap_msb = val as u8,
            XMAP9_REG_MODE_TABLE_WRITE => {
                // 32-bit write: Address (8 bits) + Data (24 bits)
                // Byte 3 (MSB) is address, Bytes 2-0 are data (Big Endian)
                let addr = (val >> 24) & 0x1F; // 5 bits
                let data = val & 0xFFFFFF;
                self.mode_table[addr as usize] = data;
            }
            XMAP9_REG_MODE_ADDR => self.mode_addr = val as u8,
            _ => {}
        }
    }

    pub fn print_status(&self, label: &str, writer: &mut dyn std::io::Write) {
        writeln!(writer, "=== {} ===", label).unwrap();
        writeln!(writer, "  config={:02x} cursor_cmap_msb={:02x} popup_cmap_msb={:02x} mode_addr={:02x} fifo_avail={:02x}",
            self.config, self.cursor_cmap_msb, self.popup_cmap_msb, self.mode_addr, self.fifo_avail).unwrap();
        writeln!(writer, "  Mode Table (32 entries):").unwrap();
        writeln!(writer, "  idx  raw      pix_mode pix_size msb_cmap cmap_base  aux_mode aux_msb aux_base  buf ovl").unwrap();
        for (i, &raw) in self.mode_table.iter().enumerate() {
            let buf      = (raw >> 0)  & 0x1;         // bit 0
            let ovl      = (raw >> 1)  & 0x1;         // bit 1
            let msb      = ((raw >> 3) & 0x1F) as u8; // bits 7:3
            let pix_mode = (raw >> 8)  & 0x3;         // bits 9:8
            let pix_size = (raw >> 10) & 0x3;         // bits 11:10
            let aux_pix  = (raw >> 16) & 0x7;         // bits 18:16
            let aux_msb  = ((raw >> 19) & 0x1F) as u8; // bits 23:19
            let pix_mode_name = match pix_mode { 0=>"CI", 1=>"RGB0", 2=>"RGB1", 3=>"RGB2", _=>"?" };
            let pix_size_name = match pix_size { 0=>"4bpp", 1=>"8bpp", 2=>"12bpp", 3=>"24bpp", _=>"?" };
            let cmap_base = if pix_mode == 0 {
                if pix_size == 2 { (msb as usize & 0x10) << 8 } else { (msb as usize) << 8 }
            } else { 0 };
            let aux_base = (aux_msb as usize) << 8;
            writeln!(writer,
                "  {:2}   {:06x}   {:<5}    {:<6}   {:02x}       {:04x}       {:x}        {:02x}     {:04x}      {} {}",
                i, raw, pix_mode_name, pix_size_name, msb, cmap_base,
                aux_pix, aux_msb, aux_base, buf, ovl).unwrap();
        }
    }
}

impl Default for Xmap9 {

    fn default() -> Self {
        Self::new()
    }
}
