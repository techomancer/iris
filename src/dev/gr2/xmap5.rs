//! XMAP5 multimode colour mapper — five instances, one per 5-pixel phase.
//!
//! Storage only: mode table (per DID), misc registers and the 8192-entry CLUT.
//! Writes to the broadcast port (`xmapall`) go to all five. The compositor
//! reads channel `x % 5`. Protocol and init values: `ignore/gr2/XMAP5.h`.

pub const XMAP_MISC: u32 = 0x00;
pub const XMAP_MODE: u32 = 0x04;
pub const XMAP_CLUT: u32 = 0x08;
pub const XMAP_CRC: u32 = 0x0c;
pub const XMAP_ADDRLO: u32 = 0x10;
pub const XMAP_ADDRHI: u32 = 0x14;
pub const XMAP_BYTECNT: u32 = 0x18;
pub const XMAP_FIFOSTATUS: u32 = 0x1c;

pub const CLUT_SIZE: usize = 8192;

/// `fifostatus` for an idle XMAP: bit 0 clear (not busy), bit 1 set (room).
/// Software waits while bit 0 is set and until bit 1 is set.
pub const FIFOSTATUS_IDLE: u8 = 0x02;

/// One XMAP5. Plain data; all-zero is a valid power-on state.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Xmap5 {
    /// 13-bit address pointer: CLUT index, DID (mode) or misc register.
    pub addr: u16,
    /// Next CLUT component to receive (0 = R, 1 = G, 2 = B).
    pub clut_phase: u8,
    /// Last value written to fifostatus (flat-panel output enables).
    pub fifostatus_w: u8,
    pub misc: [u8; 8],
    /// 32-bit mode word per DID.
    pub mode: [u32; 32],
    /// Staging for the R and G bytes of the CLUT entry being written.
    pub clut_rg: [u8; 2],
    /// CLUT entries as 0x00RRGGBB.
    pub clut: [u32; CLUT_SIZE],
}

impl Xmap5 {
    pub fn write(&mut self, reg: u32, val: u32) {
        match reg {
            XMAP_MISC => {
                self.misc[(self.addr & 7) as usize] = val as u8;
                self.addr = (self.addr + 1) & 0x1fff;
            }
            // Mode table is byte-addressed: addrlo = DID * 4 (kernel
            // gr2_retrace.c swap). A 32-bit mode access moves 4 bytes; the
            // +4 advance is inferred (the kernel always sets the address).
            XMAP_MODE => {
                self.mode[((self.addr >> 2) & 31) as usize] = val;
                self.addr = (self.addr + 4) & 0x1fff;
            }
            XMAP_CLUT => {
                let b = val as u8;
                match self.clut_phase {
                    0 | 1 => {
                        self.clut_rg[self.clut_phase as usize] = b;
                        self.clut_phase += 1;
                    }
                    _ => {
                        self.clut[self.addr as usize] = ((self.clut_rg[0] as u32) << 16)
                            | ((self.clut_rg[1] as u32) << 8)
                            | b as u32;
                        self.clut_phase = 0;
                        self.addr = (self.addr + 1) & 0x1fff;
                    }
                }
            }
            XMAP_ADDRLO => {
                self.addr = (self.addr & 0x1f00) | (val as u16 & 0xff);
                self.clut_phase = 0;
            }
            XMAP_ADDRHI => {
                self.addr = ((val as u16 & 0x1f) << 8) | (self.addr & 0xff);
                self.clut_phase = 0;
            }
            XMAP_FIFOSTATUS => self.fifostatus_w = val as u8,
            _ => {}
        }
    }

    /// 32-bit store to the CLUT port: one whole entry, R in bits 31:24, G in
    /// 23:16, B in 15:8 (the low byte is ignored), then the address advances.
    /// This is how the Xsgi DDX loads colour maps; byte stores (kernel, PROM)
    /// go through `write` one component at a time.
    pub fn write_clut_packed(&mut self, val: u32) {
        self.clut[self.addr as usize] = val >> 8;
        self.clut_phase = 0;
        self.addr = (self.addr + 1) & 0x1fff;
    }

    pub fn read(&mut self, reg: u32) -> u32 {
        match reg {
            XMAP_MISC => {
                let v = self.misc[(self.addr & 7) as usize];
                self.addr = (self.addr + 1) & 0x1fff;
                v as u32
            }
            XMAP_MODE => {
                let v = self.mode[((self.addr >> 2) & 31) as usize];
                self.addr = (self.addr + 4) & 0x1fff;
                v
            }
            XMAP_CLUT => {
                let e = self.clut[self.addr as usize];
                let v = match self.clut_phase {
                    0 => e >> 16,
                    1 => e >> 8,
                    _ => e,
                } & 0xff;
                self.clut_phase += 1;
                if self.clut_phase == 3 {
                    self.clut_phase = 0;
                    self.addr = (self.addr + 1) & 0x1fff;
                }
                v
            }
            XMAP_ADDRLO => (self.addr & 0xff) as u32,
            XMAP_ADDRHI => (self.addr >> 8) as u32,
            XMAP_FIFOSTATUS => FIFOSTATUS_IDLE as u32,
            _ => 0,
        }
    }
}
