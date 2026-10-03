//! GE11 geometry engines as their diagnostic ports see them. The engines do
//! not run; they only hold downloaded microcode, read it back, and answer
//! "started".

use super::hq3::host;
use super::plain::Ring;

/// Microcode lines an engine holds (ge11.bin is about 79K lines).
const UCODE_LINES: usize = 0x2_0000;

/// One engine's diagnostic state.
#[repr(C)]
struct GeDiag {
    /// Current diagnostic address: a microcode line (from `UCODE_BASE`) or an
    /// internal register.
    addr: u32,
    /// Which 32-bit word of the current 72-bit microcode line comes next.
    word: usize,
    ucode: [[u32; 3]; UCODE_LINES],
}

impl GeDiag {
    const UCODE_BASE: u32 = 0x20_0000;
    /// Internal register: execution control; bit 0 starts the engine.
    const EXEC_CONTROL: u32 = 0x4_0000;

    /// Microcode line at diagnostic address `addr`, if it is one.
    fn line(&self, addr: u32) -> Option<&[u32; 3]> {
        self.ucode.get(addr.wrapping_sub(Self::UCODE_BASE) as usize)
    }

    /// Words a readback starting at `addr` delivers, in order: for each pair
    /// of lines, word 0 and the top byte of the first, then word 1 of the
    /// second. A readback starting on an odd line is preceded by two words
    /// that carry nothing. (The driver's verifier reads exactly this; the
    /// order is inferred from it.)
    fn readback(&self, lines: u32) -> Vec<u32> {
        let w = |l: u32, i: usize| self.line(l).map(|x| x[i]).unwrap_or(0);
        let mut out = Vec::new();
        let mut l = self.addr;
        if l.wrapping_sub(Self::UCODE_BASE) & 1 == 1 {
            out.extend([0, 0]);
        }
        for _ in 0..lines / 2 {
            out.extend([w(l, 0), w(l, 2) & 0xFF, w(l + 1, 1)]);
            l += 2;
        }
        out
    }
}

/// Both engines' diagnostic ports and their shared readback queue. Plain
/// data, valid zeroed.
#[repr(C)]
pub struct Ge11 {
    ge: [GeDiag; 2],
    /// Pending diagnostic readback words, oldest first.
    pub out: Ring<{ 1 << 18 }>,
}

impl Ge11 {
    /// Engine `ge`'s current diagnostic address.
    pub fn diag_addr(&self, ge: usize) -> u32 {
        self.ge[ge & 1].addr
    }

    /// Engine `ge`'s microcode line `i` (from the start of microcode).
    pub fn ucode_line(&self, ge: usize, i: usize) -> Option<[u32; 3]> {
        self.ge[ge & 1].ucode.get(i).copied()
    }

    /// Whether `off` is one of the engines' diagnostic data or address ports.
    pub fn is_diag_port(off: u32) -> bool {
        host::GE_DIAG.iter().any(|&(d, a)| off == d || off == a)
    }

    /// A write to a diagnostic data or address port. True when an engine was
    /// started: its version program's answer (revision 1) is then waiting in
    /// the readback registers.
    pub fn write(&mut self, off: u32, val: u32) -> bool {
        let n = host::GE_DIAG.iter().position(|&(d, a)| off == d || off == a).unwrap();
        let (data_port, _) = host::GE_DIAG[n];
        let ge = &mut self.ge[n];
        if off != data_port {
            if val & 0x8000_0000 != 0 {
                // Read request for `val & 0x7FFF_FFFF` lines from `addr`; it
                // replaces anything a previous request left unread.
                let words = ge.readback(val & 0x7FFF_FFFF);
                self.out.clear();
                for w in words {
                    self.out.push(w);
                }
            } else {
                ge.addr = val;
                ge.word = 0;
            }
            return false;
        }
        if ge.addr >= GeDiag::UCODE_BASE {
            if let Some(line) = ge.ucode.get_mut((ge.addr - GeDiag::UCODE_BASE) as usize) {
                line[ge.word] = val;
            }
            ge.word += 1;
            if ge.word == 3 {
                ge.word = 0;
                ge.addr += 1;
            }
            false
        } else {
            ge.addr == GeDiag::EXEC_CONTROL && val & 1 != 0
        }
    }
}
