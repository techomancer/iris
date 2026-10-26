//! The Xbow crossbar, widget 0 of the Octane: the common widget block, the
//! crossbar's own registers, and a link block per port (8 to 15) whose
//! status says whether a widget is alive there. Everything else reads back
//! what was written.
//!
//! Offsets are within the widget's 16 MB window; registers are 32 bits
//! wide in the low word of each doubleword (`docs/Octane/ip30-address-map.md`).

use std::collections::HashMap;

use parking_lot::Mutex;

pub const WIDGET_ID: u32 = 0x04;

/// A widget ID register: JTAG IDCODE layout, bit 0 set.
pub const fn widget_id(rev: u32, part: u32, mfgr: u32) -> u32 {
    (rev << 28) | (part << 12) | (mfgr << 1) | 1
}

const XB_LINK_BASE: u32 = 0x100;
const XB_LINK_SIZE: u32 = 0x40;
const XB_LINK_STATUS: u32 = 0x14;
const XB_LINK_AUX_STATUS: u32 = 0x3c;
pub const XB_STAT_LINKALIVE: u32 = 0x8000_0000;
const XB_AUX_STAT_PRESENT: u32 = 0x20;

pub struct Xbow {
    present: [bool; 16],
    regs: Mutex<HashMap<u32, u32>>,
}

impl Xbow {
    /// `present` lists the widget numbers with something on their link.
    pub fn new(present: &[u32]) -> Self {
        let mut p = [false; 16];
        for &w in present {
            p[w as usize] = true;
        }
        Xbow { present: p, regs: Mutex::new(HashMap::new()) }
    }

    pub fn read32(&self, off: u32) -> u32 {
        if (XB_LINK_BASE..XB_LINK_BASE + 8 * XB_LINK_SIZE).contains(&off) {
            let port = 8 + (off - XB_LINK_BASE) / XB_LINK_SIZE;
            let reg = (off - XB_LINK_BASE) % XB_LINK_SIZE;
            let up = self.present[port as usize];
            match reg {
                XB_LINK_STATUS => return if up { XB_STAT_LINKALIVE } else { 0 },
                XB_LINK_AUX_STATUS => return if up { XB_AUX_STAT_PRESENT } else { 0 },
                _ => {}
            }
        }
        match off {
            WIDGET_ID => widget_id(3, 0x0, 0x0), // crossbow rev 1.2
            o => *self.regs.lock().get(&o).unwrap_or(&0),
        }
    }

    pub fn write32(&self, off: u32, val: u32) {
        self.regs.lock().insert(off, val);
    }

    pub fn power_on(&self) {
        self.regs.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xbow_reports_the_live_links() {
        let x = Xbow::new(&[8, 15]);
        let status = |port: u32| x.read32(0x100 + (port & 7) * 0x40 + 0x14);
        assert_eq!(status(8), XB_STAT_LINKALIVE);
        assert_eq!(status(0xf), XB_STAT_LINKALIVE);
        assert_eq!(status(0xc), 0);
        assert_eq!((x.read32(WIDGET_ID) >> 12) & 0xffff, 0);
    }
}
