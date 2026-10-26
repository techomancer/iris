//! HEART, the Octane's system controller: the memory configuration, the
//! interrupt controller, the 52-bit counter and compare, the MicroLAN
//! master, and its widget-side registers as widget 8 on the Xbow.
//!
//! Two register blocks: the processor-side one at physical 0x0ff00000
//! (`piu_read`/`piu_write`, offsets within it) and the widget-side one in
//! widget 8's window (`widget_read`/`widget_write`). Both are 64 bits wide
//! at 8-byte spacing; MEMCFG is read as 32-bit halves. The map is in
//! `docs/Octane/ip30-address-map.md`.
//!
//! Interrupts: 64 ISR bits in five levels, each level an OR into one of
//! the CPU's hardware interrupt pins when any of its bits is set in ISR
//! and processor 0's mask. The levels are written straight into the CPU's
//! interrupt word (`MipsCore::hot.interrupts`, bits 10..14).
//!
//! Registers without a model read back what was written to them.

use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::Mutex;

use crate::dev::sgi_nic::OneWire;
use crate::dev::xbow::widget_id;

// Processor-side (PIU) register offsets.
const H_MODE: u32 = 0x00000;
const H_MEMCFG0: u32 = 0x00020; // ..0x38: two 32-bit bank configs each
const H_STATUS: u32 = 0x00080;
const H_MLAN_CTL: u32 = 0x000b8;
const H_IMR0: u32 = 0x10000; // ..0x10018, one per CPU
const H_SET_ISR: u32 = 0x10020;
const H_CLR_ISR: u32 = 0x10028;
const H_ISR: u32 = 0x10030;
const H_IMSR: u32 = 0x10038;
const H_CAUSE: u32 = 0x10040;
const H_COUNT: u32 = 0x20000;
const H_COMPARE: u32 = 0x30000;
const H_PRID: u32 = 0x50000;

/// Widget-side: an XIO write here sets ISR bits (the interrupt target).
const HW_ISR: u32 = 0x80;

pub const HEART_WIDGET: u32 = 8;

const HEART_STAT_BIG_ENDIAN: u64 = 1 << 8;
const HEART_STAT_TRITON: u64 = 1 << 10;
const HEART_STAT_PROC_ACTIVE0: u64 = 1 << 4;
/// Vector 50: count has reached compare.
pub const HEART_INT_TIMER: u64 = 1 << 50;
/// HEART's counter: the 400 MHz XIO clock / 32.
pub const HEART_COUNT_HZ: u64 = 12_500_000;
const COUNT_MASK: u64 = (1 << 52) - 1;

/// HEART's five interrupt levels and the Cause.IP bit each one drives:
/// hardware interrupts 0..4 (Cause bits 10..14). Bit 15 is the R10000's
/// own Count/Compare.
const HEART_LEVELS: [(u64, u64); 5] = [
    (0x0000_0000_0000_ffff, 1 << 10),
    (0x0000_0000_ffff_0000, 1 << 11),
    (0x0003_ffff_0000_0000, 1 << 12),
    (0x0004_0000_0000_0000, 1 << 13),
    (0xfff8_0000_0000_0000, 1 << 14),
];
const HEART_IP_MASK: u64 = 0x1f << 10;

/// HEART_MEMCFG: valid, size-1 and base, in 32 MB units from RAM's start.
pub const MEMCFG_VALID: u32 = 0x8000_0000;
pub const MEMCFG_SIZE_SHIFT: u32 = 16;
pub const MEMCFG_SIZE_MASK: u32 = 0x003f_0000;
pub const MEMCFG_ADDR_MASK: u32 = 0x0000_01ff;
pub const MEMCFG_UNIT_SHIFT: u32 = 25;

/// A bank of `mb` megabytes at `base` bytes from RAM's start, in HEART's
/// MEMCFG encoding; 0 for a bank that is not there.
pub fn encode_memcfg(base: u64, mb: u32) -> u32 {
    if mb == 0 {
        return 0;
    }
    let units = ((mb as u64) << 20) >> MEMCFG_UNIT_SHIFT;
    MEMCFG_VALID
        | (((units.max(1) - 1) as u32) << MEMCFG_SIZE_SHIFT) & MEMCFG_SIZE_MASK
        | ((base >> MEMCFG_UNIT_SHIFT) as u32 & MEMCFG_ADDR_MASK)
}

struct HeartState {
    mode: u64,
    memcfg: [u32; 8],
    imr: [u64; 4],
    isr: u64,
    compare: u64,
    cause: u64,
    mlan: OneWire,
    other: HashMap<u32, u64>,
    widget: HashMap<u32, u64>,
}

pub struct Heart {
    s: Mutex<HeartState>,
    start: Instant,
    /// The CPU's interrupt word (`MipsCore::hot.interrupts`).
    cpu_irq: AtomicPtr<AtomicU64>,
    reset_memcfg: [u32; 8],
}

impl Heart {
    /// `banks_mb` is what is fitted; MEMCFG starts with the banks back to
    /// back from RAM's start, as the PROM would leave them.
    pub fn new(banks_mb: [u32; 4]) -> Self {
        let mut memcfg = [0u32; 8];
        let mut base = 0u64;
        for (i, &mb) in banks_mb.iter().enumerate() {
            memcfg[i] = encode_memcfg(base, mb);
            base += (mb as u64) << 20;
        }
        Heart {
            s: Mutex::new(HeartState {
                mode: 0,
                memcfg,
                imr: [0; 4],
                isr: 0,
                compare: u64::MAX,
                cause: 0,
                mlan: OneWire::default(),
                other: HashMap::new(),
                widget: HashMap::new(),
            }),
            start: Instant::now(),
            cpu_irq: AtomicPtr::new(std::ptr::null_mut()),
            reset_memcfg: memcfg,
        }
    }

    /// Where HEART's interrupt levels go: the CPU's interrupt word.
    pub fn set_interrupts(&self, p: *const AtomicU64) {
        self.cpu_irq.store(p as *mut AtomicU64, Ordering::Release);
    }

    pub fn power_on(&self) {
        let mut s = self.s.lock();
        s.mode = 0;
        s.memcfg = self.reset_memcfg;
        s.imr = [0; 4];
        s.isr = 0;
        s.compare = u64::MAX;
        s.cause = 0;
        s.other.clear();
        s.widget.clear();
        self.update_irq(&s);
    }

    pub fn count(&self) -> u64 {
        let ns = self.start.elapsed().as_nanos() as u64;
        (ns / (1_000_000_000 / HEART_COUNT_HZ)) & COUNT_MASK
    }

    /// The four banks' MEMCFG words as they stand.
    pub fn memcfg(&self) -> [u32; 4] {
        let s = self.s.lock();
        [s.memcfg[0], s.memcfg[1], s.memcfg[2], s.memcfg[3]]
    }

    /// Set ISR bit `vec` (a widget's interrupt packet, or a device wired
    /// straight to HEART), or clear it.
    pub fn set_vector(&self, vec: u32, level: bool) {
        let mut s = self.s.lock();
        let bit = 1u64 << (vec & 63);
        let isr = if level { s.isr | bit } else { s.isr & !bit };
        if isr != s.isr {
            s.isr = isr;
            self.update_irq(&s);
        }
    }

    /// Recompute the CPU's hardware interrupt pins from ISR and CPU 0's mask.
    fn update_irq(&self, s: &HeartState) {
        let p = self.cpu_irq.load(Ordering::Acquire);
        if p.is_null() {
            return;
        }
        let pending = s.isr & s.imr[0];
        let mut ip = 0;
        for (mask, bit) in HEART_LEVELS {
            if pending & mask != 0 {
                ip |= bit;
            }
        }
        let word = unsafe { &*p };
        word.fetch_and(!(HEART_IP_MASK & !ip), Ordering::AcqRel);
        if ip != 0 {
            word.fetch_or(ip, Ordering::AcqRel);
        }
    }

    /// Count has passed compare: raise the timer vector. Called from the
    /// machine's service thread.
    pub fn poll_timer(&self) {
        let now = self.count();
        let mut s = self.s.lock();
        if now >= s.compare && s.isr & HEART_INT_TIMER == 0 {
            s.isr |= HEART_INT_TIMER;
            self.update_irq(&s);
        }
    }

    pub fn piu_read(&self, off: u32) -> u64 {
        let s = self.s.lock();
        match off {
            H_MODE => s.mode,
            o if (H_MEMCFG0..H_MEMCFG0 + 0x20).contains(&o) => {
                let i = ((o - H_MEMCFG0) / 8) as usize * 2;
                ((s.memcfg[i] as u64) << 32) | s.memcfg[i + 1] as u64
            }
            H_STATUS => HEART_STAT_BIG_ENDIAN | HEART_STAT_TRITON | HEART_STAT_PROC_ACTIVE0 | HEART_WIDGET as u64,
            H_MLAN_CTL => s.mlan.read() as u64,
            o if (H_IMR0..H_IMR0 + 0x20).contains(&o) => s.imr[((o - H_IMR0) / 8) as usize],
            H_ISR => s.isr,
            H_IMSR => s.isr & s.imr[0],
            H_CAUSE => s.cause,
            H_COUNT => {
                drop(s);
                self.count()
            }
            H_COMPARE => s.compare,
            H_PRID => 0,
            o => *s.other.get(&o).unwrap_or(&0),
        }
    }

    /// Returns true when MEMCFG changed, so the bus can place the banks.
    pub fn piu_write(&self, off: u32, val: u64) -> bool {
        let mut s = self.s.lock();
        match off {
            H_MODE => s.mode = val,
            o if (H_MEMCFG0..H_MEMCFG0 + 0x20).contains(&o) => {
                let i = ((o - H_MEMCFG0) / 8) as usize * 2;
                s.memcfg[i] = (val >> 32) as u32;
                s.memcfg[i + 1] = val as u32;
                return true;
            }
            H_MLAN_CTL => s.mlan.write(val as u32),
            o if (H_IMR0..H_IMR0 + 0x20).contains(&o) => s.imr[((o - H_IMR0) / 8) as usize] = val,
            H_SET_ISR => s.isr |= val,
            H_CLR_ISR => s.isr &= !val,
            H_CAUSE => s.cause &= !val,
            H_COMPARE => {
                s.compare = val & COUNT_MASK;
                s.isr &= !HEART_INT_TIMER;
            }
            H_COUNT => {}
            o => {
                s.other.insert(o, val);
            }
        }
        self.update_irq(&s);
        false
    }

    pub fn widget_read(&self, off: u32) -> u64 {
        match off & !7 {
            0x00 => widget_id(4, 0xc001, 0x036) as u64, // HEART rev D
            HW_ISR => self.s.lock().isr,
            o => *self.s.lock().widget.get(&o).unwrap_or(&0),
        }
    }

    pub fn widget_write(&self, off: u32, val: u64) {
        match off & !7 {
            // An interrupt packet from a widget: its vector.
            HW_ISR => self.set_vector(val as u32, true),
            o => {
                self.s.lock().widget.insert(o, val);
            }
        }
    }

    /// Whether the guest has enabled HEART's interrupts at all (mode bit 13).
    pub fn interrupts_enabled(&self) -> bool {
        self.s.lock().mode & (1 << 13) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memcfg_encodes_the_fitted_banks() {
        let h = Heart::new([1024, 256, 0, 0]);
        let m = h.memcfg();
        assert_eq!(m[0], 0x801f_0000); // 32 units at 0
        assert_eq!(m[1], 0x8007_0020); // 8 units at 32
        assert_eq!(m[2], 0);
        assert_eq!(h.piu_read(H_MEMCFG0) >> 32, 0x801f_0000);
    }

    #[test]
    fn heart_count_runs_at_12_5_mhz() {
        let h = Heart::new([256, 0, 0, 0]);
        let a = h.piu_read(H_COUNT);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let b = h.piu_read(H_COUNT);
        let ticks = b - a;
        assert!((200_000..400_000).contains(&ticks), "{ticks} ticks in 20 ms");
    }

    #[test]
    fn heart_levels_drive_the_cause_bits() {
        let h = Heart::new([256, 0, 0, 0]);
        let word = AtomicU64::new(1 << 15); // IP7, the CPU's own: untouched
        h.set_interrupts(&word);
        h.piu_write(H_IMR0, 1 << 50 | 1 << 3); // IMR0: timer, vector 3
        h.piu_write(H_SET_ISR, 1 << 3); // vector 3: level 0
        assert_eq!(word.load(Ordering::SeqCst), 1 << 15 | 1 << 10);
        h.widget_write(HW_ISR, 50); // a widget raising the timer vector: level 3
        assert_eq!(word.load(Ordering::SeqCst), 1 << 15 | 1 << 10 | 1 << 13);
        h.piu_write(H_CLR_ISR, 1 << 3 | 1 << 50);
        assert_eq!(word.load(Ordering::SeqCst), 1 << 15);
    }

    #[test]
    fn a_memcfg_write_reports_itself() {
        let h = Heart::new([256, 0, 0, 0]);
        assert!(h.piu_write(H_MEMCFG0, (0x819f_0040u64) << 32));
        assert!(!h.piu_write(H_MODE, 1));
        assert_eq!(h.memcfg()[0], 0x819f_0040);
    }
}
