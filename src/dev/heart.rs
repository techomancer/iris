//! HEART, the Octane's processor and memory controller: the bank
//! descriptors that place SDRAM, the 64 interrupt bits that drive the
//! R10000's interrupt pins, a 52-bit counter, a MicroLAN master for the
//! processor module's ID part, and the XIO widget (widget 8) the Crossbow's
//! other widgets reach memory and interrupts through.
//!
//! It is a functional model rather than a register-exact HEART: accesses it
//! does not support are recorded (see [`Heart::unsupported`]) and the guest
//! sees a fixed answer. Each approximation is said where it lives.
//! Register offsets follow OpenBSD's `xheartreg.h` and Linux's
//! `asm/sgi/heart.h`.
//!
//! The processor sees HEART's registers at [`PIU_BASE`] and its widget
//! registers at the start of widget 8's small window; `Ip30Bus` maps
//! both to this device, which takes the full physical address. Memory is
//! `heart_mem::HeartRam`, which a descriptor write reaches through
//! [`Heart::on_memcfg`].
//!
//! Ported from hw/misc/sgi-heart.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::cpu::smp::{current_cpu, ParkSlot, HOLD_HEART, MAX_CPUS};
use crate::dev::w1;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_ERR, BUS_OK};

/// HEART's processor-interface registers.
pub const PIU_BASE: u64 = 0x0ff0_0000;
pub const PIU_SIZE: u64 = 0x0006_0008;
/// HEART's own widget registers, at the start of widget 8's small window.
pub const WIDGET_BASE: u64 = 0x1800_0000;
pub const WIDGET_REG_SIZE: u64 = 0x1000;
pub const BANKS: usize = 4;

const COUNT_MASK: u64 = (1 << 52) - 1;
/// The counter ticks at 12.5 MHz.
const COUNT_NS: u64 = 80;

const LEVEL_MASKS: [u64; 5] = [
    0x0000_0000_0000_ffff,
    0x0000_0000_ffff_0000,
    0x0003_ffff_0000_0000,
    0x0004_0000_0000_0000,
    0xfff8_0000_0000_0000,
];
/// HEART's levels 0 to 4 are the CPU's IP2 to IP6: Cause bits 10 to 14.
const CAUSE_SHIFT: u32 = 10;

/// Mode register bits the model looks at.
const MODE_INTERRUPT_ENABLE: u64 = 1 << 13;
/// Software reset, of the whole machine.
const MODE_SOFT_RESET: u64 = 1 << 22;
/// Processor n stops being served while bit `60 + n` is set, and continues
/// where it stopped when it clears.
const MODE_PROC_DISABLE_SHIFT: u32 = 60;
const MODE_ECC_GENERATE: u64 = 1 << 19;
const MODE_ECC_DOUBLE: u64 = 1 << 20;
const MODE_ECC_FORCE_WRITE: u64 = 1 << 21;
/// The memory-error interrupt bit, which only reaches the CPU while the
/// mode register enables it.
const ECC_PENDING: u64 = 1 << 63;

const ONEWIRE_DONE: u64 = 2;
const ONEWIRE_LEVEL: u64 = 1;

/// HEART's registers. Everything is behind one lock: no register access
/// calls out except to drive the CPU's interrupt word, a leaf atomic.
struct Regs {
    mode: u64,
    sdram_mode: u64,
    mem_refresh: u64,
    mem_req_arb: u64,
    flow_cell: [u64; 2],
    clock_divider: u64,
    compare: u32,
    descriptor: [u32; 8],
    mask: [u64; 4],
    pending: u64,
    ecc_cause: u64,
    ecc_address: u64,
    ecc_data: u64,
    /// The counter is `saved_count` plus the ticks since `epoch_ns`, or
    /// just `saved_count` while the machine is stopped.
    saved_count: u64,
    epoch_ns: u64,
    /// When the machine was stopped, if it is: the counter holds still, as
    /// the CPU's Count does, so a stop costs the guest no time.
    paused_at: Option<u64>,
    widget_control: u32,
    widget_request_timeout: u32,
    widget_error_mask: u32,
    onewire_control: u64,
    onewire_deadline_ns: u64,
    onewire_level: bool,
    onewire_busy: bool,
    /// The levels last driven onto each processor's IP2..IP6.
    out: [u8; MAX_CPUS],
}

impl Regs {
    fn new(now: u64) -> Self {
        Self {
            mode: 0,
            sdram_mode: 0,
            mem_refresh: 0,
            mem_req_arb: 0,
            flow_cell: [0; 2],
            clock_divider: 0,
            compare: 0,
            descriptor: [0; 8],
            mask: [0; 4],
            pending: 0,
            ecc_cause: 0,
            ecc_address: 0,
            ecc_data: 0,
            saved_count: 0,
            epoch_ns: now,
            paused_at: None,
            widget_control: 0,
            widget_request_timeout: 0,
            widget_error_mask: 0,
            onewire_control: 0,
            onewire_deadline_ns: 0,
            onewire_level: false,
            onewire_busy: false,
            out: [0; MAX_CPUS],
        }
    }

    fn counter(&self, now: u64) -> u64 {
        let now = self.paused_at.unwrap_or(now);
        (self.saved_count + now.saturating_sub(self.epoch_ns) / COUNT_NS) & COUNT_MASK
    }

    /// What processor `cpu`'s IP2..IP6 should be: its mask over the pending
    /// bits, one output per level. An inter-processor interrupt needs nothing
    /// more: the sender sets the destination's bit, which only its mask has.
    fn levels(&self, cpu: usize) -> u8 {
        let mut enabled = self.pending & self.mask[cpu];
        let mut out = 0u8;
        for (i, m) in LEVEL_MASKS.iter().enumerate().take(4) {
            if enabled & m != 0 {
                out |= 1 << i;
            }
        }
        if self.mode & MODE_INTERRUPT_ENABLE == 0 {
            enabled &= !ECC_PENDING;
        }
        if enabled & LEVEL_MASKS[4] != 0 {
            out |= 1 << 4;
        }
        out
    }

    fn banks(&self) -> [u32; BANKS] {
        [self.descriptor[0], self.descriptor[1], self.descriptor[2], self.descriptor[3]]
    }
}

/// What a machine fixes about its HEART.
#[derive(Clone, Debug)]
pub struct HeartConfig {
    /// Each bank's capacity in bytes, for the monitor.
    pub banks: [u64; BANKS],
    /// The widget identification word's revision field.
    pub widget_revision: u8,
    /// The whole identification word, when the machine pins it; 0 builds it
    /// from the revision.
    pub widget_id_64: u64,
    /// Processors fitted, from 0 up: the status register's active bits, and
    /// the outputs driven.
    pub cpus: u8,
}

impl HeartConfig {
    /// The Octane: HEART revision D (4) and its identification word.
    pub fn octane(banks: [u64; BANKS]) -> Self {
        Self { banks, widget_revision: 4, widget_id_64: 0x4c00_106d, cpus: 1 }
    }
}

/// Nanoseconds from some fixed point, for the counter and the MicroLAN slot
/// timing. The machine uses the host clock; tests step their own.
pub type Clock = Arc<dyn Fn() -> u64 + Send + Sync>;

pub fn host_clock() -> Clock {
    let start = std::time::Instant::now();
    Arc::new(move || start.elapsed().as_nanos() as u64)
}

pub struct Heart {
    regs: Mutex<Regs>,
    cfg: HeartConfig,
    clock: Clock,
    /// Each CPU's interrupt word (`MipsCore`'s), once the machine has it.
    cpu_irq: [AtomicPtr<AtomicU64>; MAX_CPUS],
    /// Where the MODE register's disable bits hold each CPU.
    cpu_park: Mutex<[Option<Arc<ParkSlot>>; MAX_CPUS]>,
    /// The guest asked for a software reset.
    reset_hook: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    /// Where descriptor writes go: the machine points it at
    /// `Ip30Bus::remap_banks`.
    memcfg_hook: Mutex<Option<Box<dyn Fn([u32; BANKS]) + Send + Sync>>>,
    /// The processor module's MicroLAN: the PM10 board record.
    mlan: Mutex<w1::Bus>,
    /// Accesses this model does not support, first 64 distinct, for the
    /// monitor: `(offset, size, write)`.
    unsupported: Mutex<Vec<(u32, u8, bool)>>,
    /// `unsupported` holds 64: a further access costs one relaxed load.
    unsupported_full: AtomicBool,
    /// Packets that reached HEART's interrupt register, for the monitor.
    pub packets: AtomicU64,
}

impl Heart {
    pub fn new(cfg: HeartConfig, clock: Clock) -> Self {
        let now = clock();
        Self {
            regs: Mutex::new(Regs::new(now)),
            cfg,
            clock,
            cpu_irq: std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())),
            cpu_park: Mutex::new(Default::default()),
            reset_hook: Mutex::new(None),
            memcfg_hook: Mutex::new(None),
            mlan: Mutex::new(w1::Bus::new()),
            unsupported: Mutex::new(Vec::new()),
            unsupported_full: AtomicBool::new(false),
            packets: AtomicU64::new(0),
        }
    }

    pub fn config(&self) -> &HeartConfig {
        &self.cfg
    }

    /// Where HEART's levels go: the CPU's interrupt word
    /// (`MipsCpu::interrupts_ptr`). HEART's five outputs, levels 0 to 4,
    /// are the CPU's IP2 to IP6 (Cause bits 10 to 14); HEART touches no
    /// other bit. The word must outlive this HEART.
    pub fn set_interrupts(&self, word: *const AtomicU64) {
        self.set_cpu_interrupts(0, word);
    }

    /// [`Self::set_interrupts`] for processor `cpu`.
    pub fn set_cpu_interrupts(&self, cpu: usize, word: *const AtomicU64) {
        let r = self.regs.lock();
        self.cpu_irq[cpu].store(word as *mut AtomicU64, Ordering::Release);
        // Publish the levels as they stand into the new word.
        self.drive(cpu, 0x1f, r.out[cpu]);
    }

    /// Where the MODE register's disable bits hold processor `cpu`.
    pub fn set_cpu_park(&self, cpu: usize, slot: Arc<ParkSlot>) {
        self.cpu_park.lock()[cpu] = Some(slot);
    }

    /// Call `f` when the guest sets the software-reset bit. On the writing
    /// CPU's thread, outside HEART's lock.
    pub fn on_soft_reset(&self, f: Box<dyn Fn() + Send + Sync>) {
        *self.reset_hook.lock() = Some(f);
    }

    /// Hold or release the CPUs whose disable bits changed from `old` to `new`.
    /// Under the register lock: the slots are leaves.
    fn apply_disables(&self, old: u64, new: u64) {
        let changed = (old ^ new) >> MODE_PROC_DISABLE_SHIFT;
        if changed == 0 {
            return;
        }
        let park = self.cpu_park.lock();
        for (cpu, slot) in park.iter().enumerate() {
            let Some(slot) = slot else { continue };
            if changed & (1 << cpu) != 0 {
                if new >> (MODE_PROC_DISABLE_SHIFT + cpu as u32) & 1 != 0 {
                    slot.hold(HOLD_HEART);
                } else {
                    slot.release(HOLD_HEART);
                }
            }
        }
    }

    /// The status register: which processors are fitted, and HEART's widget id.
    fn status(&self) -> u64 {
        let active = (1u64 << self.cfg.cpus) - 1;
        (active << 4) | 8
    }

    /// Call `f` with the four bank descriptors whenever the guest (or a
    /// reset) changes one. Called outside HEART's lock, on the thread that
    /// wrote the register.
    pub fn on_memcfg(&self, f: Box<dyn Fn([u32; BANKS]) + Send + Sync>) {
        *self.memcfg_hook.lock() = Some(f);
    }

    /// The bank descriptors as they stand.
    pub fn memcfg(&self) -> [u32; BANKS] {
        self.regs.lock().banks()
    }

    /// Put the processor module's MicroLAN on HEART's master.
    pub fn set_microlan(&self, bus: w1::Bus) {
        *self.mlan.lock() = bus;
    }

    pub fn unsupported(&self) -> Vec<(u32, u8, bool)> {
        self.unsupported.lock().clone()
    }

    pub fn register_locks(self: &Arc<Self>) {
        crate::locks::register_mutex_fn("heart::regs", self, |h| h.regs.is_locked());
        crate::locks::register_mutex_fn("heart::mlan", self, |h| h.mlan.is_locked());
    }

    fn note(&self, off: u32, size: usize, write: bool) {
        if self.unsupported_full.load(Ordering::Relaxed) {
            return;
        }
        let mut u = self.unsupported.lock();
        if !u.iter().any(|&(o, s, w)| o == off && s as usize == size && w == write) {
            if trace() {
                eprintln!("ip30: HEART: unsupported {size}-byte {} at {off:#x}", if write { "write" } else { "read" });
            }
            u.push((off, size as u8, write));
            if u.len() == 64 {
                self.unsupported_full.store(true, Ordering::Relaxed);
            }
        }
    }

    fn memcfg_changed(&self, banks: [u32; BANKS]) {
        if let Some(f) = self.memcfg_hook.lock().as_ref() {
            f(banks);
        }
    }

    /// Set the outputs in `changed` to `levels` in the CPU's word, as the
    /// IOC does on the Indy. Under the register lock: the word is a leaf.
    fn drive(&self, cpu: usize, changed: u8, levels: u8) {
        let p = self.cpu_irq[cpu].load(Ordering::Acquire);
        if p.is_null() || changed == 0 {
            return;
        }
        // SAFETY: `set_interrupts`'s contract: the word outlives HEART.
        let word = unsafe { &*p };
        let set = ((changed & levels) as u64) << CAUSE_SHIFT;
        let clear = ((changed & !levels) as u64) << CAUSE_SHIFT;
        if set != 0 {
            word.fetch_or(set, Ordering::SeqCst);
            #[cfg(feature = "idle-pause")]
            crate::cpu::idle_park::wake();
        }
        if clear != 0 {
            word.fetch_and(!clear, Ordering::SeqCst);
        }
    }

    /// Drive the CPU's pins to what the registers now say.
    fn update_irq(&self, r: &mut Regs) {
        for cpu in 0..self.cfg.cpus as usize {
            let new = r.levels(cpu);
            if new != r.out[cpu] {
                let changed = new ^ r.out[cpu];
                r.out[cpu] = new;
                self.drive(cpu, changed, new);
            }
        }
    }

    /// The level IPn (n = 2..6) is at, for tests and the monitor.
    pub fn pin(&self, n: u32) -> bool {
        assert!((2..=6).contains(&n), "HEART drives IP2 to IP6, not IP{n}");
        self.regs.lock().out[0] & (1 << (n - 2)) != 0
    }

    pub fn pending(&self) -> u64 {
        self.regs.lock().pending
    }

    // ── MicroLAN ───────────────────────────────────────

    /// One bit slot, started by a write of the command register: the part
    /// answers at once (`w1::slot`), and the done bit follows after pulse +
    /// sample, counted as microseconds (the silicon's units are not known).
    fn onewire_start(&self, r: &mut Regs, command: u64) {
        let (pulse, sample) = w1::slot::fields(command);
        r.onewire_control = command & !(ONEWIRE_DONE | ONEWIRE_LEVEL);
        r.onewire_level = w1::slot::slot(&mut self.mlan.lock(), pulse, sample);
        r.onewire_busy = true;
        r.onewire_deadline_ns = (self.clock)() + (pulse + sample) as u64 * 1000;
    }

    fn onewire_refresh(&self, r: &mut Regs) {
        if r.onewire_busy && (self.clock)() >= r.onewire_deadline_ns {
            r.onewire_busy = false;
            r.onewire_control |= ONEWIRE_DONE;
            if r.onewire_level {
                r.onewire_control |= ONEWIRE_LEVEL;
            }
        }
    }

    // ── processor-interface registers ───────────────────

    /// A PIU register read of `size` bytes (4 or 8; HEART refuses narrower
    /// accesses, `None`). Unsupported offsets read as zero and are recorded.
    pub fn piu_read(&self, off: u32, size: usize) -> Option<u64> {
        if size != 4 && size != 8 {
            return None;
        }
        let mut r = self.regs.lock();
        if (0x20..0x40).contains(&off) {
            let n = ((off - 0x20) / 4) as usize;
            if size == 4 {
                return Some(r.descriptor[n] as u64);
            }
            if n & 1 == 0 {
                return Some(((r.descriptor[n] as u64) << 32) | r.descriptor[n + 1] as u64);
            }
        }
        if (off == 0xb8 && size == 8) || (off == 0xbc && size == 4) {
            self.onewire_refresh(&mut r);
            return Some(r.onewire_control);
        }
        if size == 8 {
            let v = match off {
                0x00 => Some(r.mode),
                0x08 => Some(r.sdram_mode),
                0x10 => Some(r.mem_refresh),
                0x18 => Some(r.mem_req_arb),
                0x50 => Some(r.flow_cell[0]),
                0x58 => Some(r.flow_cell[1]),
                0x80 => Some(self.status()),
                0x98 => Some(r.ecc_address),
                0xa0 => Some(r.ecc_data),
                0xb0 => Some(r.clock_divider),
                0x10000 | 0x10008 | 0x10010 | 0x10018 => Some(r.mask[((off - 0x10000) / 8) as usize]),
                0x10030 => Some(r.pending),
                // The reader's own mask: a CPU services what it enabled.
                0x10038 => Some(r.pending & r.mask[current_cpu()]),
                0x10040 => Some(r.ecc_cause),
                0x20000 => Some(r.counter((self.clock)())),
                0x30000 => Some(r.compare as u64),
                // Who is asking.
                0x50000 => Some(current_cpu() as u64),
                _ => None,
            };
            if let Some(v) = v {
                return Some(v);
            }
        }
        drop(r);
        self.note(off, size, false);
        Some(0)
    }

    /// A PIU register write; false is a refused width. Unsupported offsets
    /// are recorded and otherwise ignored.
    pub fn piu_write(&self, off: u32, size: usize, val: u64) -> bool {
        if size != 4 && size != 8 {
            return false;
        }
        let mut r = self.regs.lock();
        if (0x20..0x40).contains(&off) {
            let n = ((off - 0x20) / 4) as usize;
            if size == 4 || n & 1 == 0 {
                if size == 4 {
                    r.descriptor[n] = val as u32;
                } else {
                    r.descriptor[n] = (val >> 32) as u32;
                    r.descriptor[n + 1] = val as u32;
                }
                let banks = r.banks();
                drop(r);
                // On the CPU thread: the window may only be remapped there.
                self.memcfg_changed(banks);
                return true;
            }
        }
        if (off == 0xb8 && size == 8) || (off == 0xbc && size == 4) {
            self.onewire_start(&mut r, val);
            return true;
        }
        if size == 8 {
            match off {
                0x00 => {
                    // ECC generation and forced writes plant errors in
                    // memory, which this model does not do.
                    if val & (MODE_ECC_GENERATE | MODE_ECC_DOUBLE | MODE_ECC_FORCE_WRITE) != 0 {
                        if trace() {
                            eprintln!("ip30: HEART: mode {val:#x} asks for ECC error injection");
                        }
                        drop(r);
                        self.note(off, size, true);
                        r = self.regs.lock();
                    }
                    let old = std::mem::replace(&mut r.mode, val);
                    self.apply_disables(old, val);
                    self.update_irq(&mut r);
                    if val & !old & MODE_SOFT_RESET != 0 {
                        drop(r);
                        if let Some(f) = self.reset_hook.lock().as_ref() {
                            f();
                        }
                    }
                    return true;
                }
                0x08 => r.sdram_mode = val,
                0x10 => r.mem_refresh = val,
                0x18 => r.mem_req_arb = val,
                0x50 => r.flow_cell[0] = val,
                0x58 => r.flow_cell[1] = val,
                0xb0 => r.clock_divider = val,
                0x10040 => r.ecc_cause &= !val,
                // Stored, not compared: the model raises no timer vector.
                0x30000 => r.compare = (val & 0xff_ffff) as u32,
                0x10000 | 0x10008 | 0x10010 | 0x10018 => {
                    r.mask[((off - 0x10000) / 8) as usize] = val;
                    self.update_irq(&mut r);
                }
                0x10020 => {
                    r.pending |= val;
                    self.update_irq(&mut r);
                }
                0x10028 => {
                    r.pending &= !val;
                    self.update_irq(&mut r);
                }
                _ => {
                    drop(r);
                    self.note(off, size, true);
                }
            }
            return true;
        }
        drop(r);
        self.note(off, size, true);
        true
    }

    // ── widget registers ───────────────────────────────

    /// HEART's widget registers as the processor and other widgets see
    /// them, `None` for an access this model refuses.
    pub fn widget_read(&self, off: u32, size: usize) -> Option<u64> {
        let r = self.regs.lock();
        match (off, size) {
            (0x00, 8) | (0x04, 4) => Some(if self.cfg.widget_id_64 != 0 {
                if size == 4 {
                    self.cfg.widget_id_64 & 0xffff_ffff
                } else {
                    self.cfg.widget_id_64
                }
            } else {
                ((self.cfg.widget_revision as u64) << 28) | (0xc001 << 12) | (0x0036 << 1)
            }),
            (0x24, 4) => Some(r.widget_control as u64),
            (0x28, 8) | (0x2c, 4) => Some(r.widget_request_timeout as u64),
            (0x14 | 0x1c | 0x44, 4) => Some(0),
            (0x60 | 0x70 | 0x78 | 0x90, 8) => Some(0),
            (0x68, 8) => Some(r.widget_error_mask as u64),
            _ => {
                drop(r);
                self.note(0x1000_0000 | off, size, false);
                None
            }
        }
    }

    pub fn widget_write(&self, off: u32, size: usize, val: u64) -> bool {
        let mut r = self.regs.lock();
        match (off, size) {
            (0x24, 4) => r.widget_control = val as u32,
            (0x28, 8) | (0x2c, 4) => r.widget_request_timeout = val as u32,
            // The error-command word, taken only as a zero write. IRIX
            // 6.5.30's bus-error probe clears it with a 64-bit store to the
            // slot (OpenBSD's map puts the 64-bit slot at +0x40, the word in
            // its low half at +0x44); that is the same zero write.
            (0x44, 4) | (0x40, 8) if val == 0 => {}
            // Nothing here produces widget errors, so status stays zero.
            (0x60, 8) => {}
            (0x68, 8) => r.widget_error_mask = val as u32,
            _ => {
                drop(r);
                self.note(0x1000_0000 | off, size, true);
                return false;
            }
        }
        true
    }

    /// An interrupt packet: a doubleword write of `value` to HEART's widget
    /// offset 0x80, bits 7:0 the vector and bit 8 set or clear. Anything
    /// else there is refused.
    pub fn packet_write(&self, off: u64, size: usize, value: u64) -> bool {
        if off != 0x80 || size != 8 || value & !0x1ff != 0 || value & 0xff >= 64 {
            self.note(0x2000_0000 | off as u32, size, true);
            return false;
        }
        self.set_vector((value & 0xff) as u8, value & 0x100 != 0);
        true
    }

    fn set_vector(&self, vector: u8, set: bool) {
        self.packets.fetch_add(1, Ordering::Relaxed);
        let mut r = self.regs.lock();
        if set {
            r.pending |= 1 << vector;
        } else {
            r.pending &= !(1 << vector);
        }
        self.update_irq(&mut r);
    }

    /// The machine stopped: hold the counter (and any MicroLAN slot in
    /// progress) still until [`Heart::resume`].
    pub fn pause(&self) {
        let now = (self.clock)();
        let mut r = self.regs.lock();
        if r.paused_at.is_none() {
            r.paused_at = Some(now);
        }
    }

    /// The machine runs again: the counter continues from where it held,
    /// and a MicroLAN slot's deadline moves out by the time stopped.
    pub fn resume(&self) {
        let now = (self.clock)();
        let mut r = self.regs.lock();
        if let Some(at) = r.paused_at.take() {
            let stopped = now.saturating_sub(at);
            r.epoch_ns += stopped;
            r.onewire_deadline_ns += stopped;
        }
    }

    /// The counter, for the monitor.
    pub fn counter(&self) -> u64 {
        self.regs.lock().counter((self.clock)())
    }

    /// The register block `addr` (a full physical address) falls in: the
    /// processor interface (true) or the widget registers, with the offset.
    #[inline]
    fn block(addr: u64) -> Option<(bool, u32)> {
        if (PIU_BASE..PIU_BASE + PIU_SIZE).contains(&addr) {
            Some((true, (addr - PIU_BASE) as u32))
        } else if (WIDGET_BASE..WIDGET_BASE + WIDGET_REG_SIZE).contains(&addr) {
            Some((false, (addr - WIDGET_BASE) as u32))
        } else {
            None
        }
    }

    fn bus_read(&self, addr: u64, size: usize) -> Option<u64> {
        match Self::block(addr)? {
            (true, off) => self.piu_read(off, size),
            (false, off) => self.widget_read(off, size),
        }
    }

    fn bus_write(&self, addr: u64, size: usize, val: u64) -> u32 {
        let ok = match Self::block(addr) {
            Some((true, off)) => self.piu_write(off, size, val),
            Some((false, off)) => self.widget_write(off, size, val),
            None => false,
        };
        if ok {
            BUS_OK
        } else {
            BUS_ERR
        }
    }
}

/// HEART as the target of interrupts: line `n` is ISR bit `n`, the vector
/// (0 to 63) a widget's interrupt packet carries or a device wired to HEART
/// raises. A true level is a set packet, false the matching clear. ISR is
/// not an OR of its inputs: two sources on one vector share the bit, as on
/// the hardware. A line beyond 63 cannot be carried by a packet and is
/// dropped. Safe from any thread; it drives the CPU's interrupt word, a
/// leaf, under HEART's own lock, and calls nothing else.
impl crate::irq::InterruptController for Heart {
    fn set_line(&self, line: u32, level: bool) {
        if line < 64 {
            self.set_vector(line as u8, level);
        } else {
            self.note(0x2000_0080, 8, true);
        }
    }
}

/// The two register blocks, at full physical addresses: the processor
/// interface at [`PIU_BASE`] and the widget registers at [`WIDGET_BASE`].
/// Anything else in the slots `Ip30Bus` gives HEART is a bus error.
impl BusDevice for Heart {
    fn read8(&self, addr: u64) -> BusRead8 {
        self.bus_read(addr, 1).map_or(BusRead8::err(), |v| BusRead8::ok(v as u8))
    }
    fn read16(&self, addr: u64) -> BusRead16 {
        self.bus_read(addr, 2).map_or(BusRead16::err(), |v| BusRead16::ok(v as u16))
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        self.bus_read(addr, 4).map_or(BusRead32::err(), |v| BusRead32::ok(v as u32))
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        self.bus_read(addr, 8).map_or(BusRead64::err(), BusRead64::ok)
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.bus_write(addr, 1, val as u64)
    }
    fn write16(&self, addr: u64, val: u16) -> u32 {
        self.bus_write(addr, 2, val as u64)
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        self.bus_write(addr, 4, val as u64)
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.bus_write(addr, 8, val)
    }
}

/// Reset: every register to zero, the counter from zero, the MicroLAN
/// master idle. Memory keeps its contents; with no descriptor valid none of
/// it is reachable until the PROM (or the machine) sets them again.
impl crate::traits::Resettable for Heart {
    fn power_on(&self) {
        let now = (self.clock)();
        let mut r = self.regs.lock();
        // A reset with the machine stopped leaves the counter held at zero.
        let paused_at = r.paused_at.map(|_| now);
        let out = r.out;
        let old_mode = r.mode;
        *r = Regs::new(now);
        r.paused_at = paused_at;
        r.out = out;
        // Every processor is served again.
        self.apply_disables(old_mode, 0);
        self.update_irq(&mut r);
        let banks = r.banks();
        drop(r);
        self.memcfg_changed(banks);
    }
}

fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("IRIS_IP30_TRACE").is_some())
}

/// HEART has no thread. Stopping it holds its counter, as a stopped
/// machine's CPU Count holds; starting it lets the counter run on.
impl crate::traits::Device for Heart {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {
        self.pause();
    }
    fn start(&self) {
        self.resume();
    }
    fn is_running(&self) -> bool {
        self.regs.lock().paused_at.is_none()
    }
    fn get_clock(&self) -> u64 {
        0
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("heart".into(), "HEART: heart [status|unsupported]".into())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "heart" {
            return Err("Command not found".into());
        }
        let out = match args.first().copied().unwrap_or("status") {
            "status" => {
                let r = self.regs.lock();
                let mut s = format!(
                    "mode {:#x} pending {:#018x} mask0 {:#018x} out IP2-6 {:05b} counter {:#x} packets {}\n",
                    r.mode,
                    r.pending,
                    r.mask[0],
                    r.out[0],
                    r.counter((self.clock)()),
                    self.packets.load(Ordering::Relaxed)
                );
                let park = self.cpu_park.lock();
                for cpu in 1..self.cfg.cpus as usize {
                    s.push_str(&format!("cpu {cpu}: mask {:#018x} out IP2-6 {:05b}\n", r.mask[cpu], r.out[cpu]));
                }
                if self.cfg.cpus > 1 {
                    for (cpu, slot) in park.iter().enumerate().take(self.cfg.cpus as usize) {
                        if let Some(slot) = slot {
                            let disabled = r.mode >> (MODE_PROC_DISABLE_SHIFT + cpu as u32) & 1 != 0;
                            s.push_str(&format!("cpu {cpu}: {} holds {:#x}{}\n",
                                if slot.is_parked() { "parked" } else { "running" },
                                slot.holds(),
                                if disabled { " (disabled in MODE)" } else { "" }));
                        }
                    }
                }
                drop(park);
                for (i, d) in r.descriptor[..BANKS].iter().enumerate() {
                    s.push_str(&format!("bank {i}: descriptor {d:#010x}, {} MiB fitted\n", self.cfg.banks[i] >> 20));
                }
                s
            }
            "unsupported" => self
                .unsupported()
                .iter()
                .map(|(o, s, w)| format!("{} {s} bytes at {o:#x}\n", if *w { "write" } else { "read" }))
                .collect(),
            other => return Err(format!("heart: unknown subcommand {other}")),
        };
        w.write_all(out.as_bytes()).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::irq::InterruptController;
    use crate::traits::Resettable;

    struct Fixture {
        heart: Heart,
        now: Arc<AtomicU64>,
    }

    impl Fixture {
        fn step_us(&self, us: u64) {
            self.now.fetch_add(us * 1000, Ordering::SeqCst);
        }
    }

    fn heart() -> Fixture {
        let now = Arc::new(AtomicU64::new(0));
        let n = now.clone();
        let clock: Clock = Arc::new(move || n.load(Ordering::SeqCst));
        let heart = Heart::new(HeartConfig::octane([128 << 20, 0, 0, 0]), clock);
        Fixture { heart, now }
    }

    fn wq(h: &Heart, off: u32, v: u64) {
        assert!(h.piu_write(off, 8, v));
    }

    fn rq(h: &Heart, off: u32) -> u64 {
        h.piu_read(off, 8).unwrap()
    }

    /// HEART holds raw pointers to `words`, so they stay together: taking a
    /// test's HEART out alone would leave it writing into freed memory.
    struct TwoCpus {
        h: Heart,
        words: Box<[AtomicU64; 2]>,
        slots: [Arc<ParkSlot>; 2],
    }

    fn two_cpu_heart() -> TwoCpus {
        let mut cfg = HeartConfig::octane([128 << 20, 0, 0, 0]);
        cfg.cpus = 2;
        let h = Heart::new(cfg, Arc::new(|| 0));
        let words = Box::new([AtomicU64::new(0), AtomicU64::new(0)]);
        let slots = [Arc::new(ParkSlot::new(&words[0])), Arc::new(ParkSlot::new(&words[1]))];
        for cpu in 0..2 {
            h.set_cpu_interrupts(cpu, &words[cpu]);
            h.set_cpu_park(cpu, slots[cpu].clone());
        }
        TwoCpus { h, words, slots }
    }

    /// Runs `f` on a thread that is CPU `cpu`, as HEART sees it.
    fn as_cpu<R: Send>(cpu: u8, f: impl FnOnce() -> R + Send) -> R {
        std::thread::scope(|s| {
            s.spawn(move || {
                crate::cpu::smp::set_current_cpu(cpu);
                f()
            })
            .join()
            .unwrap()
        })
    }

    const LEVEL2_IP4_CAUSE: u64 = 1 << 12;

    #[test]
    fn each_cpu_reads_its_own_number_and_all_see_the_fitted_ones() {
        let TwoCpus { h, .. } = &two_cpu_heart();
        assert_eq!(as_cpu(0, || rq(&h, 0x50000)), 0);
        assert_eq!(as_cpu(1, || rq(&h, 0x50000)), 1);
        assert_eq!(as_cpu(1, || rq(&h, 0x80)), 0x38);
    }

    #[test]
    fn one_fitted_cpu_reads_as_it_always_did() {
        let f = heart();
        assert_eq!(rq(&f.heart, 0x80), 0x18);
        assert_eq!(rq(&f.heart, 0x50000), 0);
    }

    #[test]
    fn an_ipi_interrupts_only_its_destination() {
        let TwoCpus { h, words, .. } = &two_cpu_heart();
        let ipi = |cpu: u32| 1u64 << (46 + cpu);
        wq(&h, 0x10000, ipi(0));
        wq(&h, 0x10008, ipi(1));
        as_cpu(0, || wq(&h, 0x10020, ipi(1)));
        assert_eq!(words[0].load(Ordering::SeqCst) & LEVEL2_IP4_CAUSE, 0);
        assert_ne!(words[1].load(Ordering::SeqCst) & LEVEL2_IP4_CAUSE, 0);
        as_cpu(1, || wq(&h, 0x10028, ipi(1)));
        assert_eq!(words[1].load(Ordering::SeqCst) & LEVEL2_IP4_CAUSE, 0);
    }

    #[test]
    fn imsr_is_masked_by_the_reader() {
        let TwoCpus { h, .. } = &two_cpu_heart();
        wq(&h, 0x10000, 0b01);
        wq(&h, 0x10008, 0b10);
        wq(&h, 0x10020, 0b11);
        assert_eq!(as_cpu(0, || rq(&h, 0x10038)), 0b01);
        assert_eq!(as_cpu(1, || rq(&h, 0x10038)), 0b10);
    }

    #[test]
    fn a_disable_bit_holds_its_cpu_and_clearing_it_lets_go() {
        let TwoCpus { h, words, slots } = &two_cpu_heart();
        let park = crate::cpu::smp::HOST_PARK_BIT;
        wq(&h, 0, 1 << 61);
        assert_ne!(words[1].load(Ordering::SeqCst) & park, 0);
        assert_eq!(words[0].load(Ordering::SeqCst) & park, 0);
        assert_eq!(slots[1].holds(), HOLD_HEART);
        wq(&h, 0, 0);
        assert_eq!(words[1].load(Ordering::SeqCst) & park, 0);
    }

    #[test]
    fn a_reset_serves_every_processor_again() {
        let TwoCpus { h, words, .. } = &two_cpu_heart();
        wq(&h, 0, 1 << 61);
        h.power_on();
        assert_eq!(words[1].load(Ordering::SeqCst) & crate::cpu::smp::HOST_PARK_BIT, 0);
    }

    #[test]
    fn the_software_reset_bit_asks_once_per_edge() {
        let TwoCpus { h, .. } = &two_cpu_heart();
        let asked = Arc::new(AtomicU64::new(0));
        let a = asked.clone();
        h.on_soft_reset(Box::new(move || {
            a.fetch_add(1, Ordering::SeqCst);
        }));
        wq(&h, 0, MODE_SOFT_RESET);
        wq(&h, 0, MODE_SOFT_RESET);
        assert_eq!(asked.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn the_unsupported_list_keeps_the_first_64_and_then_stops() {
        let f = heart();
        let h = &f.heart;
        for i in 0..100u32 {
            h.note(0x1000 + 8 * i, 8, false);
        }
        let u = h.unsupported();
        assert_eq!(u.len(), 64);
        assert_eq!(u[63], (0x1000 + 8 * 63, 8, false));
        h.note(0x9000, 4, true);
        assert_eq!(h.unsupported(), u, "a full list is left alone");
    }

    #[test]
    fn descriptor_writes_reach_memory_and_reset_clears_them() {
        let f = heart();
        let h = &f.heart;
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        h.on_memcfg(Box::new(move |b| s.lock().push(b)));
        assert_eq!(h.piu_read(0x20, 8), Some(0));
        assert!(h.piu_write(0x24, 4, 0x81bf_0040));
        assert_eq!(h.piu_read(0x20, 8), Some(0x81bf_0040));
        assert!(h.piu_write(0x28, 8, 0x8001_0000_8003_0040));
        assert_eq!(h.piu_read(0x2c, 4), Some(0x8003_0040));
        assert_eq!(h.memcfg(), [0, 0x81bf_0040, 0x8001_0000, 0x8003_0040]);
        h.power_on();
        assert_eq!(h.memcfg(), [0; 4]);
        let seen = seen.lock();
        assert_eq!(seen.len(), 3, "each write and the reset");
        assert_eq!(seen[1], [0, 0x81bf_0040, 0x8001_0000, 0x8003_0040]);
        assert_eq!(seen[2], [0; 4]);
    }

    #[test]
    fn widget_identification() {
        let f = heart();
        let h = &f.heart;
        assert_eq!(h.widget_read(4, 4), Some(0x4c00_106d));
        assert_eq!(h.widget_read(0, 8), Some(0x4c00_106d));
        assert_eq!(h.widget_read(0, 1), None);
        assert_eq!(h.widget_read(0, 4), None);
        assert_eq!(h.widget_read(8, 8), None);
    }

    #[test]
    fn the_error_command_word_takes_a_zero_write_at_either_width() {
        let f = heart();
        let h = &f.heart;
        assert!(h.widget_write(0x44, 4, 0));
        assert!(h.widget_write(0x40, 8, 0));
        assert!(!h.widget_write(0x44, 4, 1));
        assert!(!h.widget_write(0x40, 8, 1));
        assert!(!h.widget_write(0x40, 4, 0));
    }

    #[test]
    fn packets_set_and_clear_vectors_by_level() {
        let f = heart();
        let h = &f.heart;
        let (l0, l1, l2) = (1u64 << 3, 1u64 << 16, 1u64 << 42);
        assert!(!h.packet_write(0x80, 4, 0x103));
        assert!(!h.packet_write(0x80, 8, 0x140));
        assert!(!h.packet_write(0x80, 8, 0x303));
        assert!(!h.packet_write(0x88, 8, 0x103));
        assert_eq!(h.pending(), 0);
        wq(h, 0x10000, l0 | l1 | l2);
        assert!(h.packet_write(0x80, 8, 0x103));
        assert_eq!(h.pending(), l0);
        assert!(h.pin(2) && !h.pin(3) && !h.pin(4));
        assert!(h.packet_write(0x80, 8, 0x110));
        assert!(h.pin(3));
        h.set_line(42, true);
        assert_eq!(rq(h, 0x10038), l0 | l1 | l2);
        assert!(h.pin(4));
        // Masking drops the pin; the pending bit stays.
        wq(h, 0x10000, l0 | l2);
        assert_eq!(h.pending(), l0 | l1 | l2);
        assert!(!h.pin(3));
        wq(h, 0x10000, l0 | l1 | l2);
        assert!(h.pin(3));
        h.set_line(16, false);
        assert_eq!(h.pending(), l0 | l2);
        assert!(h.pin(2) && !h.pin(3) && h.pin(4));
        h.set_line(3, false);
        assert!(!h.pin(2));
        h.set_line(42, false);
        assert!(!h.pin(4));
    }

    #[test]
    fn set_and_clear_registers() {
        let f = heart();
        let h = &f.heart;
        wq(h, 0x10008, 2);
        wq(h, 0x10020, 3);
        assert_eq!(rq(h, 0x10030), 3);
        assert_eq!(rq(h, 0x10038), 0);
        wq(h, 0x10000, 1);
        assert_eq!(rq(h, 0x10038), 1);
        wq(h, 0x10028, 1);
        assert_eq!(rq(h, 0x10030), 2);
        assert_eq!(rq(h, 0x10008), 2);
    }

    #[test]
    fn timer_and_error_levels() {
        let f = heart();
        let h = &f.heart;
        let (timer, error) = (1u64 << 50, 1u64 << 51);
        wq(h, 0x10020, timer | error);
        assert!(!h.pin(5) && !h.pin(6));
        wq(h, 0x10000, timer | error);
        assert!(h.pin(5) && h.pin(6));
        wq(h, 0x10000, error);
        assert!(!h.pin(5) && h.pin(6));
        wq(h, 0x10028, error);
        assert!(!h.pin(6));
        // The memory-error bit reaches IP6 only with interrupts enabled.
        wq(h, 0x10020, ECC_PENDING);
        wq(h, 0x10000, ECC_PENDING);
        assert!(!h.pin(6));
        wq(h, 0, MODE_INTERRUPT_ENABLE);
        assert!(h.pin(6));
        h.power_on();
        assert!(!h.pin(6));
    }

    /// HEART as an interrupt controller writes its five levels into the
    /// CPU's word, Cause IP2 to IP6, and leaves every other bit alone.
    #[test]
    fn interrupt_lines_drive_the_cpu_word() {
        let f = heart();
        let h = &f.heart;
        let others = 1u64 << 15 | 1 << 63; // IP7 and a bit HEART does not own
        let word = AtomicU64::new(others);
        let ip = |n: u32| 1u64 << (n + 8);
        // Pending before the CPU exists: published when the word arrives.
        wq(h, 0x10000, u64::MAX);
        h.set_line(5, true);
        h.set_interrupts(&word);
        assert_eq!(word.load(Ordering::SeqCst), others | ip(2), "level 0 is IP2");
        for (vector, n) in [(20u32, 3u32), (40, 4), (50, 5), (60, 6)] {
            h.set_line(vector, true);
            assert_ne!(word.load(Ordering::SeqCst) & ip(n), 0, "vector {vector} raises IP{n}");
        }
        assert_eq!(word.load(Ordering::SeqCst), others | (0x1f << 10));
        // Two sources on one vector share its bit, but another vector on
        // the same level keeps the level up.
        h.set_line(21, true);
        h.set_line(20, false);
        assert_ne!(word.load(Ordering::SeqCst) & ip(3), 0, "vector 21 still holds level 1");
        h.set_line(21, false);
        assert_eq!(word.load(Ordering::SeqCst) & ip(3), 0);
        let before = h.pending();
        h.set_line(64, true);
        assert_eq!(h.pending(), before, "a line past 63 is dropped");
        wq(h, 0x10028, u64::MAX);
        assert_eq!(word.load(Ordering::SeqCst), others);
    }

    #[test]
    fn the_bus_reaches_both_register_blocks() {
        let f = heart();
        let h = &f.heart;
        assert_eq!(h.read32(WIDGET_BASE + 4).data, 0x4c00_106d);
        assert_eq!(h.read64(WIDGET_BASE).data, 0x4c00_106d);
        assert_eq!(h.read32(WIDGET_BASE + 0x1004).status, BUS_ERR, "no route back to HEART");
        assert_eq!(h.read64(PIU_BASE + 0x10030).data, 0, "pending bits");
        assert_eq!(h.read32(PIU_BASE + 0x20).data, 0);
        assert_eq!(h.read8(PIU_BASE).status, BUS_ERR, "narrow PIU accesses are refused");
        assert_eq!(h.write16(PIU_BASE, 0), BUS_ERR);
        assert_eq!(h.read64(PIU_BASE + PIU_SIZE).status, BUS_ERR, "past the registers");
        assert_eq!(h.write64(PIU_BASE + 0x10020, 1), BUS_OK);
        assert_eq!(h.pending(), 1);
    }

    #[test]
    fn the_counter_ticks_at_12_5_mhz_and_resets() {
        let f = heart();
        let h = &f.heart;
        assert_eq!(rq(h, 0x80), 0x18);
        f.now.fetch_add(800, Ordering::SeqCst);
        assert_eq!(rq(h, 0x20000), 10);
        wq(h, 0x10000, 1 << 16);
        wq(h, 0x10020, 1 << 16);
        assert!(h.pin(3));
        h.power_on();
        assert_eq!(rq(h, 0x20000), 0);
        assert_eq!(rq(h, 0x10030), 0);
        assert!(!h.pin(3));
        assert_eq!(h.piu_read(0x20, 4), Some(0));
    }

    #[test]
    fn the_counter_holds_still_while_the_machine_is_stopped() {
        use crate::traits::Device;
        let f = heart();
        let h = &f.heart;
        f.now.fetch_add(800, Ordering::SeqCst);
        assert_eq!(rq(h, 0x20000), 10);
        h.stop();
        f.now.fetch_add(30_000_000_000, Ordering::SeqCst);
        assert_eq!(rq(h, 0x20000), 10, "30 s stopped are not counted");
        h.start();
        assert_eq!(rq(h, 0x20000), 10);
        f.now.fetch_add(800, Ordering::SeqCst);
        assert_eq!(rq(h, 0x20000), 20, "and it runs on from there");
        h.start();
        assert_eq!(rq(h, 0x20000), 20, "a second start changes nothing");
    }

    #[test]
    fn narrow_piu_accesses_are_refused() {
        let f = heart();
        assert_eq!(f.heart.piu_read(0, 1), None);
        assert!(!f.heart.piu_write(0, 2, 0));
    }

    /// A part that answers presence and holds nothing else.
    struct Present;
    impl w1::Part for Present {
        fn rom(&self) -> [u8; 8] {
            [0x0b, 1, 0, 0, 0, 0, 0, 0]
        }
        fn reply(&self, _cmd: u8, _addr: u16) -> Option<Vec<u8>> {
            None
        }
    }

    #[test]
    fn microlan_slots_complete_after_their_time() {
        let f = heart();
        let h = &f.heart;
        let cmd = |pulse: u64, sample: u64| (pulse << 10) | (sample << 2);
        let reset = cmd(500, 65);
        assert_eq!(h.piu_read(0xb8, 8), Some(0));
        assert!(h.piu_write(0xb8, 8, reset));
        assert_eq!(h.piu_read(0xb8, 8).unwrap() & 3, 0);
        f.step_us(564);
        assert_eq!(h.piu_read(0xbc, 4).unwrap() & 3, 0);
        f.step_us(1);
        // No part: the wire stays high through the presence sample.
        assert_eq!(h.piu_read(0xb8, 8), Some(reset | 3));
        let read = cmd(6, 13);
        assert!(h.piu_write(0xbc, 4, read));
        f.step_us(19);
        assert_eq!(h.piu_read(0xbc, 4), Some(read | 3));
        assert!(h.piu_write(0xbc, 4, cmd(80, 30)));
        f.step_us(110);
        assert_eq!(h.piu_read(0xbc, 4).unwrap() & 3, 2);

        let mut bus = w1::Bus::new();
        bus.add(Box::new(Present));
        h.set_microlan(bus);
        assert!(h.piu_write(0xbc, 4, reset));
        f.step_us(565);
        assert_eq!(h.piu_read(0xbc, 4).unwrap() & 3, 2, "presence pulls the wire low");
    }

    /// A slot in progress when the machine stops completes as long after
    /// the restart as it had left to run.
    #[test]
    fn a_microlan_slot_waits_out_a_stop() {
        let f = heart();
        let h = &f.heart;
        assert!(h.piu_write(0xbc, 4, (6 << 10) | (13 << 2)));
        f.step_us(10);
        h.pause();
        f.step_us(1_000_000);
        h.resume();
        assert_eq!(h.piu_read(0xbc, 4).unwrap() & 2, 0, "9 us still to go");
        f.step_us(9);
        assert_eq!(h.piu_read(0xbc, 4).unwrap() & 2, 2);
    }
}
