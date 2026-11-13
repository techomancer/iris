//! The Dallas DS1685 real-time clock the Octane fits on its IOC3's byte bus,
//! with the NVRAM image it keeps in a file.
//!
//! The part sits behind two byte-wide selects: an address latch and a data
//! port, so the guest picks a register and then reads or writes it. It is the
//! O2's DS1687 model (`ds1687.rs`) behind that index and data pair, with the
//! RAM the DS1685 adds: the bank-0 user RAM, the bank-1 registers and the
//! 128-byte extended RAM where the PROM keeps part of its environment.
//!
//! # What is approximated here
//!
//! The part has no alarm or periodic interrupts, and the model number's side
//! effects are not modelled. The RAM persists in a 512-byte file, replaced
//! whole once the guest's writes to it pause, by a writer thread
//! (`IOC3-NVRAM`) of this part. Once the guest sets the clock, the clock
//! persists beside the file (see `battery_clock`).
//!
//! Ported from hw/misc/sgi-ioc3.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::dev::deadline::{decode, encode};
use crate::dev::battery_clock;
use crate::dev::ds1687::{image, MaceRtc};
use crate::snapshot::{field, field_bool, field_u8, field_u8s, StateTable};
use crate::traits::{BusDevice, Device, MachineEvent, Saveable};

/// How long the guest's NVRAM writes must pause before the file is
/// rewritten.
pub const NVRAM_QUIET: Duration = Duration::from_millis(500);
/// The file is the DS1685's 512-byte image (see `ds1687::image`).
const NVRAM_IMAGE_LEN: usize = image::LEN;
pub(crate) const NVRAM_OFF_NV: usize = image::NV;
const NVRAM_OFF_EXT: usize = image::EXT;
const NVRAM_OFF_XRAM: usize = image::XRAM;
/// Register A bit 4 selects bank 1 for registers 0x40..0x7f.
const RTC_A_DV0: u8 = 0x10;
/// Bank-1 extended RAM address and data ports.
const RTC_XRAM_ADDR: usize = 0x50;
const RTC_XRAM_DATA: usize = 0x53;
const RTC_XRAM_SIZE: usize = 128;

/// The clock, reached through an address port and a data port. The bank bit
/// switches only registers 0x40..0x7f: the four control registers and the
/// low RAM are the same cells in either bank. Bank 1 carries the extended
/// registers and, through the address and data ports at 0x50 and 0x53, 128
/// bytes of extended RAM, where the PROM keeps part of its environment.
///
/// The clock and the power-control and century registers are `MaceRtc`'s;
/// see the module notes for what the model leaves out (alarm and periodic
/// interrupts, the model number's side effects).
struct Rtc {
    dev: MaceRtc,
    index: u8,
    file: Option<std::path::PathBuf>,
    /// Bank-0 RAM at 0x40..0x7f.
    nv: [u8; 64],
    /// Bank-1 registers at 0x40..0x7f (0x48 and 0x4a live in `dev`).
    ext: [u8; 64],
    xram: [u8; RTC_XRAM_SIZE],
    /// When the RAM last changed, if the file has not caught up yet.
    dirty_since: Option<Instant>,
    /// Where the clock starts with no reference saved (`[rtc_offset]`).
    rtc_offset: crate::config::RtcOffset,
}

impl Rtc {
    fn new() -> Rtc {
        Rtc {
            dev: MaceRtc::new(),
            index: 0,
            file: None,
            nv: [0; 64],
            ext: [0; 64],
            xram: [0; RTC_XRAM_SIZE],
            dirty_since: None,
            rtc_offset: Default::default(),
        }
    }

    fn bank1(&self) -> bool {
        self.dev.peek(0x0a) & RTC_A_DV0 != 0
    }

    /// Whether the selected register is one `MaceRtc` models itself.
    fn delegated(&self) -> bool {
        let i = self.index as usize;
        i < 0x40 || (self.bank1() && (i == 0x48 || i == 0x4a))
    }

    fn read(&self) -> u8 {
        let i = self.index as usize;
        if self.delegated() {
            self.dev.read8(((i as u64) << 8) | 7).data
        } else if !self.bank1() {
            self.nv[i - 0x40]
        } else if i == RTC_XRAM_DATA {
            self.xram[self.ext[RTC_XRAM_ADDR - 0x40] as usize & (RTC_XRAM_SIZE - 1)]
        } else {
            self.ext[i - 0x40]
        }
    }

    fn write(&mut self, v: u8) {
        let i = self.index as usize;
        if self.delegated() {
            self.dev.write8(((i as u64) << 8) | 7, v);
        } else if !self.bank1() {
            self.nv[i - 0x40] = v;
        } else if i == RTC_XRAM_DATA {
            let at = self.ext[RTC_XRAM_ADDR - 0x40] as usize & (RTC_XRAM_SIZE - 1);
            self.xram[at] = v;
        } else {
            self.ext[i - 0x40] = v;
        }
        // The file catches up once the writes stop: see `Ds1685::flush_after`.
        // A write to the clock registers can set the clock, which is saved
        // beside it. Register A's cannot, and the PROM flips its bank bit
        // around every extended-RAM access.
        if i != 0x0a && self.file.is_some() {
            self.dirty_since = Some(Instant::now());
        }
    }

    fn image(&self) -> Vec<u8> {
        let mut img = vec![0u8; NVRAM_IMAGE_LEN];
        let (regs, century) = self.dev.image_registers();
        img[..0x40].copy_from_slice(&regs);
        img[NVRAM_OFF_NV..NVRAM_OFF_NV + 64].copy_from_slice(&self.nv);
        let mut ext = self.ext;
        ext[0x08] = century;
        ext[0x0a] = self.dev.peek(0x4a);
        img[NVRAM_OFF_EXT..NVRAM_OFF_EXT + 64].copy_from_slice(&ext);
        img[NVRAM_OFF_XRAM..NVRAM_OFF_XRAM + RTC_XRAM_SIZE].copy_from_slice(&self.xram);
        img
    }

    /// Load a 512-byte image. The clock registers (0x00..0x0d) are never
    /// loaded: the clock comes from its reference. Returns false for any
    /// other length.
    fn load(&mut self, img: &[u8]) -> bool {
        if img.len() != NVRAM_IMAGE_LEN {
            return false;
        }
        self.dev.load_image_registers(img);
        self.nv.copy_from_slice(&img[NVRAM_OFF_NV..NVRAM_OFF_NV + 64]);
        self.ext.copy_from_slice(&img[NVRAM_OFF_EXT..NVRAM_OFF_EXT + 64]);
        self.xram.copy_from_slice(&img[NVRAM_OFF_XRAM..NVRAM_OFF_XRAM + RTC_XRAM_SIZE]);
        // Loading never powers the machine off.
        self.dev.poke(0x4a, self.ext[0x0a] & !0x08);
        true
    }

    /// The file, the image to put in it and the clock reference to put
    /// beside it, if the RAM changed at least `quiet` ago; the RAM then
    /// counts as saved.
    fn take_dirty(&mut self, quiet: Duration) -> Option<NvramJob> {
        let since = self.dirty_since?;
        if since.elapsed() < quiet {
            return None;
        }
        self.dirty_since = None;
        Some((self.file.clone()?, self.image(), self.dev.clock_reference()))
    }
}

/// Say that an NVRAM save failed, once until a save succeeds again: the
/// guest's writes keep scheduling saves, and a full disk would otherwise
/// print on every one.
fn report_nvram_save(path: &std::path::Path, result: std::io::Result<()>) {
    static FAILING: AtomicBool = AtomicBool::new(false);
    match result {
        Ok(()) => FAILING.store(false, Ordering::Relaxed),
        Err(e) => {
            if !FAILING.swap(true, Ordering::Relaxed) {
                eprintln!("ds1685: NVRAM {}: {e}", path.display());
            }
        }
    }
}

/// Replace the NVRAM file in one step (a temporary file, synced and
/// renamed), so a kill mid-write leaves the previous image whole, then the
/// clock reference beside it, if the guest has set the clock.
fn save_nvram((path, image, clock): &NvramJob) -> std::io::Result<()> {
    let name = path
        .to_str()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "NVRAM path is not UTF-8"))?;
    crate::nv_storage::save(name, image)?;
    match clock {
        Some(c) => battery_clock::write(&battery_clock::companion(path), c),
        None => Ok(()),
    }
}

/// A rewrite of the NVRAM file waiting for the writer thread.
type NvramJob = (std::path::PathBuf, Vec<u8>, Option<Vec<u8>>);

/// The clock part: its registers and RAM, the file they persist in, and the
/// thread that writes it.
pub struct Ds1685 {
    st: Mutex<Rtc>,
    /// When the RAM last changed, where a timer can read it without the lock.
    dirty: AtomicU64,
    /// Where the guest's power-off goes: one `PowerOff` event when it writes
    /// the bit.
    events: Option<SyncSender<MachineEvent>>,
    /// The NVRAM image the writer thread has yet to put in the file.
    job: Mutex<Option<NvramJob>>,
    ready: Condvar,
    running: AtomicBool,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    me: Weak<Ds1685>,
}

impl Ds1685 {
    pub fn new(events: Option<SyncSender<MachineEvent>>) -> Arc<Ds1685> {
        Arc::new_cyclic(|me| Ds1685 {
            st: Mutex::new(Rtc::new()),
            dirty: AtomicU64::new(0),
            events,
            job: Mutex::new(None),
            ready: Condvar::new(),
            running: AtomicBool::new(false),
            thread: Mutex::new(None),
            me: me.clone(),
        })
    }

    /// Back the RAM (user RAM, bank-1 registers and the 128-byte extended
    /// RAM) with a file. A 512-byte file is loaded. The file is written as
    /// 512 bytes now and replaced whole once the guest's writes to the RAM
    /// pause (see [`Ds1685::flush_after`]). The clock runs from the reference
    /// beside the file (`<file>.clock`, the layout QEMU's model keeps) if
    /// there is one, else from where [`Ds1685::set_rtc_offset`] put it, host
    /// time by default.
    pub fn set_nvram_file(&self, path: impl Into<std::path::PathBuf>) -> std::io::Result<()> {
        let path = path.into();
        let clock = battery_clock::companion(&path);
        let job = {
            let mut rtc = self.st.lock();
            if let Ok(img) = std::fs::read(&path) {
                if !rtc.load(&img) {
                    log::warn!("ds1685: NVRAM file {} has {} bytes, not 512; ignored", path.display(), img.len());
                }
            }
            if let Some(guest) = battery_clock::read_or_warn(&clock).and_then(|r| rtc.dev.restore_clock(&r)) {
                battery_clock::note_override(&rtc.rtc_offset, &clock, guest);
            }
            rtc.file = Some(path.clone());
            rtc.dirty_since = None;
            self.dirty.store(0, Ordering::SeqCst);
            (path, rtc.image(), rtc.dev.clock_reference())
        };
        save_nvram(&job)
    }

    /// Start the clock where `[rtc_offset]` puts it, for as long as the
    /// NVRAM file keeps no clock of the guest's.
    pub fn set_rtc_offset(&self, cfg: &crate::config::RtcOffset) {
        let mut rtc = self.st.lock();
        rtc.rtc_offset = *cfg;
        rtc.dev.set_offset(battery_clock::seed(cfg));
    }

    /// The index latch, on byte-bus select 1: seven bits, with the bank from
    /// register A.
    pub fn write_index(&self, v: u8) {
        self.st.lock().index = v & 0x7f;
    }

    /// A write port on the part; the latch is what is there.
    pub fn read_index(&self) -> u8 {
        self.st.lock().index
    }

    /// The data port, on select 2.
    pub fn read_data(&self) -> u8 {
        self.st.lock().read()
    }

    pub fn write_data(&self, v: u8) {
        let powered_off = {
            let mut rtc = self.st.lock();
            let was = rtc.dev.powered_off();
            rtc.write(v);
            self.dirty.store(encode(rtc.dirty_since), Ordering::SeqCst);
            !was && rtc.dev.powered_off()
        };
        if powered_off {
            if let Some(tx) = &self.events {
                let _ = tx.try_send(MachineEvent::PowerOff);
            }
        }
    }

    /// Whether the guest has asked the clock to power the machine off.
    pub fn powered_off(&self) -> bool {
        self.st.lock().dev.powered_off()
    }

    /// When the file next needs writing, if the RAM changed since it was.
    pub fn due(&self) -> Option<Instant> {
        decode(self.dirty.load(Ordering::SeqCst)).map(|t| t + NVRAM_QUIET)
    }

    /// The debounce ran out: hand the image to the writer thread.
    pub fn hand_over(&self) {
        if let Some(job) = self.take_dirty(NVRAM_QUIET) {
            *self.job.lock() = Some(job);
            self.ready.notify_all();
        }
    }

    fn take_dirty(&self, quiet: Duration) -> Option<NvramJob> {
        let mut rtc = self.st.lock();
        let job = rtc.take_dirty(quiet);
        self.dirty.store(encode(rtc.dirty_since), Ordering::SeqCst);
        job
    }

    /// Write the NVRAM file if the RAM changed at least `quiet` ago. The
    /// PROM writes its environment a byte at a time, so a timer waits for a
    /// pause rather than rewriting the file per byte; stopping the machine
    /// and dropping the part flush with no wait. The file is written outside
    /// the part's lock.
    pub fn flush_after(&self, quiet: Duration) {
        let Some(job) = self.take_dirty(quiet) else { return };
        report_nvram_save(&job.0, save_nvram(&job));
    }

    /// [`Ds1685::flush_after`] with no wait.
    pub fn flush(&self) {
        self.flush_after(Duration::ZERO);
    }

    /// Write the NVRAM file when the timer hands over an image.
    fn writer(me: Weak<Ds1685>) {
        loop {
            let job = {
                let Some(d) = me.upgrade() else { return };
                let mut slot = d.job.lock();
                while slot.is_none() && d.running.load(Ordering::SeqCst) {
                    d.ready.wait(&mut slot);
                }
                match slot.take() {
                    Some(j) => j,
                    None => return,
                }
            };
            report_nvram_save(&job.0, save_nvram(&job));
        }
    }

    /// Let `secs` of guest time pass at once, as a loaded host lets them
    /// pass between two statements of a test.
    #[cfg(test)]
    pub(crate) fn advance_clock(&self, secs: i64) {
        self.st.lock().dev.advance_clock(secs);
    }

    /// The clock and its RAM for a snapshot, as the chip's `rtc` key.
    pub fn save_state(&self) -> toml::Value {
        let r = self.st.lock();
        StateTable::new()
            .value("clock", r.dev.save_state())
            .u8("index", r.index)
            .u8s("nv", &r.nv)
            .u8s("ext", &r.ext)
            .u8s("xram", &r.xram)
            .bool("dirty", r.dirty_since.is_some())
            .build()
    }

    pub fn load_state(&self, r: &toml::Value, now: Instant) -> Result<(), String> {
        let mut rtc = self.st.lock();
        rtc.dev.load_state(field(r, "clock")?)?;
        rtc.index = field_u8(r, "index")?;
        rtc.nv = field_u8s(r, "nv")?;
        rtc.ext = field_u8s(r, "ext")?;
        rtc.xram = field_u8s(r, "xram")?;
        // The NVRAM file catches up with the restored RAM in the usual way.
        rtc.dirty_since = (field_bool(r, "dirty")? || rtc.file.is_some()).then_some(now);
        self.dirty.store(encode(rtc.dirty_since), Ordering::SeqCst);
        Ok(())
    }

    pub fn is_locked(&self) -> bool {
        self.st.is_locked()
    }

    #[cfg(test)]
    pub(crate) fn dirty_since(&self) -> Option<Instant> {
        self.st.lock().dirty_since
    }
}

impl Device for Ds1685 {
    fn step(&self, _cycles: u64) {}

    /// Start the writer thread.
    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let me = self.me.clone();
        *self.thread.lock() = Some(
            std::thread::Builder::new()
                .name("IOC3-NVRAM".into())
                .spawn(move || Ds1685::writer(me))
                .expect("spawn ds1685 nvram thread"),
        );
    }

    /// Join the writer and flush the file.
    fn stop(&self) {
        if !self.running.swap(false, Ordering::SeqCst) {
            return;
        }
        // The writer waits on the job slot's lock: take it so the wake-up
        // cannot fall between its check of `running` and its wait.
        drop(self.job.lock());
        self.ready.notify_all();
        // The writer can hold the last reference for a moment, and then this
        // runs on it; it cannot join itself.
        if let Some(t) = self.thread.lock().take() {
            if t.thread().id() != std::thread::current().id() {
                let _ = t.join();
            }
        }
        self.flush();
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    fn get_clock(&self) -> u64 {
        0
    }
}

impl Drop for Ds1685 {
    fn drop(&mut self) {
        Device::stop(&*self);
        self.flush();
    }
}
