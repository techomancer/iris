//! The Dallas DS1685 real-time clock the Octane attaches to its IOC3's byte bus,
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
use crate::traits::MachineEvents;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use crate::clock::{decode, encode, Clock};
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
    /// When the RAM last changed (machine nanoseconds), if the file has not caught up yet.
    dirty_since: Option<u64>,
    /// Where the clock starts with no reference saved (`[rtc_offset]`).
    rtc_offset: crate::config::RtcOffset,
}

impl Rtc {
    fn new(clock: Clock) -> Rtc {
        Rtc {
            dev: MaceRtc::new(clock),
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

    fn write(&mut self, v: u8, now: u64) {
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
        // The file catches up once the writes stop: see `Ds1685::hand_over`.
        // A write to the clock registers can set the clock, which is saved
        // beside it. Register A's cannot, and the PROM flips its bank bit
        // around every extended-RAM access.
        if i != 0x0a && self.file.is_some() {
            self.dirty_since = Some(now);
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
    fn take_dirty(&mut self, quiet: Duration, now: u64) -> Option<NvramJob> {
        let since = self.dirty_since?;
        if now.saturating_sub(since) < quiet.as_nanos() as u64 {
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

/// What the writer thread shares with the part. It holds only this, so
/// dropping the part's last reference runs its `Drop`, which joins the writer.
#[derive(Default)]
struct WriterSlot {
    /// The NVRAM image the writer thread has yet to put in the file.
    job: Mutex<Option<NvramJob>>,
    ready: Condvar,
    running: AtomicBool,
}

/// The clock part: its registers and RAM, the file they persist in, and the
/// thread that writes it.
pub struct Ds1685 {
    st: Mutex<Rtc>,
    /// When the RAM last changed, where a timer can read it without the lock.
    dirty: AtomicU64,
    /// Where the guest's power-off goes: one `PowerOff` event when it writes
    /// the bit.
    events: Option<MachineEvents>,
    slot: Arc<WriterSlot>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Ds1685 {
    pub fn new(events: Option<MachineEvents>, clock: Clock) -> Arc<Ds1685> {
        Arc::new(Ds1685 {
            st: Mutex::new(Rtc::new(clock)),
            dirty: AtomicU64::new(0),
            events,
            slot: Arc::default(),
            thread: Mutex::new(None),
        })
    }

    /// Back the RAM (user RAM, bank-1 registers and the 128-byte extended
    /// RAM) with a file. A 512-byte file is loaded. The file is written as
    /// 512 bytes now and replaced whole once the guest's writes to the RAM
    /// pause (see [`Ds1685::hand_over`]). The clock runs from the reference
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

    pub fn write_data(&self, v: u8, now: u64) {
        let powered_off = {
            let mut rtc = self.st.lock();
            let was = rtc.dev.powered_off();
            rtc.write(v, now);
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
    pub fn due(&self) -> Option<u64> {
        decode(self.dirty.load(Ordering::SeqCst)).map(|t| t + NVRAM_QUIET.as_nanos() as u64)
    }

    /// The debounce ran out: hand the image to the writer thread.
    pub fn hand_over(&self, now: u64) {
        if let Some(job) = self.take_dirty(NVRAM_QUIET, now) {
            *self.slot.job.lock() = Some(job);
            self.slot.ready.notify_all();
        }
    }

    fn take_dirty(&self, quiet: Duration, now: u64) -> Option<NvramJob> {
        let mut rtc = self.st.lock();
        let job = rtc.take_dirty(quiet, now);
        self.dirty.store(encode(rtc.dirty_since), Ordering::SeqCst);
        job
    }

    /// Write the NVRAM file if the RAM changed, with no wait, outside the part's lock.
    /// The timer ([`Ds1685::hand_over`]) waits for a pause in the PROM's byte writes.
    pub fn flush(&self) {
        let Some(job) = self.take_dirty(Duration::ZERO, 0) else { return };
        report_nvram_save(&job.0, save_nvram(&job));
    }

    /// Write the NVRAM file when the timer hands over an image.
    fn writer(s: Arc<WriterSlot>) {
        loop {
            let job = {
                let mut slot = s.job.lock();
                while slot.is_none() && s.running.load(Ordering::SeqCst) {
                    s.ready.wait(&mut slot);
                }
                match slot.take() {
                    Some(j) => j,
                    None => return,
                }
            };
            report_nvram_save(&job.0, save_nvram(&job));
        }
    }

    /// Run `f` on the PROM's environment store: registers 0x0e..0x7f, then the
    /// extended RAM (242 bytes). A change is saved at the next timer pass.
    pub fn with_env_store<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> R {
        let mut rtc = self.st.lock();
        let mut img = rtc.image();
        let mut store = [&img[0x0e..NVRAM_OFF_EXT], &img[NVRAM_OFF_XRAM..]].concat();
        let before = store.clone();
        let r = f(&mut store);
        if store != before {
            let (low, high) = store.split_at(NVRAM_OFF_EXT - 0x0e);
            img[0x0e..NVRAM_OFF_EXT].copy_from_slice(low);
            img[NVRAM_OFF_XRAM..].copy_from_slice(high);
            rtc.load(&img);
            if rtc.file.is_some() {
                rtc.dirty_since = Some(0);
                self.dirty.store(encode(rtc.dirty_since), Ordering::SeqCst);
            }
        }
        r
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

    pub fn load_state(&self, r: &toml::Value, now: u64) -> Result<(), String> {
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
    pub(crate) fn dirty_since(&self) -> Option<u64> {
        self.st.lock().dirty_since
    }
}

impl Device for Ds1685 {
    fn step(&self, _cycles: u64) {}

    /// Start the writer thread.
    fn start(&self) {
        if self.slot.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let slot = self.slot.clone();
        *self.thread.lock() = Some(
            std::thread::Builder::new()
                .name("IOC3-NVRAM".into())
                .spawn(move || Ds1685::writer(slot))
                .expect("spawn ds1685 nvram thread"),
        );
    }

    /// Join the writer and flush the file.
    fn stop(&self) {
        if !self.slot.running.swap(false, Ordering::SeqCst) {
            return;
        }
        // The writer waits on the job slot's lock: take it so the wake-up
        // cannot fall between its check of `running` and its wait.
        drop(self.slot.job.lock());
        self.slot.ready.notify_all();
        if let Some(t) = self.thread.lock().take() {
            let _ = t.join();
        }
        self.flush();
    }

    fn is_running(&self) -> bool {
        self.slot.running.load(Ordering::SeqCst)
    }
}

impl Drop for Ds1685 {
    fn drop(&mut self) {
        Device::stop(&*self);
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn dropping_a_started_part_without_stop_ends_its_writer() {
        let d = Ds1685::new(None, Clock::stepped());
        d.start();
        // Wait for the writer to park holding its reference.
        let t = Instant::now();
        while Arc::strong_count(&d) < 2 && t.elapsed() < Duration::from_millis(500) {
            std::thread::yield_now();
        }
        let part = Arc::downgrade(&d);
        drop(d);
        assert!(part.upgrade().is_none(), "the writer kept the part alive");
    }
}
