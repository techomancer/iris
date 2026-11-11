//! An NVM Express controller on PCI, with one namespace.
//!
//! Written from the NVM Express Base Specification, revision 1.4, and the PCI
//! specifications. The controller registers sit behind BAR0 (a 64-bit memory
//! BAR); the host writes queue entries to host memory, rings a doorbell, and
//! reads completions from memory. The model is the specification's contract
//! and nothing else: no vendor behaviour, no guest-specific shortcuts.
//!
//! What it does: the admin command set and the NVM Read, Write, Write Zeroes
//! and Flush commands, physically contiguous queues, PRP data pointers, one
//! 512-byte-block namespace, pin-based interrupts (the PCI core has no MSI).
//! The namespace is a disk of the shared SCSI target layer (raw image, CHD,
//! copy-on-write overlay), so images, overlays, flushing and snapshots behave
//! as they do for the SCSI disks. Anything else is refused with the status
//! the specification gives an unsupported command, and counted
//! (`nvme refused` in the monitor).
//!
//! Where the model approximates, it says so:
//!
//! * Controller enable, disable and shutdown complete at once; CAP.TO is only
//!   the bound a host may wait. CSTS.RDY is set inside the CC write that sets
//!   CC.EN, so the first read after it sees the controller ready.
//! * Interrupt coalescing is stored and reported, not applied: every
//!   completion raises the pin at once.
//! * The PCI Express and Power Management capabilities are present and
//!   read-only.
//! * The namespace is limited to 2 TiB by the SCSI layer's 32-bit block
//!   address.
//! * The namespace GUID is derived from the image path.
//! * A doorbell value outside its queue is ignored and counted; no
//!   asynchronous error event is raised.
//! * The vendor and device numbers are the ones QEMU's emulated controller
//!   already uses.
//!
//! `log nvme` (or `IRIS_DEBUG_LOG=nvme`) traces what the host does to the
//! controller and what it does back, each line stamped with host time since
//! the controller was made: `state` (CC writes, CSTS changes, resets, fatal
//! errors), `admin`, `io` (doorbells, commands, completions) and `regs`
//! (every register access). Off, each costs two atomic loads.

mod cmd;
mod spec;
#[cfg(test)]
pub(crate) mod tests;

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use crate::devlog::{LogModule, NVME_LOG_ADMIN, NVME_LOG_IO, NVME_LOG_REGS, NVME_LOG_STATE};
use crate::pci::{Bar, DevFn, PciBus, PciConfig, PciDma, PciFunction, PciFunctionState, PciToPciBridge, PCI_INTA};
use crate::scsi::{ScsiBus, ScsiIdentities, ScsiIdentity};
use crate::scsi_monitor::ScsiTargets;
use crate::snapshot::{get_field, hex_u32, hex_u64, toml_u32, toml_u64};
use crate::traits::{Device, Resettable, Saveable};

pub use spec::{BAR_SIZE, MAX_IO_QUEUES};
use spec::*;

/// The controller's PCI identity: the numbers QEMU's emulated controller
/// uses, which generic NVMe drivers match by class as well. Class 01 08 02:
/// mass storage, non-volatile memory, NVM Express.
pub const PCI_VENDOR: u16 = 0x1b36;
pub const PCI_DEVICE: u16 = 0x0010;
pub const PCI_CLASS_NVME: u32 = 0x01_0802;

/// The identity of the bridge a `bridge = true` card sits behind: a DEC
/// 21150, the generic PCI-to-PCI bridge drivers know. Only its type 1 header
/// is modelled.
pub(crate) const BRIDGE_VENDOR: u16 = 0x1011;
pub(crate) const BRIDGE_DEVICE: u16 = 0x0026;

/// A trace line of category `cat` (a `NVME_LOG_*` bit), stamped with the
/// microseconds since the controller was made. Nothing is formatted unless
/// the module is on and the category is in its mask.
macro_rules! trace {
    ($n:expr, $cat:expr, $($arg:tt)*) => {
        if crate::devlog::devlog_is_active(LogModule::Nvme) && crate::devlog::devlog_mask(LogModule::Nvme) & $cat != 0 {
            crate::devlog::devlog().write(
                LogModule::Nvme,
                format_args!("t={}us {}", $n.born.elapsed().as_micros(), format_args!($($arg)*)),
            );
        }
    };
}

const IDENTITY: ScsiIdentity = ScsiIdentity::new(b"IRIS    ", b"NVME NAMESPACE  ", b"1.0 ");

/// The snapshot slot the namespace's disk is recorded under in the manifest:
/// the SCSI controllers have 0 to 31.
pub const SNAPSHOT_SLOT: u8 = 0xf0;

/// The card a `[nvme]` section describes, before it is in a slot.
pub struct Card {
    pub nvme: Arc<Nvme>,
    bridge: bool,
}

impl Card {
    /// The controller with its namespace open.
    pub fn open(cfg: &crate::config::NvmeConfig) -> Result<Card, String> {
        let nvme = Nvme::new();
        let blocks = nvme.attach_image(&cfg.path, cfg.overlay, None).map_err(|e| format!("[nvme]: {e}"))?;
        eprintln!("iris: NVMe {} ({blocks} blocks){}", cfg.path, if cfg.bridge { " behind a PCI bridge" } else { "" });
        Ok(Card { nvme, bridge: cfg.bridge })
    }

    /// Put the card in `device` of `bus`: in the slot itself, or behind a
    /// PCI-to-PCI bridge there, as device 0 of the bridge's bus.
    pub fn fit(&self, bus: &PciBus, device: u8) {
        let at = |device| DevFn { device, function: 0 };
        if self.bridge {
            let ppb = PciToPciBridge::new(bus, at(device), BRIDGE_VENDOR, BRIDGE_DEVICE);
            bus.attach(at(device), ppb.clone());
            ppb.secondary().attach(at(0), self.nvme.clone());
        } else {
            bus.attach(at(device), self.nvme.clone());
        }
    }
}

/// The snapshot manifest's record of the card's disk.
pub fn disk_ref(cfg: &crate::config::NvmeConfig, nvme: &Nvme) -> crate::snapshot::DiskRef {
    crate::snapshot::DiskRef { id: SNAPSHOT_SLOT, path: cfg.path.clone(), size_bytes: nvme.blocks() * 512 }
}

/// A submission queue (5.4).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Sq {
    base: u64,
    entries: u32,
    head: u32,
    tail: u32,
    cqid: u16,
}

/// A completion queue (5.3). `phase` is the tag the next entry carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Cq {
    base: u64,
    entries: u32,
    head: u32,
    tail: u32,
    phase: bool,
    iv: u16,
    ien: bool,
}

impl Cq {
    fn is_full(&self) -> bool {
        (self.tail + 1) % self.entries == self.head
    }

    fn has_unread(&self) -> bool {
        self.head != self.tail
    }
}

/// Feature values, as Get Features returns them (5.21.1).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Features {
    arbitration: u32,
    power_management: u32,
    temperature_threshold: u32,
    error_recovery: u32,
    interrupt_coalescing: u32,
    write_atomicity: u32,
    async_event_config: u32,
    /// The queue counts last set, zero based (NSQA in 15:0, NCQA in 31:16).
    queues: u32,
    /// Number of Queues may be set only before an I/O queue exists.
    queues_set: bool,
}

impl Default for Features {
    fn default() -> Self {
        Features {
            // Arbitration burst: no limit.
            arbitration: 7,
            power_management: 0,
            // 343 K, the over-temperature threshold's usual default.
            temperature_threshold: 0x157,
            error_recovery: 0,
            interrupt_coalescing: 0,
            write_atomicity: 0,
            async_event_config: 0,
            queues: (MAX_IO_QUEUES as u32 - 1) | (MAX_IO_QUEUES as u32 - 1) << 16,
            queues_set: false,
        }
    }
}

/// The counters the SMART / Health log page reports (5.14.1.2), in 512-byte
/// blocks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Counters {
    host_reads: u64,
    host_writes: u64,
    blocks_read: u64,
    blocks_written: u64,
}

/// A completion waiting for room in its queue.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pending {
    sqid: u16,
    cid: u16,
    dw0: u32,
    status: Status,
}

#[derive(Default)]
struct State {
    cc: u32,
    csts: u32,
    intms: u32,
    aqa: u32,
    asq: u64,
    acq: u64,
    /// Queue 0 is the admin queue.
    sqs: Vec<Option<Sq>>,
    cqs: Vec<Option<Cq>>,
    feat: Features,
    /// Command identifiers of the Async Event Request commands waiting for
    /// an event.
    aers: VecDeque<u16>,
    /// Admin completions that found their queue full.
    backlog: VecDeque<Pending>,
    counters: Counters,
    /// Where the round robin over the I/O queues resumes.
    cursor: usize,
    /// The level last driven on the pin.
    pin: bool,
    /// Counts controller resets, so work begun before one is discarded.
    generation: u64,
    /// Commands refused, by (is I/O, opcode); evidence for the monitor.
    refused: BTreeMap<(bool, u8), u64>,
    refused_features: BTreeMap<u8, u64>,
    bad_doorbells: u64,
    /// Admin commands run, and the times the controller failed (CSTS.CFS)
    /// with the last reason. Like the counters they outlive a reset, so the
    /// monitor still shows them after the host has disabled the controller.
    admin_commands: u64,
    fatal: u64,
    fatal_why: &'static str,
}

impl State {
    fn new() -> State {
        State {
            sqs: vec![None; 1 + MAX_IO_QUEUES],
            cqs: vec![None; 1 + MAX_IO_QUEUES],
            ..Default::default()
        }
    }

    fn enabled(&self) -> bool {
        self.cc & CC_EN != 0
    }

    fn fail(&mut self, why: &'static str) {
        self.csts |= CSTS_CFS;
        self.fatal += 1;
        self.fatal_why = why;
    }
}

/// Which reset (7.3.2): a Controller Reset (CC.EN 1 to 0) keeps the admin
/// queue registers; RST#, power-on and a restore do not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reset {
    Controller,
    Conventional,
}

/// One command taken from a submission queue.
struct Job {
    sqid: u16,
    addr: u64,
    generation: u64,
}

pub struct Nvme {
    st: Mutex<State>,
    /// Held by whoever is running commands, so the worker and an inline
    /// doorbell never overlap.
    exec: Mutex<()>,
    pci: PciFunctionState,
    /// The namespace, as the one target of a one-id SCSI bus.
    bus: ScsiBus,
    /// The namespace's size in 512-byte blocks; 0 with no disk attached.
    blocks: AtomicU64,
    nguid: Mutex<[u8; 16]>,
    wake: Arc<(Mutex<bool>, Condvar)>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    running: AtomicBool,
    me: Weak<Self>,
    /// The zero of the trace's timestamps.
    born: Instant,
}

impl Nvme {
    /// A controller with no namespace. Commands run inline until
    /// [`Device::start`] gives it a thread.
    pub fn new() -> Arc<Self> {
        let pm = 0x0003_4801; // Power Management, version 3, next at 0x48
        let pcie = [
            0x0002_0010, // PCI Express, version 2, endpoint
            0,           // device capabilities
            0,           // device control and status
            0x42,        // link capabilities: 5.0 GT/s, x4
            0x0042_0000, // link status: the same
        ];
        let caps: Vec<u32> = [pm, 0].into_iter().chain(pcie).collect();
        Arc::new_cyclic(|me| Nvme {
            st: Mutex::new(State::new()),
            exec: Mutex::new(()),
            pci: PciFunctionState::new(
                PciConfig::new(
                    PCI_VENDOR,
                    PCI_DEVICE,
                    PCI_CLASS_NVME,
                    1,
                    [Bar::Mem64(BAR_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused],
                    PCI_INTA,
                )
                .with_capabilities(16, &caps),
            ),
            bus: ScsiBus::new(1, ScsiIdentities { disk: IDENTITY, cdrom: IDENTITY }),
            blocks: AtomicU64::new(0),
            nguid: Mutex::new([0; 16]),
            wake: Arc::new((Mutex::new(false), Condvar::new())),
            worker: Mutex::new(None),
            running: AtomicBool::new(false),
            me: me.clone(),
            born: Instant::now(),
        })
    }

    /// Give the controller its namespace: the image at `path`, copy-on-write
    /// when `overlay` (to `overlay_path`, or `PATH.overlay`).
    pub fn attach_image(&self, path: &str, overlay: bool, overlay_path: Option<&str>) -> Result<u64, String> {
        let blocks = self
            .bus
            .attach_device(0, path, false, Vec::new(), overlay, overlay_path, false)
            .map_err(|e| format!("could not open {path}: {e}"))?;
        // The SCSI layer addresses blocks in 32 bits.
        if blocks > u64::from(u32::MAX) {
            let _ = self.bus.attach(0, Box::new(crate::scsi::ScsiDevice::new_empty_cdrom(IDENTITY)));
            return Err(format!("{path} is larger than 2 TiB, which the disk layer cannot address"));
        }
        self.blocks.store(blocks, Ordering::Release);
        *self.nguid.lock().unwrap() = nguid_of(path);
        Ok(blocks)
    }

    /// Copy the namespace's overlay into `dir` and return its dirty sectors,
    /// for a snapshot.
    pub fn export_overlays(&self, dir: &std::path::Path) -> std::io::Result<Vec<(usize, Vec<u64>)>> {
        self.bus.export_overlays(dir)
    }

    /// Put back what [`Nvme::export_overlays`] wrote.
    pub fn import_overlays(&self, dir: &std::path::Path, sets: &[(usize, Vec<u64>)]) -> std::io::Result<()> {
        self.bus.import_overlays(dir, sets)
    }

    /// The namespace's size in 512-byte blocks.
    pub fn blocks(&self) -> u64 {
        self.blocks.load(Ordering::Acquire)
    }

    fn dma(&self) -> Option<&Arc<dyn PciDma>> {
        self.pci.dma()
    }

    // ---- registers ------------------------------------------------------

    fn read_reg(&self, off: u32) -> u32 {
        let st = self.st.lock().unwrap();
        match off {
            REG_CAP => CAP as u32,
            0x04 => (CAP >> 32) as u32,
            REG_VS => VERSION,
            REG_INTMS | REG_INTMC => st.intms,
            REG_CC => st.cc,
            REG_CSTS => st.csts,
            REG_AQA => st.aqa,
            REG_ASQ => st.asq as u32,
            0x2c => (st.asq >> 32) as u32,
            REG_ACQ => st.acq as u32,
            0x34 => (st.acq >> 32) as u32,
            // NSSR (CAP.NSSRS is 0), the controller memory buffer (none),
            // and reserved space.
            _ => 0,
        }
    }

    fn write_reg(&self, off: u32, val: u32) {
        // A CC write may reset the controller, which first lets a command
        // being run finish: after the write, nothing the host gave the
        // controller before it moves memory or posts a completion.
        let _exec = (off == REG_CC).then(|| self.exec.lock().unwrap());
        let mut st = self.st.lock().unwrap();
        match off {
            REG_INTMS => {
                st.intms |= val;
                self.update_pin(&mut st);
            }
            REG_INTMC => {
                st.intms &= !val;
                self.update_pin(&mut st);
            }
            REG_CC => self.write_cc(&mut st, val),
            // The admin queue registers are plain storage (3.1.8 to 3.1.10):
            // they are read when CC.EN is next set, and a Controller Reset
            // keeps them.
            REG_AQA => st.aqa = val & 0x0fff_0fff,
            REG_ASQ => st.asq = st.asq & !0xffff_ffff | u64::from(val & !0xfff),
            0x2c => st.asq = st.asq & 0xffff_ffff | u64::from(val) << 32,
            REG_ACQ => st.acq = st.acq & !0xffff_ffff | u64::from(val & !0xfff),
            0x34 => st.acq = st.acq & 0xffff_ffff | u64::from(val) << 32,
            _ => {}
        }
    }

    fn write_cc(&self, st: &mut State, val: u32) {
        let (was, csts) = (st.enabled(), st.csts);
        let val = val & CC_WRITABLE;
        trace!(self, NVME_LOG_STATE, "CC {:#010x} -> {val:#010x} (CSTS {csts:#x})", st.cc);
        if was && val & CC_EN == 0 {
            self.reset_locked(st, Reset::Controller);
        } else {
            st.cc = val;
            if !was && val & CC_EN != 0 {
                self.enable(st);
            } else if was {
                let shn = cc_shn(val);
                if shn != 0 && st.csts & CSTS_SHST_DONE == 0 {
                    // Normal or abrupt shutdown: make what was written durable.
                    let _ = self.bus.flush();
                    st.csts |= CSTS_SHST_DONE;
                }
            }
        }
        if st.csts != csts {
            trace!(self, NVME_LOG_STATE, "CSTS {csts:#x} -> {:#x}", st.csts);
        }
    }

    /// CC.EN 0 to 1 (3.1.5): the admin queues come from AQA, ASQ and ACQ.
    fn enable(&self, st: &mut State) {
        let asqs = (st.aqa & 0xfff) + 1;
        let acqs = (st.aqa >> 16 & 0xfff) + 1;
        let why = if cc_css(st.cc) != 0 {
            Some("enable: CC.CSS is not the NVM command set")
        } else if cc_mps(st.cc) > (CAP >> 52 & 15) as u32 {
            Some("enable: CC.MPS is above CAP.MPSMAX")
        } else if cc_ams(st.cc) != 0 {
            Some("enable: CC.AMS is not round robin")
        } else if cc_iosqes(st.cc) != 6 || cc_iocqes(st.cc) != 4 {
            Some("enable: CC.IOSQES or CC.IOCQES is not 64 or 16 bytes")
        } else if asqs < 2 || acqs < 2 {
            Some("enable: AQA gives an admin queue of fewer than 2 entries")
        } else if st.asq == 0 || st.acq == 0 {
            Some("enable: ASQ or ACQ is zero")
        } else {
            None
        };
        if let Some(why) = why {
            trace!(self, NVME_LOG_STATE, "fatal: {why} (CC {:#x} AQA {:#x} ASQ {:#x} ACQ {:#x})", st.cc, st.aqa, st.asq, st.acq);
            st.fail(why);
            return;
        }
        trace!(self, NVME_LOG_STATE, "ready: admin SQ {:#x} x{asqs}, CQ {:#x} x{acqs}", st.asq, st.acq);
        st.sqs[0] = Some(Sq { base: st.asq, entries: asqs, cqid: 0, ..Default::default() });
        st.cqs[0] = Some(Cq { base: st.acq, entries: acqs, phase: true, ien: true, ..Default::default() });
        st.csts = CSTS_RDY;
    }

    /// A Controller Level Reset (7.3.2): every register and queue goes back
    /// to its power-on value, except that a Controller Reset keeps AQA, ASQ
    /// and ACQ. The PCI header is not touched. Work begun before it is
    /// discarded; the counters and the refusal records stay.
    fn reset_locked(&self, st: &mut State, kind: Reset) {
        trace!(self, NVME_LOG_STATE, "{kind:?} reset (CC {:#x} CSTS {:#x})", st.cc, st.csts);
        let old = std::mem::replace(st, State::new());
        st.generation = old.generation + 1;
        st.pin = old.pin;
        st.counters = old.counters;
        st.refused = old.refused;
        st.refused_features = old.refused_features;
        st.bad_doorbells = old.bad_doorbells;
        st.admin_commands = old.admin_commands;
        st.fatal = old.fatal;
        st.fatal_why = old.fatal_why;
        if kind == Reset::Controller {
            (st.aqa, st.asq, st.acq) = (old.aqa, old.asq, old.acq);
        }
        self.update_pin(st);
    }

    /// Drive the pin: asserted while a completion queue with interrupts
    /// enabled holds an entry the host has not read, and its vector is not
    /// masked (3.1.3, 7.5.1.1). Only vector 0 exists.
    fn update_pin(&self, st: &mut State) {
        let level = Self::pin_level(st);
        if level != st.pin {
            st.pin = level;
            if let Some(irq) = self.pci.irq() {
                irq.set(level);
            }
        }
    }

    /// A doorbell write (3.1.10).
    fn doorbell(&self, off: u32, val: u32) {
        let idx = ((off - DOORBELLS) / 4) as usize;
        let (qid, cq) = (idx / 2, idx & 1 == 1);
        let val = val & 0xffff;
        trace!(self, if qid == 0 { NVME_LOG_ADMIN } else { NVME_LOG_IO }, "doorbell {} {qid} = {val}", if cq { "CQ head" } else { "SQ tail" });
        {
            let mut st = self.st.lock().unwrap();
            if !st.enabled() || st.csts & CSTS_RDY == 0 || qid > MAX_IO_QUEUES {
                trace!(self, NVME_LOG_STATE, "doorbell at {off:#x} ignored: CC {:#x} CSTS {:#x}", st.cc, st.csts);
                return;
            }
            if cq {
                let Some(q) = st.cqs[qid].as_mut() else { return };
                if val >= q.entries {
                    st.bad_doorbells += 1;
                    return;
                }
                q.head = val;
                self.update_pin(&mut st);
            } else {
                let Some(q) = st.sqs[qid].as_mut() else { return };
                if val >= q.entries {
                    st.bad_doorbells += 1;
                    return;
                }
                q.tail = val;
            }
        }
        self.kick();
    }

    // ---- running commands -------------------------------------------------

    /// Tell the worker there may be work; with no worker, do it now.
    fn kick(&self) {
        if self.running.load(Ordering::Acquire) {
            let (m, cv) = &*self.wake;
            *m.lock().unwrap() = true;
            cv.notify_one();
        } else {
            self.process();
        }
    }

    /// Run commands until no queue has one that can complete.
    pub(crate) fn process(&self) {
        let _exec = self.exec.lock().unwrap();
        while let Some(job) = self.take() {
            self.run(job);
        }
    }

    /// The next command to run: the admin queue first, then the I/O queues
    /// in turn. A queue whose completion queue is full waits for the host.
    fn take(&self) -> Option<Job> {
        let mut st = self.st.lock().unwrap();
        self.flush_backlog(&mut st);
        if !st.backlog.is_empty() || st.csts & (CSTS_RDY | CSTS_CFS) != CSTS_RDY {
            return None;
        }
        let (n, cursor) = (st.sqs.len(), st.cursor);
        for qid in std::iter::once(0).chain((0..n - 1).map(|i| 1 + (cursor + i) % (n - 1))) {
            let Some(sq) = st.sqs[qid].as_ref() else { continue };
            if sq.head == sq.tail || st.cqs[sq.cqid as usize].as_ref().is_none_or(Cq::is_full) {
                continue;
            }
            let addr = sq.base + u64::from(sq.head) * SQE_SIZE as u64;
            let sq = st.sqs[qid].as_mut().unwrap();
            sq.head = (sq.head + 1) % sq.entries;
            if qid != 0 {
                st.cursor = qid % (n - 1);
            }
            return Some(Job { sqid: qid as u16, addr, generation: st.generation });
        }
        None
    }

    fn run(&self, job: Job) {
        let mut raw = [0u8; SQE_SIZE];
        if !self.dma().is_some_and(|d| d.dma_read(job.addr, &mut raw).is_ok()) {
            // The queue is not in memory the controller can reach: fatal.
            let mut st = self.st.lock().unwrap();
            if st.generation == job.generation {
                trace!(self, NVME_LOG_STATE, "fatal: SQ {} entry at {:#x} cannot be read", job.sqid, job.addr);
                st.fail("a submission queue entry could not be read");
            }
            return;
        }
        let sqe = Sqe::parse(&raw);
        let cat = if job.sqid == 0 { NVME_LOG_ADMIN } else { NVME_LOG_IO };
        trace!(self, cat, "SQ {} cid {} opcode {:#04x} nsid {} prp {:#x} {:#x} cdw10 {:#x} cdw11 {:#x}", job.sqid, sqe.cid, sqe.opcode, sqe.nsid, sqe.prp1, sqe.prp2, sqe.cdw[0], sqe.cdw[1]);
        let done = if job.sqid == 0 {
            self.st.lock().unwrap().admin_commands += 1;
            self.admin(&sqe)
        } else {
            Some(self.nvm(&sqe))
        };
        if let Some((status, dw0)) = done {
            let mut st = self.st.lock().unwrap();
            if st.generation == job.generation {
                trace!(self, cat, "SQ {} cid {} done: status {:#x} dw0 {dw0:#x}", job.sqid, sqe.cid, status.0);
                self.post(&mut st, job.sqid, sqe.cid, dw0, status);
            } else {
                trace!(self, cat, "SQ {} cid {} done after a reset: dropped", job.sqid, sqe.cid);
            }
        }
    }

    /// Put a completion in the queue of submission queue `sqid` (4.6), or
    /// hold it for room.
    fn post(&self, st: &mut State, sqid: u16, cid: u16, dw0: u32, status: Status) {
        st.backlog.push_back(Pending { sqid, cid, dw0, status });
        self.flush_backlog(st);
    }

    fn flush_backlog(&self, st: &mut State) {
        let mut wrote = false;
        while let Some(p) = st.backlog.front().cloned() {
            let Some(sq) = st.sqs[p.sqid as usize].clone() else {
                st.backlog.pop_front();
                continue;
            };
            let Some(cq) = st.cqs[sq.cqid as usize].as_mut() else {
                st.backlog.pop_front();
                continue;
            };
            if cq.is_full() {
                break;
            }
            let at = cq.base + u64::from(cq.tail) * CQE_SIZE as u64;
            let mut e = [0u8; CQE_SIZE];
            e[0..4].copy_from_slice(&p.dw0.to_le_bytes());
            e[8..12].copy_from_slice(&(sq.head | u32::from(p.sqid) << 16).to_le_bytes());
            let dw3 = u32::from(p.cid) | u32::from(p.status.field() | cq.phase as u16) << 16;
            e[12..16].copy_from_slice(&dw3.to_le_bytes());
            // The dword holding the phase tag goes last, so a host polling
            // it never sees a half-written entry.
            let ok = self.dma().is_some_and(|d| d.dma_write(at, &e[..12]).is_ok() && d.dma_write(at + 12, &e[12..]).is_ok());
            if !ok {
                trace!(self, NVME_LOG_STATE, "fatal: CQ {} entry at {at:#x} cannot be written", sq.cqid);
                st.fail("a completion queue entry could not be written");
                st.backlog.clear();
                break;
            }
            cq.tail += 1;
            if cq.tail == cq.entries {
                cq.tail = 0;
                cq.phase = !cq.phase;
            }
            st.backlog.pop_front();
            wrote = true;
        }
        if wrote {
            self.update_pin(st);
        }
    }

    fn worker_loop(me: Weak<Self>, wake: Arc<(Mutex<bool>, Condvar)>) {
        loop {
            {
                let (m, cv) = &*wake;
                let mut pending = m.lock().unwrap();
                while !*pending {
                    let (g, t) = cv.wait_timeout(pending, Duration::from_millis(50)).unwrap();
                    pending = g;
                    if t.timed_out() {
                        break;
                    }
                }
                *pending = false;
            }
            let Some(s) = me.upgrade() else { return };
            if !s.running.load(Ordering::Acquire) {
                return;
            }
            s.process();
        }
    }

    /// RST# or power-on: the controller returns to its reset state, once a
    /// command being run has finished.
    fn controller_reset(&self) {
        let _exec = self.exec.lock().unwrap();
        let mut st = self.st.lock().unwrap();
        self.reset_locked(&mut st, Reset::Conventional);
    }

    /// The `nvme` monitor command's report.
    fn status_lines(&self) -> Vec<String> {
        let st = self.st.lock().unwrap();
        let mut v = vec![format!(
            "NVMe controller: CC={:#010x} CSTS={:#04x} INTMS={:#x} pin={} namespace {} blocks",
            st.cc,
            st.csts,
            st.intms,
            st.pin as u8,
            self.blocks()
        )];
        for (i, q) in st.sqs.iter().enumerate() {
            if let Some(q) = q {
                v.push(format!("  SQ {i}: base {:#x} {} entries head {} tail {} -> CQ {}", q.base, q.entries, q.head, q.tail, q.cqid));
            }
        }
        for (i, q) in st.cqs.iter().enumerate() {
            if let Some(q) = q {
                v.push(format!("  CQ {i}: base {:#x} {} entries head {} tail {} phase {} ien {}", q.base, q.entries, q.head, q.tail, q.phase as u8, q.ien as u8));
            }
        }
        let c = &st.counters;
        v.push(format!(
            "  reads {} ({} blocks) writes {} ({} blocks) bad doorbell values {}",
            c.host_reads, c.blocks_read, c.host_writes, c.blocks_written, st.bad_doorbells
        ));
        v.push(format!(
            "  admin commands {} fatal errors {}{}",
            st.admin_commands,
            st.fatal,
            if st.fatal == 0 { String::new() } else { format!(" (last: {})", st.fatal_why) }
        ));
        v
    }

    fn refused_lines(&self) -> Vec<String> {
        let st = self.st.lock().unwrap();
        let mut v: Vec<String> = st
            .refused
            .iter()
            .map(|((io, op), n)| format!("{} opcode {op:#04x}: {n}", if *io { "I/O" } else { "admin" }))
            .collect();
        v.extend(st.refused_features.iter().map(|(f, n)| format!("feature {f:#04x}: {n}")));
        if v.is_empty() {
            v.push("nothing refused".into());
        }
        v
    }
}

/// A namespace GUID that is stable for one image path (FNV-1a, two seeds).
fn nguid_of(path: &str) -> [u8; 16] {
    let fnv = |seed: u64| path.bytes().fold(seed, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3));
    let mut g = [0u8; 16];
    g[..8].copy_from_slice(&fnv(0xcbf2_9ce4_8422_2325).to_be_bytes());
    g[8..].copy_from_slice(&fnv(0x8422_2325_cbf2_9ce4).to_be_bytes());
    g
}

impl PciFunction for Nvme {
    fn state(&self) -> &PciFunctionState {
        &self.pci
    }

    fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32 {
        if bar != 0 || off >= BAR_SIZE {
            return 0;
        }
        let dword = if off >= DOORBELLS { 0 } else { self.read_reg(off & !3) };
        let shift = (off & 3) * 8;
        let v = (dword >> shift) & (u32::MAX >> (32 - 8 * width as u32));
        trace!(self, NVME_LOG_REGS, "read {width} at {off:#x} = {v:#x}");
        v
    }

    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize) {
        if bar != 0 || off >= BAR_SIZE {
            return;
        }
        if off >= DOORBELLS {
            self.doorbell(off & !3, val);
            return;
        }
        trace!(self, NVME_LOG_REGS, "write {width} at {off:#x} = {val:#x}");
        let shift = (off & 3) * 8;
        let mask = (u32::MAX >> (32 - 8 * width as u32)) << shift;
        let val = if width == 4 { val } else { (self.read_reg(off & !3) & !mask) | (val << shift & mask) };
        self.write_reg(off & !3, val);
    }

    fn reset(&self) {
        self.controller_reset();
    }
}

impl Device for Nvme {
    fn step(&self, _cycles: u64) {}

    fn get_clock(&self) -> u64 {
        0
    }

    /// Stop the worker; a command it is running finishes first, so a
    /// snapshot sees none half done. Commands run inline until `start`.
    fn stop(&self) {
        let mut worker = self.worker.lock().unwrap();
        self.running.store(false, Ordering::Release);
        if let Some(w) = worker.take() {
            let (m, cv) = &*self.wake;
            *m.lock().unwrap() = true;
            cv.notify_one();
            let _ = w.join();
        }
    }

    /// Give the controller its own thread, so disk I/O no longer runs on the
    /// CPU's.
    fn start(&self) {
        let mut worker = self.worker.lock().unwrap();
        if self.running.swap(true, Ordering::AcqRel) {
            return;
        }
        let (me, wake) = (self.me.clone(), self.wake.clone());
        *worker = Some(
            std::thread::Builder::new()
                .name("NVMe".into())
                .spawn(move || Self::worker_loop(me, wake))
                .expect("nvme: worker thread"),
        );
        // Work that arrived while it was stopped.
        self.kick();
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![(
            "nvme".into(),
            "NVMe controller: nvme status | nvme refused | nvme cow status | nvme cow commit | nvme cow reset".into(),
        )]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write as _;
        if cmd != "nvme" {
            return Err("Command not found".into());
        }
        let lines = match args.first().copied() {
            Some("status") => self.status_lines(),
            Some("refused") => self.refused_lines(),
            Some("cow") => return crate::scsi_monitor::cow_command(self, &args[1..], &mut *w),
            _ => return Err("Usage: nvme status | nvme refused | nvme cow status | nvme cow commit [0] | nvme cow reset [0]".into()),
        };
        for l in lines {
            let _ = writeln!(w, "{l}");
        }
        Ok(())
    }
}

/// The namespace is a SCSI target, so the monitor's `cow` commands, the
/// disks' flush and the snapshot's overlays reach it through its bus.
impl ScsiTargets for Nvme {
    fn scsi_bus(&self) -> &ScsiBus {
        &self.bus
    }
}

/// The controller's registers, queues, features and counters. The namespace's
/// contents go with the snapshot's overlays ([`ScsiBus::export_overlays`]).
/// Save with the worker stopped.
impl Saveable for Nvme {
    fn save_state(&self) -> toml::Value {
        use toml::Value as V;
        let st = self.st.lock().unwrap();
        let mut t = toml::map::Map::new();
        t.insert("cc".into(), hex_u32(st.cc));
        t.insert("csts".into(), hex_u32(st.csts));
        t.insert("intms".into(), hex_u32(st.intms));
        t.insert("aqa".into(), hex_u32(st.aqa));
        t.insert("asq".into(), hex_u64(st.asq));
        t.insert("acq".into(), hex_u64(st.acq));
        let sqs = st.sqs.iter().enumerate().filter_map(|(i, q)| q.as_ref().map(|q| (i, q))).map(|(i, q)| {
            let mut m = toml::map::Map::new();
            m.insert("id".into(), hex_u32(i as u32));
            m.insert("base".into(), hex_u64(q.base));
            m.insert("entries".into(), hex_u32(q.entries));
            m.insert("head".into(), hex_u32(q.head));
            m.insert("tail".into(), hex_u32(q.tail));
            m.insert("cqid".into(), hex_u32(q.cqid.into()));
            V::Table(m)
        });
        t.insert("sqs".into(), V::Array(sqs.collect()));
        let cqs = st.cqs.iter().enumerate().filter_map(|(i, q)| q.as_ref().map(|q| (i, q))).map(|(i, q)| {
            let mut m = toml::map::Map::new();
            m.insert("id".into(), hex_u32(i as u32));
            m.insert("base".into(), hex_u64(q.base));
            m.insert("entries".into(), hex_u32(q.entries));
            m.insert("head".into(), hex_u32(q.head));
            m.insert("tail".into(), hex_u32(q.tail));
            m.insert("phase".into(), V::Boolean(q.phase));
            m.insert("iv".into(), hex_u32(q.iv.into()));
            m.insert("ien".into(), V::Boolean(q.ien));
            V::Table(m)
        });
        t.insert("cqs".into(), V::Array(cqs.collect()));
        let f = &st.feat;
        let feats = [f.arbitration, f.power_management, f.temperature_threshold, f.error_recovery, f.interrupt_coalescing, f.write_atomicity, f.async_event_config, f.queues];
        t.insert("features".into(), V::Array(feats.iter().map(|&v| hex_u32(v)).collect()));
        t.insert("queues_set".into(), V::Boolean(f.queues_set));
        t.insert("aers".into(), V::Array(st.aers.iter().map(|&c| hex_u32(c.into())).collect()));
        let backlog = st.backlog.iter().map(|p| {
            V::Array(vec![hex_u32(p.sqid.into()), hex_u32(p.cid.into()), hex_u32(p.dw0), hex_u32(p.status.0.into())])
        });
        t.insert("backlog".into(), V::Array(backlog.collect()));
        let c = &st.counters;
        t.insert("counters".into(), V::Array([c.host_reads, c.host_writes, c.blocks_read, c.blocks_written].iter().map(|&v| hex_u64(v)).collect()));
        t.insert("cursor".into(), hex_u32(st.cursor as u32));
        V::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut st = self.st.lock().unwrap();
        let generation = st.generation + 1;
        *st = State::new();
        st.generation = generation;
        let u32f = |v: &toml::Value, k: &str| get_field(v, k).and_then(toml_u32).unwrap_or(0);
        let u64f = |v: &toml::Value, k: &str| get_field(v, k).and_then(toml_u64).unwrap_or(0);
        st.cc = u32f(v, "cc");
        st.csts = u32f(v, "csts");
        st.intms = u32f(v, "intms");
        st.aqa = u32f(v, "aqa");
        st.asq = u64f(v, "asq");
        st.acq = u64f(v, "acq");
        let list = |k: &str| get_field(v, k).and_then(|a| a.as_array()).cloned().unwrap_or_default();
        for q in list("sqs") {
            let id = u32f(&q, "id") as usize;
            let slot = st.sqs.get_mut(id).ok_or_else(|| format!("nvme: saved submission queue {id}"))?;
            *slot = Some(Sq { base: u64f(&q, "base"), entries: u32f(&q, "entries"), head: u32f(&q, "head"), tail: u32f(&q, "tail"), cqid: u32f(&q, "cqid") as u16 });
        }
        for q in list("cqs") {
            let id = u32f(&q, "id") as usize;
            let slot = st.cqs.get_mut(id).ok_or_else(|| format!("nvme: saved completion queue {id}"))?;
            *slot = Some(Cq {
                base: u64f(&q, "base"),
                entries: u32f(&q, "entries"),
                head: u32f(&q, "head"),
                tail: u32f(&q, "tail"),
                phase: get_field(&q, "phase").and_then(|b| b.as_bool()).unwrap_or(true),
                iv: u32f(&q, "iv") as u16,
                ien: get_field(&q, "ien").and_then(|b| b.as_bool()).unwrap_or(false),
            });
        }
        let feats: Vec<u32> = list("features").iter().map(|x| toml_u32(x).unwrap_or(0)).collect();
        if feats.len() == 8 {
            st.feat = Features {
                arbitration: feats[0],
                power_management: feats[1],
                temperature_threshold: feats[2],
                error_recovery: feats[3],
                interrupt_coalescing: feats[4],
                write_atomicity: feats[5],
                async_event_config: feats[6],
                queues: feats[7],
                queues_set: get_field(v, "queues_set").and_then(|b| b.as_bool()).unwrap_or(false),
            };
        }
        st.aers = list("aers").iter().map(|x| toml_u32(x).unwrap_or(0) as u16).collect();
        st.backlog = list("backlog")
            .iter()
            .filter_map(|p| p.as_array())
            .filter(|p| p.len() == 4)
            .map(|p| {
                let n = |i: usize| toml_u32(&p[i]).unwrap_or(0);
                Pending { sqid: n(0) as u16, cid: n(1) as u16, dw0: n(2), status: Status(n(3) as u16) }
            })
            .collect();
        let counters: Vec<u64> = list("counters").iter().map(|x| toml_u64(x).unwrap_or(0)).collect();
        if counters.len() == 4 {
            st.counters = Counters { host_reads: counters[0], host_writes: counters[1], blocks_read: counters[2], blocks_written: counters[3] };
        }
        st.cursor = u32f(v, "cursor") as usize;
        // The pin follows the queues; drive it whatever it was.
        st.pin = !Self::pin_level(&st);
        self.update_pin(&mut st);
        Ok(())
    }
}

impl Nvme {
    fn pin_level(st: &State) -> bool {
        st.intms & 1 == 0 && st.cqs.iter().flatten().any(|c| c.ien && c.has_unread())
    }
}

/// The bus resets it ([`PciFunction::reset`]); power-on is a reset.
impl Resettable for Nvme {
    fn power_on(&self) {
        self.controller_reset();
    }
}
