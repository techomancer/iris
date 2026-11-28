//! The SCSI bus: what sits on it, and the table every controller keeps.
//!
//! A controller (the WD33C93A, the AIC-7880, the ISP1040, ISP12160 and
//! ISP2200, and the NVMe controller for its one namespace) is registers and
//! DMA; the targets behind it do not depend on which chip asks. [`ScsiTarget`]
//! is what varies on the bus, and [`ScsiBus`] is the shared mechanism: the
//! id-to-target table with the attach and query methods, which each
//! controller owns one of and reaches its targets through.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, MutexGuard};

use super::{ScsiDevice, ScsiIdentity, ScsiRequest, ScsiResponse};
use crate::block_node::{open_image, ChangerNode, ImageKind};
use crate::dev::daynaport::DaynaPort;

/// A device on a SCSI bus, as the controller drives it: a CDB goes in, data
/// and a status come out.
///
/// A disk or CD-ROM is a [`ScsiDevice`], a DaynaPort Ethernet adapter a
/// [`DaynaPort`]. The contract is `request` and the lifecycle (`power_on`,
/// `start`, `stop`). Everything else is optional behaviour that a
/// kind of target has or does not (a copy-on-write layer, a disc changer),
/// defaulting to "not this target"; the controllers and the monitor ask for
/// the behaviour, never for the type.
pub trait ScsiTarget: Send {
    /// Run one command; `req.data_in` is what a write sends the target.
    fn request(&mut self, req: &ScsiRequest) -> io::Result<ScsiResponse>;

    /// Return to the state a power-up leaves, as the controller's own
    /// reset does.
    fn power_on(&mut self) {}

    /// Start any thread the target runs of its own.
    fn start(&mut self) {}

    /// Stop it, with no command part way through.
    fn stop(&mut self) {}

    /// Take the bus's counters for this target's id, to count the reads and
    /// writes it carries out. A target that moves no blocks ignores them.
    fn set_activity(&mut self, _activity: Arc<DiskActivity>) {}

    /// Read `buf.len()` bytes of the medium at byte `offset`, as the guest
    /// would see them (a copy-on-write overlay or a CHD diff included), for a
    /// front-end that browses a running machine's disk. Read-only; a target
    /// with no block medium says `Unsupported`.
    fn read_media(&mut self, _offset: u64, _buf: &mut [u8]) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "this target has no block medium"))
    }

    /// The medium's size in bytes, `None` with no block medium or no disc.
    fn media_size(&self) -> Option<u64> { None }

    /// A count that moves whenever what [`ScsiTarget::read_media`] returns
    /// may have changed (a write, a disc change, a rolled-back overlay), so
    /// a read made in pieces can tell that it saw one medium throughout.
    fn media_generation(&self) -> u64 { 0 }

    /// Whether READ(6) and WRITE(6) move packets, not blocks, so a
    /// controller must not read their count as a number of sectors.
    fn moves_packets(&self) -> bool { false }

    /// The target's lines in the monitor's `scsi status`, for the target at
    /// `id`.
    fn status(&self, _id: usize) -> Vec<String> { Vec::new() }

    /// The longer report of `scsi dayna`, for a network adapter.
    fn adapter_status(&self) -> Vec<String> { Vec::new() }

    /// The logical block size, for a target that has one.
    fn block_size(&self) -> Option<u64> { None }

    /// A SCSI bus reset reached this target, which owes the initiator a
    /// unit attention on its next command. The ISP12160 calls this when the
    /// guest resets either of its buses, and the Fibre Channel adapter when
    /// the host resets a loop, target or LUN; the WD33C93A, AIC-7880 and
    /// ISP1040 models never told their targets about a reset and still do not.
    fn bus_reset(&mut self) {}

    /// The AIC-7880's bus reset: block size back to what that controller
    /// leaves, with a CD-ROM at 512 bytes if `cdrom_512`. Only the AIC-7880
    /// calls this.
    fn reset_block_size(&mut self, _cdrom_512: bool) {}

    /// Put the mode parameters a guest can change back to what the target
    /// powers up with, as a bus reset does. Every controller that resets its
    /// targets calls this.
    fn reset_mode_parameters(&mut self) {}

    /// Make what the target has accepted so far survive the process: for a
    /// copy-on-write disk, the overlay and the record of which sectors it
    /// holds. Nothing to do for a target with no storage.
    fn flush(&mut self) -> io::Result<()> { Ok(()) }

    // ---- copy-on-write layer -----------------------------------------

    /// Uncommitted sectors, `None` when the target has no such layer.
    fn cow_dirty(&self) -> Option<usize> { None }

    /// What `cow status` says of the layer's uncommitted writes, `None` when
    /// the target has no such layer.
    fn cow_summary(&self) -> Option<String> { None }

    /// Merge the layer into the base image; how much was committed.
    fn cow_commit(&mut self) -> io::Result<usize> { Ok(0) }

    /// Discard the layer's writes.
    fn cow_reset(&mut self) -> io::Result<()> { Ok(()) }

    /// Copy the layer to `dest` and return its dirty sectors; `None` when
    /// there is no layer.
    fn cow_export(&mut self, _dest: &std::path::Path) -> io::Result<Option<Vec<u64>>> { Ok(None) }

    /// Replace the layer with `source`, holding `dirty`.
    fn cow_import(&mut self, _source: &std::path::Path, _dirty: Vec<u64>) -> io::Result<()> { Ok(()) }

    /// (base, diff) of a CHD whose diff holds changes to fold back.
    fn pending_chd_sync(&self) -> Option<(std::path::PathBuf, std::path::PathBuf)> { None }

    /// [`ScsiTarget::pending_chd_sync`], releasing the files so the base can
    /// be rebuilt.
    fn take_pending_chd_sync(&mut self) -> Option<(std::path::PathBuf, std::path::PathBuf)> { None }

    /// Put back what [`ScsiTarget::take_pending_chd_sync`] released, on the
    /// base as the fold left it. A target that released nothing does nothing.
    fn reattach_chd(&mut self) -> io::Result<()> { Ok(()) }

    /// The drive's tray and queue of discs, `None` unless this is a CD-ROM.
    fn changer(&mut self) -> Option<&mut ChangerNode> { None }
}


/// A DaynaPort answers its own vendor command set, which reuses the READ(6)
/// and WRITE(6) opcodes for packet receive and transmit; it answers no
/// storage command at all.
impl ScsiTarget for DaynaPort {
    fn request(&mut self, req: &ScsiRequest) -> io::Result<ScsiResponse> {
        let mut response = DaynaPort::request(self, req)?;
        if let super::ScsiDataLength::Fixed(max_len) = req.data_len {
            response.data.truncate(max_len.min(response.data.len()));
        }
        Ok(response)
    }
    fn power_on(&mut self) { DaynaPort::power_on(self) }
    fn start(&mut self) { DaynaPort::start(self) }
    fn stop(&mut self) { DaynaPort::stop(self) }
    fn moves_packets(&self) -> bool { true }
    fn status(&self, _id: usize) -> Vec<String> { self.status_lines().into_iter().take(1).collect() }
    fn adapter_status(&self) -> Vec<String> { self.status_lines() }
}

/// What a controller's disks and CD-ROMs report in INQUIRY. The command set
/// is the same on every machine; who a machine says its drives are is its
/// own, so the controller gives them to its bus.
#[derive(Clone, Copy, Debug)]
pub struct ScsiIdentities {
    pub disk: ScsiIdentity,
    pub cdrom: ScsiIdentity,
}

/// Reads and writes a target has completed since it was attached, for a
/// front-end's activity light. The counters only go up; whoever shows them
/// compares two samples. Reading takes no lock, so a sample never waits for
/// a command that holds the bus.
#[derive(Default)]
pub struct DiskActivity {
    reads: AtomicU64,
    read_bytes: AtomicU64,
    writes: AtomicU64,
    write_bytes: AtomicU64,
}

/// One sample of a [`DiskActivity`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskCounts {
    /// Completed read commands and the bytes they returned.
    pub reads: u64,
    pub read_bytes: u64,
    /// Completed write commands and the bytes they stored.
    pub writes: u64,
    pub write_bytes: u64,
}

impl DiskActivity {
    pub fn read(&self, bytes: u64) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.read_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn write(&self, bytes: u64) {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.write_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn counts(&self) -> DiskCounts {
        DiskCounts {
            reads: self.reads.load(Ordering::Relaxed),
            read_bytes: self.read_bytes.load(Ordering::Relaxed),
            writes: self.writes.load(Ordering::Relaxed),
            write_bytes: self.write_bytes.load(Ordering::Relaxed),
        }
    }
}

/// The most [`ScsiBus::read_media`] reads under one hold of the bus lock.
pub const READ_MEDIA_CHUNK: usize = 64 * 1024;

/// How many times [`ScsiBus::read_media`] reads in pieces before it holds the
/// bus lock for the whole read instead.
pub const READ_MEDIA_TRIES: usize = 3;

/// The targets on a bus by SCSI id, `None` where nothing answers.
pub type Targets = Vec<Option<Box<dyn ScsiTarget>>>;

/// One SCSI bus: a controller's table of targets and the methods that fill
/// and query it. Locking is the bus's own; the controller holds a bus beside
/// its register state, so a monitor command or a snapshot reaches the
/// targets without the chip's lock.
pub struct ScsiBus {
    identities: ScsiIdentities,
    targets: Mutex<Targets>,
    /// Per-id counters, outside the table's lock; a target keeps the ones
    /// for its id from [`ScsiBus::attach`] on.
    activity: Box<[Arc<DiskActivity>]>,
}

impl ScsiBus {
    /// A bus of `ids` SCSI ids (8 narrow, 16 wide) with nothing on it.
    pub fn new(ids: usize, identities: ScsiIdentities) -> Self {
        Self {
            identities,
            targets: Mutex::new(std::iter::repeat_with(|| None).take(ids).collect()),
            activity: std::iter::repeat_with(Arc::default).take(ids).collect(),
        }
    }

    /// Take the table, for a controller that has to hold it across a check
    /// and a command.
    pub fn lock(&self) -> MutexGuard<'_, Targets> {
        self.targets.lock()
    }

    /// Take the table if no command holds it, for a controller that must not
    /// wait for a disk.
    pub fn try_lock(&self) -> Option<MutexGuard<'_, Targets>> {
        self.targets.try_lock()
    }

    /// Whether the table is held; for the lock registry.
    pub fn is_locked(&self) -> bool {
        self.targets.is_locked()
    }

    /// Run `f` over the table under the lock.
    pub fn with_targets<R>(&self, f: impl FnOnce(&mut [Option<Box<dyn ScsiTarget>>]) -> R) -> R {
        f(&mut self.targets.lock())
    }

    /// Put `target` at `id`, replacing what was there.
    pub fn attach(&self, id: u8, mut target: Box<dyn ScsiTarget>) -> io::Result<()> {
        let mut targets = self.targets.lock();
        let slot = targets.get_mut(id as usize).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("SCSI id {id} out of range"))
        })?;
        target.set_activity(self.activity[id as usize].clone());
        *slot = Some(target);
        Ok(())
    }

    /// Attach what a `[scsi.N]` table describes: the image at `path` (empty
    /// for a CD-ROM with its tray empty), a CD-ROM's changer list `discs`,
    /// and copy-on-write when `overlay` is set, to `overlay_path` or the
    /// default place ([`open_image`]). The bus gives it its INQUIRY
    /// identity. Returns the size in 512-byte blocks.
    pub fn attach_device(
        &self,
        id: u8,
        path: &str,
        cdrom: bool,
        discs: Vec<String>,
        overlay: bool,
        overlay_path: Option<&str>,
        write_cache: bool,
    ) -> io::Result<u64> {
        let identity = if cdrom { self.identities.cdrom } else { self.identities.disk };
        let dev = if cdrom && path.is_empty() {
            // A drive with its tray empty answers INQUIRY and reports no
            // medium; a disc is loaded into it later.
            ScsiDevice::new_empty_cdrom(identity)
        } else {
            let kind = if cdrom { ImageKind::Cdrom } else { ImageKind::Disk };
            let node = open_image(path, kind, overlay, overlay_path)?;
            ScsiDevice::new(node, cdrom, path.to_string(), discs, identity).with_write_cache(write_cache)
        };
        let blocks = dev.size() / 512;
        self.attach(id, Box::new(dev))?;
        Ok(blocks)
    }

    /// A raw image as a disk or CD-ROM, for the controllers' tests.
    #[cfg(test)]
    pub fn attach_image(&self, id: u8, path: &std::path::Path, cdrom: bool) -> io::Result<u64> {
        self.attach_device(id, path.to_str().unwrap(), cdrom, Vec::new(), false, None, false)
    }

    /// A raw image as a disk that powers up with a write cache, for the
    /// controllers' tests.
    #[cfg(test)]
    pub fn attach_cached_image(&self, id: u8, path: &std::path::Path) -> io::Result<()> {
        self.attach_device(id, path.to_str().unwrap(), false, Vec::new(), false, None, true).map(|_| ())
    }

    /// The caching page's WCE bit as MODE SENSE(6) reports it for the disk at
    /// `id`, for the controllers' tests.
    #[cfg(test)]
    pub fn write_cache_reported(&self, id: u8) -> bool {
        let sense = ScsiRequest {
            cdb: vec![super::scsi_cmd::MODE_SENSE_6, 0x08, 0x08, 0, 255, 0],
            data_len: super::ScsiDataLength::Unlimited,
            data_in: None,
        };
        let r = self.request(id, &sense).unwrap().unwrap();
        assert_eq!(r.data[4], 0x08, "caching page");
        r.data[6] & 0x04 != 0
    }

    /// A guest's MODE SELECT(6) of the caching page with WCE `on`, for the
    /// controllers' tests.
    #[cfg(test)]
    pub fn select_write_cache(&self, id: u8, on: bool) {
        let sense = ScsiRequest {
            cdb: vec![super::scsi_cmd::MODE_SENSE_6, 0x08, 0x08, 0, 255, 0],
            data_len: super::ScsiDataLength::Unlimited,
            data_in: None,
        };
        let mut list = vec![0u8; 4];
        list.extend_from_slice(&self.request(id, &sense).unwrap().unwrap().data[4..]);
        list[4 + 2] = if on { 0x04 } else { 0x00 };
        let select = ScsiRequest {
            cdb: vec![super::scsi_cmd::MODE_SELECT_6, 0x10, 0, 0, list.len() as u8, 0],
            data_len: super::ScsiDataLength::Unlimited,
            data_in: Some(list),
        };
        assert_eq!(self.request(id, &select).unwrap().unwrap().status, 0x00);
    }

    /// Whether something answers at `id`.
    /// Every id's read and write counters, by id. Takes no lock: safe from a
    /// status poll while a command holds the bus. An id with nothing on it
    /// stays at zero.
    pub fn activity(&self) -> Vec<DiskCounts> {
        self.activity.iter().map(|a| a.counts()).collect()
    }

    /// Read the medium of the target at `id` (see [`ScsiTarget::read_media`]).
    /// The bus lock is held for at most [`READ_MEDIA_CHUNK`] bytes at a time,
    /// so a large read never stalls the guest's own commands for long. The
    /// read is still one picture of the medium: when the guest changed it
    /// between two pieces, the read starts again, and after
    /// [`READ_MEDIA_TRIES`] such tries it is made under one hold.
    pub fn read_media(&self, id: u8, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if buf.len() > READ_MEDIA_CHUNK {
            for _ in 0..READ_MEDIA_TRIES {
                if self.read_media_in_pieces(id, offset, buf)? {
                    return Ok(());
                }
            }
        }
        let mut targets = self.targets.lock();
        Self::media_target(&mut targets, id)?.read_media(offset, buf)
    }

    /// One try of a read in pieces: `false` when the medium changed under it.
    fn read_media_in_pieces(&self, id: u8, offset: u64, buf: &mut [u8]) -> io::Result<bool> {
        let mut seen = None;
        let mut done = 0usize;
        while done < buf.len() {
            let n = (buf.len() - done).min(READ_MEDIA_CHUNK);
            let mut targets = self.targets.lock();
            let target = Self::media_target(&mut targets, id)?;
            let generation = target.media_generation();
            if seen.is_some_and(|g| g != generation) {
                return Ok(false);
            }
            seen = Some(generation);
            target.read_media(offset + done as u64, &mut buf[done..done + n])?;
            drop(targets);
            done += n;
        }
        Ok(true)
    }

    fn media_target<'a>(targets: &'a mut Targets, id: u8) -> io::Result<&'a mut Box<dyn ScsiTarget>> {
        targets
            .get_mut(id as usize)
            .and_then(|t| t.as_mut())
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no target at SCSI id {id}")))
    }

    /// The size of the medium of the target at `id`.
    pub fn media_size(&self, id: u8) -> Option<u64> {
        self.targets.lock().get(id as usize).and_then(|t| t.as_ref()).and_then(|t| t.media_size())
    }

    pub fn has_target(&self, id: u8) -> bool {
        self.targets.lock().get(id as usize).is_some_and(|t| t.is_some())
    }

    /// Run a command on the target at `id`; `None` when nothing answers.
    pub fn request(&self, id: u8, req: &ScsiRequest) -> Option<io::Result<ScsiResponse>> {
        self.targets.lock().get_mut(id as usize)?.as_mut().map(|t| t.request(req))
    }

    /// Run `f` on every target, with its id.
    pub fn for_each(&self, mut f: impl FnMut(usize, &mut dyn ScsiTarget)) {
        for (id, t) in self.targets.lock().iter_mut().enumerate() {
            if let Some(t) = t {
                f(id, &mut **t);
            }
        }
    }

    /// Flush every target's storage (see [`ScsiTarget::flush`]). Every target
    /// is tried; the first error is returned.
    pub fn flush(&self) -> io::Result<()> {
        let mut first = None;
        self.for_each(|id, t| {
            if let Err(e) = t.flush() {
                eprintln!("iris: flushing SCSI id {id} failed: {e}");
                first.get_or_insert(e);
            }
        });
        first.map_or(Ok(()), Err)
    }

    /// Targets whose `.diff.chd` holds changes waiting to be folded back into
    /// the base on a clean shutdown.
    pub fn pending_chd_sync_count(&self) -> usize {
        let mut n = 0;
        self.for_each(|_, t| n += t.pending_chd_sync().is_some() as usize);
        n
    }

    /// Fold pending CHD diffs on `buses` back into their bases ("sync"),
    /// preserving each base's compression. `only` limits it to one SCSI id
    /// (on every bus); `None` means all. Each target releases its backend
    /// first, closing the CHD; the rebuild then runs with no bus lock held,
    /// since recompression is slow. `progress(done, total, fraction)` reports
    /// per-disk progress; `cancel()` stops before the next disk (the in-flight
    /// rebuild also honours it), leaving every un-synced base and diff intact.
    /// Every released disk is attached again afterwards: a synced one on its
    /// folded base, any other as it was, so the guest can resume.
    /// Returns the count synced.
    pub fn sync_chd_disks(
        buses: &[&ScsiBus],
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> io::Result<usize> {
        let mut pending = Vec::new();
        for bus in buses {
            bus.for_each(|id, t| {
                if only.is_none_or(|o| o == id) {
                    pending.extend(t.take_pending_chd_sync());
                }
            });
        }
        let total = pending.len();
        let mut done = 0usize;
        let mut folded = Ok(());
        for (base, diff) in pending {
            if cancel() {
                break;
            }
            progress(done, total, 0.0);
            folded = crate::chd_disk::flatten_diff(&base, &diff, &mut |f| progress(done, total, f), cancel);
            if folded.is_err() {
                break;
            }
            done += 1;
            progress(done, total, 1.0);
        }
        // Whatever happened, every released disk goes back: folded ones on
        // their new base, the rest on their base and diff as they were, so a
        // guest resumed after a failed or cancelled sync still has its disks.
        let mut reattached = Ok(());
        for bus in buses {
            bus.for_each(|_, t| {
                if let Err(e) = t.reattach_chd() {
                    reattached = Err(e);
                }
            });
        }
        folded.and(reattached)?;
        Ok(done)
    }

    /// Copy each target's copy-on-write overlay into `dir` as
    /// `scsi<id>.overlay`, and return the dirty sectors of each.
    pub fn export_overlays(&self, dir: &std::path::Path) -> io::Result<Vec<(usize, Vec<u64>)>> {
        let mut out = Vec::new();
        let mut err = None;
        self.for_each(|id, t| {
            if err.is_none() {
                match t.cow_export(&dir.join(format!("scsi{id}.overlay"))) {
                    Ok(Some(dirty)) => out.push((id, dirty)),
                    Ok(None) => {}
                    Err(e) => err = Some(e),
                }
            }
        });
        err.map_or(Ok(out), Err)
    }

    /// Put back the overlays [`Self::export_overlays`] wrote. A target with
    /// no entry in `dirty_sets` keeps its overlay.
    pub fn import_overlays(&self, dir: &std::path::Path, dirty_sets: &[(usize, Vec<u64>)]) -> io::Result<()> {
        let mut targets = self.targets.lock();
        for (id, dirty) in dirty_sets {
            if let Some(t) = targets.get_mut(*id).and_then(|t| t.as_mut()) {
                t.cow_import(&dir.join(format!("scsi{id}.overlay")), dirty.clone())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scsi::scsi_cmd;
    use crate::test_temp::TempPath;

    const ID: ScsiIdentity = ScsiIdentity::new(b"TEST    ", b"BUS             ", b"1.0 ");

    fn bus(ids: usize) -> ScsiBus {
        ScsiBus::new(ids, ScsiIdentities { disk: ID, cdrom: ID })
    }

    fn attach(b: &ScsiBus, id: u8, p: &std::path::Path, overlay: bool) -> io::Result<u64> {
        b.attach_device(id, p.to_str().unwrap(), false, Vec::new(), overlay, None, false)
    }

    fn image(dir: &std::path::Path, blocks: usize) -> std::path::PathBuf {
        let p = dir.join("img");
        std::fs::write(&p, vec![0u8; blocks * 512]).unwrap();
        p
    }

    fn inquiry() -> ScsiRequest {
        ScsiRequest {
            cdb: vec![scsi_cmd::INQUIRY, 0, 0, 0, 36, 0],
            data_len: crate::scsi::ScsiDataLength::Unlimited,
            data_in: None,
        }
    }

    #[test]
    fn attach_query_and_answer_with_the_buses_identity() {
        let dir = TempPath::dir("scsibus");
        let b = bus(8);
        let p = image(&dir, 8);
        assert!(!b.has_target(3));
        assert_eq!(attach(&b, 3, &p, false).unwrap(), 8);
        assert!(b.has_target(3));
        let r = b.request(3, &inquiry()).expect("target answers").unwrap();
        assert_eq!(&r.data[8..16], b"TEST    ");
        assert!(b.request(2, &inquiry()).is_none(), "an empty id does not answer");
    }

    /// The read hook returns what the medium holds, across the bus's chunk
    /// size, and refuses an empty id and a read past the end.
    #[test]
    fn read_media_reads_the_disk_in_pieces() {
        let dir = TempPath::dir("scsibus");
        let b = bus(8);
        let p = dir.join("img");
        let blocks = (READ_MEDIA_CHUNK * 3) / 512;
        let data: Vec<u8> = (0..blocks * 512).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&p, &data).unwrap();
        attach(&b, 2, &p, false).unwrap();
        assert_eq!(b.media_size(2), Some(data.len() as u64));
        let mut buf = vec![0u8; READ_MEDIA_CHUNK * 2 + 100];
        b.read_media(2, 300, &mut buf).unwrap();
        assert_eq!(buf, data[300..300 + buf.len()]);
        assert!(b.read_media(5, 0, &mut [0u8; 4]).is_err(), "nothing at id 5");
        assert!(b.read_media(2, data.len() as u64 - 2, &mut [0u8; 4]).is_err(), "past the end");
    }

    /// A medium the "guest" rewrites, every byte to the same new value, after
    /// each of the first `writes` pieces read from it.
    struct Rewritten {
        data: Vec<u8>,
        generation: u64,
        writes: usize,
    }

    impl ScsiTarget for Rewritten {
        fn request(&mut self, _req: &ScsiRequest) -> io::Result<ScsiResponse> {
            unreachable!()
        }
        fn read_media(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            buf.copy_from_slice(&self.data[offset as usize..offset as usize + buf.len()]);
            if self.writes > 0 {
                self.writes -= 1;
                let next = self.data[0].wrapping_add(1);
                self.data.fill(next);
                self.generation += 1;
            }
            Ok(())
        }
        fn media_size(&self) -> Option<u64> {
            Some(self.data.len() as u64)
        }
        fn media_generation(&self) -> u64 {
            self.generation
        }
    }

    /// A read in pieces never mixes two versions of the medium: it starts
    /// again when the medium changed between pieces, and once the guest has
    /// kept changing it for every try, it reads under one hold.
    #[test]
    fn read_media_is_one_picture_of_the_medium() {
        for writes in [0, 1, 2, READ_MEDIA_TRIES, 1000] {
            let b = bus(8);
            b.attach(1, Box::new(Rewritten { data: vec![0; READ_MEDIA_CHUNK * 4], generation: 0, writes })).unwrap();
            let mut buf = vec![0xEEu8; READ_MEDIA_CHUNK * 3 + 5];
            b.read_media(1, 7, &mut buf).unwrap();
            assert!(buf.iter().all(|&x| x == buf[0]), "{writes} writes: a torn read");
        }
    }

    /// A disk's generation stays put through reads and moves with a write.
    #[test]
    fn a_write_moves_the_media_generation() {
        let dir = TempPath::dir("scsibus");
        let b = bus(8);
        attach(&b, 2, &image(&dir, 8), false).unwrap();
        let generation = |b: &ScsiBus| b.with_targets(|t| t[2].as_ref().unwrap().media_generation());
        let req = |cdb: Vec<u8>, data_in| ScsiRequest { cdb, data_len: crate::scsi::ScsiDataLength::Unlimited, data_in };
        let before = generation(&b);
        b.request(2, &req(vec![scsi_cmd::READ_10, 0, 0, 0, 0, 0, 0, 0, 1, 0], None)).unwrap().unwrap();
        b.request(2, &inquiry()).unwrap().unwrap();
        assert_eq!(generation(&b), before, "reads leave it");
        b.request(2, &req(vec![scsi_cmd::WRITE_10, 0, 0, 0, 0, 0, 0, 0, 1, 0], Some(vec![1; 512]))).unwrap().unwrap();
        assert_ne!(generation(&b), before, "a write moves it");
    }

    /// A controller's register path can look at the bus while a command
    /// holds it, and gets nothing rather than waiting.
    #[test]
    fn try_lock_gives_nothing_while_a_command_holds_the_bus() {
        let b = bus(8);
        assert!(b.try_lock().is_some());
        let held = b.lock();
        assert!(b.try_lock().is_none());
        drop(held);
        assert!(b.try_lock().is_some());
    }

    #[test]
    fn an_id_the_bus_does_not_have_is_refused() {
        let dir = TempPath::dir("scsibus");
        let b = bus(8);
        let p = image(&dir, 1);
        let e = attach(&b, 8, &p, false).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
        assert!(attach(&bus(16), 8, &p, false).is_ok(), "a wide bus has id 8");
    }

    /// A disk's write_cache setting reaches the disk: it decides the WCE
    /// bit the guest sees.
    #[test]
    fn attach_device_gives_the_disk_its_configured_write_cache() {
        let dir = TempPath::dir("scsibus");
        let b = bus(8);
        let p = image(&dir, 8);
        for (id, on) in [(1, false), (2, true)] {
            b.attach_device(id, p.to_str().unwrap(), false, Vec::new(), false, None, on).unwrap();
            assert_eq!(b.write_cache_reported(id), on, "write_cache = {on}");
        }
    }

    fn write_block(lba: u32, byte: u8) -> ScsiRequest {
        let l = lba.to_be_bytes();
        ScsiRequest {
            cdb: vec![scsi_cmd::WRITE_10, 0, l[0], l[1], l[2], l[3], 0, 0, 1, 0],
            data_len: crate::scsi::ScsiDataLength::Unlimited,
            data_in: Some(vec![byte; 512]),
        }
    }

    fn read_block(lba: u32) -> ScsiRequest {
        let l = lba.to_be_bytes();
        ScsiRequest {
            cdb: vec![scsi_cmd::READ_10, 0, l[0], l[1], l[2], l[3], 0, 0, 1, 0],
            data_len: crate::scsi::ScsiDataLength::Unlimited,
            data_in: None,
        }
    }

    /// A front-end's disk light: completed reads and writes count on the
    /// id's own counters, a refused command and other ids do not, and a
    /// sample needs no lock, so it works while a command holds the bus.
    #[test]
    fn activity_counts_completed_reads_and_writes_per_id() {
        let dir = TempPath::dir("scsiact");
        let b = bus(8);
        attach(&b, 1, &image(&dir, 8), false).unwrap();
        assert_eq!(b.activity(), vec![DiskCounts::default(); 8]);

        b.request(1, &write_block(2, 0x55)).unwrap().unwrap();
        b.request(1, &read_block(2)).unwrap().unwrap();
        b.request(1, &read_block(3)).unwrap().unwrap();
        // Past the end: CHECK CONDITION, nothing moved.
        assert_ne!(b.request(1, &read_block(99)).unwrap().unwrap().status, 0);
        b.request(1, &inquiry()).unwrap().unwrap();

        let held = b.lock();
        let counts = b.activity();
        drop(held);
        assert_eq!(counts[1], DiskCounts { reads: 2, read_bytes: 1024, writes: 1, write_bytes: 512 });
        assert!(counts.iter().enumerate().all(|(id, c)| id == 1 || *c == DiskCounts::default()));
    }

    /// The machine can exit without dropping the bus (`process::exit` after
    /// a guest power-off), so [`ScsiBus::flush`] is what the next run's
    /// overlay depends on. `mem::forget` stands in for that exit.
    #[test]
    fn flushed_overlay_writes_survive_an_exit_that_skips_drop() {
        let dir = TempPath::dir("scsibus");
        let p = image(&dir, 4);
        let first = bus(8);
        attach(&first, 1, &p, true).unwrap();
        for (lba, byte) in [(1, 0x5A), (3, 0xC3)] {
            first.request(1, &write_block(lba, byte)).unwrap().unwrap();
        }
        first.flush().unwrap();
        std::mem::forget(first);

        let second = bus(8);
        attach(&second, 1, &p, true).unwrap();
        for (lba, byte) in [(0, 0x00), (1, 0x5A), (2, 0x00), (3, 0xC3)] {
            let r = second.request(1, &read_block(lba)).unwrap().unwrap();
            assert_eq!(r.data, vec![byte; 512], "block {lba}");
        }
        drop(second);
    }

    /// A guest's SYNCHRONIZE CACHE is its way of asking for the same.
    #[test]
    fn synchronize_cache_flushes_the_overlay() {
        let dir = TempPath::dir("scsibus");
        let p = image(&dir, 4);
        let first = bus(8);
        attach(&first, 1, &p, true).unwrap();
        first.request(1, &write_block(2, 0x77)).unwrap().unwrap();
        let sync = ScsiRequest {
            cdb: vec![scsi_cmd::SYNCHRONIZE_CACHE_10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            data_len: crate::scsi::ScsiDataLength::Unlimited,
            data_in: None,
        };
        assert_eq!(first.request(1, &sync).unwrap().unwrap().status, 0);
        std::mem::forget(first);

        let second = bus(8);
        attach(&second, 1, &p, true).unwrap();
        let r = second.request(1, &read_block(2)).unwrap().unwrap();
        assert_eq!(r.data, vec![0x77; 512]);
        drop(second);
    }

    #[test]
    fn overlays_export_and_import_by_id() {
        // The directory comes first so the bus, which writes its overlay's record
        // as it drops, goes before it.
        let dir = TempPath::dir("scsibus");
        let p = image(&dir, 4);
        let b = bus(8);
        attach(&b, 2, &p, true).unwrap();
        let ov = TempPath::dir("scsibus-ov");
        let sets = b.export_overlays(&ov).unwrap();
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].0, 2);
        b.import_overlays(&ov, &sets).unwrap();
    }

    /// A compressed CHD in `dir` attached at id 4 of a fresh bus, with
    /// sector 2 written to 0x5A through the diff.
    fn chd_bus_with_a_write(dir: &std::path::Path) -> (std::path::PathBuf, ScsiBus) {
        use libchdman_rs::hd::{create_from_reader, HdCreateOptions};
        use libchdman_rs::CHD_CODEC_ZLIB;
        let base = dir.join("base.chd");
        let logical = 128 * 1024u64;
        create_from_reader(
            std::io::Cursor::new(vec![0xABu8; logical as usize]),
            &base,
            HdCreateOptions {
                logical_size: logical,
                hunk_size: 4096,
                unit_size: 512,
                codecs: [CHD_CODEC_ZLIB, 0, 0, 0],
                geometry: None,
                ident: None,
            },
            &mut |_| {},
            &|| false,
        )
        .unwrap();
        let b = bus(8);
        attach(&b, 4, &base, false).unwrap();
        assert_eq!(b.pending_chd_sync_count(), 0, "nothing written yet");
        let write = ScsiRequest {
            cdb: vec![scsi_cmd::WRITE_10, 0, 0, 0, 0, 2, 0, 0, 1, 0],
            data_len: crate::scsi::ScsiDataLength::Unlimited,
            data_in: Some(vec![0x5Au8; 512]),
        };
        b.request(4, &write).unwrap().unwrap();
        assert_eq!(b.pending_chd_sync_count(), 1);
        (base, b)
    }

    /// What the disk at id 4 answers to READ(10) of sector 2.
    fn sector_2(b: &ScsiBus) -> Vec<u8> {
        let read = ScsiRequest {
            cdb: vec![scsi_cmd::READ_10, 0, 0, 0, 0, 2, 0, 0, 1, 0],
            data_len: crate::scsi::ScsiDataLength::Unlimited,
            data_in: None,
        };
        let r = b.request(4, &read).unwrap().unwrap();
        assert_eq!(r.status, 0, "the disk is attached and answers");
        r.data
    }

    #[test]
    fn a_chd_diff_is_counted_and_folded_on_every_bus_given() {
        let dir = TempPath::dir("scsibus");
        let (_, b1) = chd_bus_with_a_write(&dir);
        let b0 = bus(8);
        assert_eq!(b0.pending_chd_sync_count(), 0);
        let n = ScsiBus::sync_chd_disks(&[&b0, &b1], None, &mut |_, _, _| {}, &|| false).unwrap();
        assert_eq!(n, 1);
        assert_eq!(b1.pending_chd_sync_count(), 0, "folded");
        assert!(
            !crate::chd_disk::diff_path_for(&dir.join("base.chd")).exists(),
            "no empty diff is made until the disk is used again"
        );
        assert_eq!(sector_2(&b1), vec![0x5Au8; 512], "the folded disk is back for a resumed guest");
    }

    /// `cow status` counts a raw overlay's sectors, and says only whether a
    /// CHD diff holds changes, as it has no count to give.
    #[test]
    fn cow_summary_counts_sectors_only_where_there_are_sectors() {
        let dir = TempPath::dir("scsibus-summary");
        let (_, b) = chd_bus_with_a_write(&dir);
        let summary = |b: &ScsiBus, id: u8| {
            let mut s = None;
            b.for_each(|i, t| if i == id as usize { s = t.cow_summary() });
            s
        };
        assert_eq!(summary(&b, 4).as_deref(), Some("uncommitted changes"));
        let raw = image(&dir, 4);
        attach(&b, 1, &raw, true).unwrap();
        assert_eq!(summary(&b, 1).as_deref(), Some("0 dirty sectors"));
        b.request(1, &write_block(2, 0x77)).unwrap().unwrap();
        assert_eq!(summary(&b, 1).as_deref(), Some("1 dirty sector"));
        b.request(1, &write_block(3, 0x77)).unwrap().unwrap();
        assert_eq!(summary(&b, 1).as_deref(), Some("2 dirty sectors"));
    }

    #[test]
    fn a_cancelled_sync_leaves_every_disk_attached_with_its_writes() {
        let dir = TempPath::dir("scsibus-cancel");
        let (_, b) = chd_bus_with_a_write(&dir);
        let n = ScsiBus::sync_chd_disks(&[&b], None, &mut |_, _, _| {}, &|| true).unwrap();
        assert_eq!(n, 0);
        assert_eq!(b.pending_chd_sync_count(), 1, "still waiting to be folded");
        assert_eq!(sector_2(&b), vec![0x5Au8; 512]);
    }

    #[test]
    fn a_failed_fold_leaves_the_disk_attached_with_its_writes() {
        let dir = TempPath::dir("scsibus-fail");
        let (base, b) = chd_bus_with_a_write(&dir);
        // The rebuild cannot create its temporary file where a directory is.
        let mut tmp = base.as_os_str().to_owned();
        tmp.push(".synctmp.chd");
        std::fs::create_dir(&tmp).unwrap();
        assert!(ScsiBus::sync_chd_disks(&[&b], None, &mut |_, _, _| {}, &|| false).is_err());
        assert_eq!(b.pending_chd_sync_count(), 1, "still waiting to be folded");
        assert_eq!(sector_2(&b), vec![0x5Au8; 512]);
    }
}
