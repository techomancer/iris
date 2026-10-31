//! Layered block-node interface for the disk backends.
//!
//! This module is the single block-device abstraction — [`BlockNode`] — with
//! one implementation per disk backend ([`RawNode`], [`CowNode`], [`ChdNode`],
//! [`ChdCdNode`]). It replaces the former `crate::scsi::DiskBackend` enum:
//! callers hold a `Box<dyn BlockNode>` and never match on a backend kind.
//! SCSI stays LBA- and sector-oriented at the top; a node is byte-addressed and
//! owns the backend-specific arithmetic.
//!
//! The shape follows QEMU's block graph: every image is a node with an optional
//! `backing` node, and the one query that makes copy-on-write consistent across
//! formats is [`BlockNode::block_status`] — "does this layer hold these bytes,
//! and if not, where do I read them from?". Commit / rollback / dirty-count are
//! methods on that same contract.
//!
//! Nodes are byte-addressed and operate on byte ranges. The raw and overlay
//! layers use 512-byte sectors internally; reads and writes may be unaligned,
//! and a partial-sector write copies the rest of the sector up through the
//! backing chain first (read-modify-write), exactly as a real overlay must.
//!
//! # Dirty, commit, and rollback
//!
//! A node may sit over an optional *divergence layer*: a sparse `.overlay` file
//! beside a raw base, or a MAME `.diff.chd` beside a CHD base. Writes land in
//! that layer and reads fall through it. **Dirty has exactly one meaning across
//! every backend: the node holds writes that are not yet part of its durable
//! base.** Precisely, a node is dirty when its divergence layer holds at least
//! one sector that shadows the base it was opened over.
//!
//! [`BlockNode::dirty_count`] reports the size of that divergence in the units
//! the backend tracks. A raw overlay tracks its sectors exactly, so it returns
//! the number of dirty 512-byte sectors. A CHD diff is only exposed by MAME as a
//! whole-image flag, not a per-sector map, so it returns `1` while any change is
//! held and `0` otherwise. Callers that need the predicate rather than a size
//! test `dirty_count() > 0`, or [`BlockNode::is_cow`] for "has a divergence
//! layer at all".
//!
//! [`BlockNode::commit`] folds the layer into the base — raw copies its dirty
//! sectors into the base in place; CHD rebuilds the base from the merged
//! parent+diff view — and leaves the node with an empty layer.
//! [`BlockNode::rollback`] discards the layer, reverting reads to the base. A
//! node with no divergence layer has nothing to fold or discard, so the trait
//! defaults are no-ops.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::chd_disk::{ChdCd, ChdHd};

/// Sector granularity shared by the raw and overlay layers. Matches the SCSI
/// disk sector size and [`crate::cow_disk`]'s private constant.
pub const SECTOR_BYTES: usize = 512;
/// [`SECTOR_BYTES`] as a `u64`, for offset arithmetic.
pub const SECTOR_SIZE: u64 = SECTOR_BYTES as u64;

/// What one layer knows about the allocation of a byte range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockState {
    /// The range is allocated in this layer and holds real data.
    Data,
    /// The range is not held by this layer and there is no backing to fall
    /// through to, so it reads as zeroes.
    Zero,
    /// The range is not held by this layer; read it from [`BlockNode::backing`]
    /// at [`BlockStatus::backing_offset`].
    Allocated,
}

/// The answer to a [`BlockNode::block_status`] query.
///
/// A status describes a homogeneous run beginning at the queried offset, so the
/// answer always makes sense even when the requested range straddles an
/// allocation boundary: `len` is the number of bytes from the query offset for
/// which this `state` holds. Callers that need a whole range covered iterate
/// `offset += status.len` until they have consumed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockStatus {
    /// The allocation state of the run.
    pub state: BlockState,
    /// For [`BlockState::Allocated`], the byte offset to read from in the
    /// backing node. `None` for `Data` and `Zero`.
    pub backing_offset: Option<u64>,
    /// Bytes from the query offset for which `state` holds. Never zero for a
    /// non-empty query.
    pub len: u64,
}

impl BlockStatus {
    /// A run of real data held by this layer.
    pub const fn data(len: u64) -> Self {
        Self { state: BlockState::Data, backing_offset: None, len }
    }

    /// A run that reads as zeroes because no layer holds it.
    pub const fn zero(len: u64) -> Self {
        Self { state: BlockState::Zero, backing_offset: None, len }
    }

    /// A run served by the backing node at `backing_offset`.
    pub const fn allocated(backing_offset: u64, len: u64) -> Self {
        Self { state: BlockState::Allocated, backing_offset: Some(backing_offset), len }
    }
}

/// A byte-addressed node in a layered block graph.
///
/// Implementations are the raw image, the copy-on-write overlay, the hard-disk
/// CHD and the CD CHD. A node owns its storage; it is `Send` (the SCSI worker
/// already moves its backend across threads) but not `Sync`.
///
/// See the module docs for the single definition of dirty, commit and rollback
/// that every backend shares.
pub trait BlockNode: Send {
    /// Capacity in bytes.
    fn size(&self) -> u64;

    /// Read exactly `buf.len()` bytes starting at byte `offset`.
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Write exactly `buf.len()` bytes starting at byte `offset`.
    fn write(&mut self, offset: u64, buf: &[u8]) -> io::Result<()>;

    /// Push buffered writes out to durable storage.
    fn flush(&mut self) -> io::Result<()>;

    /// Discard `len` bytes at `offset`, releasing the layer's claim on them.
    /// A later read falls through to the backing node (or zeroes, if none).
    fn discard(&mut self, offset: u64, len: u64) -> io::Result<()>;

    /// Report the allocation state of `[offset, offset + len)`.
    fn block_status(&self, offset: u64, len: u64) -> io::Result<BlockStatus>;

    /// The node that serves ranges this layer does not hold, if any.
    fn backing(&self) -> Option<&dyn BlockNode>;

    /// Size of this node's uncommitted divergence, in the units the backend
    /// tracks: exact dirty 512-byte sectors for a raw overlay, or a coarse
    /// `1`/`0` for a CHD diff. `0` when nothing is uncommitted or there is no
    /// divergence layer. See the module docs for the one dirty meaning.
    fn dirty_count(&self) -> usize {
        0
    }

    /// Whether this node writes through a divergence layer (a raw `.overlay` or
    /// a CHD `.diff.chd`) that [`BlockNode::commit`] or
    /// [`BlockNode::rollback`] can act on. `false` for a plainly-written image.
    fn is_cow(&self) -> bool {
        false
    }

    /// Fold this node's divergence into its durable base and empty the layer.
    /// Returns the number of dirty units applied (raw: sectors; CHD: `1` when a
    /// diff was folded, `0` when there was nothing to fold). A no-op for a node
    /// with no divergence layer.
    fn commit(&mut self) -> io::Result<usize> {
        Ok(0)
    }

    /// Discard this node's divergence, reverting reads to the base. A no-op for
    /// a node with no divergence layer.
    fn rollback(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// `(base, overlay)` paths for this node's divergence layer, if any.
    fn overlay_paths(&self) -> Option<(PathBuf, PathBuf)> {
        None
    }

    /// `(base, overlay)` when a clean exit should auto-fold this node's diff
    /// back into the base. A CHD in copy-on-write mode keeps changes separate
    /// and returns `None`; a compressed CHD with COW off returns `Some`. Always
    /// `None` for a node with no divergence layer.
    fn pending_sync(&self) -> Option<(PathBuf, PathBuf)> {
        None
    }

    /// Copy this node's divergence layer to `dest` and return its dirty sector
    /// set (sorted), for snapshot capture. Raw overlays only; other backends
    /// leave no file and return an empty list.
    fn export_overlay(&mut self, _dest: &Path) -> io::Result<Vec<u64>> {
        Ok(Vec::new())
    }

    /// Replace this node's divergence layer with `source` and adopt `dirty` as
    /// its sector set, for snapshot restore. Raw overlays only; a no-op
    /// elsewhere.
    fn import_overlay(&mut self, _source: &Path, _dirty: Vec<u64>) -> io::Result<()> {
        Ok(())
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.to_string())
}

/// Check that `[offset, offset + len)` lies within a node of `size` bytes.
fn check_range(size: u64, offset: u64, len: u64) -> io::Result<()> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| invalid("block range overflows u64"))?;
    if end > size {
        return Err(invalid("block range past end of node"));
    }
    Ok(())
}

/// The sector range `[first, last)` covering `[offset, offset + len)`.
fn sectors_for(offset: u64, len: u64) -> (u64, u64) {
    let first = offset / SECTOR_SIZE;
    let end = offset + len;
    let last = (end + SECTOR_SIZE - 1) / SECTOR_SIZE;
    (first, last)
}

/// A plain raw image file: every byte is real data held by this layer.
pub struct RawNode {
    file: File,
    size: u64,
}

impl RawNode {
    /// Open `path` read-write and use its current length as the capacity.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let size = file.metadata()?.len();
        Ok(Self { file, size })
    }

    /// Open `path` read-only. Used for removable read-only media (CD-ROM ISOs)
    /// and for a copy-on-write base, whose contents the overlay never mutates
    /// through this node.
    pub fn open_readonly(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        let size = file.metadata()?.len();
        Ok(Self { file, size })
    }

    /// Wrap an already-open file handle. `size` is the node's capacity; the
    /// caller has the file positioned/opened as it wants.
    pub fn from_file(file: File, size: u64) -> Self {
        Self { file, size }
    }
}

impl BlockNode for RawNode {
    fn size(&self) -> u64 {
        self.size
    }

    fn read(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        check_range(self.size, offset, buf.len() as u64)?;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(buf)
    }

    fn write(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        check_range(self.size, offset, buf.len() as u64)?;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.sync_all()
    }

    fn discard(&mut self, offset: u64, len: u64) -> io::Result<()> {
        check_range(self.size, offset, len)?;
        // Raw images do not track allocation, so a discard is a zero-fill: the
        // bytes are still "data" to this layer, they just read as zeroes.
        let zeros = vec![0u8; len as usize];
        self.write(offset, &zeros)
    }

    fn block_status(&self, offset: u64, len: u64) -> io::Result<BlockStatus> {
        check_range(self.size, offset, len)?;
        Ok(BlockStatus::data(len))
    }

    fn backing(&self) -> Option<&dyn BlockNode> {
        None
    }
}

/// A copy-on-write overlay: writes are redirected to a sparse overlay file and
/// unheld ranges read through to the backing node.
pub struct CowNode {
    backing: Option<Box<dyn BlockNode>>,
    /// Base image path, so [`CowNode::commit`] can merge the overlay back into
    /// it. Derived from a `.overlay` suffix unless set explicitly.
    base_path: Option<PathBuf>,
    overlay: File,
    overlay_path: PathBuf,
    dirty: HashSet<u64>,
    /// Set by anything that changes the overlay or its sector set, cleared by
    /// a flush, so a flush with nothing new to write costs nothing.
    unflushed: bool,
    size: u64,
}

/// The base image path implied by a `….overlay` path, if any.
fn derive_base_path(overlay_path: &Path) -> Option<PathBuf> {
    overlay_path
        .to_str()
        .and_then(|s| s.strip_suffix(".overlay"))
        .map(PathBuf::from)
}

impl CowNode {
    /// Open (creating if needed) an overlay at `overlay_path` presenting `size`
    /// bytes over `backing`. The overlay starts with no held sectors, so every
    /// range reads through to the backing until first written.
    pub fn new(
        backing: Option<Box<dyn BlockNode>>,
        overlay_path: impl AsRef<Path>,
        size: u64,
    ) -> io::Result<Self> {
        let overlay_path = overlay_path.as_ref().to_path_buf();
        let overlay = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&overlay_path)?;
        // Recover the dirty set persisted by a previous run. The sidecar is the
        // load-bearing record of which sectors the host finished writing —
        // sparse bytes in the overlay alone cannot be trusted.
        let sidecar = crate::cow_disk::sidecar_path_for(&overlay_path);
        let dirty = crate::cow_disk::load_dirty_sidecar(&sidecar).unwrap_or_default();
        let base_path = derive_base_path(&overlay_path);
        let base_label = base_path
            .as_deref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<backing>".to_string());
        eprintln!(
            "iris: COW overlay active (base: {}, overlay: {}, dirty sectors: {})",
            base_label,
            overlay_path.display(),
            dirty.len()
        );
        if dirty.is_empty() && std::fs::metadata(&overlay_path).map(|m| m.len()).unwrap_or(0) > 0 {
            eprintln!("iris: note: overlay file has data but no .dirty sidecar — prior writes are not in use");
        }
        eprintln!(
            "iris: to reset disk to clean state, delete {} and {}",
            overlay_path.display(),
            sidecar.display()
        );
        Ok(Self { backing, base_path, overlay, overlay_path, dirty, unflushed: false, size })
    }

    /// Record the base image path explicitly. `add_device` knows it; `new`
    /// otherwise derives it from the `.overlay` suffix.
    pub fn with_base_path(mut self, base_path: impl Into<PathBuf>) -> Self {
        self.base_path = Some(base_path.into());
        self
    }

    /// Path of the overlay file (for diagnostics and, later, commit/export).
    pub fn overlay_path(&self) -> &Path {
        &self.overlay_path
    }

    fn sidecar_path(&self) -> PathBuf {
        crate::cow_disk::sidecar_path_for(&self.overlay_path)
    }

    /// Read one whole sector: the overlay if it holds it, else the backing, else
    /// zeroes.
    fn read_sector(&mut self, lba: u64) -> io::Result<[u8; SECTOR_BYTES]> {
        let mut buf = [0u8; SECTOR_BYTES];
        if self.dirty.contains(&lba) {
            self.overlay.seek(SeekFrom::Start(lba * SECTOR_SIZE))?;
            self.overlay.read_exact(&mut buf)?;
        } else if let Some(backing) = self.backing.as_mut() {
            backing.read(lba * SECTOR_SIZE, &mut buf)?;
        }
        Ok(buf)
    }

    /// The state of one sector, ignoring the requested range.
    fn sector_state(&self, lba: u64) -> BlockState {
        if self.dirty.contains(&lba) {
            BlockState::Data
        } else if self.backing.is_some() {
            BlockState::Allocated
        } else {
            BlockState::Zero
        }
    }
}

impl Drop for CowNode {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            eprintln!(
                "iris: COW flush on drop failed for {}: {} (writes may be lost)",
                self.overlay_path.display(),
                e
            );
        }
    }
}

impl BlockNode for CowNode {
    fn size(&self) -> u64 {
        self.size
    }

    fn read(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        check_range(self.size, offset, buf.len() as u64)?;
        if buf.is_empty() {
            return Ok(());
        }
        let end = offset + buf.len() as u64;
        let (first, last) = sectors_for(offset, buf.len() as u64);
        for lba in first..last {
            let sector = self.read_sector(lba)?;
            let s_off = lba * SECTOR_SIZE;
            let start = offset.max(s_off);
            let stop = end.min(s_off + SECTOR_SIZE);
            let src = (start - s_off) as usize;
            let dst = (start - offset) as usize;
            let n = (stop - start) as usize;
            buf[dst..dst + n].copy_from_slice(&sector[src..src + n]);
        }
        Ok(())
    }

    fn write(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        check_range(self.size, offset, buf.len() as u64)?;
        if buf.is_empty() {
            return Ok(());
        }
        let (first, last) = sectors_for(offset, buf.len() as u64);
        for lba in first..last {
            // Copy-on-write does read-modify-write: pull the current sector up
            // through the chain so the untouched bytes survive, then hand the
            // whole sector to the overlay.
            let mut sector = self.read_sector(lba)?;
            let s_off = lba * SECTOR_SIZE;
            let start = offset.max(s_off);
            let stop = (offset + buf.len() as u64).min(s_off + SECTOR_SIZE);
            let src = (start - offset) as usize;
            let dst = (start - s_off) as usize;
            let n = (stop - start) as usize;
            sector[dst..dst + n].copy_from_slice(&buf[src..src + n]);

            self.overlay.seek(SeekFrom::Start(lba * SECTOR_SIZE))?;
            self.overlay.write_all(&sector)?;
            self.dirty.insert(lba);
            self.unflushed = true;
        }
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.unflushed {
            return Ok(());
        }
        self.overlay.sync_all()?;
        crate::cow_disk::save_dirty_sidecar(&self.sidecar_path(), &self.dirty)?;
        // The backing is never written through this node, so it has nothing
        // of ours to flush.
        self.unflushed = false;
        Ok(())
    }

    fn discard(&mut self, offset: u64, len: u64) -> io::Result<()> {
        check_range(self.size, offset, len)?;
        if len == 0 {
            return Ok(());
        }
        let (first, last) = sectors_for(offset, len);
        for lba in first..last {
            self.dirty.remove(&lba);
            self.unflushed = true;
        }
        Ok(())
    }

    fn block_status(&self, offset: u64, len: u64) -> io::Result<BlockStatus> {
        check_range(self.size, offset, len)?;
        if len == 0 {
            return Ok(BlockStatus::data(0));
        }
        let (first, _) = sectors_for(offset, len);
        let state = self.sector_state(first);
        // Extend the run while the next sector has the same state and still
        // overlaps the requested range.
        let end = offset + len;
        let mut lba = first;
        loop {
            let next = lba + 1;
            if next * SECTOR_SIZE >= end || self.sector_state(next) != state {
                break;
            }
            lba = next;
        }
        let run_end = ((lba + 1) * SECTOR_SIZE).min(end);
        let covered = run_end - offset;
        Ok(match state {
            BlockState::Data => BlockStatus::data(covered),
            BlockState::Zero => BlockStatus::zero(covered),
            BlockState::Allocated => BlockStatus::allocated(offset, covered),
        })
    }

    fn backing(&self) -> Option<&dyn BlockNode> {
        self.backing.as_deref()
    }

    fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    fn is_cow(&self) -> bool {
        true
    }

    /// Merge every dirty overlay sector into the base image, then truncate the
    /// overlay so it is empty. Mirrors [`crate::cow_disk::CowDisk::commit`].
    fn commit(&mut self) -> io::Result<usize> {
        let base_path = self.base_path.clone().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "cannot determine base path from overlay path",
            )
        })?;
        let mut base_rw = OpenOptions::new().read(true).write(true).open(&base_path)?;
        let mut buf = [0u8; SECTOR_BYTES];
        let mut committed = 0usize;
        for &lba in &self.dirty {
            self.overlay.seek(SeekFrom::Start(lba * SECTOR_SIZE))?;
            self.overlay.read_exact(&mut buf)?;
            base_rw.seek(SeekFrom::Start(lba * SECTOR_SIZE))?;
            base_rw.write_all(&buf)?;
            committed += 1;
        }
        base_rw.sync_all()?;
        self.dirty.clear();
        self.unflushed = true;
        self.overlay.set_len(0)?;
        eprintln!("iris: COW committed {} sectors to {}", committed, base_path.display());
        Ok(committed)
    }

    /// Delete every overlay write and clear the on-disk dirty set. Mirrors
    /// [`crate::cow_disk::CowDisk::reset_overlay`].
    fn rollback(&mut self) -> io::Result<()> {
        self.dirty.clear();
        self.unflushed = true;
        self.overlay.set_len(0)?;
        self.overlay.seek(SeekFrom::Start(0))?;
        let _ = std::fs::remove_file(self.sidecar_path());
        Ok(())
    }

    /// Copy the current overlay to `dest` and return the dirty sector list
    /// (sorted). Mirrors [`crate::cow_disk::CowDisk::export_overlay`].
    fn export_overlay(&mut self, dest: &Path) -> io::Result<Vec<u64>> {
        self.overlay.sync_all()?;
        crate::cow_disk::reflink_or_copy(&self.overlay_path, dest)?;
        let mut dirty: Vec<u64> = self.dirty.iter().copied().collect();
        dirty.sort_unstable();
        Ok(dirty)
    }

    /// Replace the overlay contents with `source` and adopt `dirty`. Mirrors
    /// [`crate::cow_disk::CowDisk::import_overlay`].
    fn import_overlay(&mut self, source: &Path, dirty: Vec<u64>) -> io::Result<()> {
        if source.exists() {
            crate::cow_disk::reflink_or_copy(source, &self.overlay_path)?;
        } else {
            std::fs::File::create(&self.overlay_path)?;
        }
        self.overlay = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.overlay_path)?;
        self.dirty = dirty.into_iter().collect();
        self.unflushed = true;
        Ok(())
    }
}

/// A hard-disk CHD. The underlying [`ChdHd`] already presents the merged
/// parent + diff view, so this layer holds every byte it reports — there is no
/// backing node.
pub struct ChdNode {
    /// The CHD handle. `None` only transiently, while [`BlockNode::commit`] or
    /// [`BlockNode::rollback`] has closed it so `flatten_diff` can atomically
    /// rebuild the base; it is restored before the operation returns.
    hd: Option<ChdHd>,
    sector_size: u64,
    /// Written since the CHD was last closed and opened again (see
    /// [`BlockNode::flush`] below).
    unflushed: bool,
}

impl ChdNode {
    /// Open a hard-disk CHD. `cow` is the per-disk copy-on-write toggle, threaded
    /// straight through to [`ChdHd::open`].
    pub fn open(path: &str, cow: bool) -> io::Result<Self> {
        let hd = ChdHd::open(path, cow)?;
        let sector_size = u64::from(hd.sector_size());
        Ok(Self { hd: Some(hd), sector_size, unflushed: false })
    }

    fn hd(&self) -> &ChdHd {
        self.hd.as_ref().expect("CHD handle present outside commit/rollback")
    }

    fn hd_mut(&mut self) -> &mut ChdHd {
        self.hd.as_mut().expect("CHD handle present outside commit/rollback")
    }
}

impl BlockNode for ChdNode {
    fn size(&self) -> u64 {
        self.hd().size()
    }

    fn read(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let ss = self.sector_size;
        if offset % ss != 0 || (buf.len() as u64) % ss != 0 {
            return Err(invalid("CHD reads must be sector-aligned"));
        }
        check_range(self.hd().size(), offset, buf.len() as u64)?;
        let data = self.hd_mut().read_blocks(offset / ss, buf.len() / (ss as usize), ss)?;
        buf.copy_from_slice(&data);
        Ok(())
    }

    fn write(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        let ss = self.sector_size;
        if offset % ss != 0 || (buf.len() as u64) % ss != 0 {
            return Err(invalid("CHD writes must be sector-aligned"));
        }
        check_range(self.hd().size(), offset, buf.len() as u64)?;
        self.unflushed = true;
        self.hd_mut().write_sectors(offset / ss, buf)
    }

    /// libchdman has no flush: the writes it has accepted are all in its file
    /// only once it is closed, and a node that is leaked, as at a
    /// `process::exit`, never closes it (the last hunks written come back
    /// unwritten). So close the CHD and open it again, which is the one way to
    /// make them durable. It costs a reopen only when something was written
    /// since the last flush.
    fn flush(&mut self) -> io::Result<()> {
        if !self.unflushed {
            return Ok(());
        }
        let hd = self.hd.take().expect("CHD handle present outside commit/rollback");
        let base = hd.base_path().to_str().ok_or_else(|| invalid("non-UTF-8 CHD path"))?.to_string();
        let cow = hd.is_cow();
        drop(hd); // closes the file, writing out what libchdman held back
        self.hd = Some(ChdHd::open(&base, cow)?);
        self.unflushed = false;
        Ok(())
    }

    fn discard(&mut self, offset: u64, len: u64) -> io::Result<()> {
        check_range(self.hd().size(), offset, len)?;
        let zeros = vec![0u8; len as usize];
        self.write(offset, &zeros)
    }

    fn block_status(&self, offset: u64, len: u64) -> io::Result<BlockStatus> {
        check_range(self.hd().size(), offset, len)?;
        Ok(BlockStatus::data(len))
    }

    fn backing(&self) -> Option<&dyn BlockNode> {
        None
    }

    fn dirty_count(&self) -> usize {
        usize::from(self.hd().diff_dirty())
    }

    fn is_cow(&self) -> bool {
        self.hd().overlay_paths().is_some()
    }

    fn overlay_paths(&self) -> Option<(PathBuf, PathBuf)> {
        self.hd().overlay_paths()
    }

    fn pending_sync(&self) -> Option<(PathBuf, PathBuf)> {
        self.hd().pending_sync()
    }

    /// Rebuild the base from the merged parent+diff view and reopen a fresh
    /// overlay. The handle is closed first — `flatten_diff` atomically renames
    /// the rebuilt base — and reopened afterwards, so the node stays usable. On
    /// a flatten error the base and diff are left intact and the diff is
    /// reattached.
    fn commit(&mut self) -> io::Result<usize> {
        if !self.hd().diff_dirty() {
            return Ok(0);
        }
        let (base, diff) = self
            .hd()
            .overlay_paths()
            .expect("a dirty CHD reports overlay paths");
        let cow = self.hd().is_cow();
        let base_str = base
            .to_str()
            .ok_or_else(|| invalid("non-UTF-8 CHD path"))?
            .to_string();

        self.hd = None; // close so flatten_diff can rename over the base
        let folded = crate::chd_disk::flatten_diff(&base, &diff, &mut |_| {}, &|| false);
        // base + diff are intact even when the fold failed, so either way the
        // disk can be reopened; surface the fold error after restoring the node.
        self.hd = Some(ChdHd::open(&base_str, cow)?);
        self.unflushed = false;
        folded?;
        Ok(1)
    }

    /// Discard the diff (every uncommitted write) and reopen a fresh overlay
    /// over the untouched base. A CHD with no diff has nothing to roll back.
    fn rollback(&mut self) -> io::Result<()> {
        let Some((base, diff)) = self.hd().overlay_paths() else {
            return Ok(());
        };
        let cow = self.hd().is_cow();
        let base_str = base
            .to_str()
            .ok_or_else(|| invalid("non-UTF-8 CHD path"))?
            .to_string();

        self.hd = None; // close before reopening over the (untouched) base
        let _ = std::fs::remove_file(&diff);
        self.hd = Some(ChdHd::open(&base_str, cow)?);
        self.unflushed = false;
        Ok(())
    }
}

/// A CD-ROM CHD (single-track MODE1). The cooked reader presents a flat,
/// byte-addressed, read-only stream — writes are rejected. It holds every byte
/// it reports, so there is no backing node.
pub struct ChdCdNode {
    reader: ChdCd,
    size: u64,
}

impl ChdCdNode {
    /// Open a CD CHD read-only.
    pub fn open(path: &str) -> io::Result<Self> {
        let reader = ChdCd::open(path)?;
        let size = reader.size();
        Ok(Self { reader, size })
    }
}

impl BlockNode for ChdCdNode {
    fn size(&self) -> u64 {
        self.size
    }

    fn read(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        check_range(self.size, offset, buf.len() as u64)?;
        self.reader.read_at(offset, buf)
    }

    fn write(&mut self, _offset: u64, _buf: &[u8]) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::PermissionDenied, "CD CHD is read-only"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn discard(&mut self, _offset: u64, _len: u64) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::PermissionDenied, "CD CHD is read-only"))
    }

    fn block_status(&self, offset: u64, len: u64) -> io::Result<BlockStatus> {
        check_range(self.size, offset, len)?;
        Ok(BlockStatus::data(len))
    }

    fn backing(&self) -> Option<&dyn BlockNode> {
        None
    }
}

/// What an image is mounted as, which decides how its file is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageKind {
    Disk,
    Cdrom,
}

/// Open the image at `path` as a node: a CHD (hard disk or CD), a
/// copy-on-write overlay over it when `overlay` (hard disks), or the file
/// itself. `overlay_path` names the overlay file, `PATH.overlay` when `None`.
///
/// Everything that mounts an image goes through here, so a disk attached at
/// start-up and a disc swapped in later are read the same way.
pub fn open_image(
    path: &str,
    kind: ImageKind,
    overlay: bool,
    overlay_path: Option<&str>,
) -> io::Result<Box<dyn BlockNode>> {
    let cdrom = kind == ImageKind::Cdrom;
    Ok(if crate::chd_disk::is_chd(path) {
        if cdrom {
            Box::new(ChdCdNode::open(path)?)
        } else {
            // The `overlay` flag is the per-disk copy-on-write toggle: COW on
            // → always overlay (even an uncompressed base gets a diff) and
            // never auto-fold on exit (commit/roll back manually); COW off →
            // write in place (uncompressed) or a diff that auto-folds on a
            // clean exit (compressed).
            Box::new(ChdNode::open(path, overlay)?)
        }
    } else if overlay && !cdrom {
        let overlay_path = overlay_path.map(str::to_string).unwrap_or_else(|| format!("{path}.overlay"));
        let backing = RawNode::open_readonly(Path::new(path))?;
        let size = backing.size();
        Box::new(CowNode::new(Some(Box::new(backing)), &overlay_path, size)?.with_base_path(path))
    } else if cdrom {
        Box::new(RawNode::open_readonly(Path::new(path))?)
    } else {
        Box::new(RawNode::open(Path::new(path))?)
    })
}

/// Media that can be taken out and put back: a CD-ROM drive's tray.
///
/// This is a trait of its own, not part of [`BlockNode`], because removable
/// media is not always a block device. A tape drive's cartridge is removable
/// too, but a tape grows as it is written and is read in order, so it can
/// implement this without being a node.
pub trait Removable {
    /// Whether a medium is in.
    fn present(&self) -> bool;

    /// Whether the medium changed since this was last asked. Asking clears
    /// it, so a caller reports a change once.
    fn take_changed(&mut self) -> bool;

    /// Take the medium out. Returns what the drive loads in its place, `None`
    /// when it is left empty.
    fn eject(&mut self) -> io::Result<Option<String>>;

    /// Put the image at `path` in, as a hand swapping the medium would.
    fn load(&mut self, path: &str) -> io::Result<()>;
}

/// A CD-ROM drive's tray, with the discs queued for it.
///
/// The queue is paths; only the active disc is open. It is a node that
/// forwards to the active disc, so the storage commands neither know nor care
/// that the disc can change, and a drive with its tray empty is a node of
/// size 0 whose reads fail. The queue editing is the changer's own, not
/// [`Removable`]'s: a tape has no queue.
pub struct ChangerNode {
    /// The discs, the active one first.
    discs: Vec<String>,
    /// The active disc, `None` with the tray empty.
    active: Option<Box<dyn BlockNode>>,
    changed: bool,
}

impl ChangerNode {
    /// A drive with its tray empty and nothing queued.
    pub fn empty() -> Self {
        Self { discs: Vec::new(), active: None, changed: false }
    }

    /// A drive holding `node`, the image at `path`, with `discs` queued
    /// behind it. `path` is first in the queue.
    pub fn loaded(node: Box<dyn BlockNode>, path: &str, mut discs: Vec<String>) -> Self {
        if discs.first().map(String::as_str) != Some(path) {
            discs.retain(|d| d != path);
            discs.insert(0, path.to_string());
        }
        Self { discs, active: Some(node), changed: false }
    }

    /// The active disc's path, `None` with the tray empty.
    pub fn current(&self) -> Option<&str> {
        self.active.as_ref().and_then(|_| self.discs.first()).map(String::as_str)
    }

    /// The queue, the active disc first.
    pub fn discs(&self) -> &[String] {
        &self.discs
    }

    /// Take the disc out and leave the tray empty, keeping the queue.
    pub fn unload(&mut self) {
        self.active = None;
        self.changed = true;
    }

    /// Queue `path` to follow the active disc. The file has to exist; it is
    /// not opened until it is loaded.
    pub fn add(&mut self, path: String) -> Result<(), String> {
        if !Path::new(&path).exists() {
            return Err(format!("File not found: {}", path));
        }
        let at = if self.discs.is_empty() { 0 } else { 1 };
        self.discs.insert(at, path);
        Ok(())
    }

    /// Drop the queued disc at `ordinal`. Dropping the active disc (0) takes
    /// it off the list but leaves it in the drive.
    pub fn remove(&mut self, ordinal: usize) -> Result<String, String> {
        if ordinal >= self.discs.len() {
            return Err(format!("Ordinal {} out of range (0..{})", ordinal, self.discs.len()));
        }
        Ok(self.discs.remove(ordinal))
    }

    /// Make the queued disc at `ordinal` the one after the active disc.
    pub fn move_next(&mut self, ordinal: usize) -> Result<(), String> {
        if ordinal == 0 {
            return Err("Ordinal 0 is the active disc; use 'scsi eject' to advance".to_string());
        }
        if ordinal >= self.discs.len() {
            return Err(format!("Ordinal {} out of range (0..{})", ordinal, self.discs.len()));
        }
        if self.discs.len() < 2 {
            return Ok(());
        }
        let path = self.discs.remove(ordinal);
        self.discs.insert(1, path);
        Ok(())
    }

    fn disc(&mut self) -> io::Result<&mut Box<dyn BlockNode>> {
        self.active
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, "no disc in the drive"))
    }
}

impl Removable for ChangerNode {
    fn present(&self) -> bool {
        self.active.is_some()
    }

    fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    /// With none or one disc queued the tray empties and the queue is cleared;
    /// with more, the queue turns and the next disc loads. If the next disc
    /// will not open, the drive is left as it was.
    fn eject(&mut self) -> io::Result<Option<String>> {
        if self.discs.len() <= 1 {
            if let Some(prev) = self.current() {
                eprintln!("CD-ROM: ejected '{}', tray empty", prev);
            }
            self.unload();
            self.discs.clear();
            return Ok(None);
        }
        let node = open_image(&self.discs[1], ImageKind::Cdrom, false, None)?;
        let gone = self.discs.remove(0);
        self.discs.push(gone);
        self.active = Some(node);
        self.changed = true;
        let next = self.discs[0].clone();
        eprintln!("CD-ROM changer: switched to '{}'", next);
        Ok(Some(next))
    }

    /// The disc becomes the active one at the front of the queue. With an
    /// empty tray it is the only entry; with one disc loaded that disc stays
    /// queued behind it, so an eject turns the queue rather than emptying the
    /// tray; with more, a disc already queued moves to the front instead of
    /// appearing twice.
    fn load(&mut self, path: &str) -> io::Result<()> {
        let node = open_image(path, ImageKind::Cdrom, false, None)?;
        self.active = Some(node);
        self.changed = true;
        match self.discs.len() {
            0 => self.discs.push(path.to_string()),
            1 => {
                if self.discs[0] != path {
                    self.discs.insert(0, path.to_string());
                }
            }
            _ => {
                self.discs.retain(|d| d != path);
                self.discs.insert(0, path.to_string());
            }
        }
        Ok(())
    }
}

impl BlockNode for ChangerNode {
    fn size(&self) -> u64 {
        self.active.as_ref().map_or(0, |d| d.size())
    }
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.disc()?.read(offset, buf)
    }
    fn write(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.disc()?.write(offset, buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.active.as_mut().map_or(Ok(()), |d| d.flush())
    }
    fn discard(&mut self, offset: u64, len: u64) -> io::Result<()> {
        self.disc()?.discard(offset, len)
    }
    fn block_status(&self, offset: u64, len: u64) -> io::Result<BlockStatus> {
        match &self.active {
            Some(d) => d.block_status(offset, len),
            None => Err(io::Error::new(io::ErrorKind::NotConnected, "no disc in the drive")),
        }
    }
    fn backing(&self) -> Option<&dyn BlockNode> {
        self.active.as_deref()
    }
    fn dirty_count(&self) -> usize {
        self.active.as_ref().map_or(0, |d| d.dirty_count())
    }
    fn is_cow(&self) -> bool {
        self.active.as_ref().is_some_and(|d| d.is_cow())
    }
    fn commit(&mut self) -> io::Result<usize> {
        self.active.as_mut().map_or(Ok(0), |d| d.commit())
    }
    fn rollback(&mut self) -> io::Result<()> {
        self.active.as_mut().map_or(Ok(()), |d| d.rollback())
    }
    fn overlay_paths(&self) -> Option<(PathBuf, PathBuf)> {
        self.active.as_ref().and_then(|d| d.overlay_paths())
    }
    fn pending_sync(&self) -> Option<(PathBuf, PathBuf)> {
        self.active.as_ref().and_then(|d| d.pending_sync())
    }
    fn export_overlay(&mut self, dest: &Path) -> io::Result<Vec<u64>> {
        self.active.as_mut().map_or(Ok(Vec::new()), |d| d.export_overlay(dest))
    }
    fn import_overlay(&mut self, source: &Path, dirty: Vec<u64>) -> io::Result<()> {
        self.active.as_mut().map_or(Ok(()), |d| d.import_overlay(source, dirty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp_path(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "iris-blocknode-{}-{}-{}",
            tag,
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn raw_filled(byte: u8, len: usize) -> (RawNode, PathBuf) {
        let path = tmp_path("raw");
        std::fs::write(&path, vec![byte; len]).unwrap();
        (RawNode::open(&path).unwrap(), path)
    }

    /// Build a 256 KiB compressed hard-disk CHD full of `byte`, and return its
    /// path. MAME's v5 compression map asserts below one hunk, so keep it large.
    fn compressed_chd(byte: u8) -> PathBuf {
        use libchdman_rs::hd::{create_from_reader, HdCreateOptions};
        use libchdman_rs::CHD_CODEC_ZLIB;
        use std::io::Cursor;

        let base = tmp_path("chd").with_extension("chd");
        let logical = 256 * 1024u64;
        create_from_reader(
            Cursor::new(vec![byte; logical as usize]),
            &base,
            HdCreateOptions {
                logical_size: logical,
                hunk_size: 4096,
                unit_size: SECTOR_BYTES as u32,
                codecs: [CHD_CODEC_ZLIB, 0, 0, 0],
                geometry: None,
                ident: None,
            },
            &mut |_| {},
            &|| false,
        )
        .unwrap();
        base
    }

    fn cleanup(overlay: &Path) {
        let _ = std::fs::remove_file(overlay);
        let _ = std::fs::remove_file(crate::cow_disk::sidecar_path_for(overlay));
    }

    /// libchdman has no flush: what it has accepted is in the file only once
    /// it is closed, which a node left behind by `process::exit` never is.
    /// `mem::forget` stands in for that exit; the write must still be there
    /// when the CHD is opened again, both in a diff and with the COW flag, at
    /// the start, middle and end of the disk.
    #[test]
    fn chd_flushed_writes_survive_an_exit_that_skips_drop() {
        for cow in [false, true] {
            for lba in [0u64, 9, 511] {
                let base = compressed_chd(0xAB);
                let mut node = ChdNode::open(base.to_str().unwrap(), cow).unwrap();
                node.write(lba * SECTOR_SIZE, &[0x5A; SECTOR_BYTES]).unwrap();
                node.flush().unwrap();
                std::mem::forget(node);

                let mut node = ChdNode::open(base.to_str().unwrap(), cow).unwrap();
                let mut got = [0u8; SECTOR_BYTES];
                node.read(lba * SECTOR_SIZE, &mut got).unwrap();
                assert_eq!(got, [0x5A; SECTOR_BYTES], "cow={cow} lba={lba}");
                drop(node);
                let _ = std::fs::remove_file(crate::chd_disk::diff_path_for(&base));
                let _ = std::fs::remove_file(base);
            }
        }
    }

    #[test]
    fn raw_roundtrips_and_reports_data() {
        let (mut node, path) = raw_filled(0x00, 4 * SECTOR_BYTES);

        node.write(SECTOR_SIZE, &[0x11u8; SECTOR_BYTES]).unwrap();
        let mut got = [0u8; SECTOR_BYTES];
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0x11u8; SECTOR_BYTES]);

        let status = node.block_status(SECTOR_SIZE, SECTOR_SIZE).unwrap();
        assert_eq!(status.state, BlockState::Data);
        assert_eq!(status.backing_offset, None);
        assert_eq!(status.len, SECTOR_SIZE);
        assert!(node.backing().is_none(), "a raw image has no backing");

        node.discard(SECTOR_SIZE, SECTOR_SIZE).unwrap();
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0x00u8; SECTOR_BYTES], "discard zero-fills a raw node");

        node.flush().unwrap();
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn raw_rejects_ranges_past_the_end() {
        let (mut node, path) = raw_filled(0, 2 * SECTOR_BYTES);
        let mut buf = [0u8; SECTOR_BYTES];
        assert!(node.read(SECTOR_SIZE, &mut buf).is_ok());
        assert!(node.read(2 * SECTOR_SIZE, &mut buf).is_err());
        assert!(node.block_status(SECTOR_SIZE, SECTOR_SIZE + 1).is_err());
        let _ = std::fs::remove_file(path);
    }

    /// The process can exit without dropping the node (`process::exit` after a
    /// guest power-off), so a flush is all that carries the writes over to
    /// the next run. `mem::forget` stands in for that exit.
    #[test]
    fn cow_flushed_writes_survive_an_exit_that_skips_drop() {
        let (backing, backing_path) = raw_filled(0xAA, 8 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-exit");
        let size = 8 * SECTOR_SIZE;
        let mut expect = vec![0xAAu8; size as usize];

        let mut node = CowNode::new(Some(Box::new(backing)), &overlay_path, size).unwrap();
        // Not sector aligned, across a sector boundary.
        let span = vec![0x5Au8; 700];
        node.write(100, &span).unwrap();
        expect[100..800].copy_from_slice(&span);
        // A written sector of zeros is data, not a hole in the overlay file.
        node.write(6 * SECTOR_SIZE, &[0u8; SECTOR_BYTES]).unwrap();
        expect[6 * SECTOR_BYTES..7 * SECTOR_BYTES].fill(0);
        // The last sector, written in two halves.
        node.write(7 * SECTOR_SIZE, &[0x11u8; 256]).unwrap();
        node.write(7 * SECTOR_SIZE + 256, &[0x22u8; 256]).unwrap();
        expect[7 * SECTOR_BYTES..7 * SECTOR_BYTES + 256].fill(0x11);
        expect[7 * SECTOR_BYTES + 256..8 * SECTOR_BYTES].fill(0x22);

        node.flush().unwrap();
        std::mem::forget(node);

        let backing = RawNode::open_readonly(&backing_path).unwrap();
        let mut node = CowNode::new(Some(Box::new(backing)), &overlay_path, size).unwrap();
        assert_eq!(node.dirty_count(), 4, "sectors 0, 1, 6 and 7 come back as held");
        let mut got = vec![0u8; size as usize];
        node.read(0, &mut got).unwrap();
        assert!(got == expect, "reopened overlay does not read back what was written");

        drop(node);
        cleanup(&overlay_path);
        let _ = std::fs::remove_file(backing_path);
    }

    /// A guest can issue SYNCHRONIZE CACHE often; a flush with nothing new
    /// must not rewrite the sector record.
    #[test]
    fn cow_flush_with_nothing_new_leaves_the_record_alone() {
        let (backing, backing_path) = raw_filled(0, 4 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-idle");
        let mut node = CowNode::new(Some(Box::new(backing)), &overlay_path, 4 * SECTOR_SIZE).unwrap();
        let sidecar = crate::cow_disk::sidecar_path_for(&overlay_path);

        node.write(0, &[1u8; SECTOR_BYTES]).unwrap();
        node.flush().unwrap();
        assert!(sidecar.exists());
        std::fs::remove_file(&sidecar).unwrap();
        node.flush().unwrap();
        assert!(!sidecar.exists(), "an idle flush rewrote the record");

        node.discard(0, SECTOR_SIZE).unwrap();
        node.flush().unwrap();
        assert!(sidecar.exists(), "a discard is a change the record must carry");

        drop(node);
        cleanup(&overlay_path);
        let _ = std::fs::remove_file(backing_path);
    }

    #[test]
    fn cow_unwritten_range_reads_through_to_backing() {
        let (backing, backing_path) = raw_filled(0xAA, 4 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-overlay");
        let mut node =
            CowNode::new(Some(Box::new(backing)), &overlay_path, 4 * SECTOR_SIZE).unwrap();

        assert!(node.backing().is_some(), "the overlay sits on a backing node");
        assert_eq!(node.dirty_count(), 0);

        // An unwritten range reads through to the backing and reports that it is
        // allocated somewhere below this layer.
        let mut got = [0u8; SECTOR_BYTES];
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0xAAu8; SECTOR_BYTES]);
        let status = node.block_status(SECTOR_SIZE, SECTOR_SIZE).unwrap();
        assert_eq!(status.state, BlockState::Allocated);
        assert_eq!(status.backing_offset, Some(SECTOR_SIZE));
        assert_eq!(status.len, SECTOR_SIZE);

        let _ = std::fs::remove_file(backing_path);
        let _ = std::fs::remove_file(overlay_path);
    }

    #[test]
    fn cow_written_range_reads_overlay_and_reports_data() {
        let (backing, backing_path) = raw_filled(0xAA, 4 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-overlay");
        let mut node =
            CowNode::new(Some(Box::new(backing)), &overlay_path, 4 * SECTOR_SIZE).unwrap();

        node.write(SECTOR_SIZE, &[0x55u8; SECTOR_BYTES]).unwrap();
        assert_eq!(node.dirty_count(), 1);

        let mut got = [0u8; SECTOR_BYTES];
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0x55u8; SECTOR_BYTES], "the overlay shadows the backing");
        let status = node.block_status(SECTOR_SIZE, SECTOR_SIZE).unwrap();
        assert_eq!(status.state, BlockState::Data);
        assert_eq!(status.backing_offset, None);

        // A neighbouring range is still served by the backing.
        node.read(2 * SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0xAAu8; SECTOR_BYTES]);

        let _ = std::fs::remove_file(backing_path);
        let _ = std::fs::remove_file(overlay_path);
    }

    #[test]
    fn cow_discard_releases_the_overlay_range() {
        let (backing, backing_path) = raw_filled(0xAA, 4 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-overlay");
        let mut node =
            CowNode::new(Some(Box::new(backing)), &overlay_path, 4 * SECTOR_SIZE).unwrap();

        node.write(SECTOR_SIZE, &[0x55u8; SECTOR_BYTES]).unwrap();
        node.discard(SECTOR_SIZE, SECTOR_SIZE).unwrap();
        assert_eq!(node.dirty_count(), 0);

        let mut got = [0u8; SECTOR_BYTES];
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0xAAu8; SECTOR_BYTES], "discard falls back to the backing");
        let status = node.block_status(SECTOR_SIZE, SECTOR_SIZE).unwrap();
        assert_eq!(status.state, BlockState::Allocated);
        assert_eq!(status.backing_offset, Some(SECTOR_SIZE));

        let _ = std::fs::remove_file(backing_path);
        let _ = std::fs::remove_file(overlay_path);
    }

    #[test]
    fn cow_without_backing_reports_zero_and_zero_fills() {
        let overlay_path = tmp_path("cow-nobacking");
        let mut node = CowNode::new(None, &overlay_path, 2 * SECTOR_SIZE).unwrap();
        assert!(node.backing().is_none());

        let mut got = [0xFFu8; SECTOR_BYTES];
        node.read(0, &mut got).unwrap();
        assert_eq!(got, [0x00u8; SECTOR_BYTES], "no backing and no overlay reads zeroes");
        let status = node.block_status(0, SECTOR_SIZE).unwrap();
        assert_eq!(status.state, BlockState::Zero);
        assert_eq!(status.backing_offset, None);

        node.write(0, &[0x33u8; SECTOR_BYTES]).unwrap();
        assert_eq!(node.block_status(0, SECTOR_SIZE).unwrap().state, BlockState::Data);

        node.discard(0, SECTOR_SIZE).unwrap();
        assert_eq!(node.block_status(0, SECTOR_SIZE).unwrap().state, BlockState::Zero);

        let _ = std::fs::remove_file(overlay_path);
    }

    #[test]
    fn cow_range_straddling_allocated_and_data_reports_first_run() {
        let (backing, backing_path) = raw_filled(0xAA, 4 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-overlay");
        let mut node =
            CowNode::new(Some(Box::new(backing)), &overlay_path, 4 * SECTOR_SIZE).unwrap();

        // Sector 1 lives in the overlay; sectors 0 and 2 do not.
        node.write(SECTOR_SIZE, &[0x55u8; SECTOR_BYTES]).unwrap();

        let mut got = [0u8; 3 * SECTOR_BYTES];
        node.read(0, &mut got).unwrap();
        assert_eq!(&got[0..SECTOR_BYTES], &[0xAAu8; SECTOR_BYTES]);
        assert_eq!(&got[SECTOR_BYTES..2 * SECTOR_BYTES], &[0x55u8; SECTOR_BYTES]);
        assert_eq!(&got[2 * SECTOR_BYTES..], &[0xAAu8; SECTOR_BYTES]);

        let first = node.block_status(0, 3 * SECTOR_SIZE).unwrap();
        assert_eq!(first.state, BlockState::Allocated);
        assert_eq!(first.len, SECTOR_SIZE, "the run stops at the overlay boundary");

        // Stepping by `status.len` reaches the held sector.
        let second = node.block_status(first.len, 2 * SECTOR_SIZE).unwrap();
        assert_eq!(second.state, BlockState::Data);
        assert_eq!(second.len, SECTOR_SIZE);

        let _ = std::fs::remove_file(backing_path);
        let _ = std::fs::remove_file(overlay_path);
    }

    #[test]
    fn cow_partial_sector_write_preserves_neighbours() {
        let (backing, backing_path) = raw_filled(0xAA, 2 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-overlay");
        let mut node =
            CowNode::new(Some(Box::new(backing)), &overlay_path, 2 * SECTOR_SIZE).unwrap();

        // Overwrite 100 bytes in the middle of sector 0.
        node.write(100, &[0x11u8; 100]).unwrap();
        let mut got = [0u8; SECTOR_BYTES];
        node.read(0, &mut got).unwrap();
        assert_eq!(&got[0..100], &[0xAAu8; 100]);
        assert_eq!(&got[100..200], &[0x11u8; 100]);
        assert_eq!(&got[200..], &[0xAAu8; SECTOR_BYTES - 200]);

        let _ = std::fs::remove_file(backing_path);
        let _ = std::fs::remove_file(overlay_path);
    }

    #[test]
    fn chd_node_reads_writes_and_reports_data() {
        use libchdman_rs::hd::{create_from_reader, HdCreateOptions};
        use libchdman_rs::CHD_CODEC_ZLIB;
        use std::io::Cursor;

        let base = tmp_path("chd").with_extension("chd");
        // MAME's v5 compression map asserts on a logical size smaller than one
        // hunk, so use the same 256 KiB the chd_disk tests use.
        let logical = 256 * 1024u64;
        create_from_reader(
            Cursor::new(vec![0xABu8; logical as usize]),
            &base,
            HdCreateOptions {
                logical_size: logical,
                hunk_size: 4096,
                unit_size: SECTOR_BYTES as u32,
                codecs: [CHD_CODEC_ZLIB, 0, 0, 0],
                geometry: None,
                ident: None,
            },
            &mut |_| {},
            &|| false,
        )
        .unwrap();

        let mut node = ChdNode::open(base.to_str().unwrap(), false).unwrap();
        assert_eq!(node.size(), logical);
        assert!(node.backing().is_none(), "a CHD presents the merged view itself");

        let mut got = [0u8; SECTOR_BYTES];
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0xABu8; SECTOR_BYTES]);
        assert_eq!(
            node.block_status(SECTOR_SIZE, SECTOR_SIZE).unwrap().state,
            BlockState::Data
        );

        node.write(SECTOR_SIZE, &[0x5Au8; SECTOR_BYTES]).unwrap();
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0x5Au8; SECTOR_BYTES]);
        node.flush().unwrap();

        drop(node);
        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(crate::chd_disk::diff_path_for(&base));
    }

    #[test]
    fn cow_commit_merges_dirty_into_base_and_clears_overlay() {
        let (backing, backing_path) = raw_filled(0xAA, 4 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-overlay");
        let mut node = CowNode::new(Some(Box::new(backing)), &overlay_path, 4 * SECTOR_SIZE)
            .unwrap()
            .with_base_path(&backing_path);

        node.write(SECTOR_SIZE, &[0x55u8; SECTOR_BYTES]).unwrap();
        assert_eq!(node.dirty_count(), 1);

        let committed = node.commit().unwrap();
        assert_eq!(committed, 1);
        assert_eq!(node.dirty_count(), 0);
        assert_eq!(
            node.block_status(SECTOR_SIZE, SECTOR_SIZE).unwrap().state,
            BlockState::Allocated,
            "after commit the sector is served by the base again"
        );

        // The base file on disk now holds the write...
        let on_disk = std::fs::read(&backing_path).unwrap();
        assert_eq!(&on_disk[SECTOR_BYTES..2 * SECTOR_BYTES], &[0x55u8; SECTOR_BYTES]);
        // ...and the node reads it back through the backing.
        let mut got = [0u8; SECTOR_BYTES];
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0x55u8; SECTOR_BYTES]);

        node.flush().unwrap();
        let _ = std::fs::remove_file(&backing_path);
        cleanup(&overlay_path);
    }

    #[test]
    fn cow_reset_discards_overlay_and_reverts_reads() {
        let (backing, backing_path) = raw_filled(0xAA, 2 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-overlay");
        let mut node = CowNode::new(Some(Box::new(backing)), &overlay_path, 2 * SECTOR_SIZE)
            .unwrap()
            .with_base_path(&backing_path);

        node.write(0, &[0x11u8; SECTOR_BYTES]).unwrap();
        assert_eq!(node.dirty_count(), 1);
        node.rollback().unwrap();
        assert_eq!(node.dirty_count(), 0);

        let mut got = [0u8; SECTOR_BYTES];
        node.read(0, &mut got).unwrap();
        assert_eq!(got, [0xAAu8; SECTOR_BYTES], "reset falls back to the base");

        node.flush().unwrap();
        let _ = std::fs::remove_file(&backing_path);
        cleanup(&overlay_path);
    }

    #[test]
    fn cow_dirty_set_survives_reopen_via_sidecar() {
        let (backing, backing_path) = raw_filled(0xAA, 4 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-overlay");
        {
            let mut node =
                CowNode::new(Some(Box::new(backing)), &overlay_path, 4 * SECTOR_SIZE).unwrap();
            node.write(SECTOR_SIZE, &[0x55u8; SECTOR_BYTES]).unwrap();
            node.flush().unwrap(); // persists the .dirty sidecar
        }

        let backing2 = RawNode::open(&backing_path).unwrap();
        let mut node =
            CowNode::new(Some(Box::new(backing2)), &overlay_path, 4 * SECTOR_SIZE).unwrap();
        assert_eq!(node.dirty_count(), 1, "the sidecar restores the dirty set");
        let mut got = [0u8; SECTOR_BYTES];
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0x55u8; SECTOR_BYTES], "overlay bytes are read back after reopen");

        let _ = std::fs::remove_file(&backing_path);
        cleanup(&overlay_path);
    }

    #[test]
    fn cow_export_and_import_roundtrip_the_overlay() {
        let (backing, backing_path) = raw_filled(0xAA, 2 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-overlay");
        let export_path = tmp_path("cow-export");
        let mut node =
            CowNode::new(Some(Box::new(backing)), &overlay_path, 2 * SECTOR_SIZE).unwrap();
        node.write(0, &[0x42u8; SECTOR_BYTES]).unwrap();

        let dirty = node.export_overlay(&export_path).unwrap();
        assert_eq!(dirty, vec![0u64]);
        assert!(export_path.exists(), "export writes the overlay bytes");

        // Roll back, then re-import the captured overlay.
        node.rollback().unwrap();
        assert_eq!(node.dirty_count(), 0);
        node.import_overlay(&export_path, dirty).unwrap();
        assert_eq!(node.dirty_count(), 1);
        let mut got = [0u8; SECTOR_BYTES];
        node.read(0, &mut got).unwrap();
        assert_eq!(got, [0x42u8; SECTOR_BYTES]);

        let _ = std::fs::remove_file(&backing_path);
        cleanup(&overlay_path);
        let _ = std::fs::remove_file(&export_path);
    }

    #[test]
    fn chd_node_reports_pending_sync_and_overlay_paths() {
        let base = compressed_chd(0xAB);
        let mut node = ChdNode::open(base.to_str().unwrap(), false).unwrap();

        // A compressed base cannot be written in place, so it gets a diff.
        assert!(node.overlay_paths().is_some());
        assert!(node.is_cow(), "a compressed base writes through a diff regardless of the COW flag");
        assert_eq!(node.dirty_count(), 0, "nothing written yet");
        assert!(node.pending_sync().is_none());

        node.write(SECTOR_SIZE, &[0x5Au8; SECTOR_BYTES]).unwrap();
        assert_eq!(node.dirty_count(), 1, "a CHD reports a coarse 1/0 dirtiness");
        assert!(node.pending_sync().is_some(), "COW off: the diff auto-folds on exit");

        drop(node);
        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(crate::chd_disk::diff_path_for(&base));
    }

    /// The overlay contract — dirty_count / commit / rollback — is exercised
    /// through the `dyn BlockNode` the SCSI device actually holds, so no caller
    /// needs to match on a backend kind.
    #[test]
    fn cow_contract_operations_go_through_the_trait_object() {
        let (backing, backing_path) = raw_filled(0xAA, 4 * SECTOR_BYTES);
        let overlay_path = tmp_path("cow-dyn");
        let node = CowNode::new(Some(Box::new(backing)), &overlay_path, 4 * SECTOR_SIZE)
            .unwrap()
            .with_base_path(&backing_path);
        let mut node: Box<dyn BlockNode> = Box::new(node);

        assert!(node.is_cow());
        assert_eq!(node.dirty_count(), 0);

        node.write(SECTOR_SIZE, &[0x55u8; SECTOR_BYTES]).unwrap();
        assert_eq!(node.dirty_count(), 1);

        node.rollback().unwrap();
        assert_eq!(node.dirty_count(), 0);
        let mut got = [0u8; SECTOR_BYTES];
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0xAAu8; SECTOR_BYTES], "rollback falls back to the base");

        node.write(SECTOR_SIZE, &[0x55u8; SECTOR_BYTES]).unwrap();
        assert_eq!(node.commit().unwrap(), 1);
        assert_eq!(node.dirty_count(), 0);
        let on_disk = std::fs::read(&backing_path).unwrap();
        assert_eq!(&on_disk[SECTOR_BYTES..2 * SECTOR_BYTES], &[0x55u8; SECTOR_BYTES]);

        let _ = std::fs::remove_file(&backing_path);
        cleanup(&overlay_path);
    }

    /// A CHD's commit folds the diff into the base (the flatten operation) and
    /// rollback discards it, both through the trait object and both leaving the
    /// node usable.
    #[test]
    fn chd_commit_and_rollback_go_through_the_trait_object() {
        let base = compressed_chd(0xAB);
        let mut node: Box<dyn BlockNode> =
            Box::new(ChdNode::open(base.to_str().unwrap(), false).unwrap());

        node.write(SECTOR_SIZE, &[0x5Au8; SECTOR_BYTES]).unwrap();
        assert_eq!(node.dirty_count(), 1, "a write dirties the diff");
        assert_eq!(node.commit().unwrap(), 1, "commit folds the diff (flatten)");
        assert_eq!(node.dirty_count(), 0, "the fold leaves nothing uncommitted");

        // The fold is visible through the reopened node.
        let mut got = [0u8; SECTOR_BYTES];
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0x5Au8; SECTOR_BYTES]);

        // A second write, then a rollback, restores the folded base.
        node.write(SECTOR_SIZE, &[0x11u8; SECTOR_BYTES]).unwrap();
        assert_eq!(node.dirty_count(), 1);
        node.rollback().unwrap();
        assert_eq!(node.dirty_count(), 0);
        node.read(SECTOR_SIZE, &mut got).unwrap();
        assert_eq!(got, [0x5Au8; SECTOR_BYTES], "rollback restores the committed base");

        drop(node);
        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(crate::chd_disk::diff_path_for(&base));
    }

    /// An ISO of `sectors` CD sectors, each filled with `fill`.
    fn iso(tag: &str, fill: u8, sectors: usize) -> String {
        let p = tmp_path(tag);
        std::fs::write(&p, vec![fill; sectors * 2048]).unwrap();
        p.display().to_string()
    }

    fn first_byte(c: &mut ChangerNode) -> u8 {
        let mut b = [0u8; 1];
        c.read(0, &mut b).unwrap();
        b[0]
    }

    fn drive_with(paths: &[&str]) -> ChangerNode {
        let node = open_image(paths[0], ImageKind::Cdrom, false, None).unwrap();
        ChangerNode::loaded(node, paths[0], paths.iter().map(|p| p.to_string()).collect())
    }

    #[test]
    fn ejecting_the_only_disc_empties_the_tray() {
        let a = iso("only", 1, 4);
        let mut c = drive_with(&[&a]);
        assert!(c.present());
        assert_eq!(c.eject().unwrap(), None);
        assert!(!c.present());
        assert_eq!(c.size(), 0);
        assert!(c.discs().is_empty(), "no phantom entries remain");
        assert!(c.read(0, &mut [0u8; 1]).is_err(), "an empty tray reads nothing");
        assert!(c.take_changed());
    }

    #[test]
    fn ejecting_with_two_discs_turns_the_queue() {
        let (a, b) = (iso("turn-a", 1, 4), iso("turn-b", 2, 8));
        let mut c = drive_with(&[&a, &b]);
        assert_eq!(first_byte(&mut c), 1);
        assert_eq!(c.eject().unwrap().as_deref(), Some(b.as_str()));
        assert_eq!((first_byte(&mut c), c.size()), (2, 8 * 2048));
        assert_eq!(c.discs(), [b.clone(), a.clone()], "the ejected disc goes to the back");
        assert_eq!(c.eject().unwrap().as_deref(), Some(a.as_str()), "and the queue wraps");
        assert_eq!(first_byte(&mut c), 1);
    }

    #[test]
    fn loading_a_disc_raises_the_medium_change_once() {
        let (a, b) = (iso("change-a", 1, 4), iso("change-b", 2, 4));
        let mut c = ChangerNode::empty();
        assert!(!c.take_changed(), "a drive starting empty has not changed");
        c.load(&a).unwrap();
        assert!(c.take_changed());
        assert!(!c.take_changed(), "asking clears it");
        c.load(&b).unwrap();
        assert!(c.take_changed());
        assert_eq!(c.discs(), [b, a], "the first disc stays queued behind the new one");
    }

    #[test]
    fn a_chd_cd_in_the_queue_loads_on_eject() {
        use libchdman_rs::cd::{create_from_iso, CdCreateOptions};
        let a = iso("chd-a", 1, 4);
        let src = iso("chd-src", 0x42, 16);
        let chd = tmp_path("chd-cd").with_extension("chd");
        create_from_iso(Path::new(&src), &chd, CdCreateOptions::default(), &mut |_| {}, &|| false).unwrap();
        let chd = chd.display().to_string();
        let mut c = drive_with(&[&a, &chd]);
        assert_eq!(c.eject().unwrap().as_deref(), Some(chd.as_str()));
        assert_eq!((first_byte(&mut c), c.size()), (0x42, 16 * 2048), "the CHD's data, not the file's bytes");
    }

    #[test]
    fn a_disc_that_will_not_open_leaves_the_drive_as_it_was() {
        let a = iso("bad-a", 1, 4);
        let mut c = drive_with(&[&a, "/nonexistent/disc.iso"]);
        assert!(c.eject().is_err());
        assert!(c.present());
        assert_eq!(c.discs()[0], a, "the queue did not turn");
        assert_eq!(first_byte(&mut c), 1);
    }

    #[test]
    fn queueing_a_missing_file_is_refused() {
        let a = iso("queue-a", 1, 4);
        let mut c = drive_with(&[&a]);
        assert!(c.add("/nonexistent/disc.iso".into()).unwrap_err().contains("not found"));
        assert_eq!(c.discs().len(), 1);
        let b = iso("queue-b", 2, 4);
        c.add(b.clone()).unwrap();
        assert_eq!(c.discs(), [a, b], "a queued disc follows the active one, unopened");
    }
}
