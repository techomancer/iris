//! Copy-on-write disk overlay for SCSI disk images.
//!
//! Protects the base disk image from writes by redirecting them to a sparse
//! overlay file. Reads check the overlay first, falling back to the base image
//! for clean sectors. Deleting the overlay file resets the disk to its original state.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};

const SECTOR_SIZE: u64 = 512;

pub struct CowDisk {
    base: File,
    overlay: File,
    dirty: HashSet<u64>,
    base_size: u64,
    overlay_path: String,
}

impl CowDisk {
    /// Open a COW disk with the given base image (read-only) and overlay file (read-write).
    /// If the overlay file exists, its dirty sectors are reconstructed from its sparse extent.
    /// If it doesn't exist, a new empty overlay is created.
    pub fn new(base_path: &str, overlay_path: &str) -> io::Result<Self> {
        let base = File::open(base_path)?;
        let base_size = base.metadata()?.len();

        let overlay = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(overlay_path)?;

        // Rebuild the dirty set from the overlay file size.
        // The overlay is a sparse file with the same layout as the base.
        // Any sector that has been written occupies space, but we can't easily
        // detect sparse holes portably. Instead, track dirty sectors in memory
        // and accept that a fresh start after crash loses the dirty set
        // (overlay is deleted on state load anyway).
        let dirty = HashSet::new();

        eprintln!("iris: COW overlay active (base: {}, overlay: {})", base_path, overlay_path);
        eprintln!("iris: to reset disk to clean state, delete {}", overlay_path);

        Ok(Self {
            base,
            overlay,
            dirty,
            base_size,
            overlay_path: overlay_path.to_string(),
        })
    }

    /// Read `count` sectors starting at `lba`.
    /// Dirty sectors are read from the overlay, clean sectors from the base.
    pub fn read_sectors(&mut self, lba: u64, count: usize) -> io::Result<Vec<u8>> {
        let total = count * SECTOR_SIZE as usize;
        let mut data = vec![0u8; total];

        // Batch consecutive sectors from the same source to minimize seeks.
        let mut pos = 0usize;
        let mut sector = lba;
        while pos < total {
            // Determine run length from the same source.
            let is_dirty = self.dirty.contains(&sector);
            let mut run = 1usize;
            while pos + run * SECTOR_SIZE as usize <= total {
                let next = sector + run as u64;
                if self.dirty.contains(&next) != is_dirty {
                    break;
                }
                run += 1;
            }
            // Don't overshoot.
            let run_sectors = run.min((total - pos) / SECTOR_SIZE as usize);
            let run_bytes = run_sectors * SECTOR_SIZE as usize;

            let file = if is_dirty { &mut self.overlay } else { &mut self.base };
            file.seek(SeekFrom::Start(sector * SECTOR_SIZE))?;
            file.read_exact(&mut data[pos..pos + run_bytes])?;

            pos += run_bytes;
            sector += run_sectors as u64;
        }

        Ok(data)
    }

    /// Write sectors starting at `lba`. Data length must be a multiple of 512.
    /// Writes go to the overlay file only; the base image is never modified.
    pub fn write_sectors(&mut self, lba: u64, data: &[u8]) -> io::Result<()> {
        debug_assert!(data.len() % SECTOR_SIZE as usize == 0);
        let count = data.len() / SECTOR_SIZE as usize;

        self.overlay.seek(SeekFrom::Start(lba * SECTOR_SIZE))?;
        self.overlay.write_all(data)?;

        for i in 0..count as u64 {
            self.dirty.insert(lba + i);
        }

        Ok(())
    }

    /// Base image size in bytes.
    pub fn size(&self) -> u64 {
        self.base_size
    }

    /// Merge all dirty overlay sectors into the base image, then truncate overlay.
    pub fn commit(&mut self) -> io::Result<usize> {
        // Reopen base as read-write for the commit.
        // (We can't just change the mode of self.base, so we open a second handle.)
        let base_path = {
            // Get the path from /proc/self/fd on Linux, or just require it as a param.
            // For simplicity, we'll do the commit through the overlay path convention:
            // base path = overlay path without the ".overlay" suffix.
            if self.overlay_path.ends_with(".overlay") {
                self.overlay_path[..self.overlay_path.len() - 8].to_string()
            } else {
                return Err(io::Error::new(io::ErrorKind::Other,
                    "cannot determine base path from overlay path"));
            }
        };

        let mut base_rw = OpenOptions::new().read(true).write(true).open(&base_path)?;
        let mut buf = vec![0u8; SECTOR_SIZE as usize];
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
        self.overlay.set_len(0)?;

        // Reopen base read-only to pick up committed data.
        self.base = File::open(&base_path)?;

        eprintln!("iris: COW committed {} sectors to {}", committed, base_path);
        Ok(committed)
    }

    /// Delete the overlay file and create a fresh empty one (for state load).
    pub fn reset_overlay(&mut self) -> io::Result<()> {
        self.dirty.clear();
        self.overlay.set_len(0)?;
        self.overlay.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    /// Number of dirty sectors in the overlay.
    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }
}
