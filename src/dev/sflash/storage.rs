//! Exact-size file backing shared by flash command-set models.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(crate) struct FlashBacking {
    file: File,
    size: usize,
}

impl FlashBacking {
    #[cfg(test)]
    pub(crate) fn from_file(file: File, size: usize) -> Self {
        Self { file, size }
    }

    /// Open a flash store, loading its bytes or seeding a new empty file.
    /// A non-empty file must match the supplied array size and is left intact
    /// when its size is refused.
    pub(crate) fn open(path: impl AsRef<Path>, data: &mut [u8]) -> io::Result<Self> {
        let path = path.as_ref();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let len = file.metadata()?.len();
        if len == data.len() as u64 {
            file.read_exact(data)?;
        } else if len == 0 {
            file.write_all(data)?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "flash file {} is {len} bytes, expected {} (or empty); not overwriting it",
                    path.display(),
                    data.len()
                ),
            ));
        }
        Ok(Self {
            file,
            size: data.len(),
        })
    }

    /// Write one byte range through to the backing file.
    pub(crate) fn write_range(&mut self, offset: usize, bytes: &[u8]) -> io::Result<()> {
        if offset > self.size || bytes.len() > self.size - offset {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "flash backing range is outside the array",
            ));
        }
        self.file.seek(SeekFrom::Start(offset as u64))?;
        self.file.write_all(bytes)?;
        self.file.flush()
    }
}
