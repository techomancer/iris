//! The disk browser's model: reach a machine's disk image through iris's own
//! block layer, read its SGI volume header, and open a partition's filesystem
//! with rusty-backup's crates (plan, "Disk browser").
//!
//! Images are opened with `iris::block_node::open_image`, so the browser sees
//! what the guest would (a raw image, a CHD with its diff, a copy-on-write
//! overlay) and writes land where the guest's would: on a copy-on-write disk
//! in the overlay, on a CHD in its diff. The file is never opened behind the
//! block layer's back.

use std::io::{self, Read, Seek, SeekFrom, Write};

use disk_core::entry::{EntryType, FileEntry};
use disk_core::filesystem::{EditableFilesystem, Filesystem, FilesystemError};
use iris::block_node::{BlockNode, ImageKind};
use part_tables::sgi::{SgiPartitionType, SgiVolumeHeader};
use sgi_disk::SgiFsKind;

/// `Read + Write + Seek` over a block node, which reads and writes at byte
/// offsets. Reads and writes past the end are clipped, as a file's would be.
/// The filesystem crates may read or write a few bytes anywhere (XFS log
/// replay does), while a CHD takes whole sectors only: such an access is
/// widened to the sectors around it, and a write patches them.
pub struct BlockIo {
    node: Box<dyn BlockNode>,
    pos: u64,
    /// The sector size the node is accessed in.
    sector: u64,
}

impl BlockIo {
    /// Over a disk's 512-byte sectors.
    pub fn new(node: Box<dyn BlockNode>) -> Self {
        Self::with_sector(node, 512)
    }

    pub fn with_sector(node: Box<dyn BlockNode>, sector: u64) -> Self {
        Self { node, pos: 0, sector }
    }

    /// The whole sectors around `[offset, offset + len)`, within the disk.
    fn around(&self, offset: u64, len: usize) -> (u64, usize) {
        let start = offset - offset % self.sector;
        let end = (offset + len as u64).div_ceil(self.sector) * self.sector;
        (start, (end.min(self.node.size()) - start) as usize)
    }

    fn aligned(&self, offset: u64, len: usize) -> bool {
        offset % self.sector == 0 && len as u64 % self.sector == 0
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if self.aligned(offset, buf.len()) {
            return self.node.read(offset, buf);
        }
        let (start, len) = self.around(offset, buf.len());
        let mut whole = vec![0u8; len];
        self.node.read(start, &mut whole)?;
        let at = (offset - start) as usize;
        buf.copy_from_slice(&whole[at..at + buf.len()]);
        Ok(())
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        if self.aligned(offset, buf.len()) {
            return self.node.write(offset, buf);
        }
        let (start, len) = self.around(offset, buf.len());
        let mut whole = vec![0u8; len];
        self.node.read(start, &mut whole)?;
        let at = (offset - start) as usize;
        whole[at..at + buf.len()].copy_from_slice(buf);
        self.node.write(start, &whole)
    }

    pub fn size(&self) -> u64 {
        self.node.size()
    }

    /// Push buffered writes to the image (overlay, CHD diff or file).
    pub fn flush_node(&mut self) -> io::Result<()> {
        self.node.flush()
    }
}

impl Read for BlockIo {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let size = self.node.size();
        if self.pos >= size || buf.is_empty() {
            return Ok(0);
        }
        let n = buf.len().min((size - self.pos) as usize);
        self.read_at(self.pos, &mut buf[..n])?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Write for BlockIo {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let size = self.node.size();
        if self.pos >= size {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "past the end of the disk"));
        }
        let n = buf.len().min((size - self.pos) as usize);
        self.write_at(self.pos, &buf[..n])?;
        self.pos += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.node.flush()
    }
}

impl Seek for BlockIo {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let size = self.node.size() as i128;
        let next = match to {
            SeekFrom::Start(o) => o as i128,
            SeekFrom::End(d) => size + d as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
        };
        if next < 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "seek before the start"));
        }
        self.pos = next as u64;
        Ok(self.pos)
    }
}

/// Open a machine's image through the block layer. `overlay` is the disk's
/// copy-on-write setting, so edits go where the guest's would.
pub fn open_image(path: &str, cdrom: bool, overlay: bool) -> io::Result<BlockIo> {
    let kind = if cdrom { ImageKind::Cdrom } else { ImageKind::Disk };
    let node = iris::block_node::open_image(path, kind, overlay, None)?;
    Ok(BlockIo::with_sector(node, if cdrom { 2048 } else { 512 }))
}

/// What a partition holds, as far as the browser is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartKind {
    Efs,
    Xfs,
    Ext,
    /// The volume header's own little directory (sash, ide, ...): read-only.
    VolumeHeader,
    /// Nothing to browse; the reason says why.
    None,
}

impl PartKind {
    pub fn name(self) -> &'static str {
        match self {
            PartKind::Efs => "EFS",
            PartKind::Xfs => "XFS",
            PartKind::Ext => "ext",
            PartKind::VolumeHeader => "volume header",
            PartKind::None => "",
        }
    }
}

/// One slot of the volume header's partition table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionInfo {
    pub index: usize,
    /// The SGI type's name ("efs", "xfs", "raw", "volhdr").
    pub type_name: String,
    pub offset: u64,
    pub len: u64,
    pub kind: PartKind,
    /// Why it cannot be browsed, when it cannot.
    pub why_not: Option<String>,
}

/// The ext superblock magic, at 1024 + 56 bytes into the filesystem.
const EXT_MAGIC: u16 = 0xEF53;

fn looks_like_ext<R: Read + Seek>(io: &mut R, offset: u64) -> bool {
    let mut b = [0u8; 2];
    io.seek(SeekFrom::Start(offset + 1024 + 56)).is_ok() && io.read_exact(&mut b).is_ok() && u16::from_le_bytes(b) == EXT_MAGIC
}

/// Read the volume header and describe every partition in use.
pub fn partitions<R: Read + Seek>(io: &mut R) -> Result<(SgiVolumeHeader, Vec<PartitionInfo>), String> {
    let header = sgi_disk::read_volume_header(io).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (index, p) in header.partitions.iter().enumerate() {
        if p.is_empty() {
            continue;
        }
        let ptype = p.partition_type();
        let offset = p.first as u64 * 512;
        let len = p.blocks as u64 * 512;
        let (kind, why_not) = match (SgiFsKind::of(ptype), ptype) {
            (Some(SgiFsKind::Efs), _) => (PartKind::Efs, None),
            (Some(SgiFsKind::Xfs), _) => (PartKind::Xfs, None),
            (None, SgiPartitionType::VolHdr) => (PartKind::VolumeHeader, None),
            (None, SgiPartitionType::Volume) => (PartKind::None, Some(t!("browser-why-whole-volume"))),
            (None, SgiPartitionType::Raw) if index == header.swap_part_num as usize => {
                (PartKind::None, Some(t!("browser-why-swap")))
            }
            (None, _) if looks_like_ext(io, offset) => (PartKind::Ext, None),
            (None, SgiPartitionType::Raw) => (PartKind::None, Some(t!("browser-why-unformatted"))),
            (None, _) => (PartKind::None, Some(t!("browser-why-unsupported", kind = ptype.display_name()))),
        };
        out.push(PartitionInfo { index, type_name: ptype.display_name().to_string(), offset, len, kind, why_not });
    }
    Ok((header, out))
}

/// An open partition: editable, or read-only with the reason.
pub enum OpenFs {
    Editable(Box<dyn EditableFilesystem>),
    ReadOnly(Box<dyn Filesystem>, String),
}

impl OpenFs {
    pub fn fs(&mut self) -> &mut dyn Filesystem {
        match self {
            OpenFs::Editable(e) => e.as_filesystem_mut(),
            OpenFs::ReadOnly(f, _) => f.as_mut(),
        }
    }

    pub fn editable(&mut self) -> Option<&mut dyn EditableFilesystem> {
        match self {
            OpenFs::Editable(e) => Some(e.as_mut()),
            OpenFs::ReadOnly(..) => None,
        }
    }

    pub fn read_only_reason(&self) -> Option<&str> {
        match self {
            OpenFs::Editable(_) => None,
            OpenFs::ReadOnly(_, why) => Some(why),
        }
    }
}

/// An XFS volume's log, as the browser needs it: whether IRIX will replay
/// something over edits made now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XfsLog {
    /// Cleanly unmounted (or the log is on another device): edits are safe.
    Clean,
    /// Not cleanly unmounted, and the log can be replayed here.
    Replayable { transactions: usize },
    /// Not cleanly unmounted, and the log cannot be replayed here; the reason says why.
    Unrecoverable(String),
}

/// The log state of the XFS volume at `offset`, read on a borrow of `io`
/// (nothing is written).
pub fn xfs_log(io: &mut BlockIo, offset: u64) -> Result<XfsLog, String> {
    use fs_xfs::xfs::log::LogState;
    use fs_xfs::xfs::log_replay::{check_log, LogCheck};
    let mut fs = fs_xfs::xfs::XfsFilesystem::open(&mut *io, offset).map_err(|e| e.to_string())?;
    let state = fs.log_state().map_err(|e| e.to_string())?;
    let sb = fs.superblock().clone();
    drop(fs);
    Ok(match state {
        LogState::Clean | LogState::External => XfsLog::Clean,
        LogState::Dirty { .. } => match check_log(io, offset, &sb).map_err(|e| e.to_string())? {
            LogCheck::Clean => XfsLog::Clean,
            LogCheck::Replayable { transactions } => XfsLog::Replayable { transactions },
            LogCheck::Unrecoverable(why) => XfsLog::Unrecoverable(why),
        },
    })
}

/// Replay the XFS log at `offset` as a mount would and leave it clean;
/// the number of transactions applied. Writes nothing when the log cannot be
/// replayed faithfully.
pub fn replay_xfs_log(io: &mut BlockIo, offset: u64) -> Result<usize, String> {
    use fs_xfs::xfs::log_replay::{replay_log, ReplayOutcome};
    let sb = fs_xfs::xfs::XfsFilesystem::open(&mut *io, offset).map_err(|e| e.to_string())?.superblock().clone();
    match replay_log(io, offset, &sb).map_err(|e| e.to_string())? {
        ReplayOutcome::Clean => Ok(0),
        ReplayOutcome::Replayed(r) => Ok(r.transactions),
        ReplayOutcome::Unrecoverable(why) => Err(why),
    }
}

/// Open partition `p` of the image at `io`. `read_only` is the controller's
/// reason to refuse edits (a CD-ROM, a running machine); without one the
/// filesystem opens for editing if the crate can edit it, and read-only with
/// the crate's reason otherwise.
pub fn open_partition(mut io: BlockIo, header: &SgiVolumeHeader, p: &PartitionInfo, read_only: Option<String>) -> Result<OpenFs, String> {
    let ro = |fs: Box<dyn Filesystem>, why: String| Ok(OpenFs::ReadOnly(fs, why));
    match p.kind {
        PartKind::None => Err(p.why_not.clone().unwrap_or_default()),
        PartKind::VolumeHeader => ro(Box::new(VolumeHeaderFs::new(io, header)), t!("browser-ro-volume-header")),
        PartKind::Efs | PartKind::Xfs => {
            let kind = if p.kind == PartKind::Efs { SgiFsKind::Efs } else { SgiFsKind::Xfs };
            if let Some(why) = read_only {
                let fs = sgi_disk::open_kind(kind, io, p.offset, p.len).map_err(|e| e.to_string())?;
                return ro(fs, why);
            }
            if kind == SgiFsKind::Efs {
                let fs = sgi_disk::open_kind_editable(kind, io, p.offset, p.len).map_err(|e| e.to_string())?;
                return Ok(OpenFs::Editable(fs));
            }
            // The XFS crate refuses volumes it cannot maintain (IRIX version-1
            // directories, v5 features it does not know) and says why, which
            // becomes the read-only reason. Probe on a borrow, so the image is
            // still there to open read-only.
            let start = io.stream_position().unwrap_or(0);
            let probe = sgi_disk::open_kind_editable(kind, &mut io, p.offset, p.len).map(|_| ());
            match probe {
                Ok(_) => {
                    let _ = io.seek(SeekFrom::Start(start));
                    let fs = sgi_disk::open_kind_editable(kind, io, p.offset, p.len).map_err(|e| e.to_string())?;
                    Ok(OpenFs::Editable(fs))
                }
                Err(e) => {
                    let fs = sgi_disk::open_kind(kind, io, p.offset, p.len).map_err(|e| e.to_string())?;
                    ro(fs, t!("browser-ro-crate", reason = e.to_string()))
                }
            }
        }
        PartKind::Ext => {
            if let Some(why) = read_only {
                let fs = fs_unix::ext::ExtFilesystem::open(io, p.offset).map_err(|e| e.to_string())?;
                return ro(Box::new(fs), why);
            }
            let fs = fs_unix::ext::ExtFilesystem::open(io, p.offset).map_err(|e| e.to_string())?;
            Ok(OpenFs::Editable(Box::new(fs)))
        }
    }
}

/// The volume header's directory (sash, ide, symmon, ...) as a read-only
/// filesystem, so it can be listed and exported like any other.
pub struct VolumeHeaderFs {
    io: BlockIo,
    files: Vec<(String, u64, u64)>,
}

impl VolumeHeaderFs {
    pub fn new(io: BlockIo, header: &SgiVolumeHeader) -> Self {
        let files = header
            .volume_directory
            .iter()
            .filter(|e| !e.name.is_empty() && e.bytes > 0)
            .map(|e| (e.name.clone(), e.block_num as u64 * 512, e.bytes as u64))
            .collect();
        Self { io, files }
    }
}

impl Filesystem for VolumeHeaderFs {
    fn root(&mut self) -> Result<FileEntry, FilesystemError> {
        Ok(FileEntry::root())
    }

    fn list_directory(&mut self, entry: &FileEntry) -> Result<Vec<FileEntry>, FilesystemError> {
        if entry.entry_type != EntryType::Directory {
            return Ok(Vec::new());
        }
        Ok(self
            .files
            .iter()
            .enumerate()
            .map(|(i, (name, _, len))| FileEntry::new_file(name.clone(), format!("/{name}"), *len, i as u64))
            .collect())
    }

    fn read_file(&mut self, entry: &FileEntry, max_bytes: usize) -> Result<Vec<u8>, FilesystemError> {
        let (_, at, len) = self
            .files
            .get(entry.location as usize)
            .cloned()
            .ok_or_else(|| FilesystemError::NotFound(entry.name.clone()))?;
        let n = (len as usize).min(max_bytes);
        let mut buf = vec![0u8; n];
        self.io.seek(SeekFrom::Start(at))?;
        self.io.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn volume_label(&self) -> Option<&str> {
        Some("volhdr")
    }

    fn fs_type(&self) -> &str {
        "SGI volume header"
    }

    fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.2).sum()
    }

    fn used_size(&self) -> u64 {
        self.total_size()
    }
}

/// Owner and group names read from the browsed disk's own `/etc/passwd` and
/// `/etc/group`; numbers are always accepted too.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Names {
    pub users: Vec<(u32, String)>,
    pub groups: Vec<(u32, String)>,
}

impl Names {
    /// Parse `name:pw:id:...` lines.
    fn parse(text: &str) -> Vec<(u32, String)> {
        text.lines()
            .filter_map(|l| {
                let mut f = l.split(':');
                let name = f.next()?.trim();
                let _pw = f.next()?;
                let id = f.next()?.trim().parse().ok()?;
                (!name.is_empty() && !name.starts_with('#')).then(|| (id, name.to_string()))
            })
            .collect()
    }

    pub fn from_files(passwd: &str, group: &str) -> Self {
        Self { users: Self::parse(passwd), groups: Self::parse(group) }
    }

    pub fn user(&self, id: u32) -> String {
        self.users.iter().find(|(i, _)| *i == id).map(|(_, n)| n.clone()).unwrap_or_else(|| id.to_string())
    }

    pub fn group(&self, id: u32) -> String {
        self.groups.iter().find(|(i, _)| *i == id).map(|(_, n)| n.clone()).unwrap_or_else(|| id.to_string())
    }

    /// A user by name or number.
    pub fn user_id(&self, s: &str) -> Option<u32> {
        s.trim().parse().ok().or_else(|| self.users.iter().find(|(_, n)| n == s.trim()).map(|(i, _)| *i))
    }

    pub fn group_id(&self, s: &str) -> Option<u32> {
        s.trim().parse().ok().or_else(|| self.groups.iter().find(|(_, n)| n == s.trim()).map(|(i, _)| *i))
    }

    /// Read them from the root of `fs`, when it has an `/etc`.
    pub fn read(fs: &mut dyn Filesystem) -> Self {
        let read = |fs: &mut dyn Filesystem, name: &str| -> Option<String> {
            let root = fs.root().ok()?;
            let etc = fs.list_directory(&root).ok()?.into_iter().find(|e| e.name == "etc" && e.is_directory())?;
            let f = fs.list_directory(&etc).ok()?.into_iter().find(|e| e.name == name && e.is_file())?;
            String::from_utf8(fs.read_file(&f, 1 << 20).ok()?).ok()
        };
        let passwd = read(fs, "passwd").unwrap_or_default();
        let group = read(fs, "group").unwrap_or_default();
        Self::from_files(&passwd, &group)
    }
}

/// The text editor refuses files larger than this (offering Export instead).
pub const TEXT_EDIT_LIMIT: u64 = 1 << 20;

/// Whether `bytes` look like text: no NUL in the first block, and UTF-8 or
/// mostly printable.
pub fn looks_like_text(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(4096)];
    if head.contains(&0) {
        return false;
    }
    std::str::from_utf8(bytes).is_ok()
        || head.iter().filter(|&&b| b < 0x20 && !matches!(b, b'\n' | b'\r' | b'\t' | 0x0c)).count() * 20 < head.len().max(1)
}

/// `rwxr-xr-x` for a mode's permission bits, with the file type in front.
pub fn mode_string(e: &FileEntry) -> String {
    let Some(mode) = e.mode else { return String::new() };
    let t = match e.entry_type {
        EntryType::Directory => 'd',
        EntryType::Symlink => 'l',
        EntryType::Special => 'c',
        EntryType::File => '-',
    };
    let mut s = String::from(t);
    for shift in [6, 3, 0] {
        let b = (mode >> shift) & 7;
        s.push(if b & 4 != 0 { 'r' } else { '-' });
        s.push(if b & 2 != 0 { 'w' } else { '-' });
        s.push(if b & 1 != 0 { 'x' } else { '-' });
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_come_from_passwd_and_numbers_always_work() {
        let n = Names::from_files("root:x:0:0:Super-User:/:/bin/csh\n# c\nguest:x:998:998::/usr/people/guest:/bin/csh\n", "sys::0:root\nuser::20:\n");
        assert_eq!(n.user(0), "root");
        assert_eq!(n.user(5), "5");
        assert_eq!(n.group(20), "user");
        assert_eq!(n.user_id("guest"), Some(998));
        assert_eq!(n.user_id("1234"), Some(1234));
        assert_eq!(n.group_id("sys"), Some(0));
        assert_eq!(n.group_id("nobody-here"), None);
    }

    #[test]
    fn text_is_recognised_and_binaries_are_not() {
        assert!(looks_like_text(b"#!/bin/sh\necho hi\n"));
        assert!(!looks_like_text(b"\x7fELF\x01\x02\x00\x00"));
    }

    #[test]
    fn block_io_reads_writes_and_clips_at_the_end() {
        let root = crate::model::machines::testing::TempRoot::new("blockio");
        let path = root.0.join("d.img");
        std::fs::write(&path, vec![0u8; 4096]).unwrap();
        let mut io = open_image(&path.to_string_lossy(), false, false).unwrap();
        io.seek(SeekFrom::Start(4090)).unwrap();
        assert_eq!(io.write(b"abcdefghij").unwrap(), 6);
        io.flush().unwrap();
        io.seek(SeekFrom::End(-6)).unwrap();
        let mut buf = [0u8; 10];
        assert_eq!(io.read(&mut buf).unwrap(), 6);
        assert_eq!(&buf[..6], b"abcdef");
        assert_eq!(io.read(&mut buf).unwrap(), 0);
    }

    /// A node that takes whole sectors only, as a CHD does.
    struct Sectors(Vec<u8>);

    impl BlockNode for Sectors {
        fn size(&self) -> u64 {
            self.0.len() as u64
        }
        fn read(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            assert!(offset % 512 == 0 && buf.len() % 512 == 0, "read {offset} +{}", buf.len());
            buf.copy_from_slice(&self.0[offset as usize..offset as usize + buf.len()]);
            Ok(())
        }
        fn write(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
            assert!(offset % 512 == 0 && buf.len() % 512 == 0, "write {offset} +{}", buf.len());
            self.0[offset as usize..offset as usize + buf.len()].copy_from_slice(buf);
            Ok(())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
        fn discard(&mut self, _: u64, _: u64) -> io::Result<()> {
            Ok(())
        }
        fn block_status(&self, _: u64, len: u64) -> io::Result<iris::block_node::BlockStatus> {
            Ok(iris::block_node::BlockStatus { state: iris::block_node::BlockState::Data, backing_offset: None, len })
        }
        fn backing(&self) -> Option<&dyn BlockNode> {
            None
        }
    }

    #[test]
    fn block_io_reaches_a_whole_sector_disk_at_any_byte() {
        let mut io = BlockIo::new(Box::new(Sectors((0..2048u32).map(|i| i as u8).collect())));
        // A few bytes across a sector boundary: the sectors around them are
        // read, patched and written back, the bytes beside them kept.
        io.seek(SeekFrom::Start(509)).unwrap();
        io.write_all(b"XYZ!").unwrap();
        io.seek(SeekFrom::Start(500)).unwrap();
        let mut buf = [0u8; 20];
        io.read_exact(&mut buf).unwrap();
        let mut want: Vec<u8> = (500..520u32).map(|i| i as u8).collect();
        want[9..13].copy_from_slice(b"XYZ!");
        assert_eq!(&buf[..], &want[..]);
        // And at the very end of the disk.
        io.seek(SeekFrom::End(-3)).unwrap();
        io.write_all(b"end").unwrap();
        io.seek(SeekFrom::End(-5)).unwrap();
        let mut tail = [0u8; 5];
        io.read_exact(&mut tail).unwrap();
        assert_eq!(&tail, &[(2043 % 256) as u8, (2044 % 256) as u8, b'e', b'n', b'd']);
    }

    #[test]
    fn a_disk_without_a_label_says_so() {
        let root = crate::model::machines::testing::TempRoot::new("nolabel");
        let path = root.0.join("blank.img");
        std::fs::write(&path, vec![0u8; 8192]).unwrap();
        let mut io = open_image(&path.to_string_lossy(), false, false).unwrap();
        assert!(partitions(&mut io).is_err());
    }
}

