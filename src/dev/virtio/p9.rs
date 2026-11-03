//! virtio-9p (device ID 9, virtio 1.x §5.11): a host directory for the guest,
//! served as 9P2000.L.
//!
//! One queue. A driver posts a chain of device-readable buffers holding a T
//! message followed by device-writable buffers for the R message; each chain
//! is one request. The device answers them in order, on the service thread:
//! host file operations are quick enough next to what a guest does with them
//! that a worker pool would buy nothing.
//!
//! **Staying inside the share.** The guest names files by path, and may itself
//! create symlinks, so no path is ever handed to the host whole: [`Server::dir`]
//! walks from the root's descriptor one `openat(O_NOFOLLOW)` at a time, and
//! everything else is a `*at` call on the directory that returns. A symlink in
//! the share therefore cannot lead out of it: the guest reads it with Treadlink
//! and resolves the target itself, inside its own namespace.
//!
//! **Ownership.** The guest does its own permission checks from what Tgetattr
//! reports. Host files all belong to one host user, so every file reports the
//! configured `uid` and `gid`, and a guest chown succeeds without effect.

use std::collections::HashMap;
use std::ffi::{CStr, CString, OsStr, OsString};
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::{AsRawFd, FromRawFd, OwnedFd};
use std::path::Path;

use super::{Cx, VirtioCore, VirtioDevice, Waker};

pub const DEVICE_ID: u32 = 9;
/// `VIRTIO_9P_MOUNT_TAG`: config space holds the tag the guest mounts by.
pub const F_MOUNT_TAG: u64 = 1 << 0;

/// Largest message, the header and 1 MiB of data.
const MAX_MSIZE: u32 = 1024 * 1024 + 24;
const DIALECT: &[u8] = b"9P2000.L";
const NOFID: u32 = 0xffff_ffff;
/// Components one Twalk may hold.
const MAXWELEM: u16 = 16;

// Message types: T is the request, the reply is T + 1.
const RLERROR: u8 = 7;
const TSTATFS: u8 = 8;
const TLOPEN: u8 = 12;
const TLCREATE: u8 = 14;
const TSYMLINK: u8 = 16;
const TMKNOD: u8 = 18;
const TRENAME: u8 = 20;
const TREADLINK: u8 = 22;
const TGETATTR: u8 = 24;
const TSETATTR: u8 = 26;
const TXATTRWALK: u8 = 30;
const TXATTRCREATE: u8 = 32;
const TREADDIR: u8 = 40;
const TFSYNC: u8 = 50;
const TLOCK: u8 = 52;
const TGETLOCK: u8 = 54;
const TLINK: u8 = 70;
const TMKDIR: u8 = 72;
const TRENAMEAT: u8 = 74;
const TUNLINKAT: u8 = 76;
const TVERSION: u8 = 100;
const TAUTH: u8 = 102;
const TATTACH: u8 = 104;
const TFLUSH: u8 = 108;
const TWALK: u8 = 110;
const TREAD: u8 = 116;
const TWRITE: u8 = 118;
const TCLUNK: u8 = 120;
const TREMOVE: u8 = 122;

const QTDIR: u8 = 0x80;
const QTSYMLINK: u8 = 0x02;
const QTFILE: u8 = 0;

/// Everything Tgetattr can fill in: mode through blocks, and the times.
const GETATTR_BASIC: u64 = 0x7ff;

const SETATTR_MODE: u32 = 0x001;
const SETATTR_SIZE: u32 = 0x008;
const SETATTR_ATIME: u32 = 0x010;
const SETATTR_MTIME: u32 = 0x020;
const SETATTR_ATIME_SET: u32 = 0x080;
const SETATTR_MTIME_SET: u32 = 0x100;

/// The dialect's open flags are Linux's, whatever the host's are.
const L_O_ACCMODE: u32 = 0o3;
const L_O_WRONLY: u32 = 0o1;
const L_O_RDWR: u32 = 0o2;
const L_O_CREAT: u32 = 0o100;
const L_O_EXCL: u32 = 0o200;
const L_O_TRUNC: u32 = 0o1000;
const L_O_APPEND: u32 = 0o2000;
const L_AT_REMOVEDIR: u32 = 0x200;

/// V9FS_MAGIC, what Rstatfs reports as the filesystem type.
const V9FS_MAGIC: u32 = 0x0102_1997;

/// The dialect's errnos are Linux's too.
fn linux_errno(e: &io::Error) -> u32 {
    let Some(raw) = e.raw_os_error() else { return 5 };
    match raw {
        libc::EPERM => 1,
        libc::ENOENT => 2,
        libc::EINTR => 4,
        libc::EIO => 5,
        libc::ENXIO => 6,
        libc::E2BIG => 7,
        libc::EBADF => 9,
        libc::EAGAIN => 11,
        libc::ENOMEM => 12,
        libc::EACCES => 13,
        libc::EFAULT => 14,
        libc::EBUSY => 16,
        libc::EEXIST => 17,
        libc::EXDEV => 18,
        libc::ENODEV => 19,
        libc::ENOTDIR => 20,
        libc::EISDIR => 21,
        libc::EINVAL => 22,
        libc::ENFILE => 23,
        libc::EMFILE => 24,
        libc::ETXTBSY => 26,
        libc::EFBIG => 27,
        libc::ENOSPC => 28,
        libc::ESPIPE => 29,
        libc::EROFS => 30,
        libc::EMLINK => 31,
        libc::EPIPE => 32,
        libc::ERANGE => 34,
        libc::ENAMETOOLONG => 36,
        libc::ENOLCK => 37,
        libc::ENOSYS => 38,
        libc::ENOTEMPTY => 39,
        libc::ELOOP => 40,
        libc::EOVERFLOW => 75,
        libc::ENOTSUP => 95,
        libc::ESTALE => 116,
        libc::EDQUOT => 122,
        _ => 5,
    }
}

fn err(code: i32) -> io::Error {
    io::Error::from_raw_os_error(code)
}

fn check(ret: libc::c_int) -> io::Result<libc::c_int> {
    if ret < 0 { Err(io::Error::last_os_error()) } else { Ok(ret) }
}

fn cstr(name: &OsStr) -> io::Result<CString> {
    CString::new(name.as_bytes()).map_err(|_| err(libc::EINVAL))
}

fn dup(fd: &OwnedFd) -> io::Result<OwnedFd> {
    fd.try_clone()
}

fn openat(dir: &OwnedFd, name: &CStr, flags: libc::c_int, mode: libc::c_uint) -> io::Result<OwnedFd> {
    // SAFETY: `name` is NUL-terminated; a returned descriptor is ours alone.
    let fd = check(unsafe { libc::openat(dir.as_raw_fd(), name.as_ptr(), flags | libc::O_CLOEXEC, mode) })?;
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn lstatat(dir: &OwnedFd, name: &CStr) -> io::Result<libc::stat> {
    // SAFETY: `st` is written by fstatat before it is read.
    unsafe {
        let mut st: libc::stat = std::mem::zeroed();
        check(libc::fstatat(dir.as_raw_fd(), name.as_ptr(), &mut st, libc::AT_SYMLINK_NOFOLLOW))?;
        Ok(st)
    }
}

fn fstat(fd: &OwnedFd) -> io::Result<libc::stat> {
    // SAFETY: as above.
    unsafe {
        let mut st: libc::stat = std::mem::zeroed();
        check(libc::fstat(fd.as_raw_fd(), &mut st))?;
        Ok(st)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Qid {
    ty: u8,
    version: u32,
    path: u64,
}

/// Inode numbers repeat across host filesystems a share may span.
fn qid_path(dev: u64, ino: u64) -> u64 {
    ino ^ dev.rotate_left(40)
}

fn qid_of(st: &libc::stat) -> Qid {
    let ty = match st.st_mode as u32 & libc::S_IFMT as u32 {
        m if m == libc::S_IFDIR as u32 => QTDIR,
        m if m == libc::S_IFLNK as u32 => QTSYMLINK,
        _ => QTFILE,
    };
    Qid {
        ty,
        version: (st.st_mtime as u32) ^ (st.st_size as u32),
        path: qid_path(st.st_dev as u64, st.st_ino as u64),
    }
}

/// A T message being taken apart. `None` from a getter is a message that ends
/// too soon.
struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.buf.len())?;
        let s = &self.buf[self.at..end];
        self.at = end;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        self.bytes(1).map(|b| b[0])
    }
    fn u16(&mut self) -> Option<u16> {
        self.bytes(2).map(|b| u16::from_le_bytes(b.try_into().unwrap()))
    }
    fn u32(&mut self) -> Option<u32> {
        self.bytes(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn u64(&mut self) -> Option<u64> {
        self.bytes(8).map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }
    fn string(&mut self) -> Option<&'a [u8]> {
        let n = self.u16()? as usize;
        self.bytes(n)
    }
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    fn u16(&mut self, v: u16) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn string(&mut self, s: &[u8]) -> &mut Self {
        self.u16(s.len() as u16);
        self.0.extend_from_slice(s);
        self
    }
    fn qid(&mut self, q: Qid) -> &mut Self {
        self.u8(q.ty).u32(q.version).u64(q.path)
    }
}

struct DirEntry {
    name: Vec<u8>,
    qid: Qid,
    /// The dirent type byte, in Linux's numbering (the BSDs share it).
    d_type: u8,
}

#[derive(Default)]
enum Open {
    #[default]
    No,
    File { fd: OwnedFd },
    /// A directory, and the snapshot Treaddir at offset 0 last took of it, so
    /// offsets stay meaningful however the host's readdir numbers its entries.
    Dir { fd: OwnedFd, entries: Vec<DirEntry> },
}

/// What a fid stands for: a path from the root, kept as components so it can be
/// walked safely, and a descriptor once it has been opened.
#[derive(Default)]
struct Fid {
    path: Vec<OsString>,
    open: Open,
}

/// A share as configured.
#[derive(Clone, Debug)]
pub struct ShareConfig {
    pub tag: String,
    pub path: std::path::PathBuf,
    pub readonly: bool,
    pub uid: u32,
    pub gid: u32,
}

/// The 9P2000.L server, knowing nothing of virtio.
pub struct Server {
    root: Option<OwnedFd>,
    readonly: bool,
    uid: u32,
    gid: u32,
    msize: u32,
    fids: HashMap<u32, Fid>,
}

/// One accepted name: no path separator, not empty, no NUL.
fn component(name: &[u8]) -> io::Result<&OsStr> {
    if name.is_empty() || name.contains(&b'/') || name.contains(&0) {
        return Err(err(libc::EINVAL));
    }
    Ok(OsStr::from_bytes(name))
}

/// A name that creates or removes a directory entry, for which `.` and `..`
/// are no more acceptable than in the host's own calls.
fn entry_name(name: &[u8]) -> io::Result<CString> {
    let n = component(name)?;
    if name == b"." || name == b".." {
        return Err(err(libc::EINVAL));
    }
    cstr(n)
}

impl Server {
    pub fn new(root: &Path, readonly: bool, uid: u32, gid: u32) -> io::Result<Self> {
        let path = cstr(root.as_os_str())?;
        // SAFETY: `path` is NUL-terminated; the descriptor returned is ours alone.
        let fd = check(unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC) })?;
        Ok(Self {
            root: Some(unsafe { OwnedFd::from_raw_fd(fd) }),
            readonly,
            uid,
            gid,
            msize: MAX_MSIZE,
            fids: HashMap::new(),
        })
    }

    /// A server whose share could not be opened: every attach fails.
    fn absent(readonly: bool, uid: u32, gid: u32) -> Self {
        Self { root: None, readonly, uid, gid, msize: MAX_MSIZE, fids: HashMap::new() }
    }

    /// Drop every fid, as a driver reset or a new Tversion does.
    pub fn reset(&mut self) {
        self.fids.clear();
        self.msize = MAX_MSIZE;
    }

    fn root(&self) -> io::Result<&OwnedFd> {
        self.root.as_ref().ok_or_else(|| err(libc::ENOENT))
    }

    /// The directory at `path`, walked from the root with no symlink followed.
    fn dir(&self, path: &[OsString]) -> io::Result<OwnedFd> {
        let mut fd = dup(self.root()?)?;
        for c in path {
            fd = openat(&fd, &cstr(c)?, libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW, 0)?;
        }
        Ok(fd)
    }

    /// The directory holding `path` and its last component. The root is its
    /// own parent, as `.` of itself.
    fn parent(&self, path: &[OsString]) -> io::Result<(OwnedFd, CString)> {
        match path.split_last() {
            Some((last, rest)) => Ok((self.dir(rest)?, cstr(last)?)),
            None => Ok((dup(self.root()?)?, CString::new(".").unwrap())),
        }
    }

    fn fid(&self, id: u32) -> io::Result<&Fid> {
        self.fids.get(&id).ok_or_else(|| err(libc::EBADF))
    }

    fn stat_path(&self, path: &[OsString]) -> io::Result<libc::stat> {
        let (dir, name) = self.parent(path)?;
        lstatat(&dir, &name)
    }

    /// A fid's attributes, from its descriptor if it has one, so that an open
    /// file that has since been removed still answers.
    fn stat_fid(&self, f: &Fid) -> io::Result<libc::stat> {
        match &f.open {
            Open::File { fd } | Open::Dir { fd, .. } => fstat(fd),
            Open::No => self.stat_path(&f.path),
        }
    }

    fn writable(&self) -> io::Result<()> {
        if self.readonly { Err(err(libc::EROFS)) } else { Ok(()) }
    }

    fn host_flags(&self, flags: u32) -> io::Result<libc::c_int> {
        let acc = flags & L_O_ACCMODE;
        if acc != 0 || flags & (L_O_TRUNC | L_O_CREAT | L_O_APPEND) != 0 {
            self.writable()?;
        }
        let mut f = match acc {
            L_O_WRONLY => libc::O_WRONLY,
            L_O_RDWR => libc::O_RDWR,
            _ => libc::O_RDONLY,
        };
        if flags & L_O_TRUNC != 0 {
            f |= libc::O_TRUNC;
        }
        if flags & L_O_EXCL != 0 {
            f |= libc::O_EXCL;
        }
        // Deliberately not O_APPEND: Twrite carries the offset, and the
        // guest's own append position is what it asked for.
        Ok(f | libc::O_NOFOLLOW)
    }

    /// Serve one T message; the bytes returned are the R message.
    pub fn handle(&mut self, msg: &[u8]) -> Vec<u8> {
        if msg.len() < 7 {
            return Vec::new();
        }
        let ty = msg[4];
        let tag = u16::from_le_bytes([msg[5], msg[6]]);
        let mut r = Reader { buf: msg, at: 7 };
        let body = match self.dispatch(ty, &mut r) {
            Ok(body) => body,
            Err(e) => {
                let mut w = Writer::default();
                w.u32(linux_errno(&e));
                return Self::frame(RLERROR, tag, w);
            }
        };
        Self::frame(ty + 1, tag, body)
    }

    fn frame(ty: u8, tag: u16, body: Writer) -> Vec<u8> {
        let mut out = Vec::with_capacity(7 + body.0.len());
        out.extend_from_slice(&((7 + body.0.len()) as u32).to_le_bytes());
        out.push(ty);
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&body.0);
        out
    }

    fn dispatch(&mut self, ty: u8, r: &mut Reader) -> io::Result<Writer> {
        self.op(ty, r)
            .and_then(|w| w.ok_or_else(|| err(libc::EINVAL)))
    }

    /// `Ok(None)` is a message too short for its type.
    fn op(&mut self, ty: u8, r: &mut Reader) -> io::Result<Option<Writer>> {
        macro_rules! get {
            ($e:expr) => {
                match $e {
                    Some(v) => v,
                    None => return Ok(None),
                }
            };
        }
        let mut w = Writer::default();
        match ty {
            TVERSION => {
                let msize = get!(r.u32());
                let version = get!(r.string());
                self.reset();
                self.msize = msize.clamp(4096, MAX_MSIZE);
                w.u32(self.msize);
                w.string(if version == DIALECT { DIALECT } else { b"unknown" });
            }
            TAUTH => return Err(err(libc::ENOTSUP)),
            TFLUSH => {
                // Requests finish in order before the next is read, so the one
                // named has long been answered.
                get!(r.u16());
            }
            TATTACH => {
                let fid = get!(r.u32());
                let afid = get!(r.u32());
                get!(r.string()); // uname
                let aname = get!(r.string());
                get!(r.u32()); // n_uname
                w.qid(self.attach(fid, afid, aname)?);
            }
            TWALK => {
                let fid = get!(r.u32());
                let newfid = get!(r.u32());
                let n = get!(r.u16());
                if n > MAXWELEM {
                    return Err(err(libc::EINVAL));
                }
                let mut names = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    names.push(get!(r.string()));
                }
                let qids = self.walk(fid, newfid, &names)?;
                w.u16(qids.len() as u16);
                for q in qids {
                    w.qid(q);
                }
            }
            TLOPEN => {
                let fid = get!(r.u32());
                let flags = get!(r.u32());
                w.qid(self.lopen(fid, flags)?);
                w.u32(self.msize - 24);
            }
            TLCREATE => {
                let fid = get!(r.u32());
                let name = get!(r.string());
                let flags = get!(r.u32());
                let mode = get!(r.u32());
                get!(r.u32()); // gid
                w.qid(self.lcreate(fid, name, flags, mode)?);
                w.u32(self.msize - 24);
            }
            TSYMLINK => {
                let fid = get!(r.u32());
                let name = get!(r.string());
                let target = get!(r.string());
                get!(r.u32()); // gid
                w.qid(self.symlink(fid, name, target)?);
            }
            TMKNOD => {
                // Regular files come through Tlcreate; the host user may not
                // make devices at all.
                return Err(err(libc::EPERM));
            }
            TMKDIR => {
                let fid = get!(r.u32());
                let name = get!(r.string());
                let mode = get!(r.u32());
                get!(r.u32()); // gid
                w.qid(self.mkdir(fid, name, mode)?);
            }
            TRENAME => {
                let fid = get!(r.u32());
                let dfid = get!(r.u32());
                let name = get!(r.string());
                let path = self.fid(fid)?.path.clone();
                let Some((old, olddir)) = path.split_last() else { return Err(err(libc::EBUSY)) };
                let newdir = self.fid(dfid)?.path.clone();
                self.rename(olddir, old.as_bytes(), &newdir, name)?;
            }
            TRENAMEAT => {
                let olddfid = get!(r.u32());
                let oldname = get!(r.string());
                let newdfid = get!(r.u32());
                let newname = get!(r.string());
                let olddir = self.fid(olddfid)?.path.clone();
                let newdir = self.fid(newdfid)?.path.clone();
                self.rename(&olddir, oldname, &newdir, newname)?;
            }
            TUNLINKAT => {
                let dfid = get!(r.u32());
                let name = get!(r.string());
                let flags = get!(r.u32());
                self.writable()?;
                let path = self.fid(dfid)?.path.clone();
                let dir = self.dir(&path)?;
                let name = entry_name(name)?;
                let at = if flags & L_AT_REMOVEDIR != 0 { libc::AT_REMOVEDIR } else { 0 };
                // SAFETY: `name` is NUL-terminated and `dir` is open.
                check(unsafe { libc::unlinkat(dir.as_raw_fd(), name.as_ptr(), at) })?;
            }
            TLINK => {
                let dfid = get!(r.u32());
                let fid = get!(r.u32());
                let name = get!(r.string());
                self.writable()?;
                let to = self.fid(dfid)?.path.clone();
                let to = self.dir(&to)?;
                let from = self.fid(fid)?.path.clone();
                let (from_dir, from_name) = self.parent(&from)?;
                let name = entry_name(name)?;
                // SAFETY: both names are NUL-terminated and both directories open.
                check(unsafe {
                    libc::linkat(from_dir.as_raw_fd(), from_name.as_ptr(), to.as_raw_fd(), name.as_ptr(), 0)
                })?;
            }
            TREADLINK => {
                let fid = get!(r.u32());
                let path = self.fid(fid)?.path.clone();
                let (dir, name) = self.parent(&path)?;
                let mut buf = vec![0u8; 4096];
                // SAFETY: `buf` has room for the length given.
                let n = unsafe {
                    libc::readlinkat(dir.as_raw_fd(), name.as_ptr(), buf.as_mut_ptr().cast(), buf.len())
                };
                if n < 0 {
                    return Err(io::Error::last_os_error());
                }
                w.string(&buf[..n as usize]);
            }
            TGETATTR => {
                let fid = get!(r.u32());
                get!(r.u64()); // request_mask: everything is cheap
                let st = self.stat_fid(self.fid(fid)?)?;
                self.write_attr(&mut w, &st);
            }
            TSETATTR => {
                let fid = get!(r.u32());
                let valid = get!(r.u32());
                let mode = get!(r.u32());
                get!(r.u32()); // uid
                get!(r.u32()); // gid
                let size = get!(r.u64());
                let atime = (get!(r.u64()), get!(r.u64()));
                let mtime = (get!(r.u64()), get!(r.u64()));
                self.setattr(fid, valid, mode, size, atime, mtime)?;
            }
            TXATTRWALK | TXATTRCREATE | TLOCK | TGETLOCK => return Err(err(libc::ENOTSUP)),
            TREADDIR => {
                let fid = get!(r.u32());
                let offset = get!(r.u64());
                let count = get!(r.u32());
                self.readdir(fid, offset, count, &mut w)?;
            }
            TFSYNC => {
                let fid = get!(r.u32());
                match &self.fid(fid)?.open {
                    Open::File { fd } | Open::Dir { fd, .. } => {
                        // SAFETY: `fd` is open.
                        check(unsafe { libc::fsync(fd.as_raw_fd()) })?;
                    }
                    Open::No => return Err(err(libc::EBADF)),
                }
            }
            TSTATFS => {
                let fid = get!(r.u32());
                self.statfs(fid, &mut w)?;
            }
            TREAD => {
                let fid = get!(r.u32());
                let offset = get!(r.u64());
                let count = get!(r.u32());
                let Open::File { fd } = &self.fid(fid)?.open else { return Err(err(libc::EBADF)) };
                let count = count.min(self.msize - 11) as usize;
                let mut data = vec![0u8; count];
                // SAFETY: `data` has room for `count` bytes.
                let n = unsafe { libc::pread(fd.as_raw_fd(), data.as_mut_ptr().cast(), count, offset as libc::off_t) };
                if n < 0 {
                    return Err(io::Error::last_os_error());
                }
                w.u32(n as u32);
                w.0.extend_from_slice(&data[..n as usize]);
            }
            TWRITE => {
                let fid = get!(r.u32());
                let offset = get!(r.u64());
                let count = get!(r.u32());
                let data = get!(r.bytes(count as usize));
                let Open::File { fd } = &self.fid(fid)?.open else { return Err(err(libc::EBADF)) };
                // SAFETY: `data` is `count` bytes long.
                let n = unsafe { libc::pwrite(fd.as_raw_fd(), data.as_ptr().cast(), data.len(), offset as libc::off_t) };
                if n < 0 {
                    return Err(io::Error::last_os_error());
                }
                w.u32(n as u32);
            }
            TCLUNK => {
                let fid = get!(r.u32());
                self.fids.remove(&fid).ok_or_else(|| err(libc::EBADF))?;
            }
            TREMOVE => {
                let fid = get!(r.u32());
                let f = self.fids.remove(&fid).ok_or_else(|| err(libc::EBADF))?;
                self.writable()?;
                let is_dir = self.stat_fid(&f)?.st_mode as u32 & libc::S_IFMT as u32 == libc::S_IFDIR as u32;
                let (dir, name) = self.parent(&f.path)?;
                let at = if is_dir { libc::AT_REMOVEDIR } else { 0 };
                // SAFETY: `name` is NUL-terminated and `dir` is open.
                check(unsafe { libc::unlinkat(dir.as_raw_fd(), name.as_ptr(), at) })?;
            }
            _ => return Err(err(libc::ENOSYS)),
        }
        Ok(Some(w))
    }

    fn attach(&mut self, fid: u32, afid: u32, aname: &[u8]) -> io::Result<Qid> {
        if afid != NOFID {
            return Err(err(libc::EINVAL));
        }
        if self.fids.contains_key(&fid) {
            return Err(err(libc::EBADF));
        }
        // An aname selects a directory under the share, with no way out of it.
        let mut path = Vec::new();
        for c in aname.split(|&b| b == b'/').filter(|c| !c.is_empty() && *c != b".") {
            if c == b".." {
                return Err(err(libc::EACCES));
            }
            path.push(component(c)?.to_owned());
        }
        let st = fstat(&self.dir(&path)?)?;
        self.fids.insert(fid, Fid { path, open: Open::No });
        Ok(qid_of(&st))
    }

    fn walk(&mut self, fid: u32, newfid: u32, names: &[&[u8]]) -> io::Result<Vec<Qid>> {
        let start = self.fid(fid)?;
        if !matches!(start.open, Open::No) {
            return Err(err(libc::EBUSY));
        }
        if newfid != fid && self.fids.contains_key(&newfid) {
            return Err(err(libc::EBADF));
        }
        let mut path = start.path.clone();
        let mut qids = Vec::with_capacity(names.len());
        for (i, &name) in names.iter().enumerate() {
            let step = (|| -> io::Result<Qid> {
                if name == b".." {
                    path.pop(); // at the root, stays at the root
                } else if name != b"." {
                    let c = component(name)?;
                    let dir = self.dir(&path)?;
                    let st = lstatat(&dir, &cstr(c)?)?;
                    path.push(c.to_owned());
                    return Ok(qid_of(&st));
                }
                Ok(qid_of(&fstat(&self.dir(&path)?)?))
            })();
            match step {
                Ok(q) => qids.push(q),
                // A failure after the first step is reported by a short reply.
                Err(e) if i == 0 => return Err(e),
                Err(_) => return Ok(qids),
            }
        }
        self.fids.insert(newfid, Fid { path, open: Open::No });
        Ok(qids)
    }

    fn lopen(&mut self, fid: u32, flags: u32) -> io::Result<Qid> {
        let f = self.fid(fid)?;
        if !matches!(f.open, Open::No) {
            return Err(err(libc::EBUSY));
        }
        let path = f.path.clone();
        let st = self.stat_path(&path)?;
        let fmt = st.st_mode as u32 & libc::S_IFMT as u32;
        let open = if fmt == libc::S_IFDIR as u32 {
            if flags & L_O_ACCMODE != 0 {
                return Err(err(libc::EISDIR));
            }
            Open::Dir { fd: self.dir(&path)?, entries: Vec::new() }
        } else if fmt == libc::S_IFREG as u32 {
            let (dir, name) = self.parent(&path)?;
            Open::File { fd: openat(&dir, &name, self.host_flags(flags)?, 0)? }
        } else {
            // Symlinks are read with Treadlink; a host's FIFOs and devices
            // would only block this thread.
            return Err(err(libc::ELOOP));
        };
        let qid = qid_of(&st);
        self.fids.get_mut(&fid).unwrap().open = open;
        Ok(qid)
    }

    fn lcreate(&mut self, fid: u32, name: &[u8], flags: u32, mode: u32) -> io::Result<Qid> {
        self.writable()?;
        let name = entry_name(name)?;
        let f = self.fid(fid)?;
        if !matches!(f.open, Open::No) {
            return Err(err(libc::EBUSY));
        }
        let dir = self.dir(&f.path)?;
        let host = self.host_flags(flags | L_O_CREAT)? | libc::O_CREAT;
        let fd = openat(&dir, &name, host, mode & 0o7777)?;
        // SAFETY: `fd` is open.
        let _ = unsafe { libc::fchown(fd.as_raw_fd(), self.uid as libc::uid_t, self.gid as libc::gid_t) };
        let st = fstat(&fd)?;
        let f = self.fids.get_mut(&fid).unwrap();
        f.path.push(OsStr::from_bytes(name.as_bytes()).to_owned());
        f.open = Open::File { fd };
        Ok(qid_of(&st))
    }

    fn symlink(&mut self, fid: u32, name: &[u8], target: &[u8]) -> io::Result<Qid> {
        self.writable()?;
        let name = entry_name(name)?;
        let target = CString::new(target).map_err(|_| err(libc::EINVAL))?;
        let dir = self.dir(&self.fid(fid)?.path)?;
        // SAFETY: both strings are NUL-terminated and `dir` is open.
        check(unsafe { libc::symlinkat(target.as_ptr(), dir.as_raw_fd(), name.as_ptr()) })?;
        self.own(&dir, &name);
        Ok(qid_of(&lstatat(&dir, &name)?))
    }

    /// Hand a directory or symlink the guest just made to the configured `uid` and
    /// `gid` (a file is handed over where it is made). It works while IRIS may
    /// chown, as root with CAP_CHOWN does; otherwise the entry stays the host
    /// user's, as before, and nothing is reported.
    fn own(&self, dir: &OwnedFd, name: &CStr) {
        // SAFETY: `name` is NUL-terminated and `dir` is open.
        let _ = unsafe {
            libc::fchownat(dir.as_raw_fd(), name.as_ptr(), self.uid as libc::uid_t, self.gid as libc::gid_t, libc::AT_SYMLINK_NOFOLLOW)
        };
    }
    fn mkdir(&mut self, fid: u32, name: &[u8], mode: u32) -> io::Result<Qid> {
        self.writable()?;
        let name = entry_name(name)?;
        let dir = self.dir(&self.fid(fid)?.path)?;
        // SAFETY: `name` is NUL-terminated and `dir` is open.
        check(unsafe { libc::mkdirat(dir.as_raw_fd(), name.as_ptr(), (mode & 0o7777) as libc::mode_t) })?;
        self.own(&dir, &name);
        Ok(qid_of(&lstatat(&dir, &name)?))
    }

    fn rename(&mut self, olddir: &[OsString], oldname: &[u8], newdir: &[OsString], newname: &[u8]) -> io::Result<()> {
        self.writable()?;
        let old = entry_name(oldname)?;
        let new = entry_name(newname)?;
        let from = self.dir(olddir)?;
        let to = self.dir(newdir)?;
        // SAFETY: both names are NUL-terminated and both directories open.
        check(unsafe { libc::renameat(from.as_raw_fd(), old.as_ptr(), to.as_raw_fd(), new.as_ptr()) })?;
        // Fids are paths, so everything at or under the old name moves with it.
        let mut before = olddir.to_vec();
        before.push(OsStr::from_bytes(old.as_bytes()).to_owned());
        let mut after = newdir.to_vec();
        after.push(OsStr::from_bytes(new.as_bytes()).to_owned());
        for f in self.fids.values_mut() {
            if f.path.starts_with(&before) {
                let rest = f.path.split_off(before.len());
                f.path = after.clone();
                f.path.extend(rest);
            }
        }
        Ok(())
    }

    fn write_attr(&self, w: &mut Writer, st: &libc::stat) {
        w.u64(GETATTR_BASIC);
        w.qid(qid_of(st));
        w.u32(st.st_mode as u32);
        w.u32(self.uid);
        w.u32(self.gid);
        w.u64(st.st_nlink as u64);
        w.u64(st.st_rdev as u64);
        w.u64(st.st_size as u64);
        w.u64(st.st_blksize as u64);
        w.u64(st.st_blocks as u64);
        w.u64(st.st_atime as u64).u64(st.st_atime_nsec as u64);
        w.u64(st.st_mtime as u64).u64(st.st_mtime_nsec as u64);
        w.u64(st.st_ctime as u64).u64(st.st_ctime_nsec as u64);
        // btime, gen and data_version: not in the mask.
        w.u64(0).u64(0).u64(0).u64(0);
    }

    fn setattr(&mut self, fid: u32, valid: u32, mode: u32, size: u64, atime: (u64, u64), mtime: (u64, u64)) -> io::Result<()> {
        self.writable()?;
        let f = self.fid(fid)?;
        let (dir, name) = self.parent(&f.path)?;
        let fmt = self.stat_fid(f)?.st_mode as u32 & libc::S_IFMT as u32;
        if valid & SETATTR_MODE != 0 && fmt != libc::S_IFLNK as u32 {
            let mode = (mode & 0o7777) as libc::mode_t;
            match &f.open {
                Open::File { fd } | Open::Dir { fd, .. } => {
                    // SAFETY: `fd` is open.
                    check(unsafe { libc::fchmod(fd.as_raw_fd(), mode) })?;
                }
                Open::No => {
                    // SAFETY: `name` is NUL-terminated and `dir` is open.
                    check(unsafe { libc::fchmodat(dir.as_raw_fd(), name.as_ptr(), mode, 0) })?;
                }
            }
        }
        if valid & SETATTR_SIZE != 0 {
            if fmt != libc::S_IFREG as u32 {
                return Err(err(libc::EINVAL));
            }
            let fd = openat(&dir, &name, libc::O_WRONLY | libc::O_NOFOLLOW, 0).or_else(|e| match &f.open {
                // A file removed while open has no name left to open it by.
                Open::File { fd } => dup(fd),
                _ => Err(e),
            })?;
            // SAFETY: `fd` is open.
            check(unsafe { libc::ftruncate(fd.as_raw_fd(), size as libc::off_t) })?;
        }
        if valid & (SETATTR_ATIME | SETATTR_MTIME) != 0 {
            let spec = |set: bool, given: bool, v: (u64, u64)| libc::timespec {
                tv_sec: if given { v.0 as libc::time_t } else { 0 },
                tv_nsec: match (set, given) {
                    (false, _) => libc::UTIME_OMIT as _,
                    (true, false) => libc::UTIME_NOW as _,
                    (true, true) => v.1 as _,
                },
            };
            let times = [
                spec(valid & SETATTR_ATIME != 0, valid & SETATTR_ATIME_SET != 0, atime),
                spec(valid & SETATTR_MTIME != 0, valid & SETATTR_MTIME_SET != 0, mtime),
            ];
            // SAFETY: `times` holds the two entries the call reads.
            check(unsafe { libc::utimensat(dir.as_raw_fd(), name.as_ptr(), times.as_ptr(), libc::AT_SYMLINK_NOFOLLOW) })?;
        }
        Ok(())
    }

    fn scan(fd: &OwnedFd) -> io::Result<Vec<DirEntry>> {
        let dev = fstat(fd)?.st_dev as u64;
        let dup = dup(fd)?;
        // SAFETY: fdopendir takes over the duplicate on success, and closedir
        // releases it; readdir's result is read before the next call.
        unsafe {
            let raw = dup.as_raw_fd();
            let d = libc::fdopendir(raw);
            if d.is_null() {
                return Err(io::Error::last_os_error());
            }
            std::mem::forget(dup);
            libc::rewinddir(d);
            let mut out = Vec::new();
            loop {
                let e = libc::readdir(d);
                if e.is_null() {
                    break;
                }
                let name = CStr::from_ptr((*e).d_name.as_ptr()).to_bytes().to_vec();
                let d_type = (*e).d_type;
                let ty = match d_type {
                    libc::DT_DIR => QTDIR,
                    libc::DT_LNK => QTSYMLINK,
                    _ => QTFILE,
                };
                out.push(DirEntry {
                    name,
                    qid: Qid { ty, version: 0, path: qid_path(dev, (*e).d_ino as u64) },
                    d_type,
                });
            }
            libc::closedir(d);
            Ok(out)
        }
    }

    fn readdir(&mut self, fid: u32, offset: u64, count: u32, w: &mut Writer) -> io::Result<()> {
        let msize = self.msize;
        let f = self.fids.get_mut(&fid).ok_or_else(|| err(libc::EBADF))?;
        let Open::Dir { fd, entries } = &mut f.open else { return Err(err(libc::ENOTDIR)) };
        if offset == 0 {
            *entries = Self::scan(fd)?;
        }
        let room = count.min(msize - 11) as usize;
        let mut data = Writer::default();
        let start = usize::try_from(offset).unwrap_or(usize::MAX);
        for (i, e) in entries.iter().enumerate().skip(start) {
            // qid[13] offset[8] type[1] name[s]
            if data.0.len() + 24 + e.name.len() > room {
                break;
            }
            data.qid(e.qid).u64(i as u64 + 1).u8(e.d_type).string(&e.name);
        }
        w.u32(data.0.len() as u32);
        w.0.extend_from_slice(&data.0);
        Ok(())
    }

    fn statfs(&self, fid: u32, w: &mut Writer) -> io::Result<()> {
        let f = self.fid(fid)?;
        let fd = match &f.open {
            Open::File { fd } | Open::Dir { fd, .. } => dup(fd)?,
            Open::No => self.parent(&f.path)?.0,
        };
        // SAFETY: `s` is written by fstatvfs before it is read.
        let s = unsafe {
            let mut s: libc::statvfs = std::mem::zeroed();
            check(libc::fstatvfs(fd.as_raw_fd(), &mut s))?;
            s
        };
        w.u32(V9FS_MAGIC);
        w.u32(s.f_frsize as u32);
        w.u64(s.f_blocks as u64).u64(s.f_bfree as u64).u64(s.f_bavail as u64);
        w.u64(s.f_files as u64).u64(s.f_ffree as u64);
        w.u64(s.f_fsid as u64);
        w.u32(s.f_namemax as u32);
        Ok(())
    }
}

/// The device: a mount tag and a server behind one queue.
pub struct P9 {
    tag: Vec<u8>,
    server: Server,
}

impl P9 {
    pub fn new(tag: &str, server: Server) -> Self {
        Self { tag: tag.as_bytes().to_vec(), server }
    }
}

impl VirtioDevice for P9 {
    fn device_id(&self) -> u32 {
        DEVICE_ID
    }

    fn features(&self) -> u64 {
        F_MOUNT_TAG
    }

    fn num_queues(&self, _features: u64) -> usize {
        1
    }

    fn config_len(&self) -> usize {
        2 + self.tag.len()
    }

    fn config_read(&self, off: usize, buf: &mut [u8]) {
        let mut c = (self.tag.len() as u16).to_le_bytes().to_vec();
        c.extend_from_slice(&self.tag);
        let n = buf.len().min(c.len().saturating_sub(off));
        buf[..n].copy_from_slice(&c[off..off + n]);
    }

    fn reset(&mut self) {
        self.server.reset();
    }

    fn notify(&mut self, q: usize, cx: &mut Cx) {
        if q != 0 {
            return;
        }
        while let Some(chain) = cx.queues[0].pop(cx.mem) {
            // A message's own length field bounds it inside the buffers given,
            // whose sizes say nothing about it.
            let written = chain
                .read_all(cx.mem)
                .and_then(|msg| {
                    let len = msg.get(..4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)?;
                    let reply = self.server.handle(msg.get(..len)?);
                    // A reply too long for what the driver offered must not be
                    // cut: the driver would parse a truncated message.
                    if reply.len() > chain.writable_len() {
                        return None;
                    }
                    chain.write_all(cx.mem, &reply)
                })
                .unwrap_or(0);
            cx.used(0, chain.head, written as u32);
        }
    }

    fn poll(&mut self, _cx: &mut Cx) {}
}

/// Build the device for `[virtio_9p]`.
pub fn from_config(cfg: &crate::config::Virtio9pConfig) -> VirtioCore {
    let server = match Server::new(Path::new(&cfg.path), cfg.readonly, cfg.uid, cfg.gid) {
        Ok(s) => {
            eprintln!(
                "iris: virtio-9p share {:?} is {}{}",
                cfg.tag,
                cfg.path,
                if cfg.readonly { " (read-only)" } else { "" }
            );
            s
        }
        Err(e) => {
            eprintln!("iris: virtio-9p share {:?}: {}: {e}", cfg.tag, cfg.path);
            Server::absent(cfg.readonly, cfg.uid, cfg.gid)
        }
    };
    VirtioCore::new(Box::new(P9::new(&cfg.tag, server)), Waker::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs as ufs;
    use std::path::PathBuf;

    struct Rig {
        dir: PathBuf,
        s: Server,
    }

    impl Rig {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("iris-9p-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(dir.join("share")).unwrap();
            let s = Server::new(&dir.join("share"), false, 7, 8).unwrap();
            Self { dir, s }
        }
        fn share(&self) -> PathBuf {
            self.dir.join("share")
        }
        /// Send a message; returns the reply's type and body.
        fn rpc(&mut self, ty: u8, f: impl FnOnce(&mut Writer)) -> (u8, Vec<u8>) {
            let mut w = Writer::default();
            f(&mut w);
            let msg = Server::frame(ty, 1, w);
            let r = self.s.handle(&msg);
            assert_eq!(u32::from_le_bytes(r[..4].try_into().unwrap()) as usize, r.len());
            assert_eq!(u16::from_le_bytes(r[5..7].try_into().unwrap()), 1, "tag comes back");
            (r[4], r[7..].to_vec())
        }
        fn ok(&mut self, ty: u8, f: impl FnOnce(&mut Writer)) -> Vec<u8> {
            let (rt, body) = self.rpc(ty, f);
            assert_eq!(rt, ty + 1, "error {:?}", body);
            body
        }
        fn errno(&mut self, ty: u8, f: impl FnOnce(&mut Writer)) -> u32 {
            let (rt, body) = self.rpc(ty, f);
            assert_eq!(rt, RLERROR, "expected an error, got {:?}", body);
            u32::from_le_bytes(body[..4].try_into().unwrap())
        }
        fn attach(&mut self, fid: u32) {
            self.ok(TVERSION, |w| {
                w.u32(65536).string(DIALECT);
            });
            self.ok(TATTACH, |w| {
                w.u32(fid).u32(NOFID).string(b"root").string(b"").u32(0);
            });
        }
        fn walk(&mut self, fid: u32, newfid: u32, names: &[&[u8]]) -> u16 {
            let body = self.ok(TWALK, |w| {
                w.u32(fid).u32(newfid).u16(names.len() as u16);
                for n in names {
                    w.string(n);
                }
            });
            u16::from_le_bytes(body[..2].try_into().unwrap())
        }
        fn read(&mut self, fid: u32, off: u64, count: u32) -> Vec<u8> {
            let body = self.ok(TREAD, |w| {
                w.u32(fid).u64(off).u32(count);
            });
            body[4..].to_vec()
        }
        fn names(&mut self, fid: u32) -> Vec<String> {
            let mut out = Vec::new();
            let mut off = 0;
            loop {
                let body = self.ok(TREADDIR, |w| {
                    w.u32(fid).u64(off).u32(200);
                });
                let n = u32::from_le_bytes(body[..4].try_into().unwrap()) as usize;
                if n == 0 {
                    return out;
                }
                let mut r = Reader { buf: &body[4..4 + n], at: 0 };
                while r.at < n {
                    r.bytes(13).unwrap();
                    off = r.u64().unwrap();
                    r.u8().unwrap();
                    out.push(String::from_utf8(r.string().unwrap().to_vec()).unwrap());
                }
            }
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn version_negotiates_size_and_dialect() {
        let mut r = Rig::new("version");
        let b = r.ok(TVERSION, |w| {
            w.u32(8192).string(DIALECT);
        });
        assert_eq!(u32::from_le_bytes(b[..4].try_into().unwrap()), 8192);
        assert_eq!(&b[6..], DIALECT);
        let b = r.ok(TVERSION, |w| {
            w.u32(u32::MAX).string(b"9P2000");
        });
        assert_eq!(u32::from_le_bytes(b[..4].try_into().unwrap()), MAX_MSIZE);
        assert_eq!(&b[6..], b"unknown");
    }

    #[test]
    fn create_write_read_stat_remove() {
        let mut r = Rig::new("rw");
        r.attach(1);
        assert_eq!(r.walk(1, 2, &[]), 0);
        r.ok(TLCREATE, |w| {
            w.u32(2).string(b"f").u32(L_O_RDWR).u32(0o640).u32(0);
        });
        let b = r.ok(TWRITE, |w| {
            w.u32(2).u64(3).u32(5);
            w.0.extend_from_slice(b"hello");
        });
        assert_eq!(u32::from_le_bytes(b[..4].try_into().unwrap()), 5);
        assert_eq!(r.read(2, 3, 100), b"hello");
        assert_eq!(fs::read(r.share().join("f")).unwrap(), b"\0\0\0hello");

        // getattr: valid[8] qid[13] mode[4] uid[4] gid[4] nlink[8] rdev[8] size[8]
        let b = r.ok(TGETATTR, |w| {
            w.u32(2).u64(GETATTR_BASIC);
        });
        let mut g = Reader { buf: &b, at: 8 + 13 };
        assert_eq!(g.u32().unwrap() & 0o7777, 0o640);
        assert_eq!(g.u32().unwrap(), 7, "configured uid");
        assert_eq!(g.u32().unwrap(), 8, "configured gid");
        g.u64().unwrap();
        g.u64().unwrap();
        assert_eq!(g.u64().unwrap(), 8);

        r.ok(TCLUNK, |w| {
            w.u32(2);
        });
        r.ok(TUNLINKAT, |w| {
            w.u32(1).string(b"f").u32(0);
        });
        assert!(!r.share().join("f").exists());
    }

    #[test]
    fn open_file_survives_removal() {
        let mut r = Rig::new("unlinked");
        fs::write(r.share().join("f"), b"still here").unwrap();
        r.attach(1);
        r.walk(1, 2, &[b"f"]);
        r.ok(TLOPEN, |w| {
            w.u32(2).u32(0);
        });
        r.ok(TUNLINKAT, |w| {
            w.u32(1).string(b"f").u32(0);
        });
        assert_eq!(r.read(2, 0, 100), b"still here");
        r.ok(TGETATTR, |w| {
            w.u32(2).u64(GETATTR_BASIC);
        });
    }

    #[test]
    fn walk_reports_how_far_it_got() {
        let mut r = Rig::new("walk");
        fs::create_dir(r.share().join("d")).unwrap();
        r.attach(1);
        // First step fails: an error.
        assert_eq!(
            r.errno(TWALK, |w| {
                w.u32(1).u32(2).u16(1).string(b"nope");
            }),
            2
        );
        // A later step fails: the qids so far, and no new fid.
        assert_eq!(r.walk(1, 2, &[b"d", b"nope"]), 1);
        assert_eq!(
            r.errno(TGETATTR, |w| {
                w.u32(2).u64(0);
            }),
            9
        );
        assert_eq!(r.walk(1, 2, &[b"d"]), 1);
    }

    #[test]
    fn the_share_cannot_be_left() {
        let mut r = Rig::new("escape");
        fs::write(r.dir.join("secret"), b"host file").unwrap();
        fs::create_dir(r.share().join("d")).unwrap();
        ufs::symlink("../secret", r.share().join("link")).unwrap();
        ufs::symlink("..", r.share().join("up")).unwrap();
        r.attach(1);

        // `..` at the root is the root.
        assert_eq!(r.walk(1, 2, &[b"..", b".."]), 2);
        assert!(r.names_of_root_contain(2, "link"));
        // A slash in a name is refused outright.
        assert_eq!(
            r.errno(TWALK, |w| {
                w.u32(1).u32(3).u16(1).string(b"../secret");
            }),
            22
        );
        // A symlink is not followed when opened, or when walked through.
        r.walk(1, 4, &[b"link"]);
        assert_eq!(
            r.errno(TLOPEN, |w| {
                w.u32(4).u32(0);
            }),
            40
        );
        r.walk(1, 5, &[b"up"]);
        assert_eq!(r.walk(1, 6, &[b"up", b"secret"]), 1);
        // Neither aname can reach out.
        assert_eq!(
            r.errno(TATTACH, |w| {
                w.u32(7).u32(NOFID).string(b"").string(b"../").u32(0);
            }),
            13
        );
        // Readlink gives the guest the target to resolve in its own namespace.
        let b = r.ok(TREADLINK, |w| {
            w.u32(4);
        });
        assert_eq!(&b[2..], b"../secret");
    }

    impl Rig {
        fn names_of_root_contain(&mut self, fid: u32, name: &str) -> bool {
            self.walk(fid, 99, &[]);
            self.ok(TLOPEN, |w| {
                w.u32(99).u32(0);
            });
            let found = self.names(99).iter().any(|n| n == name);
            self.ok(TCLUNK, |w| {
                w.u32(99);
            });
            found
        }
    }

    #[test]
    fn readdir_resumes_from_its_offset_in_small_pieces() {
        let mut r = Rig::new("readdir");
        for i in 0..50 {
            fs::write(r.share().join(format!("file{i:02}")), b"").unwrap();
        }
        r.attach(1);
        r.walk(1, 2, &[]);
        r.ok(TLOPEN, |w| {
            w.u32(2).u32(0);
        });
        let mut names = r.names(2);
        names.sort();
        names.retain(|n| n.starts_with("file"));
        assert_eq!(names.len(), 50);
        assert_eq!(names[49], "file49");
    }

    #[test]
    fn rename_moves_fids_under_the_old_name() {
        let mut r = Rig::new("rename");
        fs::create_dir_all(r.share().join("a/b")).unwrap();
        fs::write(r.share().join("a/b/f"), b"x").unwrap();
        r.attach(1);
        r.walk(1, 2, &[b"a", b"b", b"f"]);
        r.ok(TRENAMEAT, |w| {
            w.u32(1).string(b"a").u32(1).string(b"z");
        });
        r.ok(TLOPEN, |w| {
            w.u32(2).u32(0);
        });
        assert_eq!(r.read(2, 0, 10), b"x");
        assert!(r.share().join("z/b/f").exists());
        // A name holding a path is no name.
        assert_eq!(
            r.errno(TRENAMEAT, |w| {
                w.u32(1).string(b"z").u32(1).string(b"../out");
            }),
            22
        );
    }

    #[test]
    fn links_directories_and_setattr() {
        let mut r = Rig::new("misc");
        r.attach(1);
        r.ok(TMKDIR, |w| {
            w.u32(1).string(b"d").u32(0o755).u32(0);
        });
        r.ok(TSYMLINK, |w| {
            w.u32(1).string(b"s").string(b"d").u32(0);
        });
        assert_eq!(fs::read_link(r.share().join("s")).unwrap().to_str(), Some("d"));
        fs::write(r.share().join("f"), b"123456").unwrap();
        r.walk(1, 2, &[b"f"]);
        r.ok(TLINK, |w| {
            w.u32(1).u32(2).string(b"hard");
        });
        assert_eq!(fs::read(r.share().join("hard")).unwrap(), b"123456");
        r.ok(TSETATTR, |w| {
            w.u32(2).u32(SETATTR_SIZE | SETATTR_MODE | SETATTR_MTIME | SETATTR_MTIME_SET);
            w.u32(0o600).u32(0).u32(0).u64(2);
            w.u64(0).u64(0).u64(1_000_000).u64(0);
        });
        let m = fs::metadata(r.share().join("f")).unwrap();
        assert_eq!(m.len(), 2);
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        assert_eq!(m.permissions().mode() & 0o777, 0o600);
        assert_eq!(m.mtime(), 1_000_000);
        // Removing a directory needs the flag.
        assert_ne!(
            r.errno(TUNLINKAT, |w| {
                w.u32(1).string(b"d").u32(0);
            }),
            0
        );
        r.ok(TUNLINKAT, |w| {
            w.u32(1).string(b"d").u32(L_AT_REMOVEDIR);
        });
        // statfs answers with the 9p magic.
        let b = r.ok(TSTATFS, |w| {
            w.u32(1);
        });
        assert_eq!(u32::from_le_bytes(b[..4].try_into().unwrap()), V9FS_MAGIC);
    }

    #[test]
    fn errnos_are_linux_numbers() {
        assert_eq!(linux_errno(&err(libc::ENOTEMPTY)), 39);
        assert_eq!(linux_errno(&err(libc::ELOOP)), 40);
        assert_eq!(linux_errno(&err(libc::ENOTSUP)), 95);
        assert_eq!(linux_errno(&err(libc::ENOENT)), 2);
    }

    #[test]
    fn a_read_only_share_refuses_every_write() {
        let dir = std::env::temp_dir().join(format!("iris-9p-{}-ro", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("f"), b"x").unwrap();
        let mut r = Rig { s: Server::new(&dir, true, 0, 0).unwrap(), dir: dir.clone() };
        r.attach(1);
        r.walk(1, 2, &[b"f"]);
        for flags in [L_O_WRONLY, L_O_RDWR, L_O_TRUNC] {
            assert_eq!(
                r.errno(TLOPEN, |w| {
                    w.u32(2).u32(flags);
                }),
                30
            );
        }
        assert_eq!(
            r.errno(TMKDIR, |w| {
                w.u32(1).string(b"d").u32(0o755).u32(0);
            }),
            30
        );
        assert_eq!(
            r.errno(TUNLINKAT, |w| {
                w.u32(1).string(b"f").u32(0);
            }),
            30
        );
        r.ok(TLOPEN, |w| {
            w.u32(2).u32(0);
        });
        assert_eq!(r.read(2, 0, 1), b"x");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn short_messages_are_refused_not_panicked_on() {
        let mut r = Rig::new("short");
        r.attach(1);
        for ty in [TWALK, TREAD, TWRITE, TLCREATE, TSETATTR, TREADDIR, TRENAMEAT] {
            let msg = Server::frame(ty, 1, Writer::default());
            assert_eq!(r.s.handle(&msg)[4], RLERROR);
        }
        let b = r.s.handle(&[0, 0, 0]);
        assert!(b.is_empty());
    }

    #[test]
    fn config_space_is_the_tag() {
        let d = P9::new("share", Server::absent(false, 0, 0));
        assert_eq!(d.config_len(), 7);
        let mut b = [0u8; 7];
        d.config_read(0, &mut b);
        assert_eq!(&b, b"\x05\x00share");
        let mut b = [0u8; 2];
        d.config_read(5, &mut b);
        assert_eq!(&b, b"re");
    }

    #[test]
    fn requests_come_in_on_the_queue() {
        use crate::dev::virtio::queue::tests::{make_avail, put_desc, setup, used_entry, used_idx};
        use crate::dev::virtio::{Cx, GuestMemory, VecMemory};

        let mem = VecMemory::new(0x10000);
        let mut queues = vec![setup(8)];
        let msg = {
            let mut w = Writer::default();
            w.u32(65536).string(DIALECT);
            Server::frame(TVERSION, 0xffff, w)
        };
        assert!(mem.write(0x4000, &msg));
        // Split the reply over two writable buffers, as a driver may.
        put_desc(&mem, 0, 0x4000, msg.len() as u32, 1, 1);
        put_desc(&mem, 1, 0x5000, 5, 1 | 2, 2);
        put_desc(&mem, 2, 0x6000, 100, 2, 0);
        make_avail(&mem, &queues[0], 8, 0);

        let mut dev = P9::new("t", Server::absent(false, 0, 0));
        let mut cx = Cx { queues: &mut queues, mem: &mem, interrupt: false, config_changed: false };
        dev.notify(0, &mut cx);
        assert!(cx.interrupt);
        assert_eq!(used_idx(&mem, &queues[0]), 1);
        let (head, written) = used_entry(&mem, &queues[0], 0);
        assert_eq!(head, 0);
        assert_eq!(written, 21);
        let mut first = [0u8; 5];
        mem.read(0x5000, &mut first);
        assert_eq!(first[4], TVERSION + 1);
        let mut rest = [0u8; 16];
        mem.read(0x6000, &mut rest);
        assert_eq!(&rest[8..16], DIALECT);
    }
}
