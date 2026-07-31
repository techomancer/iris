//! In-core NFS server (NFSv2 + NFSv3) over UDP, dispatched by the NAT engine.
//!
//! The whole protocol stays inside the NAT: `src/net.rs` hands guest MOUNT/NFS
//! RPC datagrams to this module and injects the reply bytes back as
//! virtual-network frames — there are **no host network sockets**. The only host
//! interaction is file I/O against the user-chosen backing folder.
//!
//! Plan + open questions: `docs/nfsudp-plan.md`. Wire structs/semantics are
//! modelled on `nfsserve` (BSD-3-Clause), re-implemented synchronously here.
//!
//! This file is built bottom-up. **Increment 1 (this commit): the
//! version-agnostic backend** — the `fileid`↔path map, path containment, the
//! faked/synthetic unix attributes, and the filesystem operations. The XDR/RPC
//! layer, the v2/v3 procedure encoders, MOUNT, the NAT wiring, and the GUI land
//! on top of this.

#![allow(dead_code)] // wire layer that consumes the backend lands in later increments

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// The reserved NFS `fileid` of the export root.
pub const ROOT_ID: u64 = 1;

/// File kind, mapped to NFS `ftype3` (and the v2 equivalent) by the wire layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Reg,
    Dir,
    Lnk,
    Other,
}

/// Synthetic POSIX attributes for one object. "Faked" deliberately: uid/gid are
/// fixed and the mode is a heuristic, so the export behaves identically on
/// Linux, macOS, and Windows (which has no unix uid/gid/mode at all).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
    pub kind: FileKind,
    /// Full mode incl. type bits (e.g. `0o040755` for a dir) — convenience for
    /// the wire layer; the type bits come from `kind`.
    pub mode: u32,
    pub nlink: u32,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub fileid: u64,
    /// (seconds, nanoseconds) since the unix epoch.
    pub atime: (u32, u32),
    pub mtime: (u32, u32),
    pub ctime: (u32, u32),
}

/// POSIX `S_IF*` type bits, for the `mode` field.
const S_IFDIR: u32 = 0o040000;
const S_IFREG: u32 = 0o100000;
const S_IFLNK: u32 = 0o120000;

/// The backing store: one exported directory, plus a stable `fileid`↔relative
/// path map so the guest's opaque file handles resolve back to host paths. Used
/// only from the NAT thread, so plain `&mut self` (no locking).
pub struct NfsBacking {
    /// Absolute path of the exported folder. All access is contained within it.
    root: PathBuf,
    next_id: u64,
    /// fileid → path relative to `root` (root itself = ROOT_ID = empty path).
    id_to_path: HashMap<u64, PathBuf>,
    /// reverse map so re-looking-up a path returns its existing id.
    path_to_id: HashMap<PathBuf, u64>,
    /// Default owner for faked attributes.
    uid: u32,
    gid: u32,
}

impl NfsBacking {
    /// Create a backing store over `root`. `root` should be an existing,
    /// absolute directory; access is confined to it.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let mut id_to_path = HashMap::new();
        let mut path_to_id = HashMap::new();
        id_to_path.insert(ROOT_ID, PathBuf::new());
        path_to_id.insert(PathBuf::new(), ROOT_ID);
        Self {
            root: root.into(),
            next_id: ROOT_ID + 1,
            id_to_path,
            path_to_id,
            uid: 0,
            gid: 0,
        }
    }

    /// Intern a root-relative path, returning a stable fileid for it.
    fn intern(&mut self, rel: PathBuf) -> u64 {
        if let Some(&id) = self.path_to_id.get(&rel) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.id_to_path.insert(id, rel.clone());
        self.path_to_id.insert(rel, id);
        id
    }

    /// The root-relative path for a fileid, if known.
    fn rel_of(&self, id: u64) -> Option<&PathBuf> {
        self.id_to_path.get(&id)
    }

    /// The absolute host path for a fileid. Guaranteed within `root` because the
    /// relative paths only ever contain validated, normal components.
    pub fn abs_of(&self, id: u64) -> Option<PathBuf> {
        self.rel_of(id).map(|rel| self.root.join(rel))
    }

    /// Whether `id` is a directory the guest can list.
    pub fn is_dir(&self, id: u64) -> bool {
        self.abs_of(id).map(|p| p.is_dir()).unwrap_or(false)
    }

    /// Synthetic attributes for `id`, or `None` if it no longer exists.
    pub fn attr(&self, id: u64) -> Option<Attr> {
        let abs = self.abs_of(id)?;
        let md = std::fs::symlink_metadata(&abs).ok()?;
        Some(self.attr_from(id, &md))
    }

    /// The fileid of `id`'s parent, clamped at the export root so `..` can never
    /// walk out of the share.
    fn parent_id(&mut self, id: u64) -> Option<u64> {
        let mut rel = self.rel_of(id)?.clone();
        if !rel.pop() {
            return Some(ROOT_ID);
        }
        Some(self.intern(rel))
    }

    /// Resolve `name` within directory `dirid`, interning the child and
    /// returning its fileid. `.` and `..` resolve to the directory and its
    /// parent — clients send `..` over the wire on every DNLC miss, so refusing
    /// it breaks relative-path navigation. Other escaping names are rejected.
    pub fn lookup(&mut self, dirid: u64, name: &[u8]) -> Option<u64> {
        if name == b"." || name == b".." {
            if !self.is_dir(dirid) {
                return None;
            }
            return if name == b"." { Some(dirid) } else { self.parent_id(dirid) };
        }
        let comp = valid_component(name)?;
        let rel = self.rel_of(dirid)?.join(&comp);
        let abs = self.root.join(&rel);
        if !abs.symlink_metadata().is_ok() {
            return None;
        }
        Some(self.intern(rel))
    }

    /// List `dirid`, returning `(name, fileid, attr)` for each entry, `.` and
    /// `..` first — POSIX readdir includes them and getcwd-style walks need the
    /// `..` fileid to match what LOOKUP interns.
    pub fn readdir(&mut self, dirid: u64) -> Option<Vec<(Vec<u8>, u64, Attr)>> {
        let dir_rel = self.rel_of(dirid)?.clone();
        let abs = self.root.join(&dir_rel);
        let mut out = Vec::new();
        for (name, id) in [(b".".to_vec(), dirid), (b"..".to_vec(), self.parent_id(dirid)?)] {
            if let Some(attr) = self.attr(id) {
                out.push((name, id, attr));
            }
        }
        for ent in std::fs::read_dir(&abs).ok()? {
            let ent = ent.ok()?;
            let name = name_bytes(&ent.file_name());
            // Skip anything that wouldn't round-trip as a safe component.
            if valid_component(&name).is_none() {
                continue;
            }
            let rel = dir_rel.join(ent.file_name());
            let id = self.intern(rel);
            if let Some(attr) = self.attr(id) {
                out.push((name, id, attr));
            }
        }
        Some(out)
    }

    /// Read up to `count` bytes at `offset` from file `id`. Returns the data and
    /// whether end-of-file was reached.
    pub fn read(&self, id: u64, offset: u64, count: u32) -> Option<(Vec<u8>, bool)> {
        use std::io::{Read, Seek, SeekFrom};
        let abs = self.abs_of(id)?;
        let mut f = std::fs::File::open(&abs).ok()?;
        let len = f.metadata().ok()?.len();
        f.seek(SeekFrom::Start(offset)).ok()?;
        let want = count as usize;
        let mut buf = vec![0u8; want];
        let mut got = 0;
        while got < want {
            match f.read(&mut buf[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(_) => break,
            }
        }
        buf.truncate(got);
        let eof = offset.saturating_add(got as u64) >= len;
        Some((buf, eof))
    }

    /// Write `data` at `offset` to file `id`, returning the post-write attrs.
    pub fn write(&mut self, id: u64, offset: u64, data: &[u8]) -> Option<Attr> {
        use std::io::{Seek, SeekFrom, Write};
        let abs = self.abs_of(id)?;
        let mut f = std::fs::OpenOptions::new().write(true).open(&abs).ok()?;
        f.seek(SeekFrom::Start(offset)).ok()?;
        f.write_all(data).ok()?;
        f.flush().ok()?;
        self.attr(id)
    }

    /// Truncate (or extend) file `id` to `size` bytes. Used by SETATTR.
    pub fn truncate(&mut self, id: u64, size: u64) -> bool {
        let Some(abs) = self.abs_of(id) else { return false };
        std::fs::OpenOptions::new()
            .write(true)
            .open(&abs)
            .and_then(|f| f.set_len(size))
            .is_ok()
    }

    /// Create an empty regular file `name` in `dirid`; returns its fileid.
    pub fn create(&mut self, dirid: u64, name: &[u8]) -> Option<u64> {
        let comp = valid_component(name)?;
        let rel = self.rel_of(dirid)?.join(&comp);
        let abs = self.root.join(&rel);
        std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(&abs).ok()?;
        Some(self.intern(rel))
    }

    /// Create directory `name` in `dirid`; returns its fileid.
    pub fn mkdir(&mut self, dirid: u64, name: &[u8]) -> Option<u64> {
        let comp = valid_component(name)?;
        let rel = self.rel_of(dirid)?.join(&comp);
        std::fs::create_dir(self.root.join(&rel)).ok()?;
        Some(self.intern(rel))
    }

    /// Remove file `name` from `dirid`.
    pub fn remove(&mut self, dirid: u64, name: &[u8]) -> bool {
        self.remove_with(dirid, name, false)
    }

    /// Remove directory `name` from `dirid`.
    pub fn rmdir(&mut self, dirid: u64, name: &[u8]) -> bool {
        self.remove_with(dirid, name, true)
    }

    fn remove_with(&mut self, dirid: u64, name: &[u8], dir: bool) -> bool {
        let Some(comp) = valid_component(name) else { return false };
        let Some(parent) = self.rel_of(dirid) else { return false };
        let rel = parent.join(&comp);
        let abs = self.root.join(&rel);
        let ok = if dir { std::fs::remove_dir(&abs) } else { std::fs::remove_file(&abs) }.is_ok();
        if ok {
            if let Some(id) = self.path_to_id.remove(&rel) {
                self.id_to_path.remove(&id);
            }
        }
        ok
    }

    /// Rename `from_name` in `from_dir` to `to_name` in `to_dir`.
    pub fn rename(&mut self, from_dir: u64, from_name: &[u8], to_dir: u64, to_name: &[u8]) -> bool {
        let (Some(fc), Some(tc)) = (valid_component(from_name), valid_component(to_name)) else {
            return false;
        };
        let (Some(fp), Some(tp)) = (self.rel_of(from_dir).cloned(), self.rel_of(to_dir).cloned()) else {
            return false;
        };
        let from_rel = fp.join(&fc);
        let to_rel = tp.join(&tc);
        if std::fs::rename(self.root.join(&from_rel), self.root.join(&to_rel)).is_err() {
            return false;
        }
        // Re-point the moved id at its new path so its handle stays valid.
        if let Some(id) = self.path_to_id.remove(&from_rel) {
            self.id_to_path.insert(id, to_rel.clone());
            self.path_to_id.insert(to_rel, id);
        }
        true
    }

    // ── synthetic attribute construction ────────────────────────────────────

    fn attr_from(&self, id: u64, md: &std::fs::Metadata) -> Attr {
        let ft = md.file_type();
        let (kind, type_bits, base) = if ft.is_dir() {
            (FileKind::Dir, S_IFDIR, 0o755)
        } else if ft.is_symlink() {
            (FileKind::Lnk, S_IFLNK, 0o777)
        } else if ft.is_file() {
            (FileKind::Reg, S_IFREG, file_perm(md))
        } else {
            (FileKind::Other, S_IFREG, 0o644)
        };
        Attr {
            kind,
            mode: type_bits | base,
            nlink: if kind == FileKind::Dir { 2 } else { 1 },
            uid: self.uid,
            gid: self.gid,
            size: md.len(),
            fileid: id,
            atime: systime(md.accessed().ok()),
            mtime: systime(md.modified().ok()),
            ctime: systime(md.modified().ok()), // no portable ctime; reuse mtime
        }
    }
}

/// Permission bits for a regular file: 0644, plus the execute bits if the host
/// marks it executable (unix only; on other platforms files are 0644).
fn file_perm(md: &std::fs::Metadata) -> u32 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if md.permissions().mode() & 0o111 != 0 {
            return 0o755;
        }
    }
    let _ = md;
    0o644
}

/// Convert a `SystemTime` into `(secs, nsecs)` since the unix epoch (0 if before
/// the epoch or unavailable).
fn systime(t: Option<SystemTime>) -> (u32, u32) {
    t.and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| (d.as_secs() as u32, d.subsec_nanos()))
        .unwrap_or((0, 0))
}

/// Validate a single path component coming off the wire. Rejects anything that
/// could escape the export (`..`, `.`, empty, embedded separators or NUL).
/// Returns the component as an `OsString`-bearing `PathBuf` fragment.
fn valid_component(name: &[u8]) -> Option<PathBuf> {
    if name.is_empty() || name == b"." || name == b".." {
        return None;
    }
    if name.iter().any(|&b| b == b'/' || b == b'\\' || b == 0) {
        return None;
    }
    let s = os_string_from_bytes(name)?;
    let pb = PathBuf::from(&s);
    // Defense in depth: the parsed fragment must be exactly one normal component.
    let mut comps = pb.components();
    match (comps.next(), comps.next()) {
        (Some(std::path::Component::Normal(_)), None) => Some(pb),
        _ => None,
    }
}

/// Bytes → OS string. On unix, filenames are arbitrary bytes; on other targets
/// they must be valid UTF-8 (a documented cross-platform limitation).
fn os_string_from_bytes(b: &[u8]) -> Option<std::ffi::OsString> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        return Some(std::ffi::OsStr::from_bytes(b).to_os_string());
    }
    #[cfg(not(unix))]
    {
        std::str::from_utf8(b).ok().map(std::ffi::OsString::from)
    }
}

/// An OS filename → bytes (inverse of `os_string_from_bytes`).
fn name_bytes(name: &std::ffi::OsStr) -> Vec<u8> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        return name.as_bytes().to_vec();
    }
    #[cfg(not(unix))]
    {
        name.to_string_lossy().into_owned().into_bytes()
    }
}

/// Path containment check used by tests and as a sanity assert.
pub fn within(root: &Path, candidate: &Path) -> bool {
    candidate.starts_with(root)
}

// ── XDR (RFC 1014) + Sun RPC (RFC 1057) wire layer ──────────────────────────
//
// Increment 2: the transport-agnostic encode/decode used by both NFSv2 and v3.
// One UDP datagram carries exactly one RPC message (no TCP record marking).

const MSG_CALL: u32 = 0;
const MSG_REPLY: u32 = 1;
const RPC_VERSION: u32 = 2;
const REPLY_ACCEPTED: u32 = 0;
const AUTH_NULL: u32 = 0;

/// RPC accept-status values (RFC 1057). We allow every host, so auth never
/// fails; these cover protocol-level outcomes only.
pub mod accept {
    pub const SUCCESS: u32 = 0;
    pub const PROG_UNAVAIL: u32 = 1;
    pub const PROG_MISMATCH: u32 = 2;
    pub const PROC_UNAVAIL: u32 = 3;
    pub const GARBAGE_ARGS: u32 = 4;
    pub const SYSTEM_ERR: u32 = 5;
}

/// Big-endian, 4-byte-aligned XDR encoder.
#[derive(Default)]
pub struct Xdr {
    buf: Vec<u8>,
}
impl Xdr {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    pub fn i32(&mut self, v: i32) {
        self.u32(v as u32);
    }
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    pub fn bool(&mut self, v: bool) {
        self.u32(v as u32);
    }
    /// Length-prefixed opaque/string, padded to a 4-byte boundary.
    pub fn opaque(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.fixed(b);
    }
    /// Fixed-length bytes (no length prefix), padded to 4 bytes.
    pub fn fixed(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
        let pad = (4 - (b.len() % 4)) % 4;
        self.buf.extend(std::iter::repeat(0u8).take(pad));
    }
    pub fn len(&self) -> usize {
        self.buf.len()
    }
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Big-endian XDR decode cursor over a borrowed datagram.
pub struct Cur<'a> {
    b: &'a [u8],
    pos: usize,
}
impl<'a> Cur<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }
    pub fn u32(&mut self) -> Option<u32> {
        let e = self.pos.checked_add(4)?;
        if e > self.b.len() {
            return None;
        }
        let v = u32::from_be_bytes(self.b[self.pos..e].try_into().ok()?);
        self.pos = e;
        Some(v)
    }
    pub fn u64(&mut self) -> Option<u64> {
        let e = self.pos.checked_add(8)?;
        if e > self.b.len() {
            return None;
        }
        let v = u64::from_be_bytes(self.b[self.pos..e].try_into().ok()?);
        self.pos = e;
        Some(v)
    }
    pub fn i32(&mut self) -> Option<i32> {
        self.u32().map(|v| v as i32)
    }
    /// Length-prefixed opaque/string (returns the bytes; consumes the pad).
    pub fn opaque(&mut self) -> Option<&'a [u8]> {
        let len = self.u32()? as usize;
        self.fixed(len)
    }
    /// Fixed-length opaque of `len` bytes, consuming the pad to a 4-byte boundary.
    pub fn fixed(&mut self, len: usize) -> Option<&'a [u8]> {
        let e = self.pos.checked_add(len)?;
        if e > self.b.len() {
            return None;
        }
        let s = &self.b[self.pos..e];
        let pad = (4 - (len % 4)) % 4;
        self.pos = (e + pad).min(self.b.len()); // tolerate a missing trailing pad
        Some(s)
    }
    pub fn skip(&mut self, n: usize) -> Option<()> {
        let e = self.pos.checked_add(n)?;
        if e > self.b.len() {
            return None;
        }
        self.pos = e;
        Some(())
    }
    /// Skip one RPC opaque_auth (`flavor` + length-prefixed `body`).
    fn skip_auth(&mut self) -> Option<()> {
        let _flavor = self.u32()?;
        self.opaque()?;
        Some(())
    }
    pub fn remaining(&self) -> &'a [u8] {
        &self.b[self.pos.min(self.b.len())..]
    }
}

/// A parsed RPC CALL header. Credentials are skipped — every host is allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RpcCall {
    pub xid: u32,
    pub prog: u32,
    pub vers: u32,
    pub proc_num: u32,
}

/// Parse the RPC CALL header from one UDP datagram, returning the header and a
/// cursor positioned at the procedure arguments. `None` if it isn't a v2 CALL.
pub fn parse_call(msg: &[u8]) -> Option<(RpcCall, Cur<'_>)> {
    let mut c = Cur::new(msg);
    let xid = c.u32()?;
    if c.u32()? != MSG_CALL {
        return None;
    }
    if c.u32()? != RPC_VERSION {
        return None;
    }
    let prog = c.u32()?;
    let vers = c.u32()?;
    let proc_num = c.u32()?;
    c.skip_auth()?; // cred
    c.skip_auth()?; // verf
    Some((RpcCall { xid, prog, vers, proc_num }, c))
}

/// Begin an accepted RPC reply (AUTH_NULL verifier), ready for the caller to
/// append the procedure result. `accept_stat` is one of [`accept`].
pub fn reply(xid: u32, accept_stat: u32) -> Xdr {
    let mut x = Xdr::new();
    x.u32(xid);
    x.u32(MSG_REPLY);
    x.u32(REPLY_ACCEPTED);
    x.u32(AUTH_NULL); // verifier flavor
    x.u32(0); // verifier body length
    x.u32(accept_stat);
    x
}

// ── NFSv3 (RFC 1813) procedures ─────────────────────────────────────────────
//
// Increment 3: the read-side procedures over the backend. Write-side
// (SETATTR/WRITE/CREATE/MKDIR/REMOVE/RMDIR/RENAME/COMMIT), NFSv2, MOUNT, and the
// NAT wiring land in later increments.

/// NFS RPC program + version.
pub const NFS_PROG: u32 = 100003;
pub const NFS_V3: u32 = 3;

// NFSv3 procedure numbers.
const PROC3_NULL: u32 = 0;
const PROC3_GETATTR: u32 = 1;
const PROC3_SETATTR: u32 = 2;
const PROC3_LOOKUP: u32 = 3;
const PROC3_ACCESS: u32 = 4;
const PROC3_READ: u32 = 6;
const PROC3_WRITE: u32 = 7;
const PROC3_CREATE: u32 = 8;
const PROC3_MKDIR: u32 = 9;
const PROC3_REMOVE: u32 = 12;
const PROC3_RMDIR: u32 = 13;
const PROC3_RENAME: u32 = 14;
const PROC3_READDIR: u32 = 16;
const PROC3_READDIRPLUS: u32 = 17;
const PROC3_FSSTAT: u32 = 18;
const PROC3_FSINFO: u32 = 19;
const PROC3_PATHCONF: u32 = 20;
const PROC3_COMMIT: u32 = 21;

// nfsstat3 values we use.
const NFS3_OK: u32 = 0;
const NFS3ERR_IO: u32 = 5;
const NFS3ERR_NOENT: u32 = 2;
const NFS3ERR_NOTDIR: u32 = 20;
const NFS3ERR_STALE: u32 = 70;
const NFS3ERR_NOTEMPTY: u32 = 66;

// fsinfo3 sizes. Reads can be large (the NAT fragments outbound); writes need
// the inbound-reassembly increment before a large wtmax is safe.
const RTMAX: u32 = 32768;
const WTMAX: u32 = 32768;
const DTPREF: u32 = 8192;
const FSF3_HOMOGENEOUS: u32 = 0x8;
const FSF3_CANSETTIME: u32 = 0x10;

const FSID: u64 = 0x4952_4953; // "IRIS"

fn ftype3(kind: FileKind) -> u32 {
    match kind {
        FileKind::Reg | FileKind::Other => 1, // NF3REG
        FileKind::Dir => 2,                   // NF3DIR
        FileKind::Lnk => 5,                   // NF3LNK
    }
}

/// File handle (`nfs_fh3`): we encode the 8-byte fileid as the opaque handle.
fn put_fh(x: &mut Xdr, fileid: u64) {
    x.opaque(&fileid.to_be_bytes());
}
fn get_fh(c: &mut Cur) -> Option<u64> {
    let h = c.opaque()?;
    (h.len() == 8).then(|| u64::from_be_bytes(h.try_into().ok().unwrap()))
}

fn put_time(x: &mut Xdr, t: (u32, u32)) {
    x.u32(t.0);
    x.u32(t.1);
}

/// `fattr3`. Mode carries the full st_mode (type + perm bits) — matches knfsd and
/// is what clients expect; `type` is the kind.
fn put_fattr3(x: &mut Xdr, a: &Attr) {
    x.u32(ftype3(a.kind));
    x.u32(a.mode);
    x.u32(a.nlink);
    x.u32(a.uid);
    x.u32(a.gid);
    x.u64(a.size);
    x.u64(a.size); // used (approximated by size)
    x.u32(0);
    x.u32(0); // rdev (specdata3)
    x.u64(FSID);
    x.u64(a.fileid);
    put_time(x, a.atime);
    put_time(x, a.mtime);
    put_time(x, a.ctime);
}

/// `post_op_attr`: a present-flag + optional `fattr3`.
fn put_post_op_attr(x: &mut Xdr, a: Option<&Attr>) {
    match a {
        Some(a) => {
            x.bool(true);
            put_fattr3(x, a);
        }
        None => x.bool(false),
    }
}

fn garbage(xid: u32) -> Vec<u8> {
    reply(xid, accept::GARBAGE_ARGS).into_bytes()
}

/// Dispatch one NFSv3 call to its handler, returning the full reply datagram.
pub fn nfs3_call(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    match call.proc_num {
        PROC3_NULL => reply(call.xid, accept::SUCCESS).into_bytes(),
        PROC3_GETATTR => nfs3_getattr(call, args, b),
        PROC3_LOOKUP => nfs3_lookup(call, args, b),
        PROC3_ACCESS => nfs3_access(call, args, b),
        PROC3_READ => nfs3_read(call, args, b),
        PROC3_READDIR => nfs3_readdir(call, args, b, false),
        PROC3_READDIRPLUS => nfs3_readdir(call, args, b, true),
        PROC3_FSINFO => nfs3_fsinfo(call, args, b),
        PROC3_FSSTAT => nfs3_fsstat(call, args, b),
        PROC3_PATHCONF => nfs3_pathconf(call, args, b),
        PROC3_SETATTR => nfs3_setattr(call, args, b),
        PROC3_WRITE => nfs3_write(call, args, b),
        PROC3_CREATE => nfs3_create(call, args, b, false),
        PROC3_MKDIR => nfs3_create(call, args, b, true),
        PROC3_REMOVE => nfs3_remove(call, args, b, false),
        PROC3_RMDIR => nfs3_remove(call, args, b, true),
        PROC3_RENAME => nfs3_rename(call, args, b),
        PROC3_COMMIT => nfs3_commit(call, args, b),
        _ => reply(call.xid, accept::PROC_UNAVAIL).into_bytes(),
    }
}

fn nfs3_getattr(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh(args) else { return garbage(call.xid) };
    let mut x = reply(call.xid, accept::SUCCESS);
    match b.attr(fid) {
        Some(a) => {
            x.u32(NFS3_OK);
            put_fattr3(&mut x, &a);
        }
        None => x.u32(NFS3ERR_STALE),
    }
    x.into_bytes()
}

fn nfs3_lookup(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(dir) = get_fh(args) else { return garbage(call.xid) };
    let name = match args.opaque() {
        Some(n) => n.to_vec(),
        None => return garbage(call.xid),
    };
    let mut x = reply(call.xid, accept::SUCCESS);
    match b.lookup(dir, &name) {
        Some(fid) => {
            x.u32(NFS3_OK);
            put_fh(&mut x, fid);
            put_post_op_attr(&mut x, b.attr(fid).as_ref()); // obj attributes
            put_post_op_attr(&mut x, b.attr(dir).as_ref()); // dir attributes
        }
        None => {
            x.u32(NFS3ERR_NOENT);
            put_post_op_attr(&mut x, b.attr(dir).as_ref());
        }
    }
    x.into_bytes()
}

fn nfs3_access(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh(args) else { return garbage(call.xid) };
    let Some(requested) = args.u32() else { return garbage(call.xid) };
    let mut x = reply(call.xid, accept::SUCCESS);
    match b.attr(fid) {
        Some(a) => {
            x.u32(NFS3_OK);
            put_post_op_attr(&mut x, Some(&a));
            x.u32(requested); // grant everything asked for (no security)
        }
        None => {
            x.u32(NFS3ERR_STALE);
            put_post_op_attr(&mut x, None);
        }
    }
    x.into_bytes()
}

fn nfs3_read(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh(args) else { return garbage(call.xid) };
    let Some(offset) = args.u64() else { return garbage(call.xid) };
    let Some(count) = args.u32() else { return garbage(call.xid) };
    let mut x = reply(call.xid, accept::SUCCESS);
    match b.read(fid, offset, count.min(RTMAX)) {
        Some((data, eof)) => {
            x.u32(NFS3_OK);
            put_post_op_attr(&mut x, b.attr(fid).as_ref());
            x.u32(data.len() as u32);
            x.bool(eof);
            x.opaque(&data);
        }
        None => {
            x.u32(NFS3ERR_IO);
            put_post_op_attr(&mut x, b.attr(fid).as_ref());
        }
    }
    x.into_bytes()
}

/// READDIR / READDIRPLUS. Entries are name-sorted so the cookie (a 1-based index)
/// is stable across calls; we page within a byte budget and set `eof` when done.
fn nfs3_readdir(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking, plus: bool) -> Vec<u8> {
    let Some(fid) = get_fh(args) else { return garbage(call.xid) };
    let Some(cookie) = args.u64() else { return garbage(call.xid) };
    if args.fixed(8).is_none() {
        return garbage(call.xid); // cookieverf
    }
    // Honor the client's reply-size limit so we never overrun its decode buffer
    // (the v2 path documents the "Can't decode result" failure this avoids).
    // READDIR carries `maxcount`; READDIRPLUS carries `dircount` then `maxcount`.
    if plus {
        args.u32(); // dircount
    }
    let maxcount = args.u32().unwrap_or(0) as usize;
    let limit = if maxcount == 0 { 16_000 } else { maxcount.clamp(512, 16_000) };
    let Some(mut entries) = b.readdir(fid) else {
        let mut x = reply(call.xid, accept::SUCCESS);
        x.u32(NFS3ERR_NOTDIR);
        put_post_op_attr(&mut x, b.attr(fid).as_ref());
        return x.into_bytes();
    };
    entries.sort_by(|p, q| p.0.cmp(&q.0));
    let dir_attr = b.attr(fid);

    let mut x = reply(call.xid, accept::SUCCESS);
    x.u32(NFS3_OK);
    put_post_op_attr(&mut x, dir_attr.as_ref());
    x.fixed(&[0u8; 8]); // cookieverf

    let mut i = cookie as usize;
    let mut budget = 0usize;
    while i < entries.len() {
        let (name, id, attr) = &entries[i];
        let est = 40 + name.len() + if plus { 96 } else { 0 };
        if budget > 0 && budget + est > limit {
            break; // page is full; client resumes from this cookie
        }
        budget += est;
        x.bool(true); // entry follows
        x.u64(*id); // fileid
        x.opaque(name);
        x.u64((i + 1) as u64); // cookie = next index
        if plus {
            put_post_op_attr(&mut x, Some(attr)); // name_attributes
            x.bool(true); // handle follows
            put_fh(&mut x, *id);
        }
        i += 1;
    }
    x.bool(false); // no more entries in this reply
    x.bool(i >= entries.len()); // eof
    x.into_bytes()
}

fn nfs3_fsinfo(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh(args) else { return garbage(call.xid) };
    let mut x = reply(call.xid, accept::SUCCESS);
    x.u32(NFS3_OK);
    put_post_op_attr(&mut x, b.attr(fid).as_ref());
    x.u32(RTMAX); // rtmax
    x.u32(RTMAX); // rtpref
    x.u32(4096); // rtmult
    x.u32(WTMAX); // wtmax
    x.u32(WTMAX); // wtpref
    x.u32(4096); // wtmult
    x.u32(DTPREF); // dtpref
    x.u64(0x7fff_ffff_ffff_ffff); // maxfilesize
    put_time(&mut x, (1, 0)); // time_delta (1s)
    x.u32(FSF3_HOMOGENEOUS | FSF3_CANSETTIME); // properties
    x.into_bytes()
}

fn nfs3_fsstat(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh(args) else { return garbage(call.xid) };
    let mut x = reply(call.xid, accept::SUCCESS);
    x.u32(NFS3_OK);
    put_post_op_attr(&mut x, b.attr(fid).as_ref());
    let big = 1u64 << 40; // faked capacity
    x.u64(big);
    x.u64(big);
    x.u64(big); // total / free / avail bytes
    let files = 1u64 << 20;
    x.u64(files);
    x.u64(files);
    x.u64(files); // total / free / avail files
    x.u32(0); // invarsec
    x.into_bytes()
}

fn nfs3_pathconf(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh(args) else { return garbage(call.xid) };
    let mut x = reply(call.xid, accept::SUCCESS);
    x.u32(NFS3_OK);
    put_post_op_attr(&mut x, b.attr(fid).as_ref());
    x.u32(1023); // linkmax
    x.u32(255); // name_max
    x.bool(true); // no_trunc
    x.bool(false); // chown_restricted
    x.bool(false); // case_insensitive
    x.bool(true); // case_preserving
    x.into_bytes()
}

// ── NFSv3 write-side procedures (increment 4) ───────────────────────────────

/// `wcc_data`: a (omitted) pre-op attr followed by the post-op attr.
fn put_wcc_data(x: &mut Xdr, post: Option<&Attr>) {
    x.bool(false); // pre_op_attr omitted
    put_post_op_attr(x, post);
}

/// Parse `sattr3`, returning the requested size if `set_size` is true (we honor
/// truncation; mode/uid/gid/atime/mtime are accepted-and-ignored). Outer `None`
/// means the structure was malformed.
fn parse_sattr3(c: &mut Cur) -> Option<Option<u64>> {
    if c.u32()? != 0 { c.u32()?; } // set_mode
    if c.u32()? != 0 { c.u32()?; } // set_uid
    if c.u32()? != 0 { c.u32()?; } // set_gid
    let size = if c.u32()? != 0 { Some(c.u64()?) } else { None };
    if c.u32()? == 2 { c.u32()?; c.u32()?; } // set_atime = SET_TO_CLIENT_TIME
    if c.u32()? == 2 { c.u32()?; c.u32()?; } // set_mtime = SET_TO_CLIENT_TIME
    Some(size)
}

fn nfs3_setattr(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh(args) else { return garbage(call.xid) };
    let Some(size) = parse_sattr3(args) else { return garbage(call.xid) };
    if let Some(sz) = size {
        b.truncate(fid, sz);
    }
    let mut x = reply(call.xid, accept::SUCCESS);
    match b.attr(fid) {
        Some(a) => {
            x.u32(NFS3_OK);
            put_wcc_data(&mut x, Some(&a));
        }
        None => {
            x.u32(NFS3ERR_STALE);
            put_wcc_data(&mut x, None);
        }
    }
    x.into_bytes()
}

fn nfs3_write(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh(args) else { return garbage(call.xid) };
    let Some(offset) = args.u64() else { return garbage(call.xid) };
    let Some(_count) = args.u32() else { return garbage(call.xid) };
    let Some(_stable) = args.u32() else { return garbage(call.xid) };
    let Some(data) = args.opaque() else { return garbage(call.xid) };
    let mut x = reply(call.xid, accept::SUCCESS);
    match b.write(fid, offset, data) {
        Some(a) => {
            x.u32(NFS3_OK);
            put_wcc_data(&mut x, Some(&a));
            x.u32(data.len() as u32); // count written
            x.u32(2); // committed = FILE_SYNC
            x.fixed(&[0u8; 8]); // write verifier
        }
        None => {
            x.u32(NFS3ERR_IO);
            put_wcc_data(&mut x, b.attr(fid).as_ref());
        }
    }
    x.into_bytes()
}

/// CREATE/MKDIR share a shape: diropargs3 then attrs we ignore. `dir` true =
/// MKDIR. On success returns post_op fh + attrs + dir wcc.
fn nfs3_create(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking, dir: bool) -> Vec<u8> {
    let Some(parent) = get_fh(args) else { return garbage(call.xid) };
    let name = match args.opaque() {
        Some(n) => n.to_vec(),
        None => return garbage(call.xid),
    };
    // The remaining args (createmode3 + sattr3, or MKDIR's sattr3) are ignored.
    let made = if dir { b.mkdir(parent, &name) } else { b.create(parent, &name) };
    let mut x = reply(call.xid, accept::SUCCESS);
    match made {
        Some(fid) => {
            x.u32(NFS3_OK);
            x.bool(true); // post_op_fh3: handle follows
            put_fh(&mut x, fid);
            put_post_op_attr(&mut x, b.attr(fid).as_ref());
            put_wcc_data(&mut x, b.attr(parent).as_ref());
        }
        None => {
            x.u32(NFS3ERR_IO);
            x.bool(false); // no handle
            put_post_op_attr(&mut x, None);
            put_wcc_data(&mut x, b.attr(parent).as_ref());
        }
    }
    x.into_bytes()
}

/// REMOVE/RMDIR: diropargs3 → dir wcc. `dir` true = RMDIR.
fn nfs3_remove(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking, dir: bool) -> Vec<u8> {
    let Some(parent) = get_fh(args) else { return garbage(call.xid) };
    let name = match args.opaque() {
        Some(n) => n.to_vec(),
        None => return garbage(call.xid),
    };
    let ok = if dir { b.rmdir(parent, &name) } else { b.remove(parent, &name) };
    let mut x = reply(call.xid, accept::SUCCESS);
    x.u32(if ok {
        NFS3_OK
    } else if dir {
        NFS3ERR_NOTEMPTY
    } else {
        NFS3ERR_NOENT
    });
    put_wcc_data(&mut x, b.attr(parent).as_ref());
    x.into_bytes()
}

fn nfs3_rename(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(from_dir) = get_fh(args) else { return garbage(call.xid) };
    let from_name = match args.opaque() {
        Some(n) => n.to_vec(),
        None => return garbage(call.xid),
    };
    let Some(to_dir) = get_fh(args) else { return garbage(call.xid) };
    let to_name = match args.opaque() {
        Some(n) => n.to_vec(),
        None => return garbage(call.xid),
    };
    let ok = b.rename(from_dir, &from_name, to_dir, &to_name);
    let mut x = reply(call.xid, accept::SUCCESS);
    x.u32(if ok { NFS3_OK } else { NFS3ERR_IO });
    put_wcc_data(&mut x, b.attr(from_dir).as_ref()); // fromdir wcc
    put_wcc_data(&mut x, b.attr(to_dir).as_ref()); // todir wcc
    x.into_bytes()
}

fn nfs3_commit(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh(args) else { return garbage(call.xid) };
    let _ = args.u64(); // offset
    let _ = args.u32(); // count
    let mut x = reply(call.xid, accept::SUCCESS);
    x.u32(NFS3_OK); // writes are synchronous, so COMMIT is a no-op success
    put_wcc_data(&mut x, b.attr(fid).as_ref());
    x.fixed(&[0u8; 8]); // write verifier
    x.into_bytes()
}

// ── NFSv2 (RFC 1094) procedures ─────────────────────────────────────────────
//
// Increment 5: the parallel v2 encoding — 32-bit attrs, fixed 32-byte handles,
// v2 procedure numbers — over the same backend. Served when the guest mounts
// NFSv2 (IRIX 5.3).

pub const NFS_V2: u32 = 2;

// v2 procedure numbers (these differ from v3).
const PROC2_NULL: u32 = 0;
const PROC2_GETATTR: u32 = 1;
const PROC2_SETATTR: u32 = 2;
const PROC2_LOOKUP: u32 = 4;
const PROC2_READ: u32 = 6;
const PROC2_WRITE: u32 = 8;
const PROC2_CREATE: u32 = 9;
const PROC2_REMOVE: u32 = 10;
const PROC2_RENAME: u32 = 11;
const PROC2_MKDIR: u32 = 14;
const PROC2_RMDIR: u32 = 15;
const PROC2_READDIR: u32 = 16;
const PROC2_STATFS: u32 = 17;

const NFS2_OK: u32 = 0; // v2 nfsstat shares numeric values with v3 for our cases

fn ftype2(kind: FileKind) -> u32 {
    match kind {
        FileKind::Reg | FileKind::Other => 1,
        FileKind::Dir => 2,
        FileKind::Lnk => 5,
    }
}

/// v2 file handle: a fixed 32-byte opaque — we store the 8-byte fileid + zeros.
fn put_fh2(x: &mut Xdr, fileid: u64) {
    let mut h = [0u8; 32];
    h[..8].copy_from_slice(&fileid.to_be_bytes());
    x.fixed(&h);
}
fn get_fh2(c: &mut Cur) -> Option<u64> {
    let h = c.fixed(32)?;
    Some(u64::from_be_bytes(h[..8].try_into().ok()?))
}

/// v2 timeval is (seconds, *micro*seconds); convert our nanoseconds.
fn put_time2(x: &mut Xdr, t: (u32, u32)) {
    x.u32(t.0);
    x.u32(t.1 / 1000);
}

/// v2 `fattr` (all 32-bit).
fn put_fattr2(x: &mut Xdr, a: &Attr) {
    x.u32(ftype2(a.kind));
    x.u32(a.mode);
    x.u32(a.nlink);
    x.u32(a.uid);
    x.u32(a.gid);
    x.u32(a.size as u32); // v2 size is 32-bit (>4 GiB unsupported)
    x.u32(4096); // blocksize
    x.u32(0); // rdev
    x.u32(((a.size + 511) / 512) as u32); // blocks
    x.u32(FSID as u32);
    x.u32(a.fileid as u32);
    put_time2(x, a.atime);
    put_time2(x, a.mtime);
    put_time2(x, a.ctime);
}

/// Parse a v2 `sattr`, returning the size if set (`0xFFFFFFFF` = don't set).
fn parse_sattr2(c: &mut Cur) -> Option<Option<u64>> {
    c.u32()?; // mode
    c.u32()?; // uid
    c.u32()?; // gid
    let size = c.u32()?;
    c.u64()?; // atime (sec + usec)
    c.u64()?; // mtime
    Some((size != 0xFFFF_FFFF).then_some(size as u64))
}

/// Dispatch one NFSv2 call.
pub fn nfs2_call(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    match call.proc_num {
        PROC2_NULL => reply(call.xid, accept::SUCCESS).into_bytes(),
        PROC2_GETATTR => nfs2_getattr(call, args, b),
        PROC2_SETATTR => nfs2_setattr(call, args, b),
        PROC2_LOOKUP => nfs2_lookup(call, args, b),
        PROC2_READ => nfs2_read(call, args, b),
        PROC2_WRITE => nfs2_write(call, args, b),
        PROC2_CREATE => nfs2_create(call, args, b, false),
        PROC2_MKDIR => nfs2_create(call, args, b, true),
        PROC2_REMOVE => nfs2_remove(call, args, b, false),
        PROC2_RMDIR => nfs2_remove(call, args, b, true),
        PROC2_RENAME => nfs2_rename(call, args, b),
        PROC2_READDIR => nfs2_readdir(call, args, b),
        PROC2_STATFS => nfs2_statfs(call, args, b),
        _ => reply(call.xid, accept::PROC_UNAVAIL).into_bytes(),
    }
}

fn nfs2_attr_reply(call: &RpcCall, b: &mut NfsBacking, fid: u64) -> Vec<u8> {
    let mut x = reply(call.xid, accept::SUCCESS);
    match b.attr(fid) {
        Some(a) => {
            x.u32(NFS2_OK);
            put_fattr2(&mut x, &a);
        }
        None => x.u32(NFS3ERR_STALE),
    }
    x.into_bytes()
}

fn nfs2_getattr(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh2(args) else { return garbage(call.xid) };
    nfs2_attr_reply(call, b, fid)
}

fn nfs2_setattr(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh2(args) else { return garbage(call.xid) };
    let Some(size) = parse_sattr2(args) else { return garbage(call.xid) };
    if let Some(sz) = size {
        b.truncate(fid, sz);
    }
    nfs2_attr_reply(call, b, fid)
}

fn nfs2_lookup(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(dir) = get_fh2(args) else { return garbage(call.xid) };
    let name = match args.opaque() {
        Some(n) => n.to_vec(),
        None => return garbage(call.xid),
    };
    let mut x = reply(call.xid, accept::SUCCESS);
    match b.lookup(dir, &name) {
        Some(fid) => {
            x.u32(NFS2_OK);
            put_fh2(&mut x, fid);
            if let Some(a) = b.attr(fid) {
                put_fattr2(&mut x, &a);
            }
        }
        None => x.u32(NFS3ERR_NOENT),
    }
    x.into_bytes()
}

fn nfs2_read(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh2(args) else { return garbage(call.xid) };
    let Some(offset) = args.u32() else { return garbage(call.xid) };
    let Some(count) = args.u32() else { return garbage(call.xid) };
    let _total = args.u32();
    let mut x = reply(call.xid, accept::SUCCESS);
    match b.read(fid, offset as u64, count.min(8192)) {
        Some((data, _eof)) => {
            x.u32(NFS2_OK);
            if let Some(a) = b.attr(fid) {
                put_fattr2(&mut x, &a);
            }
            x.opaque(&data);
        }
        None => x.u32(NFS3ERR_IO),
    }
    x.into_bytes()
}

fn nfs2_write(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh2(args) else { return garbage(call.xid) };
    let _begin = args.u32();
    let Some(offset) = args.u32() else { return garbage(call.xid) };
    let _total = args.u32();
    let Some(data) = args.opaque() else { return garbage(call.xid) };
    b.write(fid, offset as u64, data);
    nfs2_attr_reply(call, b, fid)
}

fn nfs2_create(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking, dir: bool) -> Vec<u8> {
    let Some(parent) = get_fh2(args) else { return garbage(call.xid) };
    let name = match args.opaque() {
        Some(n) => n.to_vec(),
        None => return garbage(call.xid),
    };
    // sattr ignored.
    let made = if dir { b.mkdir(parent, &name) } else { b.create(parent, &name) };
    let mut x = reply(call.xid, accept::SUCCESS);
    match made {
        Some(fid) => {
            x.u32(NFS2_OK);
            put_fh2(&mut x, fid);
            if let Some(a) = b.attr(fid) {
                put_fattr2(&mut x, &a);
            }
        }
        None => x.u32(NFS3ERR_IO),
    }
    x.into_bytes()
}

fn nfs2_remove(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking, dir: bool) -> Vec<u8> {
    let Some(parent) = get_fh2(args) else { return garbage(call.xid) };
    let name = match args.opaque() {
        Some(n) => n.to_vec(),
        None => return garbage(call.xid),
    };
    let ok = if dir { b.rmdir(parent, &name) } else { b.remove(parent, &name) };
    let mut x = reply(call.xid, accept::SUCCESS);
    x.u32(if ok { NFS2_OK } else { NFS3ERR_NOENT });
    x.into_bytes()
}

fn nfs2_rename(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fd) = get_fh2(args) else { return garbage(call.xid) };
    let fname = match args.opaque() {
        Some(n) => n.to_vec(),
        None => return garbage(call.xid),
    };
    let Some(td) = get_fh2(args) else { return garbage(call.xid) };
    let tname = match args.opaque() {
        Some(n) => n.to_vec(),
        None => return garbage(call.xid),
    };
    let ok = b.rename(fd, &fname, td, &tname);
    let mut x = reply(call.xid, accept::SUCCESS);
    x.u32(if ok { NFS2_OK } else { NFS3ERR_IO });
    x.into_bytes()
}

fn nfs2_readdir(call: &RpcCall, args: &mut Cur, b: &mut NfsBacking) -> Vec<u8> {
    let Some(fid) = get_fh2(args) else { return garbage(call.xid) };
    let Some(cookie_bytes) = args.fixed(4) else { return garbage(call.xid) };
    let cookie = u32::from_be_bytes(cookie_bytes.try_into().unwrap()) as usize;
    let count = args.u32().unwrap_or(0) as usize;
    let mut x = reply(call.xid, accept::SUCCESS);
    let Some(mut entries) = b.readdir(fid) else {
        x.u32(NFS3ERR_NOTDIR);
        return x.into_bytes();
    };
    entries.sort_by(|p, q| p.0.cmp(&q.0));
    x.u32(NFS2_OK);
    // Page so the reply fits BOTH the client's `count`-sized decode buffer and a
    // single unfragmented UDP datagram. NFSv2/BSD-derived clients (IRIX) size
    // their receive buffer to `count`; overrunning it truncates the datagram on
    // their side and the XDR decode fails ("Can't decode result"). The directory
    // is continued across calls via the cookie, so a small per-reply budget just
    // costs a few more READDIR round-trips. 1400 keeps us under the 1472-byte
    // single-datagram UDP payload after the ~36 bytes of header + terminators.
    let limit = count.clamp(512, 1400);
    let mut i = cookie;
    let mut budget = 0usize;
    while i < entries.len() {
        let (name, id, _attr) = &entries[i];
        let est = 24 + name.len();
        if budget > 0 && budget + est > limit {
            break;
        }
        budget += est;
        x.bool(true); // entry follows
        x.u32(*id as u32); // fileid
        x.opaque(name);
        x.fixed(&((i as u32) + 1).to_be_bytes()); // nfscookie (opaque[4])
        i += 1;
    }
    x.bool(false); // end of entries
    x.bool(i >= entries.len()); // eof
    x.into_bytes()
}

fn nfs2_statfs(call: &RpcCall, args: &mut Cur, _b: &mut NfsBacking) -> Vec<u8> {
    let Some(_fid) = get_fh2(args) else { return garbage(call.xid) };
    let mut x = reply(call.xid, accept::SUCCESS);
    x.u32(NFS2_OK);
    x.u32(8192); // tsize (optimum transfer size)
    x.u32(4096); // bsize
    let big = 1u32 << 20;
    x.u32(big); // blocks
    x.u32(big); // bfree
    x.u32(big); // bavail
    x.into_bytes()
}

// ── MOUNT protocol (RFC 1813 App. I / RFC 1094 App. A) ──────────────────────
//
// Increment 6: MNT hands the guest the root file handle. We export a single
// directory and allow everyone, so the requested path is ignored.

pub const MOUNT_PROG: u32 = 100005;
pub const MOUNT_V1: u32 = 1; // for NFSv2
pub const MOUNT_V3: u32 = 3; // for NFSv3

const MNTPROC_NULL: u32 = 0;
const MNTPROC_MNT: u32 = 1;
const MNTPROC_DUMP: u32 = 2;
const MNTPROC_UMNT: u32 = 3;
const MNTPROC_UMNTALL: u32 = 4;
const MNTPROC_EXPORT: u32 = 5;

/// Dispatch one MOUNT call.
pub fn mount_call(call: &RpcCall, args: &mut Cur) -> Vec<u8> {
    match call.proc_num {
        MNTPROC_NULL => reply(call.xid, accept::SUCCESS).into_bytes(),
        MNTPROC_MNT => {
            let _path = args.opaque(); // export path ignored — single export
            let mut x = reply(call.xid, accept::SUCCESS);
            if call.vers == MOUNT_V1 {
                x.u32(0); // fhstatus = OK
                put_fh2(&mut x, ROOT_ID); // v2 fhandle (fixed 32 bytes)
            } else {
                x.u32(0); // mountstat3 = MNT3_OK
                put_fh(&mut x, ROOT_ID); // v3 fhandle3 (opaque)
                x.u32(1); // one auth flavor follows
                x.u32(AUTH_NULL);
            }
            x.into_bytes()
        }
        MNTPROC_UMNT | MNTPROC_UMNTALL => reply(call.xid, accept::SUCCESS).into_bytes(),
        MNTPROC_DUMP => {
            let mut x = reply(call.xid, accept::SUCCESS);
            x.bool(false); // empty mount list
            x.into_bytes()
        }
        MNTPROC_EXPORT => {
            let mut x = reply(call.xid, accept::SUCCESS);
            x.bool(true); // one export entry follows
            x.opaque(b"/"); // ex_dir
            x.bool(false); // ex_groups: none -> everyone
            x.bool(false); // no further entries
            x.into_bytes()
        }
        _ => reply(call.xid, accept::PROC_UNAVAIL).into_bytes(),
    }
}

// ── server: program/version dispatch + duplicate-request cache ──────────────

/// Which NFS version(s) to serve. `Auto` answers whatever the guest mounts with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NfsVersion {
    #[default]
    Auto,
    V2,
    V3,
}

/// Duplicate-request cache: NFS-over-UDP clients retransmit on timeout, so
/// non-idempotent ops (WRITE/CREATE/REMOVE/...) must not be re-applied. We cache
/// recent replies by xid and replay them. Bounded, FIFO-evicted.
struct Drc {
    cap: usize,
    order: std::collections::VecDeque<u32>,
    map: HashMap<u32, Vec<u8>>,
}
impl Drc {
    fn new(cap: usize) -> Self {
        Self { cap, order: std::collections::VecDeque::new(), map: HashMap::new() }
    }
    fn get(&self, xid: u32) -> Option<&Vec<u8>> {
        self.map.get(&xid)
    }
    fn put(&mut self, xid: u32, reply: Vec<u8>) {
        if self.map.contains_key(&xid) {
            return;
        }
        while self.order.len() >= self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        self.order.push_back(xid);
        self.map.insert(xid, reply);
    }
}

/// The in-core NFS server: one export + the dedup cache. The NAT calls
/// [`handle`](Self::handle) with each guest MOUNT/NFS RPC datagram.
pub struct NfsServer {
    backing: NfsBacking,
    drc: Drc,
    version: NfsVersion,
}

impl NfsServer {
    pub fn new(root: impl Into<PathBuf>, version: NfsVersion) -> Self {
        Self { backing: NfsBacking::new(root), drc: Drc::new(256), version }
    }

    /// Process one RPC call datagram, returning the reply datagram (or `None` if
    /// it isn't a parseable RPC call to ignore).
    pub fn handle(&mut self, msg: &[u8]) -> Option<Vec<u8>> {
        let (call, mut args) = parse_call(msg)?;
        let idem = is_idempotent(&call);
        if !idem {
            if let Some(cached) = self.drc.get(call.xid) {
                return Some(cached.clone());
            }
        }
        let out = self.dispatch(&call, &mut args);
        if !idem {
            self.drc.put(call.xid, out.clone());
        }
        Some(out)
    }

    fn dispatch(&mut self, call: &RpcCall, args: &mut Cur) -> Vec<u8> {
        if call.prog == MOUNT_PROG {
            return mount_call(call, args);
        }
        if call.prog == NFS_PROG {
            match call.vers {
                NFS_V3 if self.version != NfsVersion::V2 => {
                    return nfs3_call(call, args, &mut self.backing);
                }
                NFS_V2 if self.version != NfsVersion::V3 => {
                    return nfs2_call(call, args, &mut self.backing);
                }
                _ => return reply(call.xid, accept::PROG_MISMATCH).into_bytes(),
            }
        }
        reply(call.xid, accept::PROG_UNAVAIL).into_bytes()
    }
}

/// Whether a procedure is safe to re-run (so it skips the dedup cache). Version-
/// aware because v2 and v3 number their procedures differently.
fn is_idempotent(call: &RpcCall) -> bool {
    if call.prog != NFS_PROG {
        return true;
    }
    let non_idempotent: &[u32] = if call.vers == NFS_V2 {
        &[PROC2_SETATTR, PROC2_WRITE, PROC2_CREATE, PROC2_REMOVE, PROC2_RENAME, PROC2_MKDIR, PROC2_RMDIR]
    } else {
        &[PROC3_SETATTR, PROC3_WRITE, PROC3_CREATE, PROC3_MKDIR, PROC3_REMOVE, PROC3_RMDIR, PROC3_RENAME]
    };
    !non_idempotent.contains(&call.proc_num)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique temp export dir (no external tempfile crate).
    fn temp_export() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "iris_nfs_test_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn root_is_a_dir_with_id_1() {
        let root = temp_export();
        let b = NfsBacking::new(&root);
        assert_eq!(b.abs_of(ROOT_ID).unwrap(), root);
        assert!(b.is_dir(ROOT_ID));
        let a = b.attr(ROOT_ID).unwrap();
        assert_eq!(a.kind, FileKind::Dir);
        assert_eq!(a.fileid, ROOT_ID);
        assert_eq!(a.uid, 0);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn lookup_intern_and_attrs() {
        let root = temp_export();
        std::fs::write(root.join("hello.txt"), b"hi there").unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        let mut b = NfsBacking::new(&root);

        let fid = b.lookup(ROOT_ID, b"hello.txt").unwrap();
        assert_eq!(b.lookup(ROOT_ID, b"hello.txt").unwrap(), fid, "stable id");
        let a = b.attr(fid).unwrap();
        assert_eq!(a.kind, FileKind::Reg);
        assert_eq!(a.size, 8);
        assert_eq!(a.mode & 0o170000, S_IFREG);

        let did = b.lookup(ROOT_ID, b"sub").unwrap();
        assert!(b.is_dir(did));
        assert!(b.lookup(ROOT_ID, b"nope").is_none());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn read_write_roundtrip() {
        let root = temp_export();
        let mut b = NfsBacking::new(&root);
        let fid = b.create(ROOT_ID, b"f.bin").unwrap();
        let attr = b.write(fid, 0, b"abcdefgh").unwrap();
        assert_eq!(attr.size, 8);
        let (data, eof) = b.read(fid, 2, 3).unwrap();
        assert_eq!(data, b"cde");
        assert!(!eof);
        let (rest, eof) = b.read(fid, 5, 100).unwrap();
        assert_eq!(rest, b"fgh");
        assert!(eof);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn readdir_lists_children() {
        let root = temp_export();
        std::fs::write(root.join("a"), b"1").unwrap();
        std::fs::write(root.join("b"), b"22").unwrap();
        std::fs::create_dir(root.join("d")).unwrap();
        let mut b = NfsBacking::new(&root);
        let mut names: Vec<Vec<u8>> = b.readdir(ROOT_ID).unwrap().into_iter().map(|(n, _, _)| n).collect();
        names.sort();
        assert_eq!(
            names,
            vec![b".".to_vec(), b"..".to_vec(), b"a".to_vec(), b"b".to_vec(), b"d".to_vec()]
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn dot_and_dotdot_navigate_without_escaping() {
        let root = temp_export();
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        let mut b = NfsBacking::new(&root);
        let a_id = b.lookup(ROOT_ID, b"a").unwrap();
        let ab_id = b.lookup(a_id, b"b").unwrap();

        assert_eq!(b.lookup(ab_id, b"."), Some(ab_id));
        assert_eq!(b.lookup(ab_id, b".."), Some(a_id), "`..` is the id LOOKUP interned");
        assert_eq!(b.lookup(a_id, b".."), Some(ROOT_ID));
        assert_eq!(b.lookup(ROOT_ID, b".."), Some(ROOT_ID), "`..` stops at the export root");

        // `..` in a listing must carry the parent's fileid — getcwd walks up by
        // matching the child's fileid against the parent's entries.
        let ents = b.readdir(ab_id).unwrap();
        for (_, id, at) in &ents {
            assert_eq!(at.fileid, *id, "entry fileid must match its attr");
        }
        let named: Vec<(Vec<u8>, u64)> = ents.into_iter().map(|(n, id, _)| (n, id)).collect();
        assert!(named.contains(&(b".".to_vec(), ab_id)));
        assert!(named.contains(&(b"..".to_vec(), a_id)));

        // A non-directory has neither, and the mutating ops still refuse both.
        let f = b.create(ROOT_ID, b"f").unwrap();
        assert!(b.lookup(f, b".").is_none());
        assert!(b.mkdir(ROOT_ID, b"..").is_none());
        assert!(!b.rmdir(ROOT_ID, b".."));
        assert!(!b.rename(ROOT_ID, b"..", ROOT_ID, b"z"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn mkdir_remove_rename() {
        let root = temp_export();
        let mut b = NfsBacking::new(&root);
        let d = b.mkdir(ROOT_ID, b"dir").unwrap();
        assert!(b.is_dir(d));
        let f = b.create(d, b"x").unwrap();
        assert!(b.attr(f).is_some());
        assert!(b.rename(d, b"x", ROOT_ID, b"y"));
        assert!(root.join("y").exists());
        assert!(!root.join("dir/x").exists());
        assert!(b.remove(ROOT_ID, b"y"));
        assert!(b.rmdir(ROOT_ID, b"dir"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rejects_traversal_components() {
        assert!(valid_component(b"..").is_none());
        assert!(valid_component(b".").is_none());
        assert!(valid_component(b"").is_none());
        assert!(valid_component(b"a/b").is_none());
        assert!(valid_component(b"a\\b").is_none());
        assert!(valid_component(b"ok.txt").is_some());

        // lookup must refuse to escape the export root: `..` at the root is the
        // root itself, and no name resolves through a separator.
        let root = temp_export();
        let mut b = NfsBacking::new(&root);
        assert_eq!(b.lookup(ROOT_ID, b".."), Some(ROOT_ID));
        assert!(b.lookup(ROOT_ID, b"../etc").is_none());
        std::fs::remove_dir_all(&root).ok();
    }

    // ── wire layer (increment 2) ────────────────────────────────────────────

    #[test]
    fn xdr_roundtrip() {
        let mut x = Xdr::new();
        x.u32(0xdead_beef);
        x.u64(0x0102_0304_0506_0708);
        x.opaque(b"abc"); // 4 (len) + 3 + 1 pad = 8
        let bytes = x.into_bytes();
        assert_eq!(bytes.len(), 4 + 8 + 8);
        let mut c = Cur::new(&bytes);
        assert_eq!(c.u32(), Some(0xdead_beef));
        assert_eq!(c.u64(), Some(0x0102_0304_0506_0708));
        assert_eq!(c.opaque(), Some(&b"abc"[..]));
        assert_eq!(c.u32(), None); // exhausted
    }

    #[test]
    fn parse_rpc_call_skips_auth() {
        let mut x = Xdr::new();
        x.u32(0x1122_3344); // xid
        x.u32(0); // CALL
        x.u32(2); // rpcvers
        x.u32(100003); // NFS program
        x.u32(3); // v3
        x.u32(1); // GETATTR
        x.u32(0); x.u32(0); // cred: AUTH_NULL, len 0
        x.u32(0); x.u32(0); // verf: AUTH_NULL, len 0
        x.u32(0xCAFE_BABE); // one arg word
        let msg = x.into_bytes();
        let (call, mut args) = parse_call(&msg).unwrap();
        assert_eq!(call, RpcCall { xid: 0x1122_3344, prog: 100003, vers: 3, proc_num: 1 });
        assert_eq!(args.u32(), Some(0xCAFE_BABE), "cursor lands on the args");
    }

    #[test]
    fn parse_rejects_non_call_and_bad_version() {
        let mut reply = Xdr::new();
        reply.u32(1); reply.u32(1); // xid, REPLY (not CALL)
        assert!(parse_call(&reply.into_bytes()).is_none());

        let mut badver = Xdr::new();
        badver.u32(1); badver.u32(0); badver.u32(3); // CALL but rpcvers=3
        assert!(parse_call(&badver.into_bytes()).is_none());
    }

    #[test]
    fn reply_header_bytes() {
        let bytes = reply(0x1122_3344, accept::SUCCESS).into_bytes();
        let mut c = Cur::new(&bytes);
        assert_eq!(c.u32(), Some(0x1122_3344)); // xid
        assert_eq!(c.u32(), Some(1)); // REPLY
        assert_eq!(c.u32(), Some(0)); // MSG_ACCEPTED
        assert_eq!(c.u32(), Some(0)); // verf flavor AUTH_NULL
        assert_eq!(c.u32(), Some(0)); // verf len
        assert_eq!(c.u32(), Some(0)); // accept_stat SUCCESS
    }

    // ── NFSv3 procedures (increment 3) ──────────────────────────────────────

    /// Build an NFSv3 RPC call with AUTH_NULL and pre-encoded `args`.
    fn build_call(proc_num: u32, args: &[u8]) -> Vec<u8> {
        let mut x = Xdr::new();
        x.u32(0x42); // xid
        x.u32(0); // CALL
        x.u32(2); // rpcvers
        x.u32(NFS_PROG);
        x.u32(NFS_V3);
        x.u32(proc_num);
        x.u32(0); x.u32(0); // cred AUTH_NULL
        x.u32(0); x.u32(0); // verf AUTH_NULL
        x.fixed(args); // args are already 4-aligned, so no extra pad
        x.into_bytes()
    }

    /// Run one call against a backing store, returning (accept_stat, cursor at
    /// the procedure result).
    fn run(b: &mut NfsBacking, proc_num: u32, args: &[u8]) -> (u32, Vec<u8>) {
        let req = build_call(proc_num, args);
        let (rpc, mut argcur) = parse_call(&req).unwrap();
        let reply_bytes = nfs3_call(&rpc, &mut argcur, b);
        // Validate + strip the 6-word accepted-reply header.
        let mut c = Cur::new(&reply_bytes);
        assert_eq!(c.u32(), Some(0x42)); // xid echoed
        assert_eq!(c.u32(), Some(1)); // REPLY
        assert_eq!(c.u32(), Some(0)); // ACCEPTED
        c.u32(); c.u32(); // verf
        let stat = c.u32().unwrap();
        let off = reply_bytes.len() - c.remaining().len();
        (stat, reply_bytes[off..].to_vec())
    }

    #[test]
    fn getattr_root_is_dir() {
        let root = temp_export();
        let mut b = NfsBacking::new(&root);
        let mut a = Xdr::new();
        put_fh(&mut a, ROOT_ID);
        let (stat, res) = run(&mut b, PROC3_GETATTR, &a.into_bytes());
        assert_eq!(stat, accept::SUCCESS);
        let mut r = Cur::new(&res);
        assert_eq!(r.u32(), Some(NFS3_OK)); // nfsstat3
        assert_eq!(r.u32(), Some(2)); // ftype3 == NF3DIR
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn lookup_then_read() {
        let root = temp_export();
        std::fs::write(root.join("hello.txt"), b"hello world").unwrap();
        let mut b = NfsBacking::new(&root);

        let mut a = Xdr::new();
        put_fh(&mut a, ROOT_ID);
        a.opaque(b"hello.txt");
        let (stat, res) = run(&mut b, PROC3_LOOKUP, &a.into_bytes());
        assert_eq!(stat, accept::SUCCESS);
        let mut r = Cur::new(&res);
        assert_eq!(r.u32(), Some(NFS3_OK));
        let fid = u64::from_be_bytes(r.opaque().unwrap().try_into().unwrap()); // object fh

        let mut a = Xdr::new();
        put_fh(&mut a, fid);
        a.u64(0); // offset
        a.u32(5); // count
        let (_stat, res) = run(&mut b, PROC3_READ, &a.into_bytes());
        let mut r = Cur::new(&res);
        assert_eq!(r.u32(), Some(NFS3_OK));
        // The read data is carried as the final opaque; assert the first 5 bytes
        // of the file appear in the reply.
        assert!(res.windows(5).any(|w| w == b"hello"), "read returned the data");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn readdir_lists_entries() {
        let root = temp_export();
        std::fs::write(root.join("a.txt"), b"1").unwrap();
        std::fs::write(root.join("b.txt"), b"2").unwrap();
        let mut b = NfsBacking::new(&root);
        let mut a = Xdr::new();
        put_fh(&mut a, ROOT_ID);
        a.u64(0); // cookie
        a.fixed(&[0u8; 8]); // cookieverf
        a.u32(8192); // count
        let (stat, res) = run(&mut b, PROC3_READDIR, &a.into_bytes());
        assert_eq!(stat, accept::SUCCESS);
        assert!(res.windows(5).any(|w| w == b"a.txt"));
        assert!(res.windows(5).any(|w| w == b"b.txt"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn fsinfo_advertises_sizes() {
        let root = temp_export();
        let mut b = NfsBacking::new(&root);
        let mut a = Xdr::new();
        put_fh(&mut a, ROOT_ID);
        let (stat, res) = run(&mut b, PROC3_FSINFO, &a.into_bytes());
        assert_eq!(stat, accept::SUCCESS);
        let mut r = Cur::new(&res);
        assert_eq!(r.u32(), Some(NFS3_OK));
        // skip post_op_attr (present + fattr3): bool + 21 words (fattr3 is 84 bytes)
        assert_eq!(r.u32(), Some(1)); // attrs follow
        for _ in 0..21 { r.u32(); } // fattr3 = 84 bytes = 21 words
        assert_eq!(r.u32(), Some(RTMAX)); // rtmax
        std::fs::remove_dir_all(&root).ok();
    }

    // ── write procedures + DRC (increment 4) ────────────────────────────────

    /// Build an NFSv3 call with a chosen xid.
    fn call_xid(xid: u32, proc_num: u32, args: &[u8]) -> Vec<u8> {
        let mut x = Xdr::new();
        x.u32(xid); x.u32(0); x.u32(2);
        x.u32(NFS_PROG); x.u32(NFS_V3); x.u32(proc_num);
        x.u32(0); x.u32(0); x.u32(0); x.u32(0); // AUTH_NULL cred+verf
        x.fixed(args);
        x.into_bytes()
    }

    fn lookup_fh(s: &mut NfsServer, name: &[u8]) -> u64 {
        let mut a = Xdr::new();
        put_fh(&mut a, ROOT_ID);
        a.opaque(name);
        let r = s.handle(&call_xid(1, PROC3_LOOKUP, &a.into_bytes())).unwrap();
        let mut c = Cur::new(&r);
        for _ in 0..6 { c.u32(); } // reply header
        assert_eq!(c.u32(), Some(NFS3_OK));
        u64::from_be_bytes(c.opaque().unwrap().try_into().unwrap())
    }

    #[test]
    fn write_through_server_and_drc_dedup() {
        let root = temp_export();
        std::fs::write(root.join("f.bin"), b"").unwrap();
        let mut s = NfsServer::new(&root, NfsVersion::Auto);
        let fid = lookup_fh(&mut s, b"f.bin");

        let write_args = |data: &[u8]| {
            let mut w = Xdr::new();
            put_fh(&mut w, fid);
            w.u64(0); // offset
            w.u32(data.len() as u32); // count
            w.u32(2); // stable = FILE_SYNC
            w.opaque(data);
            w.into_bytes()
        };

        s.handle(&call_xid(100, PROC3_WRITE, &write_args(b"AAAA"))).unwrap();
        assert_eq!(std::fs::read(root.join("f.bin")).unwrap(), b"AAAA");

        // Same xid, different data: the DRC must replay, NOT re-apply.
        s.handle(&call_xid(100, PROC3_WRITE, &write_args(b"BBBB"))).unwrap();
        assert_eq!(std::fs::read(root.join("f.bin")).unwrap(), b"AAAA", "retransmit deduped");

        // A fresh xid does apply.
        s.handle(&call_xid(101, PROC3_WRITE, &write_args(b"BBBB"))).unwrap();
        assert_eq!(std::fs::read(root.join("f.bin")).unwrap(), b"BBBB");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn create_and_remove_via_server() {
        let root = temp_export();
        let mut s = NfsServer::new(&root, NfsVersion::Auto);

        let mut a = Xdr::new();
        put_fh(&mut a, ROOT_ID);
        a.opaque(b"new.txt"); // diropargs3 (createmode/sattr3 are ignored by the handler)
        let r = s.handle(&call_xid(1, PROC3_CREATE, &a.into_bytes())).unwrap();
        let mut c = Cur::new(&r);
        for _ in 0..6 { c.u32(); }
        assert_eq!(c.u32(), Some(NFS3_OK));
        assert!(root.join("new.txt").exists());

        let mut rm = Xdr::new();
        put_fh(&mut rm, ROOT_ID);
        rm.opaque(b"new.txt");
        let r = s.handle(&call_xid(2, PROC3_REMOVE, &rm.into_bytes())).unwrap();
        let mut c = Cur::new(&r);
        for _ in 0..6 { c.u32(); }
        assert_eq!(c.u32(), Some(NFS3_OK));
        assert!(!root.join("new.txt").exists());

        std::fs::remove_dir_all(&root).ok();
    }

    // ── NFSv2 (increment 5) ─────────────────────────────────────────────────

    fn call2(xid: u32, proc_num: u32, args: &[u8]) -> Vec<u8> {
        let mut x = Xdr::new();
        x.u32(xid); x.u32(0); x.u32(2);
        x.u32(NFS_PROG); x.u32(NFS_V2); x.u32(proc_num);
        x.u32(0); x.u32(0); x.u32(0); x.u32(0);
        x.fixed(args);
        x.into_bytes()
    }

    #[test]
    fn v2_getattr_lookup_read_write() {
        let root = temp_export();
        std::fs::write(root.join("v2.txt"), b"hello v2").unwrap();
        let mut s = NfsServer::new(&root, NfsVersion::Auto);

        let mut a = Xdr::new();
        put_fh2(&mut a, ROOT_ID);
        let r = s.handle(&call2(1, PROC2_GETATTR, &a.into_bytes())).unwrap();
        let mut c = Cur::new(&r);
        for _ in 0..6 { c.u32(); }
        assert_eq!(c.u32(), Some(NFS2_OK));
        assert_eq!(c.u32(), Some(2)); // ftype2 == NFDIR

        let mut a = Xdr::new();
        put_fh2(&mut a, ROOT_ID);
        a.opaque(b"v2.txt");
        let r = s.handle(&call2(2, PROC2_LOOKUP, &a.into_bytes())).unwrap();
        let mut c = Cur::new(&r);
        for _ in 0..6 { c.u32(); }
        assert_eq!(c.u32(), Some(NFS2_OK));
        let fid = u64::from_be_bytes(c.fixed(32).unwrap()[..8].try_into().unwrap());

        let mut a = Xdr::new();
        put_fh2(&mut a, fid);
        a.u32(0); a.u32(5); a.u32(0); // offset, count, totalcount
        let r = s.handle(&call2(3, PROC2_READ, &a.into_bytes())).unwrap();
        assert!(r.windows(5).any(|w| w == b"hello"));

        let mut a = Xdr::new();
        put_fh2(&mut a, fid);
        a.u32(0); a.u32(0); a.u32(4); // beginoffset, offset, totalcount
        a.opaque(b"XYZW");
        s.handle(&call2(4, PROC2_WRITE, &a.into_bytes())).unwrap();
        assert_eq!(&std::fs::read(root.join("v2.txt")).unwrap()[..4], b"XYZW");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn v2_readdir_pages_within_one_datagram() {
        let root = temp_export();
        // Enough entries that the listing can't fit one UDP datagram, so the
        // reply MUST be paged. This is the Mac-Downloads case that produced
        // "NFS2 readdir failed ... Can't decode result": the old fixed 8000-byte
        // budget overran the client's count-sized buffer and spilled into
        // multiple IP fragments.
        let n: usize = 200;
        for i in 0..n {
            std::fs::write(root.join(format!("file_{i:03}")), b"x").unwrap();
        }
        let mut s = NfsServer::new(&root, NfsVersion::Auto);

        // Decode one v2 READDIR reply: strip the 6-word RPC header + status, walk
        // the entry list, return (names, last_cookie, eof).
        fn decode(reply: &[u8]) -> (Vec<Vec<u8>>, [u8; 4], bool) {
            let mut c = Cur::new(reply);
            for _ in 0..6 { c.u32(); } // RPC accepted-reply header
            assert_eq!(c.u32(), Some(NFS2_OK));
            let mut names = Vec::new();
            let mut last_cookie = [0u8; 4];
            while c.u32() == Some(1) { // value_follows; 0 ends the list
                c.u32(); // fileid
                names.push(c.opaque().unwrap().to_vec());
                last_cookie = c.fixed(4).unwrap().try_into().unwrap();
            }
            let eof = c.u32() == Some(1);
            (names, last_cookie, eof)
        }

        let mut all = Vec::new();
        let mut cookie = [0u8; 4];
        let mut pages: u32 = 0;
        loop {
            let mut a = Xdr::new();
            put_fh2(&mut a, ROOT_ID);
            a.fixed(&cookie); // nfscookie
            a.u32(8192); // count — a typical IRIX NFSv2 readdir size
            let r = s.handle(&call2(100 + pages, PROC2_READDIR, &a.into_bytes())).unwrap();
            // The whole reply must fit one unfragmented UDP datagram (≤ 1472).
            assert!(r.len() <= 1472, "readdir page {pages} is {} bytes — would fragment", r.len());
            let (names, last, eof) = decode(&r);
            assert!(!names.is_empty(), "each page must make progress");
            all.extend(names);
            pages += 1;
            if eof { break; }
            cookie = last;
            assert!(pages < 50, "paging should terminate");
        }
        assert!(pages > 1, "the listing should have needed multiple pages");
        all.sort();
        all.dedup();
        assert_eq!(all.len(), n + 2, "every entry (plus . and ..) returned exactly once");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn mount_returns_root_handle() {
        let root = temp_export();
        let mut s = NfsServer::new(&root, NfsVersion::Auto);

        // MOUNT v3 MNT "/" -> mountstat3 OK + fhandle3(root) + auth flavors.
        let mut req = Xdr::new();
        req.u32(7); req.u32(0); req.u32(2);
        req.u32(MOUNT_PROG); req.u32(MOUNT_V3); req.u32(MNTPROC_MNT);
        req.u32(0); req.u32(0); req.u32(0); req.u32(0);
        req.opaque(b"/");
        let r = s.handle(&req.into_bytes()).unwrap();
        let mut c = Cur::new(&r);
        for _ in 0..6 { c.u32(); } // reply header
        assert_eq!(c.u32(), Some(0)); // MNT3_OK
        let fh = c.opaque().unwrap();
        assert_eq!(u64::from_be_bytes(fh.try_into().unwrap()), ROOT_ID);
        assert_eq!(c.u32(), Some(1)); // one auth flavor
        assert_eq!(c.u32(), Some(0)); // AUTH_NULL

        // MOUNT v1 MNT -> fhstatus OK + 32-byte fhandle.
        let mut req = Xdr::new();
        req.u32(8); req.u32(0); req.u32(2);
        req.u32(MOUNT_PROG); req.u32(MOUNT_V1); req.u32(MNTPROC_MNT);
        req.u32(0); req.u32(0); req.u32(0); req.u32(0);
        req.opaque(b"/");
        let r = s.handle(&req.into_bytes()).unwrap();
        let mut c = Cur::new(&r);
        for _ in 0..6 { c.u32(); }
        assert_eq!(c.u32(), Some(0)); // fhstatus OK
        let fh = c.fixed(32).unwrap();
        assert_eq!(u64::from_be_bytes(fh[..8].try_into().unwrap()), ROOT_ID);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn version_gating() {
        // A V3-only server rejects a v2 call with PROG_MISMATCH.
        let root = temp_export();
        let mut s = NfsServer::new(&root, NfsVersion::V3);
        let mut a = Xdr::new();
        put_fh2(&mut a, ROOT_ID);
        let r = s.handle(&call2(1, PROC2_GETATTR, &a.into_bytes())).unwrap();
        let mut c = Cur::new(&r);
        c.u32(); c.u32(); c.u32(); c.u32(); c.u32(); // xid, REPLY, ACCEPTED, verf x2
        assert_eq!(c.u32(), Some(accept::PROG_MISMATCH));
        std::fs::remove_dir_all(&root).ok();
    }
}
