//! Bounded, no-follow corpus discovery and atomic generator output.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const MAX_MANIFEST_BYTES: usize = 64 * 1024;
pub(super) const MAX_README_BYTES: usize = 64 * 1024;
pub(super) const MAX_RECORDING_BYTES: usize = 256 * 1024;
pub(super) const OUTPUT_ANCHOR: &str = ".rex3-corpus-root";

pub(super) fn size_limit(name: &str) -> usize {
    match name {
        "manifest.toml" => MAX_MANIFEST_BYTES,
        "README.md" => MAX_README_BYTES,
        _ => MAX_RECORDING_BYTES,
    }
}

fn checked_relative(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty() || !path.components().all(|c| matches!(c, Component::Normal(_)))
    {
        return Err("corpus directory must be a nonempty relative path without ..".into());
    }
    Ok(())
}

fn checked_name(name: &str) -> Result<(), String> {
    let mut components = Path::new(name).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(format!("invalid corpus filename {name:?}"));
    }
    Ok(())
}

fn checked_output_name(name: &str) -> Result<(), String> {
    checked_name(name)?;
    if name.contains(':') || name.contains('\0') {
        return Err(format!("invalid corpus output filename {name:?}"));
    }
    if name.eq_ignore_ascii_case(OUTPUT_ANCHOR) {
        return Err(format!(
            "{OUTPUT_ANCHOR} is a reserved corpus output anchor"
        ));
    }
    Ok(())
}

fn without_anchor(
    mut names: Vec<String>,
    expected: &BTreeSet<String>,
) -> Result<Vec<String>, String> {
    if expected
        .iter()
        .any(|name| name.eq_ignore_ascii_case(OUTPUT_ANCHOR))
    {
        return Err(format!("{OUTPUT_ANCHOR} is not corpus content"));
    }
    if !names.iter().any(|name| name == OUTPUT_ANCHOR) {
        return Err(format!(
            "missing required corpus output anchor {OUTPUT_ANCHOR}"
        ));
    }
    names.retain(|name| name != OUTPUT_ANCHOR);
    Ok(names)
}

#[derive(Clone, Copy, Debug)]
struct EntryMetadata {
    len: u64,
    regular: bool,
    reparse: bool,
}

fn check_regular(name: &str, meta: EntryMetadata) -> Result<(), String> {
    if meta.reparse {
        return Err(format!("{name}: symlink or reparse point is forbidden"));
    }
    if !meta.regular {
        return Err(format!("{name}: corpus entry is not a regular file"));
    }
    Ok(())
}

fn check_size(name: &str, len: u64, limit: usize) -> Result<(), String> {
    if len > limit as u64 {
        return Err(format!("{name}: corpus file exceeds {limit} bytes"));
    }
    Ok(())
}

fn read_bounded(reader: impl Read, len: u64, limit: usize, name: &str) -> Result<Vec<u8>, String> {
    check_size(name, len, limit)?;
    let mut bytes = Vec::with_capacity(len as usize);
    reader
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{name}: {error}"))?;
    check_size(name, bytes.len() as u64, limit)?;
    Ok(bytes)
}

fn read_named_entries(
    names: Vec<String>,
    expected: &BTreeSet<String>,
    mut read: impl FnMut(&str, usize) -> Result<Vec<u8>, String>,
) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut seen = BTreeSet::new();
    for name in names {
        checked_name(&name)?;
        if !expected.contains(&name) {
            return Err(format!("unexpected corpus file {name}"));
        }
        if !seen.insert(name.clone()) {
            return Err(format!("duplicate corpus directory entry {name}"));
        }
    }
    for name in expected {
        if !seen.contains(name) {
            return Err(match name.as_str() {
                "manifest.toml" => "missing required corpus manifest.toml".into(),
                "README.md" => "missing required corpus README.md".into(),
                _ => format!("missing named corpus file {name}"),
            });
        }
    }
    let mut files = BTreeMap::new();
    for name in seen {
        let bytes = read(&name, size_limit(&name))?;
        files.insert(name, bytes);
    }
    Ok(files)
}

fn push_expected_name(
    names: &mut Vec<String>,
    name: String,
    expected: &BTreeSet<String>,
) -> Result<(), String> {
    checked_name(&name)?;
    if !expected.contains(&name) {
        return Err(format!("unexpected corpus file {name}"));
    }
    if names.contains(&name) {
        return Err(format!("duplicate corpus directory entry {name}"));
    }
    names.push(name);
    Ok(())
}

pub(super) fn read_files(
    root: &Path,
    relative: &Path,
    expected: &BTreeSet<String>,
) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let directory = SafeDirectory::open_anchored(root, relative).map_err(|error| {
        format!(
            "required REX3 corpus directory {}: {error}",
            root.join(relative).display()
        )
    })?;
    let mut allowed = expected.clone();
    allowed.insert(OUTPUT_ANCHOR.to_owned());
    let names = without_anchor(directory.names(&allowed)?, expected)?;
    read_named_entries(names, expected, |name, limit| {
        let before = directory
            .metadata(name)
            .map_err(|error| format!("{name}: {error}"))?;
        check_regular(name, before)?;
        check_size(name, before.len, limit)?;
        let file = directory
            .open_file(name)
            .map_err(|error| format!("{name}: {error}"))?;
        let opened = file
            .metadata()
            .map(entry_metadata)
            .map_err(|error| format!("{name}: {error}"))?;
        check_regular(name, opened)?;
        read_bounded(file, opened.len, limit, name)
    })
}

pub(super) fn write_files(
    root: &Path,
    relative: &Path,
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<(), String> {
    for name in files.keys() {
        checked_output_name(name)?;
    }
    let directory = SafeDirectory::open_anchored(root, relative)?;
    write_open_directory(&directory, files)
}

pub(super) fn write_generated(
    root: &Path,
    relative: &Path,
    generate: impl FnOnce() -> BTreeMap<String, Vec<u8>>,
) -> Result<(), String> {
    let directory = SafeDirectory::open_anchored(root, relative)?;
    let files = generate();
    for name in files.keys() {
        checked_output_name(name)?;
    }
    write_open_directory(&directory, &files)
}

fn write_open_directory(
    directory: &SafeDirectory,
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<(), String> {
    for name in files.keys() {
        match directory.metadata(name) {
            Ok(meta) => check_regular(name, meta)?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("{name}: {error}")),
        }
    }
    for (name, bytes) in files {
        directory.atomic_write(name, bytes)?;
        eprintln!("{name}: {} bytes", bytes.len());
    }
    Ok(())
}

struct SafeDirectory {
    #[cfg(unix)]
    dir: File,
    #[cfg(windows)]
    _parents: Vec<File>,
    _anchor: Option<File>,
}

impl SafeDirectory {
    fn open(root: &Path, relative: &Path) -> Result<Self, String> {
        checked_relative(relative)?;
        if !root.is_absolute() {
            return Err("corpus repository root must be absolute".into());
        }
        Self::open_from_root(root, relative)
            .map_err(|error| format!("{}: {error}", root.join(relative).display()))
    }

    fn open_anchored(root: &Path, relative: &Path) -> Result<Self, String> {
        let mut directory = Self::open(root, relative)?;
        directory._anchor = Some(directory.open_anchor().map_err(|error| {
            format!("required pre-existing corpus anchor {OUTPUT_ANCHOR}: {error}")
        })?);
        Ok(directory)
    }

    fn atomic_write(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
        let (temp_name, file) = (0..128)
            .find_map(|_| {
                let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
                let temp_name = format!(".rex3-corpus-{}-{id}.part", std::process::id());
                match self.create_new(&temp_name) {
                    Ok(file) => Some(Ok((temp_name, file))),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .ok_or_else(|| format!("{name}: could not reserve an atomic output file"))?
            .map_err(|error| format!("{name}: {error}"))?;
        let mut cleanup = UncommittedFile {
            directory: self,
            name: temp_name,
            file,
            committed: false,
        };
        cleanup
            .file
            .write_all(bytes)
            .and_then(|()| cleanup.file.sync_all())
            .map_err(|error| format!("{name}: {error}"))?;
        self.rename(&cleanup.file, &cleanup.name, name)
            .map_err(|error| format!("{name}: atomic replacement failed: {error}"))?;
        cleanup.committed = true;
        Ok(())
    }
}

struct UncommittedFile<'a> {
    directory: &'a SafeDirectory,
    name: String,
    file: File,
    committed: bool,
}

impl Drop for UncommittedFile<'_> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.directory.remove_open(&self.file, &self.name);
        }
    }
}

#[cfg(unix)]
fn entry_metadata(meta: std::fs::Metadata) -> EntryMetadata {
    EntryMetadata {
        len: meta.len(),
        regular: meta.is_file(),
        reparse: meta.file_type().is_symlink(),
    }
}

#[cfg(windows)]
fn entry_metadata(meta: std::fs::Metadata) -> EntryMetadata {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    EntryMetadata {
        len: meta.len(),
        regular: meta.is_file(),
        reparse: meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0,
    }
}

#[cfg(unix)]
mod platform {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::mem::MaybeUninit;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    fn c_name(name: &std::ffi::OsStr) -> io::Result<CString> {
        CString::new(name.as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "filename contains NUL"))
    }

    fn open_at(dir: &File, name: &std::ffi::OsStr, flags: i32, mode: u32) -> io::Result<File> {
        let name = c_name(name)?;
        let fd = unsafe { libc::openat(dir.as_raw_fd(), name.as_ptr(), flags, mode) };
        if fd < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(unsafe { File::from_raw_fd(fd) })
        }
    }

    fn stat_at(dir: &File, name: &std::ffi::OsStr) -> io::Result<libc::stat> {
        let name = c_name(name)?;
        let mut stat = MaybeUninit::<libc::stat>::uninit();
        if unsafe {
            libc::fstatat(
                dir.as_raw_fd(),
                name.as_ptr(),
                stat.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } != 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(unsafe { stat.assume_init() })
        }
    }

    #[cfg(any(
        target_os = "linux",
        target_os = "emscripten",
        target_os = "hurd",
        target_os = "dragonfly"
    ))]
    unsafe fn clear_errno() {
        unsafe { *libc::__errno_location() = 0 };
    }

    #[cfg(any(target_vendor = "apple", target_os = "freebsd"))]
    unsafe fn clear_errno() {
        unsafe { *libc::__error() = 0 };
    }

    #[cfg(any(target_os = "android", target_os = "netbsd", target_os = "openbsd"))]
    unsafe fn clear_errno() {
        unsafe { *libc::__errno() = 0 };
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "emscripten",
        target_os = "hurd",
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "android",
        target_os = "netbsd",
        target_os = "openbsd"
    )))]
    unsafe fn clear_errno() {
        // Unsupported targets fail closed rather than treating an I/O error as EOF.
    }

    struct DirectoryStream(*mut libc::DIR);

    impl Drop for DirectoryStream {
        fn drop(&mut self) {
            unsafe { libc::closedir(self.0) };
        }
    }

    impl SafeDirectory {
        pub(super) fn open_from_root(root: &Path, relative: &Path) -> io::Result<Self> {
            // Resolve trusted checkout aliases before enforcing no-follow beneath it.
            let mut dir = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC)
                .open(root)?;
            let canonical_root = root.canonicalize()?;
            let canonical_dir = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(canonical_root)?;
            let actual = dir.metadata()?;
            let canonical = canonical_dir.metadata()?;
            if (actual.dev(), actual.ino()) != (canonical.dev(), canonical.ino()) {
                return Err(io::Error::other(
                    "checkout root changed during canonicalization",
                ));
            }
            for component in relative.components() {
                match component {
                    Component::Normal(name) => {
                        dir = open_at(
                            &dir,
                            name,
                            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                            0,
                        )?;
                    }
                    _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid root")),
                }
            }
            Ok(Self { dir, _anchor: None })
        }

        pub(super) fn open_anchor(&self) -> io::Result<File> {
            let name = std::ffi::OsStr::new(OUTPUT_ANCHOR);
            let anchor = open_at(
                &self.dir,
                name,
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
                0,
            )?;
            let opened = anchor.metadata()?;
            let entry = stat_at(&self.dir, name)?;
            if !opened.is_file()
                || entry.st_mode & libc::S_IFMT != libc::S_IFREG
                || (opened.dev(), opened.ino()) != (entry.st_dev as u64, entry.st_ino as u64)
            {
                return Err(io::Error::other(
                    "anchor is not the same regular directory entry",
                ));
            }
            Ok(anchor)
        }

        pub(super) fn names(&self, expected: &BTreeSet<String>) -> Result<Vec<String>, String> {
            #[cfg(not(any(
                target_os = "linux",
                target_os = "emscripten",
                target_os = "hurd",
                target_vendor = "apple",
                target_os = "freebsd",
                target_os = "dragonfly",
                target_os = "android",
                target_os = "netbsd",
                target_os = "openbsd"
            )))]
            return Err("safe corpus directory enumeration is unsupported on this platform".into());

            let dir = open_at(
                &self.dir,
                std::ffi::OsStr::new("."),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0,
            )
            .map_err(|error| format!("open corpus directory: {error}"))?;
            use std::os::fd::IntoRawFd;
            let fd = dir.into_raw_fd();
            let stream = unsafe { libc::fdopendir(fd) };
            if stream.is_null() {
                let error = io::Error::last_os_error();
                unsafe { libc::close(fd) };
                return Err(format!("read corpus directory: {error}"));
            }
            let stream = DirectoryStream(stream);
            let mut names = Vec::new();
            loop {
                unsafe { clear_errno() };
                let entry = unsafe { libc::readdir(stream.0) };
                if entry.is_null() {
                    let error = io::Error::last_os_error();
                    if error.raw_os_error() == Some(0) {
                        break;
                    }
                    return Err(format!("read corpus directory: {error}"));
                }
                let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr().cast()) }
                    .to_str()
                    .map_err(|_| "non-UTF-8 corpus filename")?;
                if name != "." && name != ".." {
                    push_expected_name(&mut names, name.to_owned(), expected)?;
                }
            }
            Ok(names)
        }

        pub(super) fn metadata(&self, name: &str) -> io::Result<EntryMetadata> {
            let stat = stat_at(&self.dir, std::ffi::OsStr::new(name))?;
            let kind = stat.st_mode & libc::S_IFMT;
            Ok(EntryMetadata {
                len: u64::try_from(stat.st_size).unwrap_or(u64::MAX),
                regular: kind == libc::S_IFREG,
                reparse: kind == libc::S_IFLNK,
            })
        }

        pub(super) fn open_file(&self, name: &str) -> io::Result<File> {
            open_at(
                &self.dir,
                std::ffi::OsStr::new(name),
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
                0,
            )
        }

        pub(super) fn create_new(&self, name: &str) -> io::Result<File> {
            open_at(
                &self.dir,
                std::ffi::OsStr::new(name),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o666,
            )
        }

        pub(super) fn rename(&self, _file: &File, from: &str, to: &str) -> io::Result<()> {
            let from = c_name(std::ffi::OsStr::new(from))?;
            let to = c_name(std::ffi::OsStr::new(to))?;
            let result = unsafe {
                libc::renameat(
                    self.dir.as_raw_fd(),
                    from.as_ptr(),
                    self.dir.as_raw_fd(),
                    to.as_ptr(),
                )
            };
            if result == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }

        pub(super) fn remove(&self, name: &str) -> io::Result<()> {
            let name = c_name(std::ffi::OsStr::new(name))?;
            let result = unsafe { libc::unlinkat(self.dir.as_raw_fd(), name.as_ptr(), 0) };
            if result == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }

        pub(super) fn remove_open(&self, _file: &File, name: &str) -> io::Result<()> {
            self.remove(name)
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::{AsRawHandle, FromRawHandle};

    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    const FILE_SHARE_READ: u32 = 1;
    const FILE_SHARE_WRITE: u32 = 2;
    const FILE_SHARE_DELETE: u32 = 4;
    const DELETE: u32 = 0x0001_0000;
    const SYNCHRONIZE: u32 = 0x0010_0000;
    const FILE_READ_ATTRIBUTES: u32 = 0x80;
    const GENERIC_READ: u32 = 0x8000_0000;
    const GENERIC_WRITE: u32 = 0x4000_0000;
    const FILE_OPEN: u32 = 1;
    const FILE_CREATE: u32 = 2;
    const FILE_DIRECTORY_FILE: u32 = 0x1;
    const FILE_SYNCHRONOUS_IO_NONALERT: u32 = 0x20;
    const FILE_NON_DIRECTORY_FILE: u32 = 0x40;
    const FILE_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    const OBJ_CASE_INSENSITIVE: u32 = 0x40;

    #[repr(C)]
    struct UnicodeString {
        length: u16,
        maximum_length: u16,
        buffer: *mut u16,
    }

    #[repr(C)]
    struct ObjectAttributes {
        length: u32,
        root_directory: *mut c_void,
        object_name: *mut UnicodeString,
        attributes: u32,
        security_descriptor: *mut c_void,
        security_quality_of_service: *mut c_void,
    }

    #[repr(C)]
    struct FileRenameInfo {
        replace_if_exists: u32,
        root_directory: *mut c_void,
        file_name_length: u32,
        file_name: [u16; 1],
    }

    #[repr(C)]
    struct IoStatusBlock {
        status_or_pointer: usize,
        information: usize,
    }

    #[repr(C)]
    struct FileInformation {
        attributes: u32,
        creation_time: [u32; 2],
        access_time: [u32; 2],
        write_time: [u32; 2],
        volume_serial: u32,
        size_high: u32,
        size_low: u32,
        link_count: u32,
        index_high: u32,
        index_low: u32,
    }

    #[repr(C)]
    struct FileIdInformation {
        volume_serial: u64,
        file_id: [u8; 16],
    }

    #[derive(Debug, Eq, PartialEq)]
    enum FileIdentity {
        Full(u64, [u8; 16]),
        Legacy(u32, u64),
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetFileInformationByHandle(
            file: *mut c_void,
            class: i32,
            info: *const c_void,
            len: u32,
        ) -> i32;
        fn GetFileInformationByHandle(file: *mut c_void, info: *mut FileInformation) -> i32;
        fn GetFileInformationByHandleEx(
            file: *mut c_void,
            class: i32,
            info: *mut c_void,
            len: u32,
        ) -> i32;
    }

    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtCreateFile(
            file: *mut *mut c_void,
            desired_access: u32,
            attributes: *mut ObjectAttributes,
            status: *mut IoStatusBlock,
            allocation_size: *mut c_void,
            file_attributes: u32,
            share_access: u32,
            disposition: u32,
            options: u32,
            ea_buffer: *mut c_void,
            ea_length: u32,
        ) -> i32;
        fn NtQueryDirectoryFile(
            file: *mut c_void,
            event: *mut c_void,
            apc_routine: *mut c_void,
            apc_context: *mut c_void,
            status: *mut IoStatusBlock,
            info: *mut c_void,
            len: u32,
            class: i32,
            single_entry: u8,
            name: *mut UnicodeString,
            restart_scan: u8,
        ) -> i32;
        fn NtSetInformationFile(
            file: *mut c_void,
            status: *mut IoStatusBlock,
            info: *mut c_void,
            len: u32,
            class: i32,
        ) -> i32;
        fn RtlNtStatusToDosError(status: i32) -> u32;
    }

    fn nt_error(status: i32) -> io::Error {
        io::Error::from_raw_os_error(unsafe { RtlNtStatusToDosError(status) as i32 })
    }

    fn open_relative(
        directory: &File,
        name: &std::ffi::OsStr,
        access: u32,
        share: u32,
        disposition: u32,
        options: u32,
    ) -> io::Result<File> {
        let mut wide: Vec<u16> = name.encode_wide().collect();
        let length = wide
            .len()
            .checked_mul(2)
            .and_then(|len| u16::try_from(len).ok())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "filename too long"))?;
        let mut name = UnicodeString {
            length,
            maximum_length: length,
            buffer: wide.as_mut_ptr(),
        };
        let mut attributes = ObjectAttributes {
            length: std::mem::size_of::<ObjectAttributes>() as u32,
            root_directory: directory.as_raw_handle(),
            object_name: &mut name,
            attributes: OBJ_CASE_INSENSITIVE,
            security_descriptor: std::ptr::null_mut(),
            security_quality_of_service: std::ptr::null_mut(),
        };
        let mut status_block = IoStatusBlock {
            status_or_pointer: 0,
            information: 0,
        };
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            NtCreateFile(
                &mut handle,
                access | SYNCHRONIZE,
                &mut attributes,
                &mut status_block,
                std::ptr::null_mut(),
                0x80, // FILE_ATTRIBUTE_NORMAL
                share,
                disposition,
                options | FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT,
                std::ptr::null_mut(),
                0,
            )
        };
        if status < 0 {
            Err(nt_error(status))
        } else {
            Ok(unsafe { File::from_raw_handle(handle) })
        }
    }

    fn set_file_information(
        file: &File,
        class: i32,
        info: *const c_void,
        len: u32,
    ) -> io::Result<()> {
        if unsafe { SetFileInformationByHandle(file.as_raw_handle(), class, info, len) } == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn checked_full_identity(volume: u64, id: [u8; 16]) -> io::Result<FileIdentity> {
        let id_value = u128::from_le_bytes(id);
        if matches!(id_value, 0 | 0xffff_ffff_ffff_ffff | u128::MAX) {
            return Err(io::Error::other("file identity is unavailable"));
        }
        Ok(FileIdentity::Full(volume, id))
    }

    fn checked_legacy_identity(volume: u32, id: u64) -> io::Result<FileIdentity> {
        if matches!(id, 0 | u64::MAX) {
            return Err(io::Error::other("file identity is unavailable"));
        }
        Ok(FileIdentity::Legacy(volume, id))
    }

    fn select_file_identity(
        full: io::Result<(u64, [u8; 16])>,
        legacy: impl FnOnce() -> io::Result<(u32, u64)>,
    ) -> io::Result<FileIdentity> {
        match full {
            Ok((volume, id)) => checked_full_identity(volume, id),
            // Fall back only when the full-width query is unsupported.
            Err(error) if matches!(error.raw_os_error(), Some(1 | 50 | 87 | 120 | 124)) => {
                let (volume, id) = legacy()?;
                checked_legacy_identity(volume, id)
            }
            Err(error) => Err(error),
        }
    }

    fn file_identity(file: &File) -> io::Result<FileIdentity> {
        let mut full = std::mem::MaybeUninit::<FileIdInformation>::uninit();
        let full = if unsafe {
            GetFileInformationByHandleEx(
                file.as_raw_handle(),
                18, // FileIdInfo
                full.as_mut_ptr().cast(),
                std::mem::size_of::<FileIdInformation>() as u32,
            )
        } != 0
        {
            let full = unsafe { full.assume_init() };
            Ok((full.volume_serial, full.file_id))
        } else {
            Err(io::Error::last_os_error())
        };
        select_file_identity(full, || {
            let mut info = std::mem::MaybeUninit::<FileInformation>::uninit();
            if unsafe { GetFileInformationByHandle(file.as_raw_handle(), info.as_mut_ptr()) } == 0 {
                return Err(io::Error::last_os_error());
            }
            let info = unsafe { info.assume_init() };
            Ok((
                info.volume_serial,
                ((info.index_high as u64) << 32) | info.index_low as u64,
            ))
        })
    }

    fn open_directory(path: &Path) -> io::Result<File> {
        OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)
    }

    fn check_directory(file: &File) -> io::Result<()> {
        let meta = file.metadata()?;
        if entry_metadata(meta.clone()).reparse || !meta.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "directory is a reparse point or not a directory",
            ));
        }
        Ok(())
    }

    fn parse_directory_names(
        bytes: &[u8],
        names: &mut Vec<String>,
        expected: &BTreeSet<String>,
    ) -> Result<(), String> {
        let mut offset = 0usize;
        loop {
            let header_end = offset
                .checked_add(12)
                .ok_or("invalid corpus directory entry offset")?;
            let header = bytes
                .get(offset..header_end)
                .ok_or("truncated corpus directory entry")?;
            let next = u32::from_ne_bytes(header[0..4].try_into().unwrap()) as usize;
            let name_len = u32::from_ne_bytes(header[8..12].try_into().unwrap()) as usize;
            let name_end = header_end
                .checked_add(name_len)
                .filter(|&end| end <= bytes.len() && name_len % 2 == 0)
                .ok_or("invalid corpus directory filename length")?;
            let wide: Vec<u16> = bytes[header_end..name_end]
                .chunks_exact(2)
                .map(|pair| u16::from_ne_bytes([pair[0], pair[1]]))
                .collect();
            let name = String::from_utf16(&wide).map_err(|_| "non-UTF-8 corpus filename")?;
            if name != "." && name != ".." {
                push_expected_name(names, name, expected)?;
            }
            if next == 0 {
                return Ok(());
            }
            let next_offset = offset.checked_add(next);
            if next < 12 + name_len
                || next % 4 != 0
                || next_offset.is_none_or(|next_offset| next_offset >= bytes.len())
            {
                return Err("invalid next corpus directory entry".into());
            }
            offset = next_offset.unwrap();
        }
    }

    #[test]
    fn rex3_corpus_windows_identity_rejects_unavailable_ids() {
        assert_eq!(std::mem::size_of::<FileIdInformation>(), 24);
        for id in [0u64, u64::MAX] {
            assert!(checked_legacy_identity(7, id).is_err());
            assert!(
                select_file_identity(Err(io::Error::from_raw_os_error(50)), || { Ok((7, id)) })
                    .is_err()
            );
        }
        for id in [0u128, u64::MAX as u128, u128::MAX] {
            assert!(checked_full_identity(7, id.to_le_bytes()).is_err());
            assert!(select_file_identity(Ok((7, id.to_le_bytes())), || {
                panic!("an unavailable full ID must not fall back")
            })
            .is_err());
        }
        let low = 0x1234_5678_9abc_def0u64;
        let first = checked_full_identity(7, ((1u128 << 64) | low as u128).to_le_bytes()).unwrap();
        let second = checked_full_identity(7, ((2u128 << 64) | low as u128).to_le_bytes()).unwrap();
        assert_ne!(first, second, "the upper 64 bits must affect identity");
        assert_ne!(first, checked_legacy_identity(7, low).unwrap());
        assert!(checked_full_identity(7, ((1u128 << 64) | u64::MAX as u128).to_le_bytes()).is_ok());
        for error in [1, 50, 87, 120, 124] {
            assert_eq!(
                select_file_identity(Err(io::Error::from_raw_os_error(error)), || Ok((7, low)))
                    .unwrap(),
                FileIdentity::Legacy(7, low)
            );
        }
        let denied = select_file_identity(Err(io::Error::from_raw_os_error(5)), || {
            panic!("an access error must not fall back")
        })
        .unwrap_err();
        assert_eq!(denied.raw_os_error(), Some(5));
        assert_eq!(
            select_file_identity(Ok((7, ((1u128 << 64) | low as u128).to_le_bytes())), || {
                panic!("a supported full-width ID must not fall back")
            })
            .unwrap(),
            first
        );

        let root = open_directory(Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();
        let identity = file_identity(&root).expect("checkout must have a usable file ID");
        eprintln!(
            "checkout identity source: {}",
            match identity {
                FileIdentity::Full(..) => "128-bit",
                FileIdentity::Legacy(..) => "64-bit fallback",
            }
        );
    }

    #[test]
    fn rex3_corpus_windows_directory_records_are_bounded() {
        fn record(name: &str, has_next: bool) -> Vec<u8> {
            let wide: Vec<_> = name.encode_utf16().collect();
            let mut bytes = vec![0; (12 + wide.len() * 2 + 3) & !3];
            if has_next {
                let next = bytes.len() as u32;
                bytes[0..4].copy_from_slice(&next.to_ne_bytes());
            }
            bytes[8..12].copy_from_slice(&((wide.len() * 2) as u32).to_ne_bytes());
            for (index, word) in wide.iter().enumerate() {
                bytes[12 + index * 2..14 + index * 2].copy_from_slice(&word.to_ne_bytes());
            }
            bytes
        }

        let expected = [
            "fixture.txt".to_owned(),
            OUTPUT_ANCHOR.to_owned(),
            "x".to_owned(),
        ]
        .into_iter()
        .collect();
        let mut valid = record("fixture.txt", true);
        valid.extend(record(OUTPUT_ANCHOR, false));
        let mut names = Vec::new();
        parse_directory_names(&valid, &mut names, &expected).unwrap();
        assert_eq!(names, ["fixture.txt", OUTPUT_ANCHOR]);

        for malformed in [
            vec![0; 11],
            {
                let mut bytes = record("x", false);
                bytes[8..12].copy_from_slice(&1u32.to_ne_bytes());
                bytes
            },
            {
                let mut bytes = record("x", false);
                bytes[8..12].copy_from_slice(&u32::MAX.to_ne_bytes());
                bytes
            },
            {
                let mut bytes = record("x", true);
                bytes[0..4].copy_from_slice(&4u32.to_ne_bytes());
                bytes
            },
            record("x", true),
        ] {
            assert!(parse_directory_names(&malformed, &mut Vec::new(), &expected).is_err());
        }
        assert!(
            parse_directory_names(&record("unexpected", false), &mut Vec::new(), &expected)
                .unwrap_err()
                .contains("unexpected corpus file")
        );
        let mut duplicate = record("fixture.txt", true);
        duplicate.extend(record("fixture.txt", false));
        assert!(
            parse_directory_names(&duplicate, &mut Vec::new(), &expected)
                .unwrap_err()
                .contains("duplicate corpus directory entry")
        );
    }

    impl SafeDirectory {
        pub(super) fn open_from_root(root: &Path, relative: &Path) -> io::Result<Self> {
            Self::open_from_root_after_pin(root, relative, || Ok(()))
        }

        pub(super) fn open_from_root_after_pin(
            root: &Path,
            relative: &Path,
            after_pin: impl FnOnce() -> io::Result<()>,
        ) -> io::Result<Self> {
            // Follow trusted checkout aliases once; all later traversal uses this handle.
            let trusted_root = open_directory(root)?;
            check_directory(&trusted_root)?;
            file_identity(&trusted_root)?;
            after_pin()?;
            let mut parents = vec![trusted_root];
            for component in relative.components() {
                let Component::Normal(name) = component else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "invalid relative path",
                    ));
                };
                let parent = parents.last().unwrap();
                check_directory(parent)?;
                let dir = open_relative(
                    parent,
                    name,
                    GENERIC_READ,
                    FILE_SHARE_READ,
                    FILE_OPEN,
                    FILE_DIRECTORY_FILE,
                )?;
                check_directory(&dir)?;
                file_identity(&dir)?;
                check_directory(parent)?;
                parents.push(dir);
            }
            Ok(Self {
                _parents: parents,
                _anchor: None,
            })
        }

        pub(super) fn open_anchor(&self) -> io::Result<File> {
            let directory = self._parents.last().unwrap();
            check_directory(directory)?;
            let name = std::ffi::OsStr::new(OUTPUT_ANCHOR);
            let anchor = open_relative(
                directory,
                name,
                GENERIC_READ,
                FILE_SHARE_READ,
                FILE_OPEN,
                FILE_NON_DIRECTORY_FILE,
            )?;
            let opened = entry_metadata(anchor.metadata()?);
            check_regular(OUTPUT_ANCHOR, opened).map_err(io::Error::other)?;
            let entry = open_relative(
                directory,
                name,
                FILE_READ_ATTRIBUTES,
                FILE_SHARE_READ,
                FILE_OPEN,
                FILE_NON_DIRECTORY_FILE,
            )?;
            check_regular(OUTPUT_ANCHOR, entry_metadata(entry.metadata()?))
                .map_err(io::Error::other)?;
            if file_identity(&anchor)? != file_identity(&entry)? {
                return Err(io::Error::other("anchor identity changed while opening"));
            }
            check_directory(directory)?;
            Ok(anchor)
        }

        pub(super) fn names(&self, expected: &BTreeSet<String>) -> Result<Vec<String>, String> {
            let directory = self._parents.last().unwrap();
            let mut names = Vec::new();
            let mut buffer = vec![0u64; 8192];
            let mut restart = true;
            loop {
                let mut status_block = IoStatusBlock {
                    status_or_pointer: 0,
                    information: 0,
                };
                let status = unsafe {
                    NtQueryDirectoryFile(
                        directory.as_raw_handle(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        &mut status_block,
                        buffer.as_mut_ptr().cast(),
                        (buffer.len() * std::mem::size_of::<u64>()) as u32,
                        12, // FileNamesInformation
                        0,
                        std::ptr::null_mut(),
                        u8::from(restart),
                    )
                };
                restart = false;
                if status as u32 == 0x8000_0006 {
                    break; // STATUS_NO_MORE_FILES
                }
                if status < 0 {
                    return Err(format!("read corpus directory: {}", nt_error(status)));
                }
                let used = status_block.information;
                if used == 0 || used > buffer.len() * std::mem::size_of::<u64>() {
                    return Err("invalid corpus directory enumeration length".into());
                }
                let bytes =
                    unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), used) };
                parse_directory_names(bytes, &mut names, expected)?;
            }
            Ok(names)
        }

        pub(super) fn metadata(&self, name: &str) -> io::Result<EntryMetadata> {
            let file = open_relative(
                self._parents.last().unwrap(),
                std::ffi::OsStr::new(name),
                FILE_READ_ATTRIBUTES,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                FILE_OPEN,
                0,
            )?;
            file.metadata().map(entry_metadata)
        }

        pub(super) fn open_file(&self, name: &str) -> io::Result<File> {
            open_relative(
                self._parents.last().unwrap(),
                std::ffi::OsStr::new(name),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                FILE_OPEN,
                FILE_NON_DIRECTORY_FILE,
            )
        }

        pub(super) fn create_new(&self, name: &str) -> io::Result<File> {
            open_relative(
                self._parents.last().unwrap(),
                std::ffi::OsStr::new(name),
                GENERIC_WRITE | DELETE,
                FILE_SHARE_READ,
                FILE_CREATE,
                FILE_NON_DIRECTORY_FILE,
            )
        }

        pub(super) fn rename(&self, file: &File, _from: &str, to: &str) -> io::Result<()> {
            let name: Vec<u16> = std::ffi::OsStr::new(to).encode_wide().collect();
            let name_len = name
                .len()
                .checked_mul(std::mem::size_of::<u16>())
                .and_then(|len| u32::try_from(len).ok())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "filename too long"))?;
            let offset = std::mem::offset_of!(FileRenameInfo, file_name);
            let len = offset
                .checked_add(name_len as usize)
                .map(|len| len.max(std::mem::size_of::<FileRenameInfo>()))
                .and_then(|len| u32::try_from(len).ok())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "filename too long"))?;
            // u64 storage provides the alignment required by FILE_RENAME_INFO.
            let mut buffer = vec![0u64; (len as usize).div_ceil(std::mem::size_of::<u64>())];
            unsafe {
                let info = buffer.as_mut_ptr().cast::<FileRenameInfo>();
                (*info).replace_if_exists = 1;
                (*info).root_directory = std::ptr::null_mut();
                (*info).file_name_length = name_len;
                std::ptr::copy_nonoverlapping(
                    name.as_ptr(),
                    buffer.as_mut_ptr().cast::<u8>().add(offset).cast(),
                    name.len(),
                );
            }
            let mut status_block = IoStatusBlock {
                status_or_pointer: 0,
                information: 0,
            };
            // A bare name renames within the source's directory. The open
            // temporary file denies writes/deletes, so it cannot be moved away.
            let status = unsafe {
                NtSetInformationFile(
                    file.as_raw_handle(),
                    &mut status_block,
                    buffer.as_mut_ptr().cast(),
                    len,
                    10, // FileRenameInformation
                )
            };
            if status < 0 {
                Err(io::Error::from_raw_os_error(unsafe {
                    RtlNtStatusToDosError(status) as i32
                }))
            } else {
                Ok(())
            }
        }

        pub(super) fn remove(&self, name: &str) -> io::Result<()> {
            let file = open_relative(
                self._parents.last().unwrap(),
                std::ffi::OsStr::new(name),
                DELETE,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                FILE_OPEN,
                FILE_NON_DIRECTORY_FILE,
            )?;
            let delete: u8 = 1;
            set_file_information(&file, 4, std::ptr::from_ref(&delete).cast(), 1)
        }

        pub(super) fn remove_open(&self, file: &File, _name: &str) -> io::Result<()> {
            let delete: u8 = 1;
            set_file_information(file, 4, std::ptr::from_ref(&delete).cast(), 1)
        }
    }
}

#[test]
fn rex3_corpus_unknown_names_are_rejected_before_any_content_read() {
    let expected = ["manifest.toml", "fixture.txt"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    let mut reads = 0;
    let error = read_named_entries(
        vec![
            "manifest.toml".into(),
            "fixture.txt".into(),
            "unexpected".into(),
        ],
        &expected,
        |_, _| {
            reads += 1;
            Ok(Vec::new())
        },
    )
    .unwrap_err();
    assert!(error.contains("unexpected corpus file"));
    assert_eq!(reads, 0);

    let mut names = Vec::new();
    push_expected_name(&mut names, "manifest.toml".into(), &expected).unwrap();
    assert!(
        push_expected_name(&mut names, "unexpected".into(), &expected)
            .unwrap_err()
            .contains("unexpected corpus file")
    );
    assert!(
        push_expected_name(&mut names, "manifest.toml".into(), &expected)
            .unwrap_err()
            .contains("duplicate corpus directory entry")
    );
    assert_eq!(names, ["manifest.toml"]);
}

#[test]
fn rex3_corpus_size_limits_reject_metadata_and_bound_reads() {
    struct CountingReader {
        read: usize,
    }
    impl Read for CountingReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.read += buf.len();
            buf.fill(0);
            Ok(buf.len())
        }
    }
    for name in ["manifest.toml", "README.md", "fixture.txt"] {
        let limit = size_limit(name);
        let mut reader = CountingReader { read: 0 };
        assert!(read_bounded(&mut reader, limit as u64 + 1, limit, name)
            .unwrap_err()
            .contains("exceeds"));
        assert_eq!(
            reader.read, 0,
            "{name}: oversized metadata must precede reading"
        );
    }
    let mut reader = CountingReader { read: 0 };
    assert!(read_bounded(&mut reader, 0, 32, "fixture.txt")
        .unwrap_err()
        .contains("exceeds"));
    assert_eq!(reader.read, 33, "a growing file is read at most limit + 1");
}

#[test]
fn rex3_corpus_reparse_and_nonregular_entries_are_rejected() {
    for meta in [
        EntryMetadata {
            len: 1,
            regular: false,
            reparse: true,
        },
        EntryMetadata {
            len: 1,
            regular: true,
            reparse: true,
        },
        EntryMetadata {
            len: 1,
            regular: false,
            reparse: false,
        },
    ] {
        assert!(check_regular("fixture.txt", meta).is_err());
    }
}

#[test]
fn rex3_corpus_anchor_is_required_reserved_and_not_content() {
    let expected = BTreeSet::from(["fixture.txt".to_owned()]);
    assert!(without_anchor(vec!["fixture.txt".into()], &expected)
        .unwrap_err()
        .contains("missing required corpus output anchor"));
    let names =
        without_anchor(vec![OUTPUT_ANCHOR.into(), "fixture.txt".into()], &expected).unwrap();
    let mut reads = Vec::new();
    let files = read_named_entries(names, &expected, |name, _| {
        reads.push(name.to_owned());
        Ok(b"fixture".to_vec())
    })
    .unwrap();
    assert_eq!(reads, ["fixture.txt"]);
    assert!(!files.contains_key(OUTPUT_ANCHOR));
    assert!(checked_output_name(OUTPUT_ANCHOR)
        .unwrap_err()
        .contains("reserved"));
    assert!(checked_output_name(".REX3-CORPUS-ROOT").is_err());
    assert!(checked_output_name(".rex3-corpus-root:stream").is_err());
    assert!(without_anchor(
        vec![OUTPUT_ANCHOR.into()],
        &BTreeSet::from([OUTPUT_ANCHOR.to_owned()])
    )
    .is_err());
}

fn provision_test_anchor(directory: &Path) {
    let mut anchor = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(OUTPUT_ANCHOR))
        .expect("provision test anchor before opening the directory");
    anchor
        .write_all(b"independent filesystem-test anchor\n")
        .unwrap();
}

#[cfg(windows)]
fn open_test_junction(link: &Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    OpenOptions::new()
        .access_mode(0x100) // FILE_WRITE_ATTRIBUTES, not GENERIC_WRITE
        .share_mode(1 | 2 | 4)
        .custom_flags(0x0200_0000 | 0x0020_0000)
        .open(link)
}

#[cfg(windows)]
fn set_test_junction_on_handle(dir: &File, target: &Path) -> io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::IO::DeviceIoControl;

    let target = target.to_str().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "non-Unicode junction target")
    })?;
    let substitute = if let Some(path) = target.strip_prefix(r"\\?\UNC\") {
        format!(r"\??\UNC\{path}")
    } else if let Some(path) = target.strip_prefix(r"\\?\") {
        format!(r"\??\{path}")
    } else if let Some(path) = target.strip_prefix(r"\\") {
        format!(r"\??\UNC\{path}")
    } else {
        format!(r"\??\{target}")
    };
    let substitute: Vec<u16> = substitute.encode_utf16().collect();
    let print: Vec<u16> = target.encode_utf16().collect();
    let substitute_len = u16::try_from(substitute.len() * 2)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "junction target too long"))?;
    let print_len = u16::try_from(print.len() * 2)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "junction target too long"))?;
    let data_len = 8u16
        .checked_add(substitute_len)
        .and_then(|n| n.checked_add(2))
        .and_then(|n| n.checked_add(print_len))
        .and_then(|n| n.checked_add(2))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "junction target too long"))?;
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&0xa000_0003u32.to_le_bytes()); // IO_REPARSE_TAG_MOUNT_POINT
    buffer.extend_from_slice(&data_len.to_le_bytes());
    buffer.extend_from_slice(&0u16.to_le_bytes());
    buffer.extend_from_slice(&0u16.to_le_bytes());
    buffer.extend_from_slice(&substitute_len.to_le_bytes());
    buffer.extend_from_slice(&(substitute_len + 2).to_le_bytes());
    buffer.extend_from_slice(&print_len.to_le_bytes());
    for word in substitute.into_iter().chain([0]).chain(print).chain([0]) {
        buffer.extend_from_slice(&word.to_le_bytes());
    }
    let mut returned = 0u32;
    let status = unsafe {
        DeviceIoControl(
            dir.as_raw_handle().cast(),
            0x0009_00a4, // FSCTL_SET_REPARSE_POINT
            buffer.as_ptr().cast(),
            buffer.len() as u32,
            std::ptr::null_mut(),
            0,
            &mut returned,
            std::ptr::null_mut(),
        )
    };
    if status == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(windows)]
fn set_test_junction(link: &Path, target: &Path) -> io::Result<()> {
    let dir = open_test_junction(link)?;
    set_test_junction_on_handle(&dir, target)
}

#[cfg(windows)]
fn create_test_junction(link: &Path, target: &Path) -> io::Result<()> {
    std::fs::create_dir(link)?;
    let result = set_test_junction(link, target);
    if result.is_err() {
        let _ = std::fs::remove_dir(link);
    }
    result
}

#[cfg(windows)]
fn junction_unavailable(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(1 | 5 | 50 | 1314))
}

#[cfg(windows)]
fn change_test_list_deny(path: &Path, deny: bool) -> io::Result<()> {
    let system_root =
        std::env::var_os("SystemRoot").ok_or_else(|| io::Error::other("SystemRoot is not set"))?;
    let icacls = PathBuf::from(system_root)
        .join("System32")
        .join("icacls.exe");
    let mut command = std::process::Command::new(icacls);
    command.arg(path);
    if deny {
        command.args(["/deny", "*S-1-1-0:(RD)"]);
    } else {
        command.args(["/remove:d", "*S-1-1-0"]);
    }
    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "icacls failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(())
}

#[cfg(windows)]
struct TestListDeny(PathBuf);

#[cfg(windows)]
impl Drop for TestListDeny {
    fn drop(&mut self) {
        if let Err(error) = change_test_list_deny(&self.0, false) {
            eprintln!("could not restore filesystem-test ACL: {error}");
        }
    }
}

#[cfg(windows)]
#[test]
#[ignore = "internal child for the explicitly selected filesystem security test"]
fn rex3_corpus_junction_mutator_child() {
    let link = std::env::var_os("REX3_CORPUS_MUTATOR_LINK").expect("child link");
    let target = std::env::var_os("REX3_CORPUS_MUTATOR_TARGET").expect("child target");
    let dir = open_test_junction(Path::new(&link)).expect("attribute-only directory open");
    eprintln!("FILE_WRITE_ATTRIBUTES directory open succeeded");
    let result = set_test_junction_on_handle(&dir, Path::new(&target));
    match std::env::var("REX3_CORPUS_MUTATOR_EXPECT").as_deref() {
        Ok("blocked") => {
            let error = result.expect_err("the held directory accepted a junction");
            assert_eq!(error.raw_os_error(), Some(145), "{error}");
            eprintln!("junction mutation blocked while the anchor keeps the directory nonempty");
        }
        Ok("nonempty") => {
            let error = result.expect_err("the directory with its anchor accepted a junction");
            assert_eq!(error.raw_os_error(), Some(145), "{error}");
            eprintln!("junction mutation still blocked after guard release while anchor remains");
        }
        Ok("allowed") => {
            result
                .expect("junction mutation should succeed after guard release and anchor removal");
            eprintln!("junction mutation succeeded after guard release and anchor removal");
        }
        Ok("unanchored") => {
            result.expect("attribute-only mutation should succeed without an anchor");
            eprintln!("attribute-only mutation succeeded against the unanchored directory guard");
        }
        Ok("unavailable") => {
            let error = result.expect_err("the junction capability probe disagreed with the child");
            assert!(
                junction_unavailable(&error),
                "unexpected junction failure: {error}"
            );
            eprintln!("post-release junction mutation unavailable on this filesystem: {error}");
        }
        _ => panic!("missing child expectation"),
    }
}

#[cfg(windows)]
fn attempt_junction_from_child(link: &Path, target: &Path, expected: &str) {
    let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "rex3_corpus_junction_mutator_child",
            "--ignored",
            "--nocapture",
        ])
        .env("REX3_CORPUS_MUTATOR_LINK", link)
        .env("REX3_CORPUS_MUTATOR_TARGET", target)
        .env("REX3_CORPUS_MUTATOR_EXPECT", expected)
        .output()
        .expect("start junction-mutator process");
    assert!(
        output.status.success(),
        "junction-mutator process failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("running 1 test") && stdout.contains("1 passed"),
        "junction-mutator selection did not execute exactly one test: {stdout}"
    );
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
}

#[test]
#[ignore = "requires an explicitly selected writable build directory"]
fn rex3_corpus_filesystem_security() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let base = std::env::var_os("REX3_CORPUS_IO_DIR")
        .expect("set REX3_CORPUS_IO_DIR to an existing relative build directory");
    let base = Path::new(&base);
    let parent = SafeDirectory::open(root, base).expect("safe filesystem-test parent");
    let relative = base.join(format!("rex3-corpus-io-{}", std::process::id()));
    let scratch = root.join(&relative);
    std::fs::create_dir(&scratch).expect("create a new, uniquely named filesystem-test directory");
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).expect("remove filesystem-test directory");
        }
    }
    let _cleanup = Cleanup(scratch.clone());
    let inside = scratch.join("inside");
    let outside = scratch.join("outside");
    std::fs::create_dir(&inside).unwrap();
    std::fs::create_dir(&outside).unwrap();
    let guard = outside.join("guard.txt");
    std::fs::write(&guard, b"outside remains unchanged").unwrap();
    provision_test_anchor(&inside);
    provision_test_anchor(&outside);

    let missing = scratch.join("missing-anchor");
    let missing_relative = relative.join("missing-anchor");
    std::fs::create_dir(&missing).unwrap();
    let fixture = BTreeMap::from([("fixture.txt".to_owned(), b"fixture".to_vec())]);
    let fixture_names = BTreeSet::from(["fixture.txt".to_owned()]);
    assert!(write_files(root, &missing_relative, &fixture)
        .unwrap_err()
        .contains("required pre-existing corpus anchor"));
    assert!(read_files(root, &missing_relative, &fixture_names)
        .unwrap_err()
        .contains("required pre-existing corpus anchor"));
    let generated_without_anchor = std::cell::Cell::new(false);
    assert!(write_generated(root, &missing_relative, || {
        generated_without_anchor.set(true);
        fixture.clone()
    })
    .unwrap_err()
    .contains("required pre-existing corpus anchor"));
    assert!(!generated_without_anchor.get());
    assert!(!missing.join(OUTPUT_ANCHOR).exists());
    assert!(!missing.join("fixture.txt").exists());
    let reserved = BTreeMap::from([(OUTPUT_ANCHOR.to_owned(), b"changed".to_vec())]);
    assert!(write_files(root, &relative.join("inside"), &reserved)
        .unwrap_err()
        .contains("reserved"));
    assert_eq!(
        std::fs::read(inside.join(OUTPUT_ANCHOR)).unwrap(),
        b"independent filesystem-test anchor\n"
    );

    #[cfg(windows)]
    let junctions_available = {
        let probe = scratch.join("junction-capability");
        match create_test_junction(&probe, &outside) {
            Ok(()) => {
                std::fs::remove_dir(probe).unwrap();
                true
            }
            Err(error) if junction_unavailable(&error) => {
                eprintln!("Windows junction checks unavailable: {error}");
                false
            }
            Err(error) => panic!("probe junction capability: {error}"),
        }
    };

    #[cfg(windows)]
    if junctions_available {
        let original = scratch.join("checkout-original");
        let substituted = scratch.join("checkout-substituted");
        let alias = scratch.join("checkout-retargeted");
        for (root, contents) in [
            (&original, b"original checkout".as_slice()),
            (&substituted, b"substituted checkout".as_slice()),
        ] {
            let corpus = root.join("corpus");
            std::fs::create_dir_all(&corpus).unwrap();
            provision_test_anchor(&corpus);
            std::fs::write(corpus.join("fixture.txt"), contents).unwrap();
        }
        create_test_junction(&alias, &original).unwrap();
        let mut pinned =
            SafeDirectory::open_from_root_after_pin(&alias, Path::new("corpus"), || {
                std::fs::remove_dir(&alias)?;
                create_test_junction(&alias, &substituted)
            })
            .expect("traverse from the originally opened checkout");
        pinned._anchor = Some(pinned.open_anchor().unwrap());
        let mut contents = Vec::new();
        pinned
            .open_file("fixture.txt")
            .unwrap()
            .read_to_end(&mut contents)
            .unwrap();
        assert_eq!(contents, b"original checkout");
        assert_eq!(
            std::fs::read(alias.join("corpus").join("fixture.txt")).unwrap(),
            b"substituted checkout",
            "the checkout alias must actually have been retargeted"
        );
        pinned
            .atomic_write("generated.txt", b"anchored output")
            .unwrap();
        assert_eq!(
            std::fs::read(original.join("corpus").join("generated.txt")).unwrap(),
            b"anchored output"
        );
        assert!(!substituted.join("corpus").join("generated.txt").exists());
        drop(pinned);
        std::fs::remove_dir(&alias).unwrap();
        eprintln!("retargeted checkout alias did not redirect handle-relative I/O");
    }

    #[cfg(windows)]
    {
        let restricted = scratch.join("nonlistable-parent");
        let checkout = restricted.join("checkout");
        let corpus = checkout.join("corpus");
        std::fs::create_dir_all(&corpus).unwrap();
        provision_test_anchor(&corpus);
        std::fs::write(corpus.join("fixture.txt"), b"before").unwrap();
        let acl_guard = TestListDeny(restricted.clone());
        change_test_list_deny(&restricted, true).expect("deny listing the checkout parent");
        let error = std::fs::read_dir(&restricted).unwrap_err();
        assert_eq!(error.raw_os_error(), Some(5), "parent must not be listable");
        assert_eq!(
            read_files(&checkout, Path::new("corpus"), &fixture_names)
                .unwrap()
                .get("fixture.txt")
                .unwrap(),
            b"before"
        );
        write_files(&checkout, Path::new("corpus"), &fixture).unwrap();
        assert_eq!(
            std::fs::read(corpus.join("fixture.txt")).unwrap(),
            b"fixture"
        );
        drop(acl_guard);
        std::fs::read_dir(&restricted).expect("restore listing the checkout parent");
        eprintln!("corpus read and replacement succeeded under a non-listable parent");
    }

    let bad_anchor_dir = scratch.join("bad-anchor");
    let bad_anchor_relative = relative.join("bad-anchor");
    std::fs::create_dir(&bad_anchor_dir).unwrap();
    #[cfg(unix)]
    let bad_anchor_created = {
        std::os::unix::fs::symlink(&guard, bad_anchor_dir.join(OUTPUT_ANCHOR)).unwrap();
        true
    };
    #[cfg(windows)]
    let bad_anchor_created = if junctions_available {
        create_test_junction(&bad_anchor_dir.join(OUTPUT_ANCHOR), &outside).unwrap();
        true
    } else {
        false
    };
    if bad_anchor_created {
        assert!(read_files(root, &bad_anchor_relative, &fixture_names)
            .unwrap_err()
            .contains("required pre-existing corpus anchor"));
        assert!(write_files(root, &bad_anchor_relative, &fixture)
            .unwrap_err()
            .contains("required pre-existing corpus anchor"));
        assert_eq!(std::fs::read(&guard).unwrap(), b"outside remains unchanged");
    }

    // The link is above the checkout root, not in the protected relative path.
    let checkout_parent = root.parent().expect("checkout parent");
    let checkout_link = scratch.join("checkout-parent-link");
    #[cfg(unix)]
    let alias_root = {
        std::os::unix::fs::symlink(checkout_parent, &checkout_link)
            .expect("create checkout-ancestor symlink");
        Some(checkout_link.join(root.file_name().expect("checkout directory name")))
    };
    #[cfg(windows)]
    let alias_root = if junctions_available {
        create_test_junction(&checkout_link, checkout_parent)
            .expect("create checkout-ancestor junction");
        Some(checkout_link.join(root.file_name().expect("checkout directory name")))
    } else {
        None
    };
    let alias_output = scratch.join("alias-output");
    std::fs::create_dir(&alias_output).unwrap();
    provision_test_anchor(&alias_output);
    let alias_output_relative = relative.join("alias-output");
    let alias_files = BTreeMap::from([("fixture.txt".to_owned(), b"alias output".to_vec())]);
    let alias_expected = BTreeSet::from(["fixture.txt".to_owned()]);
    if let Some(alias_root) = &alias_root {
        super::corpus::assert_corpus_via_checkout_alias(alias_root);
        write_files(alias_root, &alias_output_relative, &alias_files).unwrap();
        write_files(alias_root, &alias_output_relative, &alias_files).unwrap();
        assert_eq!(
            read_files(alias_root, &alias_output_relative, &alias_expected).unwrap(),
            alias_files
        );

        let checkout_root_link = scratch.join("checkout-root-link");
        #[cfg(unix)]
        std::os::unix::fs::symlink(root, &checkout_root_link)
            .expect("create checkout-root symlink");
        #[cfg(windows)]
        create_test_junction(&checkout_root_link, root).expect("create checkout-root junction");
        super::corpus::assert_corpus_via_checkout_alias(&checkout_root_link);
        write_files(&checkout_root_link, &alias_output_relative, &alias_files).unwrap();
        assert_eq!(
            read_files(&checkout_root_link, &alias_output_relative, &alias_expected).unwrap(),
            alias_files
        );
        #[cfg(unix)]
        std::fs::remove_file(&checkout_root_link).unwrap();
        #[cfg(windows)]
        std::fs::remove_dir(&checkout_root_link).unwrap();
    }

    let generated_dir = scratch.join("generated-output");
    let generated_relative = relative.join("generated-output");
    std::fs::create_dir(&generated_dir).unwrap();
    provision_test_anchor(&generated_dir);
    let anchor_before = std::fs::read(generated_dir.join(OUTPUT_ANCHOR)).unwrap();
    let generator = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["rex3_corpus_generate", "--ignored", "--nocapture"])
        .env("REX3_CORPUS_OUT", &generated_relative)
        .output()
        .expect("start the generator with a pre-provisioned anchor");
    let generator_stdout = String::from_utf8_lossy(&generator.stdout);
    assert!(
        generator.status.success()
            && generator_stdout.contains("running 1 test")
            && generator_stdout.contains("1 passed"),
        "temporary generator did not execute successfully: stdout={generator_stdout} stderr={}",
        String::from_utf8_lossy(&generator.stderr)
    );
    assert_eq!(
        std::fs::read(generated_dir.join(OUTPUT_ANCHOR)).unwrap(),
        anchor_before,
        "the generator must not modify its pre-existing anchor"
    );
    super::corpus::assert_generated_output(root, &generated_relative);

    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;

        let held_path = scratch.join("held-output");
        let held_relative = relative.join("held-output");
        std::fs::create_dir(&held_path).unwrap();
        provision_test_anchor(&held_path);
        let held = SafeDirectory::open_anchored(root, &held_relative).expect("hold output anchor");
        held.atomic_write("created.txt", b"first").unwrap();
        held.atomic_write("created.txt", b"replacement").unwrap();
        assert_eq!(
            std::fs::read(held_path.join("created.txt")).unwrap(),
            b"replacement"
        );
        held.remove("created.txt").unwrap();
        let pending = held.create_new("pending.part").unwrap();
        let write_error = OpenOptions::new()
            .write(true)
            .open(held_path.join("pending.part"))
            .unwrap_err();
        assert_eq!(write_error.raw_os_error(), Some(32));
        let delete_error = OpenOptions::new()
            .access_mode(0x0001_0000)
            .open(held_path.join("pending.part"))
            .unwrap_err();
        assert_eq!(delete_error.raw_os_error(), Some(32));
        held.remove_open(&pending, "pending.part").unwrap();
        drop(pending);
        assert_eq!(std::fs::read_dir(&held_path).unwrap().count(), 1);
        let anchor_delete = std::fs::remove_file(held_path.join(OUTPUT_ANCHOR)).unwrap_err();
        assert_eq!(anchor_delete.raw_os_error(), Some(32));
        let anchor_rename = std::fs::rename(
            held_path.join(OUTPUT_ANCHOR),
            held_path.join("moved-anchor"),
        )
        .unwrap_err();
        assert_eq!(anchor_rename.raw_os_error(), Some(32));

        // The separate process can open with FILE_WRITE_ATTRIBUTES despite sharing flags.
        if junctions_available {
            attempt_junction_from_child(&held_path, &outside, "blocked");
        } else {
            eprintln!("attribute-only junction mutation checks unavailable");
        }
        drop(held);
        write_generated(root, &held_relative, || {
            let error = std::fs::remove_file(held_path.join(OUTPUT_ANCHOR)).unwrap_err();
            assert_eq!(error.raw_os_error(), Some(32));
            if junctions_available {
                attempt_junction_from_child(&held_path, &outside, "blocked");
            }
            BTreeMap::new()
        })
        .unwrap();
        if junctions_available {
            attempt_junction_from_child(&held_path, &outside, "nonempty");
        }
        std::fs::remove_file(held_path.join(OUTPUT_ANCHOR)).unwrap();
        attempt_junction_from_child(
            &held_path,
            &outside,
            if junctions_available {
                "allowed"
            } else {
                "unavailable"
            },
        );
        std::fs::remove_dir(&held_path).unwrap();

        if junctions_available {
            let race_path = scratch.join("unanchored-race");
            let race_relative = relative.join("unanchored-race");
            std::fs::create_dir(&race_path).unwrap();
            let unanchored = SafeDirectory::open(root, &race_relative).unwrap();
            attempt_junction_from_child(&race_path, &outside, "unanchored");
            assert!(
                unanchored.open_file(OUTPUT_ANCHOR).is_err(),
                "handle-relative open must not follow the newly installed junction"
            );
            assert!(
                unanchored.metadata("guard.txt").is_err(),
                "handle-relative metadata must not reach the junction target"
            );
            assert!(
                unanchored.open_anchor().is_err(),
                "an anchor outside the changed directory must not be trusted"
            );
            drop(unanchored);
            std::fs::remove_dir(&race_path).unwrap();
        }
    }

    #[cfg(unix)]
    {
        let held_path = scratch.join("unix-held");
        let moved_path = scratch.join("unix-moved");
        std::fs::create_dir(&held_path).unwrap();
        provision_test_anchor(&held_path);
        let held = SafeDirectory::open_anchored(root, &relative.join("unix-held")).unwrap();
        std::fs::rename(&held_path, &moved_path).unwrap();
        std::os::unix::fs::symlink(&outside, &held_path).unwrap();
        // Unix descriptors do not prevent unlink, but operations remain relative to the held directory.
        std::fs::remove_file(moved_path.join(OUTPUT_ANCHOR)).unwrap();
        held.atomic_write("anchored.txt", b"inside held directory")
            .unwrap();
        assert_eq!(
            std::fs::read(moved_path.join("anchored.txt")).unwrap(),
            b"inside held directory"
        );
        assert!(!outside.join("anchored.txt").exists());
        std::fs::remove_file(&held_path).unwrap();
    }

    let file_link = inside.join("fixture.txt");
    #[cfg(unix)]
    let file_is_symlink = {
        std::os::unix::fs::symlink(&guard, &file_link).expect("create file symlink");
        Some(true)
    };
    #[cfg(windows)]
    let file_is_symlink = match std::os::windows::fs::symlink_file(&guard, &file_link) {
        Ok(()) => Some(true),
        Err(error) if junction_unavailable(&error) => {
            if junctions_available {
                create_test_junction(&file_link, &outside).expect("create unprivileged junction");
                eprintln!(
                    "file-symlink privilege unavailable; testing a directory junction instead"
                );
                Some(false)
            } else {
                eprintln!("file-symlink and junction checks unavailable: {error}");
                None
            }
        }
        Err(error) => panic!("create file symlink: {error}"),
    };
    let inside_relative = relative.join("inside");
    let expected = BTreeSet::from(["fixture.txt".to_owned()]);
    let generated = BTreeMap::from([("fixture.txt".to_owned(), b"new bytes".to_vec())]);
    std::fs::write(inside.join("unexpected"), b"not a corpus file").unwrap();
    assert!(read_files(root, &inside_relative, &expected)
        .unwrap_err()
        .contains("unexpected corpus file"));
    std::fs::remove_file(inside.join("unexpected")).unwrap();
    if let Some(file_is_symlink) = file_is_symlink {
        assert!(read_files(root, &inside_relative, &expected)
            .unwrap_err()
            .contains("symlink or reparse"));
        assert!(write_files(root, &inside_relative, &generated)
            .unwrap_err()
            .contains("symlink or reparse"));
        assert_eq!(std::fs::read(&guard).unwrap(), b"outside remains unchanged");
        let replaced = {
            // Simulate a destination becoming a link after the preflight check.
            let directory = SafeDirectory::open_anchored(root, &inside_relative).unwrap();
            let result = directory.atomic_write("fixture.txt", b"replacement");
            if file_is_symlink {
                result.unwrap();
                true
            } else {
                result.is_ok()
            }
        };
        assert_eq!(std::fs::read_dir(&inside).unwrap().count(), 2);
        assert_eq!(std::fs::read(&guard).unwrap(), b"outside remains unchanged");
        if replaced {
            assert_eq!(std::fs::read(&file_link).unwrap(), b"replacement");
            std::fs::remove_file(&file_link).unwrap();
        } else {
            std::fs::remove_dir(&file_link).unwrap();
        }
    }
    std::fs::hard_link(&guard, &file_link).unwrap();
    write_files(root, &inside_relative, &generated).unwrap();
    assert_eq!(std::fs::read(&guard).unwrap(), b"outside remains unchanged");
    assert_eq!(std::fs::read(&file_link).unwrap(), b"new bytes");
    std::fs::remove_file(&file_link).unwrap();

    let dir_link = scratch.join("directory-link");
    #[cfg(unix)]
    let dir_link_created = {
        std::os::unix::fs::symlink(&outside, &dir_link).expect("create directory symlink");
        true
    };
    #[cfg(windows)]
    let dir_link_created = if junctions_available {
        create_test_junction(&dir_link, &outside).expect("create directory junction");
        true
    } else {
        false
    };
    if dir_link_created {
        assert!(SafeDirectory::open(root, &relative.join("directory-link")).is_err());
        if let Some(alias_root) = &alias_root {
            assert!(read_files(alias_root, &relative.join("directory-link"), &expected).is_err());
            assert!(write_files(alias_root, &relative.join("directory-link"), &generated).is_err());
        }
        assert!(write_files(root, &relative.join("directory-link"), &generated).is_err());
        assert_eq!(std::fs::read(&guard).unwrap(), b"outside remains unchanged");
        #[cfg(unix)]
        std::fs::remove_file(&dir_link).unwrap();
        #[cfg(windows)]
        std::fs::remove_dir(&dir_link).unwrap();
    }

    for name in ["manifest.toml", "fixture.txt"] {
        let path = inside.join(name);
        File::create(&path)
            .unwrap()
            .set_len(size_limit(name) as u64 + 1)
            .unwrap();
        let expected = BTreeSet::from([name.to_owned()]);
        assert!(read_files(root, &inside_relative, &expected)
            .unwrap_err()
            .contains("exceeds"));
        std::fs::remove_file(path).unwrap();
    }
    assert_eq!(std::fs::read_dir(&inside).unwrap().count(), 1);
    if alias_root.is_some() {
        #[cfg(unix)]
        std::fs::remove_file(&checkout_link).unwrap();
        #[cfg(windows)]
        std::fs::remove_dir(&checkout_link).unwrap();
    }
    drop(parent);
}
