//! The disk browser (D23): one disk or CD-ROM of a machine, its partitions,
//! a folder listing, and import, export and edits, each a job with progress
//! and cancel.
//!
//! The controller decides what is read-only and says why: CD-ROMs always; a
//! running machine (read through the machine as the guest sees it,
//! `model::handle::LiveDisk`); XFS the crate cannot edit; a filesystem whose
//! check failed, until the user chooses to go on.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use disk_core::entry::{EntryType, FileEntry};
use disk_core::filesystem::{CreateDirectoryOptions, CreateFileOptions};

use super::icons::Icon;
use super::{Answer, ChoiceSpec, ConfirmSpec, Effect, Enabled, PickSpec, Severity, TextSpec};
use crate::model::disk_browser::{self as model, BlockIo, Names, OpenFs, PartKind, PartitionInfo};
use crate::model::job::{Job, Notify, Progress, Unit};

/// Which device of which machine is being browsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrowseTarget {
    pub machine: String,
    /// "Disk 1" or "CD-ROM 4".
    pub label: String,
    /// The image path, absolute.
    pub path: String,
    pub cdrom: bool,
    /// The disk's copy-on-write setting: edits go to its overlay.
    pub overlay: bool,
}

/// A row of the file list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub name: String,
    pub icon: Icon,
    pub is_dir: bool,
    pub size: String,
    pub mode: String,
    pub owner: String,
    pub group: String,
    pub date: String,
}

/// A row of the partition list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartRow {
    pub index: usize,
    /// "Partition 0".
    pub name: String,
    /// "EFS, 1.9 GB" or "swap: not browsable".
    pub detail: String,
    pub browsable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrowserAction {
    SelectPartition(usize),
    /// Open the row (a folder: go in; a file: edit it when it is text).
    Activate(usize),
    Up,
    /// Go to a depth in the path bar (0 is the partition's root).
    GoTo(usize),
    Refresh,
    /// The rows selected, by index.
    Select(Vec<usize>),
    Export,
    /// `gzip` false for a plain `.tar`.
    ExportArchive { gzip: bool },
    /// Ask for host files to import.
    Import,
    /// Ask for a host folder to import.
    ImportFolder,
    /// Host paths dropped on the list; `into` is a folder row they were dropped on.
    Drop { paths: Vec<PathBuf>, into: Option<usize> },
    /// Copy a `.tardist` or tar in as one file instead of unpacking it.
    SetUnpackArchives(bool),
    /// Convert imported text files' line endings to Unix (LF), for IRIX.
    SetUnixLineEndings(bool),
    EditText,
    /// The editor's text changed.
    EditorChanged(String),
    EditorSave,
    EditorClose,
    Permissions,
    Owner,
    Rename,
    Move,
    NewFolder,
    Delete,
    Check,
    /// Edit a filesystem whose check failed anyway.
    EditAnyway,
    /// Replay an XFS log IRIX left unreplayed, as a mount would.
    ReplayLog,
    Cancel,
    Answer(Answer),
}

/// The open text file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Editor {
    pub name: String,
    pub text: String,
    pub dirty: bool,
    /// "LF" or "CRLF".
    pub line_endings: &'static str,
    pub read_only: bool,
}

/// Widget ids of the browser window. Every shell names its controls with
/// these and maps them back with [`action_for`].
pub mod ids {
    pub const WINDOW: &str = "browser.window";
    pub const TABLE: &str = "browser.table";
    pub const UP: &str = "browser.up";
    pub const REFRESH: &str = "browser.refresh";
    pub const IMPORT: &str = "browser.import-files";
    pub const IMPORT_FOLDER: &str = "browser.import-folder-button";
    pub const EXPORT: &str = "browser.export-button";
    pub const EXPORT_TAR: &str = "browser.export-tar-button";
    pub const EXPORT_TGZ: &str = "browser.export-tgz-button";
    pub const NEW_FOLDER: &str = "browser.new-folder-button";
    pub const RENAME: &str = "browser.rename-button";
    pub const PERMISSIONS: &str = "browser.permissions";
    pub const OWNER: &str = "browser.owner";
    pub const MOVE: &str = "browser.move";
    pub const DELETE: &str = "browser.delete-button";
    pub const EDIT: &str = "browser.edit";
    pub const CHECK: &str = "browser.check";
    pub const EDIT_ANYWAY: &str = "browser.edit-anyway-button";
    pub const REPLAY_LOG: &str = "browser.replay-log-button";
    pub const CANCEL: &str = "browser.cancel";
    pub const UNPACK: &str = "browser.unpack";
    pub const UNIX_LINE_ENDINGS: &str = "browser.unix-line-endings";
    pub const EDITOR_SAVE: &str = "browser.editor-save";
    pub const EDITOR_CLOSE: &str = "browser.editor-close";
    pub const EDITOR_TEXT: &str = "browser.editor-text";
    /// Followed by the partition's position in [`super::DiskBrowserController::partitions`].
    pub const PART: &str = "browser.part:";
    /// Followed by the depth in the path bar.
    pub const PATH: &str = "browser.path:";
}

/// A button of the browser's toolbar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tool {
    pub id: &'static str,
    pub label: String,
    pub icon: Icon,
    pub action: BrowserAction,
}

/// The browser's toolbar, in order; `None` is a gap between groups.
pub fn tools() -> Vec<Option<Tool>> {
    let t = |id, label: String, icon, action| Some(Tool { id, label, icon, action });
    vec![
        t(ids::UP, t!("browser-tb-up"), Icon::Up, BrowserAction::Up),
        t(ids::REFRESH, t!("browser-tb-refresh"), Icon::Refresh, BrowserAction::Refresh),
        None,
        t(ids::IMPORT, t!("browser-tb-import"), Icon::Import, BrowserAction::Import),
        t(ids::IMPORT_FOLDER, t!("browser-tb-import-folder"), Icon::Import, BrowserAction::ImportFolder),
        t(ids::EXPORT, t!("browser-tb-export"), Icon::Export, BrowserAction::Export),
        t(ids::EXPORT_TGZ, t!("browser-tb-export-tgz"), Icon::Export, BrowserAction::ExportArchive { gzip: true }),
        t(ids::EXPORT_TAR, t!("browser-tb-export-tar"), Icon::Export, BrowserAction::ExportArchive { gzip: false }),
        None,
        t(ids::NEW_FOLDER, t!("browser-tb-new-folder"), Icon::NewFolder, BrowserAction::NewFolder),
        t(ids::EDIT, t!("browser-tb-edit"), Icon::Edit, BrowserAction::EditText),
        t(ids::RENAME, t!("browser-tb-rename"), Icon::Rename, BrowserAction::Rename),
        t(ids::MOVE, t!("browser-tb-move"), Icon::Folder, BrowserAction::Move),
        t(ids::PERMISSIONS, t!("browser-tb-permissions"), Icon::Permissions, BrowserAction::Permissions),
        t(ids::OWNER, t!("browser-tb-owner"), Icon::Permissions, BrowserAction::Owner),
        t(ids::DELETE, t!("browser-tb-delete"), Icon::Delete, BrowserAction::Delete),
        None,
        t(ids::CHECK, t!("browser-tb-check"), Icon::Check, BrowserAction::Check),
    ]
}

/// The file table's columns after the icon: their ids (the [`Row`] field
/// each shows) and titles.
pub fn columns() -> Vec<(&'static str, String)> {
    vec![
        ("name", t!("browser-col-name")),
        ("size", t!("browser-col-size")),
        ("mode", t!("browser-col-mode")),
        ("owner", t!("browser-col-owner")),
        ("group", t!("browser-col-group")),
        ("date", t!("browser-col-date")),
    ]
}

/// The action a browser control with `id` means. `value` is a toggle's state
/// (the unpack checkbox); other controls ignore it.
pub fn action_for(id: &str, value: bool) -> Option<BrowserAction> {
    if let Some(i) = id.strip_prefix(ids::PART).and_then(|i| i.parse().ok()) {
        return Some(BrowserAction::SelectPartition(i));
    }
    if let Some(d) = id.strip_prefix(ids::PATH).and_then(|d| d.parse().ok()) {
        return Some(BrowserAction::GoTo(d));
    }
    if let Some(t) = tools().into_iter().flatten().find(|t| t.id == id) {
        return Some(t.action);
    }
    Some(match id {
        ids::EDIT_ANYWAY => BrowserAction::EditAnyway,
        ids::REPLAY_LOG => BrowserAction::ReplayLog,
        ids::CANCEL => BrowserAction::Cancel,
        ids::UNPACK => BrowserAction::SetUnpackArchives(value),
        ids::UNIX_LINE_ENDINGS => BrowserAction::SetUnixLineEndings(value),
        ids::EDITOR_SAVE => BrowserAction::EditorSave,
        ids::EDITOR_CLOSE => BrowserAction::EditorClose,
        _ => return None,
    })
}

const P_EXPORT: &str = "browser.export";
const P_EXPORT_TAR: &str = "browser.export-tar";
const P_IMPORT: &str = "browser.import";
const P_IMPORT_FOLDER: &str = "browser.import-folder";
const P_IMPORT_GO: &str = "browser.import-go";
const P_UNPACK: &str = "browser.unpack-archives";
const P_CHMOD: &str = "browser.chmod";
const P_CHOWN: &str = "browser.chown";
const P_RENAME: &str = "browser.rename";
const P_NEW_FOLDER: &str = "browser.new-folder";
const P_DELETE: &str = "browser.delete";
const P_EDIT_ANYWAY: &str = "browser.edit-anyway";
const P_REPLAY: &str = "browser.replay-log";

type Shared = Arc<Mutex<Option<OpenFs>>>;

/// File-type bits a new inode's mode must carry: the crates store the mode
/// given as-is, so bare permission bits would make a "special" file.
/// clanker/gui/disk-core-create-modes-carry-the-file-type.md
const S_IFREG: u32 = 0o100000;
const S_IFDIR: u32 = 0o040000;

/// What a finished job hands back.
enum Done {
    Opened { names: Names, check: Option<String>, log: Option<model::XfsLog> },
    /// The XFS log was replayed: transactions applied.
    Replayed(usize),
    Listed(Vec<FileEntry>),
    Exported(u64),
    Imported(u64),
    Text(String, Vec<u8>),
    Edited,
    /// The editor's file was saved: its entry now.
    Saved(FileEntry),
    Checked(String, bool),
}

pub struct DiskBrowserController {
    target: BrowseTarget,
    /// The running machine's disk, read through the machine; `None` reads
    /// the image file (the machine is stopped).
    live: Option<crate::model::handle::LiveDisk>,
    /// Why nothing here may write, when nothing may.
    read_only: Option<String>,
    header: Option<part_tables::sgi::SgiVolumeHeader>,
    parts: Vec<PartitionInfo>,
    error: Option<String>,
    selected_part: Option<usize>,
    fs: Shared,
    path: Vec<FileEntry>,
    entries: Vec<FileEntry>,
    selection: Vec<usize>,
    names: Names,
    /// The open filesystem's check failed and the user has not said go on.
    check_failed: Option<String>,
    /// The open XFS volume's log, when it is one.
    xfs_log: Option<model::XfsLog>,
    unpack_archives: bool,
    /// Imported text files get Unix line endings.
    unix_line_endings: bool,
    /// The open text file: the editor, the file's entry as it is now
    /// (refreshed after each save), and the folder it lives in (not the one
    /// shown, which can change while the editor is open).
    editor: Option<(Editor, FileEntry, FileEntry)>,
    job: Option<(Job<Done>, bool)>,
    pending_import: Vec<PathBuf>,
    import_into: Option<FileEntry>,
    /// The archives in the pending import were asked about (unpack or copy).
    unpack_asked: bool,
    /// Partitions tried while looking for one that opens.
    tried: Vec<usize>,
    notify: Arc<dyn Notify>,
}

pub(crate) fn human(n: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", U[i])
    }
}

fn name_rule(spec: &TextSpec, s: &str) -> Result<(), String> {
    let t = s.trim();
    if t.is_empty() || t == spec.initial {
        return Err(String::new());
    }
    if t.contains('/') || t == "." || t == ".." {
        return Err(t!("browser-bad-name"));
    }
    if spec.taken.iter().any(|n| n == t) {
        return Err(t!("browser-name-taken"));
    }
    Ok(())
}

fn mode_rule(_: &TextSpec, s: &str) -> Result<(), String> {
    u32::from_str_radix(s.trim(), 8).ok().filter(|m| *m <= 0o7777).map(|_| ()).ok_or_else(|| t!("browser-bad-mode"))
}

fn owner_rule(spec: &TextSpec, s: &str) -> Result<(), String> {
    let (u, g) = s.trim().split_once(':').unwrap_or((s.trim(), ""));
    // `taken` carries the disk's known names as "u:NAME" and "g:NAME".
    let known = |kind: &str, v: &str| v.parse::<u32>().is_ok() || spec.taken.iter().any(|n| n == &format!("{kind}:{v}"));
    if u.is_empty() || !known("u", u) || (!g.is_empty() && !known("g", g)) {
        return Err(t!("browser-bad-owner"));
    }
    Ok(())
}

impl DiskBrowserController {
    /// Open the browser on `target`. With `live` (the machine is running) the
    /// disk is read through the machine, read-only.
    pub fn open(target: BrowseTarget, live: Option<crate::model::handle::LiveDisk>, notify: Arc<dyn Notify>) -> Self {
        let read_only = if target.cdrom {
            Some(t!("browser-ro-cdrom"))
        } else if live.is_some() {
            Some(t!("browser-ro-running"))
        } else {
            None
        };
        let mut c = Self {
            target,
            live,
            read_only,
            header: None,
            parts: Vec::new(),
            error: None,
            selected_part: None,
            fs: Arc::default(),
            path: Vec::new(),
            entries: Vec::new(),
            selection: Vec::new(),
            names: Names::default(),
            check_failed: None,
            xfs_log: None,
            unpack_archives: true,
            unix_line_endings: false,
            editor: None,
            job: None,
            pending_import: Vec::new(),
            import_into: None,
            unpack_asked: false,
            tried: Vec::new(),
            notify,
        };
        match c.image() {
            Ok(mut io) => match model::partitions(&mut io) {
                Ok((h, parts)) => {
                    c.header = Some(h);
                    c.parts = parts;
                    // Start on the first partition with a filesystem.
                    if let Some(i) = c.parts.iter().position(|p| matches!(p.kind, PartKind::Efs | PartKind::Xfs | PartKind::Ext)) {
                        c.tried.push(i);
                        c.select_partition(i);
                    }
                }
                Err(e) => c.error = Some(t!("browser-no-label", error = e)),
            },
            Err(e) => c.error = Some(t!("browser-open-failed", error = e.to_string())),
        }
        c
    }

    fn image(&self) -> std::io::Result<BlockIo> {
        match &self.live {
            Some(l) => Ok(BlockIo::with_sector(Box::new(l.clone()), if self.target.cdrom { 2048 } else { 512 })),
            None => model::open_image(&self.target.path, self.target.cdrom, self.target.overlay),
        }
    }

    /// Reading a running machine's disk through the machine.
    pub fn live(&self) -> bool {
        self.live.is_some()
    }

    // ---- read accessors -----------------------------------------------------

    pub fn target(&self) -> &BrowseTarget {
        &self.target
    }

    pub fn title(&self) -> String {
        t!("browser-title", machine = self.target.machine.as_str(), device = self.target.label.as_str())
    }

    /// A problem with the whole device (no label, cannot open).
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn partitions(&self) -> Vec<PartRow> {
        self.parts
            .iter()
            .map(|p| PartRow {
                index: p.index,
                name: t!("browser-partition", index = p.index),
                detail: match &p.why_not {
                    Some(why) => format!("{}: {why}", p.type_name),
                    None => format!("{}, {}", p.kind.name(), human(p.len)),
                },
                browsable: p.why_not.is_none(),
            })
            .collect()
    }

    pub fn selected_partition(&self) -> Option<usize> {
        self.selected_part
    }

    /// The path bar: the partition, then each folder.
    pub fn path(&self) -> Vec<String> {
        let mut v = vec![self.selected_part.map(|i| t!("browser-partition", index = self.parts[i].index)).unwrap_or_default()];
        v.extend(self.path.iter().skip(1).map(|e| e.name.clone()));
        v
    }

    pub fn rows(&self) -> Vec<Row> {
        self.entries
            .iter()
            .map(|e| Row {
                name: e.name.chars().filter(|c| !c.is_control()).collect(),
                icon: if e.is_directory() { Icon::Folder } else { Icon::Edit },
                is_dir: e.is_directory(),
                size: if e.is_directory() { String::new() } else { human(e.size) },
                mode: model::mode_string(e),
                owner: e.uid.map(|u| self.names.user(u)).unwrap_or_default(),
                group: e.gid.map(|g| self.names.group(g)).unwrap_or_default(),
                date: e.modified.clone().unwrap_or_default(),
            })
            .collect()
    }

    pub fn selection(&self) -> &[usize] {
        &self.selection
    }

    /// Why edits are refused, if they are: the window says it.
    pub fn read_only_reason(&self) -> Option<String> {
        if let Some(r) = &self.read_only {
            return Some(r.clone());
        }
        if let Some(r) = &self.check_failed {
            return Some(r.clone());
        }
        self.fs.lock().ok().and_then(|g| g.as_ref().and_then(|f| f.read_only_reason().map(str::to_string)))
    }

    pub fn editor(&self) -> Option<&Editor> {
        self.editor.as_ref().map(|(e, _, _)| e)
    }

    pub fn job(&self) -> Option<super::JobSnapshot> {
        self.job.as_ref().map(|(j, _)| j.snapshot())
    }

    /// A job that changes the disk is running: power on must wait.
    pub fn writing(&self) -> bool {
        self.job.as_ref().is_some_and(|(j, w)| *w && j.is_running())
    }

    pub fn unpack_archives(&self) -> bool {
        self.unpack_archives
    }

    pub fn unix_line_endings(&self) -> bool {
        self.unix_line_endings
    }

    /// Whether `action` can run now, and if not why.
    pub fn enabled(&self, action: &BrowserAction) -> Enabled {
        let busy = self.job.as_ref().is_some_and(|(j, _)| j.is_running());
        let open = self.fs.lock().map(|g| g.is_some()).unwrap_or(false);
        let writable = || match self.read_only_reason() {
            Some(r) => Enabled::No(r),
            None => Enabled::Yes,
        };
        let one = self.selection.len() == 1;
        let some = !self.selection.is_empty();
        if busy && !matches!(action, BrowserAction::Cancel | BrowserAction::EditorChanged(_) | BrowserAction::EditorClose) {
            return Enabled::No(t!("browser-busy"));
        }
        if !open && !matches!(action, BrowserAction::SelectPartition(_) | BrowserAction::Cancel | BrowserAction::Answer(_)) {
            return Enabled::No(t!("browser-no-partition"));
        }
        match action {
            BrowserAction::Export | BrowserAction::ExportArchive { .. } => Enabled::Yes,
            BrowserAction::Import | BrowserAction::ImportFolder | BrowserAction::Drop { .. } | BrowserAction::NewFolder => {
                writable()
            }
            BrowserAction::Delete => {
                if !some {
                    return Enabled::No(t!("browser-select-first"));
                }
                writable()
            }
            BrowserAction::Permissions | BrowserAction::Owner => {
                if !some {
                    return Enabled::No(t!("browser-select-first"));
                }
                writable()
            }
            BrowserAction::Rename => {
                if !one {
                    return Enabled::No(t!("browser-select-one"));
                }
                writable()
            }
            BrowserAction::EditText => {
                let Some(e) = (one).then(|| &self.entries[self.selection[0]]) else {
                    return Enabled::No(t!("browser-select-one"));
                };
                if !e.is_file() {
                    return Enabled::No(t!("browser-not-a-file"));
                }
                if e.size > model::TEXT_EDIT_LIMIT {
                    return Enabled::No(t!("browser-too-big", limit = human(model::TEXT_EDIT_LIMIT)));
                }
                Enabled::Yes
            }
            // Waits for a rusty-backup release with a move; not faked with copy and delete.
            BrowserAction::Move => Enabled::No(t!("browser-move-not-yet")),
            BrowserAction::EditorSave => match self.editor.as_ref() {
                Some((e, _, _)) if e.read_only => writable(),
                Some(_) => Enabled::Yes,
                None => Enabled::No(String::new()),
            },
            BrowserAction::EditAnyway => Enabled::when(self.check_failed.is_some(), String::new),
            // A CD-ROM or a running machine: the controller's own reason.
            BrowserAction::ReplayLog => match (&self.read_only, &self.xfs_log) {
                (Some(r), _) => Enabled::No(r.clone()),
                (None, Some(model::XfsLog::Replayable { .. })) => Enabled::Yes,
                (None, _) => Enabled::No(String::new()),
            },
            BrowserAction::Cancel => Enabled::when(busy, String::new),
            BrowserAction::SelectPartition(_)
            | BrowserAction::Activate(_)
            | BrowserAction::Up
            | BrowserAction::GoTo(_)
            | BrowserAction::Refresh
            | BrowserAction::Select(_)
            | BrowserAction::SetUnpackArchives(_)
            | BrowserAction::SetUnixLineEndings(_)
            | BrowserAction::EditorChanged(_)
            | BrowserAction::EditorClose
            | BrowserAction::Check
            | BrowserAction::Answer(_) => Enabled::Yes,
        }
    }

    // ---- jobs ---------------------------------------------------------------

    fn run<F>(&mut self, label: String, writes: bool, body: F)
    where
        F: FnOnce(&mut OpenFs, &crate::model::job::JobCtx<Done>) -> Result<Done, String> + Send + 'static,
    {
        let fs = self.fs.clone();
        let job = Job::spawn(label, true, self.notify.clone(), move |ctx| {
            let mut g = fs.lock().map_err(|_| "the disk is unavailable".to_string())?;
            let fs = g.as_mut().ok_or_else(|| "no partition is open".to_string())?;
            let r = body(fs, ctx);
            if writes {
                if let Some(e) = fs.editable() {
                    e.sync_metadata().map_err(|e| e.to_string())?;
                }
            }
            r
        });
        self.job = Some((job, writes));
    }

    /// Replay the selected XFS volume's log, then reopen it.
    fn replay_log(&mut self) {
        let Some(p) = self.selected_part.and_then(|i| self.parts.get(i).cloned()) else { return };
        self.close_fs();
        let Ok(mut io) = self.image() else {
            self.error = Some(t!("browser-open-failed", error = "image"));
            return;
        };
        let job = Job::spawn(t!("browser-replaying"), false, self.notify.clone(), move |_| {
            model::replay_xfs_log(&mut io, p.offset).map(Done::Replayed)
        });
        self.job = Some((job, true));
    }

    fn select_partition(&mut self, i: usize) {
        let Some(p) = self.parts.get(i).cloned() else { return };
        if p.why_not.is_some() {
            return;
        }
        self.close_fs();
        self.selected_part = Some(i);
        self.path.clear();
        self.entries.clear();
        self.selection.clear();
        self.check_failed = None;
        self.xfs_log = None;
        let (Some(header), Ok(mut io)) = (self.header.clone(), self.image()) else {
            self.error = Some(t!("browser-open-failed", error = "image"));
            return;
        };
        let read_only = self.read_only.clone();
        let fs = self.fs.clone();
        // Opening reads the superblock and, for an editable volume, checks it:
        // a job, like everything that touches the image.
        let job = Job::spawn(t!("browser-opening"), true, self.notify.clone(), move |ctx| {
            // XFS: edits over a log IRIX has yet to replay would be undone by
            // that replay, so a volume not cleanly unmounted opens read-only.
            let log = (p.kind == PartKind::Xfs).then(|| model::xfs_log(&mut io, p.offset)).transpose()?;
            let read_only = read_only.or_else(|| match &log {
                Some(model::XfsLog::Replayable { transactions }) => Some(t!("browser-xfs-dirty", count = *transactions)),
                Some(model::XfsLog::Unrecoverable(why)) => Some(t!("browser-xfs-unrecoverable", reason = why.as_str())),
                _ => None,
            });
            let mut open = model::open_partition(io, &header, &p, read_only)?;
            ctx.phase(t!("browser-reading-names"));
            let names = Names::read(open.fs());
            let mut check = None;
            if open.editable().is_some() && p.kind != PartKind::VolumeHeader {
                ctx.phase(t!("browser-checking"));
                if let Some(Ok(r)) = open.fs().fsck() {
                    if !r.errors.is_empty() {
                        check = Some(t!("browser-check-failed-ro", count = r.errors.len()));
                    }
                }
            }
            *fs.lock().map_err(|_| "lock")? = Some(open);
            Ok(Done::Opened { names, check, log })
        });
        self.job = Some((job, false));
    }

    fn list(&mut self) {
        let Some(dir) = self.path.last().cloned() else { return };
        self.run(t!("browser-listing"), false, move |fs, _| {
            let mut v = fs.fs().list_directory(&dir).map_err(|e| e.to_string())?;
            v.retain(|e| e.name != "." && e.name != "..");
            v.sort_by(|a, b| b.is_directory().cmp(&a.is_directory()).then(a.name.cmp(&b.name)));
            Ok(Done::Listed(v))
        });
    }

    fn close_fs(&mut self) {
        if let Ok(mut g) = self.fs.lock() {
            if let Some(f) = g.as_mut() {
                if let Some(e) = f.editable() {
                    let _ = e.sync_metadata();
                }
            }
            *g = None;
        }
    }

    /// Cancel the running job and write the filesystem's metadata: before
    /// the window closes, and before the machine powers on. `false` when the
    /// job did not stop within a few seconds (it holds the disk): nothing is
    /// closed then, and the caller tries again later rather than waiting on
    /// the disk's lock without end.
    pub fn close(&mut self) -> bool {
        if let Some((j, _)) = &self.job {
            j.cancel();
            if !j.wait(std::time::Duration::from_secs(5)) {
                return false;
            }
        }
        self.job = None;
        self.close_fs();
        true
    }

    fn selected(&self) -> Vec<FileEntry> {
        self.selection.iter().filter_map(|i| self.entries.get(*i).cloned()).collect()
    }

    fn here(&self) -> Option<FileEntry> {
        self.path.last().cloned()
    }

    fn names_here(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    // ---- actions --------------------------------------------------------------

    pub fn apply(&mut self, action: BrowserAction) -> Vec<Effect> {
        let mut fx = Vec::new();
        if let Enabled::No(why) = self.enabled(&action) {
            if !why.is_empty() {
                fx.push(Effect::Toast(why));
            }
            return fx;
        }
        match action {
            BrowserAction::SelectPartition(i) => self.select_partition(i),
            BrowserAction::Activate(i) => {
                let Some(e) = self.entries.get(i).cloned() else { return fx };
                if e.is_directory() {
                    self.path.push(e);
                    self.selection.clear();
                    self.list();
                } else {
                    self.selection = vec![i];
                    return self.apply(BrowserAction::EditText);
                }
            }
            BrowserAction::Up => {
                if self.path.len() > 1 {
                    self.path.pop();
                    self.selection.clear();
                    self.list();
                }
            }
            BrowserAction::GoTo(depth) => {
                self.path.truncate(depth + 1);
                self.selection.clear();
                self.list();
            }
            BrowserAction::Refresh => self.list(),
            BrowserAction::Select(v) => self.selection = v.into_iter().filter(|i| *i < self.entries.len()).collect(),
            BrowserAction::Export => fx.push(Effect::PickFolder(PickSpec::open(P_EXPORT, t!("browser-export-title")))),
            BrowserAction::ExportArchive { gzip } => {
                let base = self.selected().first().map(|e| e.name.clone()).unwrap_or_else(|| "export".into());
                let ext = if gzip { "tgz" } else { "tar" };
                fx.push(Effect::PickFile(
                    PickSpec::save(P_EXPORT_TAR, t!("browser-export-archive-title"), format!("{base}.{ext}"))
                        .filter("tar", &[ext]),
                ));
            }
            BrowserAction::Import => {
                fx.push(Effect::PickFile(PickSpec { multiple: true, ..PickSpec::open(P_IMPORT, t!("browser-import-title")) }))
            }
            BrowserAction::ImportFolder => fx.push(Effect::PickFolder(PickSpec::open(P_IMPORT_FOLDER, t!("browser-import-title")))),
            BrowserAction::Drop { paths, into } => {
                // A drop does not copy at once: it asks, with the options and
                // the free space, like Import.
                let dest = into.and_then(|i| self.entries.get(i).filter(|e| e.is_directory()).cloned());
                fx.push(self.import_question(paths, dest));
            }
            BrowserAction::SetUnpackArchives(on) => self.unpack_archives = on,
            BrowserAction::SetUnixLineEndings(on) => self.unix_line_endings = on,
            BrowserAction::EditText => {
                let Some(e) = self.selected().into_iter().next() else { return fx };
                self.run(t!("browser-reading"), false, move |fs, _| {
                    let data = fs.fs().read_file(&e, model::TEXT_EDIT_LIMIT as usize + 1).map_err(|e| e.to_string())?;
                    Ok(Done::Text(e.name.clone(), data))
                });
            }
            BrowserAction::EditorChanged(text) => {
                if let Some((ed, _, _)) = &mut self.editor {
                    if ed.text != text {
                        ed.text = text;
                        ed.dirty = true;
                    }
                }
            }
            BrowserAction::EditorSave => self.save_text(),
            BrowserAction::EditorClose => self.editor = None,
            BrowserAction::Permissions => {
                let m = self.selected().first().and_then(|e| e.mode).unwrap_or(0o644) & 0o7777;
                fx.push(Effect::AskText(TextSpec {
                    purpose: P_CHMOD,
                    title: t!("browser-permissions-title"),
                    prompt: t!("browser-permissions-prompt"),
                    initial: format!("{m:o}"),
                    ok_label: t!("common-apply"),
                    taken: Vec::new(),
                    error: None,
                    validator: mode_rule,
                }));
            }
            BrowserAction::Owner => {
                let e = self.selected().into_iter().next();
                let initial = e
                    .map(|e| format!("{}:{}", self.names.user(e.uid.unwrap_or(0)), self.names.group(e.gid.unwrap_or(0))))
                    .unwrap_or_default();
                let taken = self
                    .names
                    .users
                    .iter()
                    .map(|(_, n)| format!("u:{n}"))
                    .chain(self.names.groups.iter().map(|(_, n)| format!("g:{n}")))
                    .collect();
                fx.push(Effect::AskText(TextSpec {
                    purpose: P_CHOWN,
                    title: t!("browser-owner-title"),
                    prompt: t!("browser-owner-prompt"),
                    initial,
                    ok_label: t!("common-apply"),
                    taken,
                    error: None,
                    validator: owner_rule,
                }));
            }
            BrowserAction::Rename => {
                let Some(e) = self.selected().into_iter().next() else { return fx };
                fx.push(Effect::AskText(TextSpec {
                    purpose: P_RENAME,
                    title: t!("browser-rename-title"),
                    prompt: t!("browser-name-prompt"),
                    initial: e.name.clone(),
                    ok_label: t!("modal-rename-confirm"),
                    taken: self.names_here(),
                    error: None,
                    validator: name_rule,
                }));
            }
            BrowserAction::Move => {}
            BrowserAction::NewFolder => fx.push(Effect::AskText(TextSpec {
                purpose: P_NEW_FOLDER,
                title: t!("library-new-folder"),
                prompt: t!("browser-name-prompt"),
                initial: String::new(),
                ok_label: t!("common-ok"),
                taken: self.names_here(),
                error: None,
                validator: name_rule,
            })),
            BrowserAction::Delete => {
                let sel = self.selected();
                let list = sel.iter().map(|e| e.name.as_str()).take(8).collect::<Vec<_>>().join(", ");
                let dirs = sel.iter().any(|e| e.is_directory());
                fx.push(Effect::Confirm(ConfirmSpec {
                    purpose: P_DELETE,
                    title: t!("browser-delete-title", count = sel.len()),
                    message: if dirs {
                        t!("browser-delete-body-dirs", names = list)
                    } else {
                        t!("browser-delete-body", names = list)
                    },
                    confirm_label: t!("library-delete-ok"),
                    destructive: true,
                }));
            }
            BrowserAction::Check => {
                self.run(t!("browser-checking"), false, |fs, _| match fs.fs().fsck() {
                    None => Ok(Done::Checked(t!("browser-check-unsupported"), true)),
                    Some(Err(e)) => Err(e.to_string()),
                    Some(Ok(r)) => {
                        let mut text = if r.errors.is_empty() {
                            t!("browser-check-clean", warnings = r.warnings.len())
                        } else {
                            t!("browser-check-errors", errors = r.errors.len(), warnings = r.warnings.len())
                        };
                        for i in r.errors.iter().chain(r.warnings.iter()).take(20) {
                            text.push_str(&format!("\n{}: {}", i.code, i.message));
                        }
                        Ok(Done::Checked(text, r.errors.is_empty()))
                    }
                });
            }
            BrowserAction::EditAnyway => fx.push(Effect::Confirm(ConfirmSpec {
                purpose: P_EDIT_ANYWAY,
                title: t!("browser-edit-anyway-title"),
                message: t!("browser-edit-anyway-body"),
                confirm_label: t!("browser-edit-anyway-ok"),
                destructive: true,
            })),
            BrowserAction::ReplayLog => {
                let n = match &self.xfs_log {
                    Some(model::XfsLog::Replayable { transactions }) => *transactions,
                    _ => 0,
                };
                fx.push(Effect::Confirm(ConfirmSpec {
                    purpose: P_REPLAY,
                    title: t!("browser-replay-title"),
                    message: t!("browser-replay-body", count = n),
                    confirm_label: t!("browser-replay-ok"),
                    destructive: false,
                }));
            }
            BrowserAction::Cancel => {
                if let Some((j, _)) = &self.job {
                    j.cancel();
                }
            }
            BrowserAction::Answer(a) => self.answer(a, &mut fx),
        }
        fx
    }

    fn import_question(&mut self, paths: Vec<PathBuf>, into: Option<FileEntry>) -> Effect {
        // An archive (a `.tardist`, a tar) is unpacked or copied as one file,
        // and the user says which before anything else.
        if !self.unpack_asked {
            let archives = archives_in(&paths);
            if !archives.is_empty() {
                self.pending_import = paths;
                self.import_into = into;
                let shown: Vec<&str> = archives.iter().take(5).map(String::as_str).collect();
                let mut names = shown.join(", ");
                if archives.len() > shown.len() {
                    names.push_str(&t!("browser-unpack-more", count = archives.len() - shown.len()));
                }
                return Effect::Choose(ChoiceSpec {
                    purpose: P_UNPACK,
                    title: t!("browser-unpack-title"),
                    message: t!("browser-unpack-body", count = archives.len(), names = names),
                    choices: vec![t!("common-cancel"), t!("browser-import-as-file"), t!("browser-import-unpack")],
                    dismiss: 0,
                    default: Some(2),
                    destructive: vec![],
                });
            }
        }
        let dest = into.clone().or_else(|| self.here()).map(|e| if e.path.is_empty() { "/".into() } else { e.path }).unwrap_or_default();
        let bytes: u64 = paths.iter().map(|p| host_size(p)).sum();
        let free = self
            .fs
            .lock()
            .ok()
            .and_then(|mut g| g.as_mut().and_then(|f| f.editable().and_then(|e| e.free_space().ok())));
        let mut message = t!("browser-import-body", count = paths.len(), size = human(bytes), dest = dest.as_str());
        if let Some(free) = free {
            message.push('\n');
            message.push_str(&t!("browser-free-space", free = human(free)));
            if bytes > free {
                message.push('\n');
                message.push_str(&t!("browser-no-room"));
            }
        }
        let taken: Vec<String> = self.names_here();
        let clashes: Vec<String> = paths
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .filter(|n| into.is_none() && taken.contains(n))
            .collect();
        if !clashes.is_empty() {
            message.push('\n');
            message.push_str(&t!("browser-import-clash", names = clashes.join(", ")));
        }
        self.pending_import = paths;
        self.import_into = into;
        Effect::Choose(ChoiceSpec {
            purpose: P_IMPORT_GO,
            title: t!("browser-import-title"),
            message,
            choices: vec![t!("common-cancel"), t!("browser-import-skip-existing"), t!("browser-import-replace")],
            dismiss: 0,
            default: Some(1),
            destructive: vec![2],
        })
    }

    fn import(&mut self, replace: bool) {
        let paths = std::mem::take(&mut self.pending_import);
        let Some(dest) = self.import_into.take().or_else(|| self.here()) else { return };
        let unpack = self.unpack_archives;
        let endings = if self.unix_line_endings { rb_fsimporter::LineEndings::Unix } else { rb_fsimporter::LineEndings::Keep };
        // New files take the destination folder's owner and group.
        let (uid, gid) = (dest.uid, dest.gid);
        self.run(t!("browser-importing"), true, move |fs, ctx| {
            let efs = fs.editable().ok_or_else(|| t!("browser-read-only"))?;
            let total: u64 = paths.iter().map(|p| host_size(p)).sum();
            let mut done = 0u64;
            for p in &paths {
                if ctx.cancelled() {
                    break;
                }
                let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                ctx.phase(name.clone());
                let existing = efs.list_directory(&dest).map_err(|e| e.to_string())?.into_iter().find(|e| e.name == name);
                if p.is_dir() {
                    let target = match existing {
                        Some(e) if e.is_directory() => e,
                        Some(_) if !replace => continue,
                        Some(e) => {
                            efs.delete_recursive(&dest, &e).map_err(|e| e.to_string())?;
                            efs.create_directory(&dest, &name, &CreateDirectoryOptions { mode: Some(S_IFDIR | 0o755), uid, gid, ..Default::default() })
                                .map_err(|e| e.to_string())?
                        }
                        None => efs
                            .create_directory(&dest, &name, &CreateDirectoryOptions { mode: Some(S_IFDIR | 0o755), uid, gid, ..Default::default() })
                            .map_err(|e| e.to_string())?,
                    };
                    let opts = dir_options(unpack, replace, endings);
                    struct H<'a>(&'a crate::model::job::JobCtx<Done>, u64, u64);
                    impl rb_fsimporter::Host for H<'_> {
                        fn cancelled(&self) -> bool {
                            self.0.cancelled()
                        }
                        fn progress(&self, s: &rb_fsimporter::ImportStats) {
                            self.0.progress(Progress::Determinate { done: self.1 + s.total_bytes, total: self.2, unit: Unit::Bytes });
                        }
                    }
                    let s = rb_fsimporter::import_dir_with(efs, &target, p, &opts, None, &H(ctx, done, total))
                        .map_err(|e| e.to_string())?;
                    // The importer applies the host's owner with its mode; the
                    // decided default is the host's mode and the destination's
                    // owner, so put the owner right afterwards.
                    if let (Some(u), Some(g)) = (uid, gid) {
                        chown_tree(efs, &target, u, g).map_err(|e| e.to_string())?;
                    }
                    done += s.total_bytes;
                } else if unpack && fs_tar::tar_import::looks_like_tar_archive(p) {
                    // Unpacked into a folder named after the archive, as an
                    // archive inside an imported folder is.
                    let folder = unpacked_name(&name);
                    let existing = efs.list_directory(&dest).map_err(|e| e.to_string())?.into_iter().find(|e| e.name == folder);
                    let target = match existing {
                        Some(e) if e.is_directory() => e,
                        Some(_) if !replace => continue,
                        Some(e) => {
                            efs.delete_recursive(&dest, &e).map_err(|e| e.to_string())?;
                            efs.create_directory(&dest, &folder, &CreateDirectoryOptions { mode: Some(S_IFDIR | 0o755), uid, gid, ..Default::default() })
                                .map_err(|e| e.to_string())?
                        }
                        None => efs
                            .create_directory(&dest, &folder, &CreateDirectoryOptions { mode: Some(S_IFDIR | 0o755), uid, gid, ..Default::default() })
                            .map_err(|e| e.to_string())?,
                    };
                    let opts = dir_options(true, replace, endings).shared;
                    let raw = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                    fs_tar::tar_import::import_tar_from_path(efs, &target, p, &opts, &|_| {}).map_err(|e| e.to_string())?;
                    if let (Some(u), Some(g)) = (uid, gid) {
                        chown_tree(efs, &target, u, g).map_err(|e| e.to_string())?;
                    }
                    done += raw;
                    ctx.progress(Progress::Determinate { done, total, unit: Unit::Bytes });
                } else {
                    if let Some(e) = existing {
                        if !replace {
                            continue;
                        }
                        efs.delete_recursive(&dest, &e).map_err(|e| e.to_string())?;
                    }
                    let mode = host_mode(p);
                    let raw = std::fs::metadata(p).map(|m| m.len()).map_err(|e| e.to_string())?;
                    let opts = CreateFileOptions { mode: Some(S_IFREG | mode.unwrap_or(0o644)), uid, gid, ..Default::default() };
                    // Text files, when asked, get Unix line endings on the way
                    // in; the crates decide what counts as text (never an archive).
                    let (len, convert) = if endings == rb_fsimporter::LineEndings::Keep {
                        (raw, false)
                    } else {
                        let mut f = std::fs::File::open(p).map_err(|e| e.to_string())?;
                        fs_tar::line_endings::imported_len(endings, &name, &mut f).map_err(|e| e.to_string())?
                    };
                    let f = std::fs::File::open(p).map_err(|e| e.to_string())?;
                    if convert {
                        let mut r = fs_tar::line_endings::LineEndingReader::new(f, endings);
                        efs.create_file(&dest, &name, &mut r, len, &opts).map_err(|e| e.to_string())?;
                    } else {
                        let mut f = f;
                        efs.create_file(&dest, &name, &mut f, len, &opts).map_err(|e| e.to_string())?;
                    }
                    done += raw;
                    ctx.progress(Progress::Determinate { done, total, unit: Unit::Bytes });
                }
            }
            Ok(Done::Imported(done))
        });
    }

    fn save_text(&mut self) {
        let Some((ed, entry, dir)) = self.editor.clone() else { return };
        let bytes = if ed.line_endings == "CRLF" { ed.text.replace('\n', "\r\n").into_bytes() } else { ed.text.into_bytes() };
        self.run(t!("browser-saving"), true, move |fs, _| {
            let efs = fs.editable().ok_or_else(|| t!("browser-read-only"))?;
            // Write a new file, then replace the old one, then put back its
            // owner, mode and date, so a failed write leaves the old file.
            let tmp = format!(".{}.iris-edit", entry.name);
            let mut r = &bytes[..];
            let opts = CreateFileOptions { mode: Some(S_IFREG | entry.mode.unwrap_or(0o644) & 0o7777), uid: entry.uid, gid: entry.gid, ..Default::default() };
            let new = efs.create_file(&dir, &tmp, &mut r, bytes.len() as u64, &opts).map_err(|e| e.to_string())?;
            efs.delete_entry(&dir, &entry).map_err(|e| e.to_string())?;
            efs.rename(&dir, &new, &entry.name).map_err(|e| e.to_string())?;
            let renamed = efs
                .list_directory(&dir)
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|e| e.name == entry.name)
                .ok_or_else(|| "the saved file is missing".to_string())?;
            // The whole mode: on EFS it carries the file type too, and
            // writing only the permission bits would make it a special file.
            if let Some(m) = entry.mode {
                let _ = efs.set_permissions(&renamed, m);
            }
            if let (Some(u), Some(g)) = (entry.uid, entry.gid) {
                let _ = efs.set_owner(&renamed, u, g);
            }
            if let Some(t) = entry.modified_unix {
                let _ = efs.set_dates(&renamed, t as u32, t as u32, 0);
            }
            // The file as it is now (a new inode): the next save replaces this
            // one, not the entry the editor opened with, which is gone.
            let now = efs
                .list_directory(&dir)
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|e| e.name == entry.name)
                .unwrap_or(renamed);
            Ok(Done::Saved(now))
        });
    }

    fn answer(&mut self, a: Answer, fx: &mut Vec<Effect>) {
        match a {
            Answer::Picked { purpose: P_EXPORT, paths } => {
                let Some(dest) = paths.into_iter().next() else { return };
                let entries = if self.selection.is_empty() { self.here().into_iter().collect() } else { self.selected() };
                self.run(t!("browser-exporting"), false, move |fs, ctx| {
                    struct H<'a>(&'a crate::model::job::JobCtx<Done>);
                    impl rb_fsexporter::Host for H<'_> {
                        fn cancelled(&self) -> bool {
                            self.0.cancelled()
                        }
                        fn wrote(&mut self, _: &FileEntry, _: u64, t: &rb_fsexporter::Summary) {
                            self.0.progress(Progress::Counting { done: t.bytes, unit: Unit::Bytes });
                        }
                    }
                    let mut opts = rb_fsexporter::Options::new(rb_fsexporter::ResourceForkMode::DataForkOnly);
                    opts.posix_modes = true;
                    let mut bytes = 0;
                    for e in &entries {
                        ctx.phase(e.name.clone());
                        bytes += rb_fsexporter::extract(fs.fs(), e, &dest, &opts, &mut H(ctx)).map_err(|e| e.to_string())?.bytes;
                    }
                    Ok(Done::Exported(bytes))
                });
            }
            Answer::Picked { purpose: P_EXPORT_TAR, paths } => {
                let Some(out) = paths.into_iter().next() else { return };
                let entries = if self.selection.is_empty() { self.here().into_iter().collect() } else { self.selected() };
                let compression = fs_tar::tar_export::TarCompression::infer_from_path(&out);
                self.run(t!("browser-exporting"), false, move |fs, ctx| {
                    let file = std::fs::File::create(&out).map_err(|e| e.to_string())?;
                    let mut w = std::io::BufWriter::new(file);
                    let mut bytes = 0;
                    for e in &entries {
                        ctx.phase(e.name.clone());
                        let prefix = if e.path.is_empty() || e.path == "/" { String::new() } else { e.name.clone() };
                        let progress = |s: &fs_tar::tar_export::TarExportStats| {
                            ctx.progress(Progress::Counting { done: s.total_bytes, unit: Unit::Bytes })
                        };
                        let s = fs_tar::tar_export::export_tar(
                            fs.fs(),
                            e,
                            &prefix,
                            &mut w,
                            compression,
                            &Default::default(),
                            &progress,
                        )
                        .map_err(|e| e.to_string())?;
                        bytes += s.total_bytes;
                    }
                    use std::io::Write as _;
                    w.flush().map_err(|e| e.to_string())?;
                    Ok(Done::Exported(bytes))
                });
            }
            Answer::Picked { purpose: P_IMPORT | P_IMPORT_FOLDER, paths } => {
                if !paths.is_empty() {
                    fx.push(self.import_question(paths, None));
                }
            }
            Answer::Chose { purpose: P_UNPACK, index } => match index {
                1 | 2 => {
                    self.unpack_archives = index == 2;
                    self.unpack_asked = true;
                    let paths = std::mem::take(&mut self.pending_import);
                    let into = self.import_into.take();
                    fx.push(self.import_question(paths, into));
                }
                _ => {
                    self.pending_import.clear();
                    self.import_into = None;
                }
            },
            Answer::Chose { purpose: P_IMPORT_GO, index } => {
                self.unpack_asked = false;
                match index {
                    1 => self.import(false),
                    2 => self.import(true),
                    _ => self.pending_import.clear(),
                }
            }
            Answer::Text { purpose: P_CHMOD, text: Some(t) } => {
                let Ok(mode) = u32::from_str_radix(t.trim(), 8) else { return };
                let sel = self.selected();
                self.run(t!("browser-applying"), true, move |fs, _| {
                    let efs = fs.editable().ok_or_else(|| t!("browser-read-only"))?;
                    for e in &sel {
                        let keep = e.mode.unwrap_or(0) & !0o7777;
                        efs.set_permissions(e, keep | mode).map_err(|e| e.to_string())?;
                    }
                    Ok(Done::Edited)
                });
            }
            Answer::Text { purpose: P_CHOWN, text: Some(t) } => {
                let (u, g) = t.trim().split_once(':').unwrap_or((t.trim(), ""));
                let (Some(uid), gid) = (self.names.user_id(u), self.names.group_id(g)) else { return };
                let sel = self.selected();
                self.run(t!("browser-applying"), true, move |fs, _| {
                    let efs = fs.editable().ok_or_else(|| t!("browser-read-only"))?;
                    for e in &sel {
                        efs.set_owner(e, uid, gid.or(e.gid).unwrap_or(0)).map_err(|e| e.to_string())?;
                    }
                    Ok(Done::Edited)
                });
            }
            Answer::Text { purpose: P_RENAME, text: Some(new) } => {
                let (Some(dir), Some(e)) = (self.here(), self.selected().into_iter().next()) else { return };
                self.run(t!("browser-applying"), true, move |fs, _| {
                    let efs = fs.editable().ok_or_else(|| t!("browser-read-only"))?;
                    efs.rename(&dir, &e, new.trim()).map_err(|e| e.to_string())?;
                    Ok(Done::Edited)
                });
            }
            Answer::Text { purpose: P_NEW_FOLDER, text: Some(name) } => {
                let Some(dir) = self.here() else { return };
                let (uid, gid) = (dir.uid, dir.gid);
                self.run(t!("browser-applying"), true, move |fs, _| {
                    let efs = fs.editable().ok_or_else(|| t!("browser-read-only"))?;
                    efs.create_directory(&dir, name.trim(), &CreateDirectoryOptions { mode: Some(S_IFDIR | 0o755), uid, gid, ..Default::default() })
                        .map_err(|e| e.to_string())?;
                    Ok(Done::Edited)
                });
            }
            Answer::Confirmed { purpose: P_DELETE, yes: true } => {
                let (Some(dir), sel) = (self.here(), self.selected()) else { return };
                self.run(t!("browser-deleting"), true, move |fs, ctx| {
                    let efs = fs.editable().ok_or_else(|| t!("browser-read-only"))?;
                    for e in &sel {
                        if ctx.cancelled() {
                            break;
                        }
                        ctx.phase(e.name.clone());
                        if e.entry_type == EntryType::Directory {
                            efs.delete_recursive(&dir, e)
                        } else {
                            efs.delete_entry(&dir, e)
                        }
                        .map_err(|e| e.to_string())?;
                    }
                    Ok(Done::Edited)
                });
            }
            Answer::Confirmed { purpose: P_EDIT_ANYWAY, yes: true } => self.check_failed = None,
            Answer::Confirmed { purpose: P_REPLAY, yes: true } => self.replay_log(),
            // Cancelled prompts, and answers that are not this window's.
            Answer::Text { .. } | Answer::Confirmed { .. } | Answer::Picked { .. } | Answer::Chose { .. } => {}
        }
    }

    /// Drain the finished job.
    pub fn tick(&mut self) -> Vec<Effect> {
        let mut fx = Vec::new();
        let Some(outcome) = self.job.as_ref().and_then(|(j, _)| j.outcome()) else { return fx };
        let wrote = self.job.take().map(|(_, w)| w).unwrap_or(false);
        match outcome {
            Ok(Done::Opened { names, check, log }) => {
                self.xfs_log = log;
                self.tried.clear();
                self.names = names;
                self.check_failed = check;
                let root = self.fs.lock().ok().and_then(|mut g| g.as_mut().and_then(|f| f.fs().root().ok()));
                if let Some(root) = root {
                    self.path = vec![root];
                    self.list();
                }
            }
            Ok(Done::Listed(v)) => {
                self.entries = v;
                self.selection.retain(|i| *i < self.entries.len());
            }
            Ok(Done::Exported(n)) => fx.push(Effect::Toast(t!("browser-exported", size = human(n)))),
            Ok(Done::Imported(n)) => {
                fx.push(Effect::Toast(t!("browser-imported", size = human(n))));
                self.list();
            }
            Ok(Done::Text(name, data)) => {
                if !model::looks_like_text(&data) || data.len() as u64 > model::TEXT_EDIT_LIMIT {
                    fx.push(Effect::Alert {
                        severity: Severity::Info,
                        title: name,
                        message: t!("browser-not-text"),
                    });
                } else {
                    let text = String::from_utf8_lossy(&data).into_owned();
                    let crlf = text.contains("\r\n");
                    let entry = self.selected().into_iter().next();
                    if let (Some(entry), Some(dir)) = (entry, self.here()) {
                        self.editor = Some((
                            Editor {
                                name,
                                text: if crlf { text.replace("\r\n", "\n") } else { text },
                                dirty: false,
                                line_endings: if crlf { "CRLF" } else { "LF" },
                                read_only: self.read_only_reason().is_some(),
                            },
                            entry,
                            dir,
                        ));
                    }
                }
            }
            Ok(Done::Replayed(n)) => {
                fx.push(Effect::Toast(t!("browser-replayed", count = n)));
                // Open it again: now clean, so editable.
                if let Some(i) = self.selected_part {
                    self.select_partition(i);
                }
            }
            Ok(Done::Saved(now)) => {
                if let Some((ed, entry, _)) = &mut self.editor {
                    ed.dirty = false;
                    *entry = now;
                }
                self.list();
            }
            Ok(Done::Edited) => self.list(),
            Ok(Done::Checked(text, clean)) => fx.push(Effect::Alert {
                severity: if clean { Severity::Info } else { Severity::Warning },
                title: t!("browser-check-title"),
                message: text,
            }),
            Err(e) if self.path.is_empty() && self.entries.is_empty() && self.selected_part.is_some() && !self.tried.is_empty() => {
                // A partition that would not open: say why on its row, and
                // fall through to the next one with a filesystem.
                let i = self.selected_part.take().unwrap_or(0);
                self.parts[i].why_not = Some(e.clone());
                self.parts[i].kind = PartKind::None;
                let next = self.parts.iter().enumerate().position(|(j, p)| {
                    !self.tried.contains(&j) && matches!(p.kind, PartKind::Efs | PartKind::Xfs | PartKind::Ext)
                });
                match next {
                    Some(j) => {
                        self.tried.push(j);
                        self.select_partition(j);
                    }
                    None => fx.push(Effect::Alert { severity: Severity::Error, title: t!("browser-failed"), message: e }),
                }
            }
            Err(e) if e == "cancelled" => {
                fx.push(Effect::Toast(t!("browser-cancelled")));
                if wrote {
                    self.list();
                }
            }
            Err(e) => {
                fx.push(Effect::Alert { severity: Severity::Error, title: t!("browser-failed"), message: e });
                if wrote {
                    self.list();
                }
            }
        }
        fx
    }
}

impl Drop for DiskBrowserController {
    fn drop(&mut self) {
        self.close();
    }
}

/// Give everything under `dir` the owner `uid:gid`.
fn chown_tree(
    efs: &mut dyn disk_core::filesystem::EditableFilesystem,
    dir: &FileEntry,
    uid: u32,
    gid: u32,
) -> Result<(), disk_core::filesystem::FilesystemError> {
    for e in efs.list_directory(dir)? {
        if e.name == "." || e.name == ".." {
            continue;
        }
        if e.uid != Some(uid) || e.gid != Some(gid) {
            efs.set_owner(&e, uid, gid)?;
        }
        if e.is_directory() {
            chown_tree(efs, &e, uid, gid)?;
        }
    }
    Ok(())
}

/// The archives among `paths` and inside the folders there, by name. Files
/// are sniffed for a tar; folders are walked by name, as far as a few
/// thousand entries.
fn archives_in(paths: &[PathBuf]) -> Vec<String> {
    fn by_name(name: &str) -> bool {
        let n = name.to_ascii_lowercase();
        [".tardist", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.zst"].iter().any(|x| n.ends_with(x))
    }
    let mut out = Vec::new();
    let mut seen = 0usize;
    let mut stack: Vec<PathBuf> = Vec::new();
    for p in paths {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if p.is_dir() {
            stack.push(p.clone());
        } else if fs_tar::tar_import::looks_like_tar_archive(p) {
            out.push(name);
        }
    }
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            seen += 1;
            if seen > 5000 {
                return out;
            }
            let path = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            match e.file_type() {
                Ok(t) if t.is_dir() => stack.push(path),
                Ok(t) if t.is_file() && by_name(&name) => out.push(name),
                _ => {}
            }
        }
    }
    out
}

/// The folder an archive unpacks into: its name without the archive's
/// extension (`foo-1.2.tar.gz` -> `foo-1.2`, `bar.tardist` -> `bar`).
fn unpacked_name(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    for x in [".tar.gz", ".tar.bz2", ".tar.zst", ".tardist", ".tgz", ".tar"] {
        if lower.ends_with(x) && name.len() > x.len() {
            return name[..name.len() - x.len()].to_string();
        }
    }
    format!("{name}.d")
}

fn dir_options(unpack: bool, replace: bool, endings: rb_fsimporter::LineEndings) -> rb_fsimporter::DirImportOptions {
    let mut o = rb_fsimporter::DirImportOptions::default();
    o.expand_archives = unpack;
    o.shared.line_endings = endings;
    o.shared.apply_permissions = true;
    o.shared.conflict =
        if replace { fs_tar::import_sink::ImportConflict::Overwrite } else { fs_tar::import_sink::ImportConflict::Skip };
    o
}

fn host_size(p: &Path) -> u64 {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => std::fs::read_dir(p).map(|rd| rd.flatten().map(|e| host_size(&e.path())).sum()).unwrap_or(0),
        Ok(m) => m.len(),
        Err(_) => 0,
    }
}

/// The host file's permission bits, for a new file (plan: the host's mode).
fn host_mode(p: &Path) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(p).ok().map(|m| m.permissions().mode() & 0o7777)
    }
    #[cfg(not(unix))]
    {
        let _ = p;
        Some(0o644)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_tool_id_maps_back_to_its_action() {
        for t in super::tools().into_iter().flatten() {
            assert_eq!(super::action_for(t.id, false), Some(t.action), "{}", t.id);
            assert!(!t.label.is_empty());
        }
        assert_eq!(super::action_for("browser.part:2", false), Some(super::BrowserAction::SelectPartition(2)));
        assert_eq!(super::action_for("browser.path:1", false), Some(super::BrowserAction::GoTo(1)));
        assert_eq!(super::action_for(super::ids::UNPACK, true), Some(super::BrowserAction::SetUnpackArchives(true)));
        assert_eq!(super::action_for("browser.nonsense", false), None);
    }

    use super::*;
    use crate::model::job::NoWake;
    use std::time::{Duration, Instant};

    fn settle(b: &mut DiskBrowserController) -> Vec<Effect> {
        let mut fx = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            fx.extend(b.tick());
            if b.job.is_none() || Instant::now() > deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        fx.extend(b.tick());
        fx
    }

    fn find(b: &DiskBrowserController, name: &str) -> Option<usize> {
        b.rows().iter().position(|r| r.name == name)
    }

    fn go(b: &mut DiskBrowserController, name: &str) {
        let i = find(b, name).unwrap_or_else(|| panic!("no {name} in {:?}", b.rows().iter().map(|r| &r.name).collect::<Vec<_>>()));
        b.apply(BrowserAction::Activate(i));
        settle(b);
    }

    fn select(b: &mut DiskBrowserController, name: &str) {
        let i = find(b, name).unwrap_or_else(|| panic!("no {name}"));
        b.apply(BrowserAction::Select(vec![i]));
    }

    fn answer(b: &mut DiskBrowserController, a: Answer) -> Vec<Effect> {
        let mut fx = b.apply(BrowserAction::Answer(a));
        fx.extend(settle(b));
        fx
    }

    fn no_errors(fx: &[Effect]) {
        for e in fx {
            if let Effect::Alert { severity: Severity::Error, message, .. } = e {
                panic!("{message}");
            }
        }
    }

    /// Archives are found among picked files by their content and inside
    /// picked folders by name; each unpacks into a folder named after it.
    #[test]
    fn archives_are_found_and_named_for_unpacking() {
        let root = crate::model::machines::testing::TempRoot::new("archives");
        let tar = root.0.join("odd-name.bin");
        let mut header = vec![0u8; 1024];
        header[..4].copy_from_slice(b"file");
        header[257..262].copy_from_slice(b"ustar");
        std::fs::write(&tar, &header).unwrap();
        let text = root.0.join("notes.txt");
        std::fs::write(&text, "not an archive").unwrap();
        let folder = root.0.join("pkgs");
        std::fs::create_dir_all(folder.join("deep")).unwrap();
        std::fs::write(folder.join("deep/gcc.tardist"), b"").unwrap();
        std::fs::write(folder.join("readme"), b"").unwrap();
        let mut found = archives_in(&[tar, text, folder]);
        found.sort();
        assert_eq!(found, ["gcc.tardist", "odd-name.bin"]);
        assert_eq!(unpacked_name("gcc.tardist"), "gcc");
        assert_eq!(unpacked_name("foo-1.2.tar.gz"), "foo-1.2");
        assert_eq!(unpacked_name("X.TGZ"), "X");
        assert_eq!(unpacked_name("odd-name.bin"), "odd-name.bin.d");
    }

    #[test]
    fn a_cdrom_is_read_only_and_a_blank_disk_has_no_label() {
        let root = crate::model::machines::testing::TempRoot::new("browser");
        let p = root.0.join("blank.img");
        std::fs::write(&p, vec![0u8; 1 << 16]).unwrap();
        let t = BrowseTarget { machine: "m".into(), label: "CD".into(), path: p.to_string_lossy().into(), cdrom: true, overlay: false };
        let b = DiskBrowserController::open(t, None, Arc::new(NoWake));
        assert!(b.error().is_some());
        assert_eq!(b.enabled(&BrowserAction::Move), Enabled::No(t!("browser-no-partition")));
    }

    /// Every action on a copy of a real IRIX disk. Run with
    /// `IRIS_BROWSER_TEST_IMAGE=/path/to/copy.img cargo test -p iris-gui-core real_irix_disk -- --ignored`.
    /// It changes the image: give it a copy.
    #[test]
    #[ignore]
    fn real_irix_disk() {
        let Ok(image) = std::env::var("IRIS_BROWSER_TEST_IMAGE") else { return };
        let host = crate::model::machines::testing::TempRoot::new("browser-host");
        let t = BrowseTarget { machine: "indy".into(), label: "Disk 1".into(), path: image, cdrom: false, overlay: false };
        let mut b = DiskBrowserController::open(t, None, Arc::new(NoWake));
        assert!(b.error().is_none(), "{:?}", b.error());
        eprintln!("partitions: {:#?}", b.partitions());
        no_errors(&settle(&mut b));
        // An XFS volume IRIX did not shut down cleanly: replay its log first,
        // as the user would with Replay Log; it reopens editable.
        eprintln!("xfs log on open: {:?}", b.xfs_log);
        if matches!(b.xfs_log, Some(crate::model::disk_browser::XfsLog::Replayable { .. })) {
            assert!(b.read_only_reason().is_some(), "a dirty XFS log opens read-only");
            assert_eq!(b.enabled(&BrowserAction::ReplayLog), Enabled::Yes);
            no_errors(&answer(&mut b, Answer::Confirmed { purpose: P_REPLAY, yes: true }));
            settle(&mut b);
            eprintln!("xfs log after replay: {:?}", b.xfs_log);
            assert_eq!(b.xfs_log, Some(crate::model::disk_browser::XfsLog::Clean), "the replay left the log clean");
        }
        // What the disk's own check said before anything was changed.
        let mut before = None;
        if b.check_failed.is_some() {
            b.apply(BrowserAction::Check);
            for e in settle(&mut b) {
                if let Effect::Alert { message, .. } = e {
                    eprintln!("check on open: {message}");
                    before = Some(message);
                }
            }
            // The check's verdict was reported; go on as a user choosing Edit Anyway would.
            b.apply(BrowserAction::Answer(Answer::Confirmed { purpose: P_EDIT_ANYWAY, yes: true }));
        }
        assert_eq!(b.read_only_reason(), None, "the root opens for editing");
        let rows = b.rows();
        eprintln!("root: {:?}", rows.iter().map(|r| &r.name).collect::<Vec<_>>());
        go(&mut b, "etc");
        let passwd = &b.rows()[find(&b, "passwd").unwrap()];
        eprintln!("passwd: {passwd:?}");
        assert_eq!(passwd.owner, "root", "owner names come from the disk's /etc/passwd");

        // Edit /etc/motd: change it and save.
        if find(&b, "motd").is_some() {
            select(&mut b, "motd");
            b.apply(BrowserAction::EditText);
            no_errors(&settle(&mut b));
            let before = b.editor().expect("motd opens in the editor").clone();
            b.apply(BrowserAction::EditorChanged(format!("{}Edited by the IRIS disk browser.\n", before.text)));
            b.apply(BrowserAction::EditorSave);
            no_errors(&settle(&mut b));
            let motd = &b.rows()[find(&b, "motd").unwrap()];
            assert_eq!((motd.mode.as_str(), motd.owner.as_str()), (passwd_like_mode(&before), "root"));
            // Save again with the editor still open: it replaces the file as
            // it is now (the first save gave it a new inode).
            b.apply(BrowserAction::EditorChanged(format!("{}Saved twice.\n", before.text)));
            b.apply(BrowserAction::EditorSave);
            no_errors(&settle(&mut b));
            // And again from another folder: it still saves into /etc.
            b.apply(BrowserAction::GoTo(0));
            settle(&mut b);
            let third = format!("{}Saved from the root.\n", before.text);
            b.apply(BrowserAction::EditorChanged(third.clone()));
            b.apply(BrowserAction::EditorSave);
            no_errors(&settle(&mut b));
            assert!(find(&b, "motd").is_none(), "nothing was written into the root");
            assert!(!b.rows().iter().any(|r| r.name.ends_with(".iris-edit")), "no temporary file in the root");
            b.apply(BrowserAction::EditorClose);
            go(&mut b, "etc");
            assert!(!b.rows().iter().any(|r| r.name.ends_with(".iris-edit")), "no temporary file in /etc");
            select(&mut b, "motd");
            b.apply(BrowserAction::EditText);
            no_errors(&settle(&mut b));
            assert_eq!(b.editor().expect("motd opens again").text, third, "the last save is what /etc/motd holds");
            b.apply(BrowserAction::EditorClose);
        }

        // Up to the root, then into usr/tmp (or tmp) for the rest.
        b.apply(BrowserAction::GoTo(0));
        settle(&mut b);
        // Not /tmp: IRIX empties it at boot, and the boot test reads these back.
        let work = "var";
        go(&mut b, work);

        // New folder, rename it, chmod and chown it.
        b.apply(BrowserAction::NewFolder);
        no_errors(&answer(&mut b, Answer::Text { purpose: P_NEW_FOLDER, text: Some("iris-test".into()) }));
        select(&mut b, "iris-test");
        b.apply(BrowserAction::Rename);
        no_errors(&answer(&mut b, Answer::Text { purpose: P_RENAME, text: Some("iris-renamed".into()) }));
        select(&mut b, "iris-renamed");
        no_errors(&answer(&mut b, Answer::Text { purpose: P_CHMOD, text: Some("750".into()) }));
        select(&mut b, "iris-renamed");
        no_errors(&answer(&mut b, Answer::Text { purpose: P_CHOWN, text: Some("guest:user".into()) }));
        let row = &b.rows()[find(&b, "iris-renamed").unwrap()];
        assert_eq!(row.mode, "drwxr-x---");
        eprintln!("chown result: {row:?}");

        // Import a host folder with a file and a tar into it.
        go(&mut b, "iris-renamed");
        let src = host.0.join("pkg");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("hello.txt"), "hello from the host\n").unwrap();
        std::fs::write(src.join("sub/data.bin"), vec![7u8; 100_000]).unwrap();
        std::fs::write(host.0.join("single.txt"), "one file\n").unwrap();
        // A DOS text file and a folder file with CRLF: Unix line endings on import.
        std::fs::write(host.0.join("dos.txt"), "line one\r\nline two\r\n").unwrap();
        std::fs::write(src.join("crlf.txt"), "a\r\nb\r\n").unwrap();
        b.apply(BrowserAction::SetUnixLineEndings(true));
        b.apply(BrowserAction::Answer(Answer::Picked { purpose: P_IMPORT_FOLDER, paths: vec![src.clone(), host.0.join("single.txt"), host.0.join("dos.txt")] }));
        no_errors(&answer(&mut b, Answer::Chose { purpose: P_IMPORT_GO, index: 1 }));
        b.apply(BrowserAction::SetUnixLineEndings(false));
        // Converted: 20 bytes with CRLF, 18 with LF.
        let dos = &b.rows()[find(&b, "dos.txt").unwrap()];
        assert_eq!(dos.size, "18 B", "a single file's CRLF became LF");
        assert!(find(&b, "pkg").is_some() && find(&b, "single.txt").is_some(), "{:?}", b.rows());
        // New files take the destination folder's owner (iris-renamed is guest:user).
        let pkg = &b.rows()[find(&b, "pkg").unwrap()];
        assert_eq!((pkg.owner.as_str(), pkg.group.as_str()), ("guest", "user"));
        go(&mut b, "pkg");
        go(&mut b, "sub");
        let data = &b.rows()[find(&b, "data.bin").unwrap()];
        assert_eq!((data.owner.as_str(), data.group.as_str()), ("guest", "user"), "{data:?}");
        b.apply(BrowserAction::GoTo(2));
        settle(&mut b);

        // A .tardist asks first: unpacked into a folder named after it, or
        // copied in as it is.
        let dist = host.0.join("dist");
        std::fs::create_dir_all(&dist).unwrap();
        for (f, body) in [("tool", "product\n"), ("tool.idb", "f 0644 root sys usr/bin/tool\n"), ("tool.sw", "payload\n")] {
            std::fs::write(dist.join(f), body).unwrap();
        }
        let tardist = host.0.join("tool.tardist");
        let ok = std::process::Command::new("tar").arg("-cf").arg(&tardist).arg("-C").arg(&dist).args(["tool", "tool.idb", "tool.sw"]).status().unwrap();
        assert!(ok.success());
        let fx = b.apply(BrowserAction::Answer(Answer::Picked { purpose: P_IMPORT, paths: vec![tardist.clone()] }));
        assert!(fx.iter().any(|e| matches!(e, Effect::Choose(c) if c.purpose == P_UNPACK)), "a tardist asks: {fx:?}");
        let fx = answer(&mut b, Answer::Chose { purpose: P_UNPACK, index: 2 });
        assert!(fx.iter().any(|e| matches!(e, Effect::Choose(c) if c.purpose == P_IMPORT_GO)), "then the import question");
        no_errors(&answer(&mut b, Answer::Chose { purpose: P_IMPORT_GO, index: 1 }));
        assert!(find(&b, "tool.tardist").is_none(), "unpacked, not copied");
        go(&mut b, "tool");
        for f in ["tool", "tool.idb", "tool.sw"] {
            let r = &b.rows()[find(&b, f).unwrap_or_else(|| panic!("{f} unpacked"))];
            assert_eq!((r.owner.as_str(), r.group.as_str()), ("guest", "user"), "{r:?}");
        }
        b.apply(BrowserAction::GoTo(2));
        settle(&mut b);
        // Copy as one file.
        b.apply(BrowserAction::Answer(Answer::Picked { purpose: P_IMPORT, paths: vec![tardist.clone()] }));
        answer(&mut b, Answer::Chose { purpose: P_UNPACK, index: 1 });
        no_errors(&answer(&mut b, Answer::Chose { purpose: P_IMPORT_GO, index: 1 }));
        let copied = &b.rows()[find(&b, "tool.tardist").expect("copied as one file")];
        assert!(copied.mode.starts_with('-'), "a plain file: {copied:?}");

        // Export it back and compare.
        b.apply(BrowserAction::Select(vec![find(&b, "pkg").unwrap()]));
        let out = host.0.join("out");
        std::fs::create_dir_all(&out).unwrap();
        no_errors(&answer(&mut b, Answer::Picked { purpose: P_EXPORT, paths: vec![out.clone()] }));
        assert_eq!(std::fs::read_to_string(out.join("pkg/hello.txt")).unwrap(), "hello from the host\n");
        assert_eq!(std::fs::read(out.join("pkg/sub/data.bin")).unwrap(), vec![7u8; 100_000]);

        // Export as a tar.gz.
        let tgz = host.0.join("pkg.tgz");
        no_errors(&answer(&mut b, Answer::Picked { purpose: P_EXPORT_TAR, paths: vec![tgz.clone()] }));
        assert!(std::fs::metadata(&tgz).unwrap().len() > 100);

        // Delete the single file, then leave the folder in place for IRIX to check.
        select(&mut b, "single.txt");
        no_errors(&answer(&mut b, Answer::Confirmed { purpose: P_DELETE, yes: true }));
        assert!(find(&b, "single.txt").is_none());

        // The filesystem's own check after all that.
        b.apply(BrowserAction::Check);
        let fx = settle(&mut b);
        for e in &fx {
            if let Effect::Alert { message, severity, .. } = e {
                eprintln!("check ({severity:?}): {message}");
                // Nothing new: clean, or exactly what was there before.
                assert!(*severity == Severity::Info || before.as_deref() == Some(message.as_str()), "{message}");
            }
        }
        b.close();
    }

    fn passwd_like_mode(_: &Editor) -> &'static str {
        "-rw-r--r--"
    }
}
