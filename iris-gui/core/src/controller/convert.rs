//! Converting a machine's compressed CHD disk so that stopping it is quick
//! (`model::chd_convert`). A disk compressed with lzma or zstd, whose rebuild
//! after every stop takes minutes, is asked about when it is attached
//! (Settings, New Machine) and before the machine starts, and pointed out on
//! the machine page, until the user converts it or keeps it as it is
//! (`GuiSettings::convert_declined`). The
//! Devices menu offers it for any compressed CHD. zlib comes first: it stays
//! small. Uncompressed never rebuilds, but its file can be very large.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::model::chd_convert::{self, ChdFormat, ChdHeader};
use crate::model::job::Job;
use crate::model::machines;

use super::app::{AppAction, AppController};
use super::{Answer, ChoiceSpec, Effect, Enabled, JobSnapshot};

/// Which format a compressed disk converts to.
pub const CONVERT_FORMAT: &str = "app.convert-format";
/// Cancels a running conversion (the status bar's button).
pub const CANCEL_ID: &str = "app.convert-cancel";

/// A conversion in progress.
pub(crate) struct Conversion {
    path: String,
    label: String,
    format: ChdFormat,
    before: u64,
    job: Job<u64>,
    /// Asked for at Power On: start this machine when it is done.
    start_after: Option<String>,
}

/// Why the format question was asked, which sets its last choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Asking {
    /// Convert…: the formats and Cancel.
    Menu,
    /// The disk was just attached: the formats and Keep As Is.
    Attached,
    /// Power On: the formats, Start Without Converting and Cancel.
    Start,
}

/// A format question waiting for its answer.
pub(crate) struct Asked {
    key: u8,
    formats: Vec<ChdFormat>,
    asking: Asking,
    path: String,
}

/// A CHD just attached to the selected machine, waiting for the disk probe
/// to read its header.
pub(crate) struct AttachCheck {
    machine: String,
    key: u8,
    path: String,
    since: Instant,
}

/// How long an attached disk's header may take to read before it is let go.
const ATTACH_WAIT: Duration = Duration::from_secs(30);

impl AppController {
    /// The selected machine's `[scsi.KEY]` disk as a CHD: its label, path
    /// and header, once the disk probe has read it.
    fn chd_disk(&self, key: u8) -> Option<(String, String, ChdHeader)> {
        self.chd_disk_with(key, false)
    }

    /// As `chd_disk`, reading the header itself when the probe has not yet
    /// (an explicit Convert or Power On, which reads the disk anyway).
    fn chd_disk_now(&self, key: u8) -> Option<(String, String, ChdHeader)> {
        self.chd_disk_with(key, true)
    }

    fn chd_disk_with(&self, key: u8, now: bool) -> Option<(String, String, ChdHeader)> {
        let cfg = self.config();
        let d = cfg.disk(key)?;
        if d.is_cdrom() || d.path.trim().is_empty() {
            return None;
        }
        let dir = self.machine_dir()?;
        let path = machines::resolve(&dir, &d.path).to_string_lossy().into_owned();
        self.probe.readable(&path);
        let header = match self.probe.chd_header(&path) {
            Some(h) => h,
            None if now && !self.probe.checked(&path) => chd_convert::header(std::path::Path::new(&path))?,
            None => return None,
        };
        let label = t!("convert-disk-label", id = crate::model::hardware::scsi_name(key));
        Some((label, path, header))
    }

    /// Whether `[scsi.KEY]` is a compressed CHD, and so can be converted.
    pub fn convertible(&self, key: u8) -> bool {
        self.chd_disk(key).is_some_and(|(_, _, h)| h.compressed())
    }

    /// Whether the machine page should suggest converting `[scsi.KEY]`:
    /// compressed with a codec that is slow to rebuild, and not kept as it
    /// is by the user.
    pub fn suggest_convert(&self, key: u8) -> bool {
        self.chd_disk(key).is_some_and(|(_, path, h)| h.slow() && !self.prefs().convert_declined.contains(&path))
    }

    /// Stop suggesting to convert `[scsi.KEY]`.
    pub(crate) fn decline_convert(&mut self, key: u8, fx: &mut Vec<Effect>) {
        if let Some((_, path, _)) = self.chd_disk(key) {
            self.decline_path(path, fx);
        }
    }

    fn decline_path(&mut self, path: String, fx: &mut Vec<Effect>) {
        if self.prefs_mut().convert_declined.insert(path) {
            self.save_prefs(fx);
        }
    }

    /// The selected machine's hard disk paths (absolute), by key.
    pub(crate) fn disk_paths(&self) -> BTreeMap<u8, String> {
        let Some(dir) = self.machine_dir() else { return BTreeMap::new() };
        crate::model::hardware::scsi_disks(self.config())
            .into_iter()
            .filter(|(_, d)| !d.is_cdrom() && !d.path.trim().is_empty())
            .map(|(k, d)| (k, machines::resolve(&dir, &d.path).to_string_lossy().into_owned()))
            .collect()
    }

    /// Watch the CHDs that are in `disk_paths` now but were not in `before`:
    /// once their headers are read, a slow one is asked about.
    pub(crate) fn watch_attached(&mut self, before: &BTreeMap<u8, String>) {
        let Some(machine) = self.selected().map(str::to_string) else { return };
        for (key, path) in self.disk_paths() {
            let chd = path.to_ascii_lowercase().ends_with(".chd");
            if chd && before.get(&key) != Some(&path) {
                self.probe.readable(&path);
                self.attach_checks.retain(|c| c.path != path);
                self.attach_checks.push(AttachCheck { machine: machine.clone(), key, path, since: Instant::now() });
            }
        }
    }

    /// Ask about one attached disk whose header has been read.
    fn attach_tick(&mut self, fx: &mut Vec<Effect>) {
        if self.attach_checks.is_empty() || self.pending_convert.is_some() || self.conversion.is_some() {
            return;
        }
        let selected = self.selected().map(str::to_string);
        let paths = self.disk_paths();
        let mut ask = None;
        let mut checks = std::mem::take(&mut self.attach_checks);
        checks.retain(|c| {
            // The disk must still be where it was attached.
            if selected.as_deref() != Some(&c.machine) || paths.get(&c.key) != Some(&c.path) {
                return false;
            }
            match self.probe.chd_header(&c.path) {
                Some(h) => {
                    if ask.is_none() && h.slow() && !self.prefs().convert_declined.contains(&c.path) {
                        ask = Some(c.key);
                    }
                    false
                }
                None => !self.probe.checked(&c.path) && c.since.elapsed() < ATTACH_WAIT,
            }
        });
        self.attach_checks = checks;
        if let Some(key) = ask {
            if !self.session().is_running() {
                self.ask(key, Asking::Attached, fx);
            }
        }
    }

    /// Before Power On: a slow CHD the user has not answered for is asked
    /// about first. `true` when it asked (the start waits for the answer).
    pub(crate) fn ask_before_start(&mut self, fx: &mut Vec<Effect>) -> bool {
        if self.conversion.is_some() {
            return false;
        }
        let keys: Vec<u8> = self.config().disks().iter().map(|d| d.0).collect();
        for key in keys {
            let slow = self
                .chd_disk_now(key)
                .is_some_and(|(_, path, h)| h.slow() && !self.prefs().convert_declined.contains(&path));
            if slow {
                self.ask(key, Asking::Start, fx);
                return self.pending_convert.is_some();
            }
        }
        false
    }

    pub(crate) fn convert_enabled(&self) -> Enabled {
        if self.conversion.is_some() {
            return Enabled::No(t!("convert-busy"));
        }
        if self.session().is_running() {
            return Enabled::No(t!("convert-needs-stopped"));
        }
        self.can_change_machine()
    }

    /// The running conversion, for the status bar.
    pub fn conversion(&self) -> Option<JobSnapshot> {
        self.conversion.as_ref().map(|c| c.job.snapshot())
    }

    /// Ask which format `[scsi.KEY]` converts to.
    pub(crate) fn ask_convert(&mut self, key: u8, fx: &mut Vec<Effect>) {
        self.ask(key, Asking::Menu, fx);
    }

    fn ask(&mut self, key: u8, asking: Asking, fx: &mut Vec<Effect>) {
        if let Enabled::No(why) = self.convert_enabled() {
            fx.push(Effect::Toast(why));
            return;
        }
        let Some((label, path, h)) = self.chd_disk_now(key).filter(|(_, _, h)| h.compressed()) else {
            fx.push(Effect::Toast(t!("convert-not-compressed")));
            return;
        };
        let human = super::disk_browser::human;
        let now = h.codecs.join(", ");
        let mut message = t!("convert-intro", disk = label.as_str(), codecs = now.as_str(), size = human(h.file_bytes));
        let mut formats = Vec::new();
        let mut choices = Vec::new();
        if !h.zlib_only() {
            formats.push(ChdFormat::Zlib);
            choices.push(t!("convert-to-zlib"));
            message.push_str("\n\n");
            message.push_str(&t!("convert-zlib-body", size = human(chd_convert::room_needed(&h, ChdFormat::Zlib))));
        }
        formats.push(ChdFormat::Uncompressed);
        choices.push(t!("convert-to-uncompressed"));
        message.push_str("\n\n");
        message.push_str(&t!("convert-uncompressed-body", size = human(h.logical_bytes)));
        message.push_str("\n\n");
        message.push_str(&t!("convert-outro"));
        let title = match asking {
            Asking::Menu => {
                choices.push(t!("common-cancel"));
                t!("convert-title", disk = label.as_str())
            }
            Asking::Attached => {
                message.push_str("\n\n");
                message.push_str(&t!("convert-keep-body"));
                choices.push(t!("convert-keep"));
                t!("convert-title", disk = label.as_str())
            }
            Asking::Start => {
                message.push_str("\n\n");
                message.push_str(&t!("convert-start-body"));
                choices.push(t!("convert-start-anyway"));
                choices.push(t!("common-cancel"));
                t!("convert-start-title", disk = label.as_str())
            }
        };
        fx.push(Effect::Choose(ChoiceSpec {
            purpose: CONVERT_FORMAT,
            title,
            message,
            // Closing the question: Cancel, or Keep As Is after attaching.
            dismiss: choices.len() - 1,
            default: Some(0),
            destructive: Vec::new(),
            choices,
        }));
        self.pending_convert = Some(Asked { key, formats, asking, path });
    }

    /// The format question's answer.
    pub(crate) fn convert_answer(&mut self, a: &Answer, fx: &mut Vec<Effect>) -> bool {
        let Answer::Chose { purpose: CONVERT_FORMAT, index } = a else { return false };
        let Some(asked) = self.pending_convert.take() else { return true };
        // The answer is for the disk it asked about: if another machine is
        // selected now, or that slot holds another disk, act on nothing.
        let same = self.chd_disk_with(asked.key, false).is_some_and(|(_, path, _)| path == asked.path);
        if !same {
            if *index <= asked.formats.len() {
                fx.push(Effect::Toast(t!("convert-disk-changed")));
            }
            return true;
        }
        if let Some(&format) = asked.formats.get(*index) {
            let start_after = (asked.asking == Asking::Start).then(|| self.selected().map(str::to_string)).flatten();
            self.start_convert(asked.key, format, start_after, fx);
            return true;
        }
        // The choice after the formats: Keep As Is, or Start Without
        // Converting (which keeps it too); Cancel after that.
        if *index == asked.formats.len() {
            match asked.asking {
                Asking::Attached => self.decline_path(asked.path, fx),
                Asking::Start => {
                    self.decline_path(asked.path, fx);
                    self.power_on(false, fx);
                }
                Asking::Menu => {}
            }
        }
        true
    }

    fn start_convert(&mut self, key: u8, format: ChdFormat, start_after: Option<String>, fx: &mut Vec<Effect>) {
        if let Enabled::No(why) = self.convert_enabled() {
            fx.push(Effect::Toast(why));
            return;
        }
        let Some((label, path, h)) = self.chd_disk_now(key) else { return };
        // The browser holds the image open; it must not see the file change.
        if !self.close_browser(fx) {
            return;
        }
        let need = chd_convert::room_needed(&h, format);
        if let Some(free) = chd_convert::free_bytes(std::path::Path::new(&path)) {
            if free < need {
                let human = super::disk_browser::human;
                fx.push(Effect::Toast(t!("convert-no-room", need = human(need), free = human(free))));
                return;
            }
        }
        let job_path = PathBuf::from(&path);
        let Some(job_diff) = self.machine_dir().zip(self.config().disk(key)).map(|(dir, d)| machines::chd_diff(&dir, &d.path)) else {
            return;
        };
        let job = Job::spawn(t!("convert-running", disk = label.as_str()), true, self.notify(), move |ctx| {
            let mut progress = |f: f32| {
                ctx.progress(super::Progress::Determinate { done: (f * 1000.0) as u64, total: 1000, unit: super::Unit::Items })
            };
            chd_convert::convert(&job_path, &job_diff, format, &mut progress, &|| ctx.cancelled()).map_err(|e| e.to_string())
        });
        self.conversion = Some(Conversion { path, label, format, before: h.file_bytes, job, start_after });
    }

    pub(crate) fn cancel_convert(&mut self) {
        if let Some(c) = &self.conversion {
            c.job.cancel();
        }
    }

    /// A finished conversion's message; called every tick.
    pub(crate) fn convert_tick(&mut self, fx: &mut Vec<Effect>) {
        self.attach_tick(fx);
        let Some(c) = &mut self.conversion else { return };
        let snap = c.job.snapshot();
        if snap.is_running() {
            return;
        }
        let cancelled = snap.cancel_requested;
        let result = c.job.outcome();
        let Some(c) = self.conversion.take() else { return };
        self.probe.forget(&c.path);
        let human = super::disk_browser::human;
        let converted = matches!(result, Some(Ok(_)));
        fx.push(Effect::Toast(match result {
            Some(Ok(size)) => {
                let format = match c.format {
                    ChdFormat::Zlib => t!("convert-format-zlib"),
                    ChdFormat::Uncompressed => t!("convert-format-uncompressed"),
                };
                t!("convert-done", disk = c.label.as_str(), format = format, size = human(size), before = human(c.before))
            }
            _ if cancelled => t!("convert-cancelled", disk = c.label.as_str()),
            Some(Err(e)) => t!("convert-failed", disk = c.label.as_str(), error = e),
            None => t!("convert-failed", disk = c.label.as_str(), error = "?"),
        }));
        // Asked for at Power On: start now, if that machine is still the one.
        if converted && c.start_after.is_some() && c.start_after.as_deref() == self.selected() && !self.session().is_running() {
            self.power_on(false, fx);
        }
    }
}

/// The convert actions' ids map here from `commands.rs`.
pub(crate) fn action_for(id: &str) -> Option<AppAction> {
    if id == CANCEL_ID {
        return Some(AppAction::CancelConversion);
    }
    if let Some(key) = id.strip_prefix(super::ids::item::DECLINE_CONVERT).and_then(|k| k.parse().ok()) {
        return Some(AppAction::DeclineConvert { key });
    }
    let key = id.strip_prefix(super::ids::item::CONVERT_DISK)?.parse().ok()?;
    Some(AppAction::ConvertDisk { key })
}
