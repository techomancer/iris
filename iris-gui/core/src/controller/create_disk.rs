//! Making a blank disk image, as a form every shell draws from the same
//! field kinds as the settings (D8).
//!
//! The image is written on a job with progress and cancel; only its
//! attachment to the machine is part of the settings draft, so discarding
//! the draft keeps a created image for later. Nothing is ever replaced.

use std::path::PathBuf;
use std::sync::Arc;

use super::settings::{self, Field, FieldKind, Value};
use super::{Answer, Effect, Enabled, JobSnapshot, Notify, PickSpec};
use crate::model::disk_image::{self, Format};
use crate::model::job::{Job, Progress, Unit};

/// Widget ids of the form.
pub mod field {
    pub const FORMAT: &str = "create-disk.format";
    pub const SIZE: &str = "create-disk.size";
    pub const FILE: &str = "create-disk.file";
    pub const NOTE: &str = "create-disk.note";
}

/// The picker's purpose, so the app routes its answer back here.
pub const PICK: &str = "create-disk.pick";

#[derive(Debug, Clone, PartialEq)]
pub enum CreateDiskAction {
    Set { field: &'static str, value: Value },
    /// Ask where to put it.
    Pick,
    Create,
    Cancel,
    Close,
    Answer(Answer),
}

/// Making one blank image for one SCSI target.
pub struct CreateDiskController {
    /// The target the finished image attaches to.
    pub key: u8,
    /// Its name, for the window's title ("SCSI 1").
    pub name: String,
    format: Format,
    mb: u64,
    path: String,
    /// Where a relative name lands: the machine's folder.
    dir: Option<PathBuf>,
    job: Option<Job<PathBuf>>,
    notify: Arc<dyn Notify>,
    /// The image that was written, for the app to attach.
    done: Option<PathBuf>,
}

impl CreateDiskController {
    pub fn new(key: u8, name: String, dir: Option<PathBuf>, suggested: &str, notify: Arc<dyn Notify>) -> Self {
        Self {
            key,
            name,
            format: Format::Raw,
            mb: disk_image::DEFAULT_MB,
            path: suggested.to_string(),
            dir,
            job: None,
            notify,
            done: None,
        }
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// The form: the format, the size and where it goes.
    pub fn fields(&self) -> Vec<Field> {
        let formats: Vec<String> = Format::ALL
            .iter()
            .map(|f| match f {
                Format::Raw => t!("create-disk-format-raw"),
                Format::Chd => t!("create-disk-format-chd"),
            })
            .collect();
        let running = self.is_running();
        let on = |yes: bool| Enabled::when(yes, || t!("create-disk-running", file = self.path.as_str()));
        let mut f = vec![
            Field {
                id: field::FORMAT,
                label: t!("create-disk-format"),
                help: None,
                kind: FieldKind::Choice {
                    options: formats,
                    selected: Format::ALL.iter().position(|f| *f == self.format).unwrap_or(0),
                },
                enabled: on(!running),
            },
            Field {
                id: field::SIZE,
                label: t!("create-disk-size"),
                help: None,
                kind: FieldKind::Number {
                    value: self.mb as f64,
                    min: disk_image::MIN_MB as f64,
                    max: disk_image::MAX_MB as f64,
                    step: 256.0,
                    unit: t!("create-disk-size-unit"),
                },
                enabled: on(!running),
            },
            Field {
                id: field::FILE,
                label: t!("create-disk-file"),
                help: None,
                kind: FieldKind::Path { value: self.path.clone(), pick: settings::PickKind::File { filters: Vec::new() } },
                enabled: on(!running),
            },
        ];
        if let Some(why) = self.can_create().reason() {
            f.push(Field {
                id: field::NOTE,
                label: String::new(),
                help: None,
                kind: FieldKind::Note(super::Severity::Warning, why.to_string()),
                enabled: Enabled::Yes,
            });
        }
        f
    }

    /// Whether Create can run, and if not why.
    pub fn can_create(&self) -> Enabled {
        if self.is_running() {
            return Enabled::No(t!("create-disk-running", file = self.path.as_str()));
        }
        if self.path.trim().is_empty() {
            return Enabled::No(t!("create-disk-needs-file"));
        }
        if self.mb < disk_image::MIN_MB || self.mb > disk_image::MAX_MB {
            return Enabled::No(t!("create-disk-needs-size"));
        }
        let full = self.full_path();
        if let Some(why) = disk_image::refusal(&full) {
            return Enabled::No(why);
        }
        Enabled::Yes
    }

    /// Where the image goes: a relative name lands in the machine's folder.
    pub fn full_path(&self) -> PathBuf {
        let p = self.format.with_extension(std::path::Path::new(self.path.trim()));
        match (&self.dir, p.is_absolute()) {
            (Some(dir), false) => dir.join(p),
            _ => p,
        }
    }

    pub fn is_running(&self) -> bool {
        self.job.as_ref().is_some_and(|j| j.is_running())
    }

    pub fn job(&self) -> Option<JobSnapshot> {
        self.job.as_ref().map(|j| j.snapshot())
    }

    /// The finished image, taken once so the app attaches it just the once.
    pub fn take_done(&mut self) -> Option<PathBuf> {
        self.done.take()
    }

    pub fn apply(&mut self, action: CreateDiskAction) -> Vec<Effect> {
        let mut fx = Vec::new();
        match action {
            CreateDiskAction::Set { field: field::FORMAT, value: Value::Index(i) } => {
                if let Some(&f) = Format::ALL.get(i) {
                    self.format = f;
                    // Keep the name honest about what it will be.
                    if !self.path.trim().is_empty() {
                        self.path = f.with_extension(std::path::Path::new(&self.path)).to_string_lossy().into_owned();
                    }
                }
            }
            CreateDiskAction::Set { field: field::SIZE, value: Value::Number(mb) } => {
                self.mb = (mb.max(0.0) as u64).clamp(disk_image::MIN_MB, disk_image::MAX_MB);
            }
            CreateDiskAction::Set { field: field::FILE, value: Value::Text(p) } => self.path = p,
            CreateDiskAction::Set { .. } => {}
            CreateDiskAction::Pick => {
                let name = self.full_path().file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                fx.push(Effect::PickFile(
                    PickSpec::save(PICK, t!("create-disk-title"), name).in_dir(self.dir.clone()),
                ));
            }
            CreateDiskAction::Answer(Answer::Picked { purpose: PICK, paths }) => {
                if let Some(p) = paths.into_iter().next() {
                    // The dialog may offer to replace a file; we never do.
                    self.path = p.to_string_lossy().into_owned();
                    if let Some(f) = Format::of(&p) {
                        self.format = f;
                    }
                }
            }
            CreateDiskAction::Answer(_) => {}
            CreateDiskAction::Create => {
                if !self.can_create().is_yes() {
                    return fx;
                }
                let (path, format, mb) = (self.full_path(), self.format, self.mb);
                let label = t!("create-disk-running", file = path.display().to_string());
                self.job = Some(Job::spawn(label, true, self.notify.clone(), move |ctx| {
                    let mut progress = |f: f32| {
                        ctx.progress(Progress::Determinate { done: (f * 1000.0) as u64, total: 1000, unit: Unit::Items })
                    };
                    disk_image::create(&path, format, mb, &mut progress, &|| ctx.cancelled()).map(|()| path)
                }));
            }
            CreateDiskAction::Cancel => {
                if let Some(j) = &self.job {
                    j.cancel();
                }
            }
            CreateDiskAction::Close => {}
        }
        fx
    }

    /// Drain a finished job; the effects are its toast or its error.
    pub fn tick(&mut self) -> Vec<Effect> {
        let mut fx = Vec::new();
        let Some(job) = &self.job else { return fx };
        if job.is_running() {
            return fx;
        }
        // Cancelled: the form stays up for another go, with nothing to report.
        if job.snapshot().status == crate::model::job::Status::Cancelled {
            self.job = None;
            return fx;
        }
        match job.outcome() {
            Some(Ok(path)) => {
                fx.push(Effect::Toast(t!("create-disk-done", file = path.display().to_string())));
                self.done = Some(path);
                self.job = None;
            }
            Some(Err(e)) => {
                fx.push(Effect::Alert {
                    severity: super::Severity::Error,
                    title: t!("create-disk-title"),
                    message: e,
                });
                self.job = None;
            }
            None => {}
        }
        fx
    }
}
