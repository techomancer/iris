//! Controllers: one per task, shared by every view.
//!
//! A view (the egui app, the AppKit shell, later GTK4 and Win32) owns a
//! controller, sends it user intents as `Action`s, draws from its read
//! accessors, and carries out the [`Effect`]s it returns (file pickers,
//! confirmations, alerts, the clipboard), replying with another `Action` where
//! the effect has an answer.
//!
//! Rules (rusty-backup's CONTRIBUTING.md "Architecture policy (MVC)", adopted
//! by `clanker/plans/iris-newgui-plan.md`):
//! - No toolkit here: no egui, eframe, rfd, objc2, gtk or Win32 UI.
//!   `iris-gui/scripts/check-boundaries.sh` enforces it.
//! - No raw threads: slow work is a [`crate::model::job::Job`].
//! - `Action` and [`Effect`] enums are exhaustive and views `match` them with no
//!   `_` arm, so a new variant fails to compile in every view that ignores it.
//! - Every actionable element has a stable widget id ([`ids`]), which each view
//!   sets as its accessibility identifier.
//! - Shortcuts are data in [`keys`], never decided in a view.
//! - Controls are icons ([`icons`]) with tooltips ([`labels`]); the shell picks
//!   the glyph.

pub mod app;
pub mod bench;
pub mod camera;
pub mod commands;
pub mod convert;
pub mod create_disk;
pub mod debug;
pub mod disk_browser;
pub mod display;
pub mod icons;
pub mod ids;
pub mod keys;
pub mod labels;
pub mod network;
pub mod new_machine;
pub mod overview;
mod power;
pub mod help;
pub mod preferences;
pub mod profile;
pub mod serial;
pub mod session;
pub mod settings;

pub use crate::model::job::{NoWake, Notify, Progress, Snapshot as JobSnapshot, Status as JobStatus, Unit};

use std::path::PathBuf;

/// The shape every controller shares.
pub trait Controller {
    /// User intents, such as `PowerOn`, `Select(name)` or `Cancel`.
    type Action;

    /// Handle one user intent. Never blocks: slow work starts a job.
    fn apply(&mut self, action: Self::Action) -> Vec<Effect>;

    /// Called by the shell when woken or about to draw: drains finished jobs
    /// and worker events.
    fn tick(&mut self) -> Vec<Effect>;
}

/// Whether a control can be used now, and if not, why. The reason is the
/// tooltip a disabled control shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Enabled {
    Yes,
    No(String),
}

impl Enabled {
    pub fn is_yes(&self) -> bool {
        matches!(self, Enabled::Yes)
    }

    /// The reason a control is disabled, if it is.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Enabled::Yes => None,
            Enabled::No(r) => Some(r),
        }
    }

    /// `Yes` when `ok`, otherwise `No(reason())`.
    pub fn when(ok: bool, reason: impl FnOnce() -> String) -> Self {
        if ok {
            Enabled::Yes
        } else {
            Enabled::No(reason())
        }
    }
}

/// How serious a message is; the shell picks the colour or icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

/// Something only the shell can do. Any answer comes back as the emitting
/// controller's matching `Action`, carrying the spec's `purpose`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// An open or save file dialog (`spec.save`).
    PickFile(PickSpec),
    PickFolder(PickSpec),
    Confirm(ConfirmSpec),
    /// A one-line text prompt.
    AskText(TextSpec),
    /// One of several buttons.
    Choose(ChoiceSpec),
    /// A transient message (status line or toast).
    Toast(String),
    /// A message the user must dismiss.
    Alert { severity: Severity, title: String, message: String },
    OpenUrl(String),
    /// Open a file with the host's default application (a machine's TOML in
    /// the user's editor).
    OpenPath(PathBuf),
    /// Show a file or folder in the host's file manager.
    RevealPath(PathBuf),
    CopyToClipboard(String),
    /// Show the machine settings at a page (`ids::settings`).
    OpenSettings(&'static str),
    /// Quit the application; every guard (running machine, disk sync) passed.
    Quit,
}

/// The user's answer to an [`Effect`] that asked something. Every controller
/// that emits such an effect has an `Answer(Answer)` action; the shell routes the
/// answer back to the controller that asked, carrying the spec's `purpose`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    /// A file or folder dialog closed; empty when cancelled.
    Picked { purpose: &'static str, paths: Vec<PathBuf> },
    Confirmed { purpose: &'static str, yes: bool },
    /// A text prompt closed; `None` when cancelled.
    Text { purpose: &'static str, text: Option<String> },
    /// A [`ChoiceSpec`] button, by index.
    Chose { purpose: &'static str, index: usize },
}

impl Answer {
    pub fn purpose(&self) -> &'static str {
        match self {
            Answer::Picked { purpose, .. }
            | Answer::Confirmed { purpose, .. }
            | Answer::Text { purpose, .. }
            | Answer::Chose { purpose, .. } => purpose,
        }
    }
}

/// A file or folder picker; `purpose` says which `Action` the answer belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickSpec {
    pub purpose: &'static str,
    pub title: String,
    /// `(name, extensions)` pairs; empty means any file.
    pub filters: Vec<(String, Vec<String>)>,
    pub save: bool,
    /// The name a save dialog suggests; empty for none.
    pub file_name: String,
    /// The folder the dialog opens in, when there is a sensible one.
    pub start_dir: Option<PathBuf>,
    /// Allow choosing several files.
    pub multiple: bool,
}

impl PickSpec {
    pub fn open(purpose: &'static str, title: impl Into<String>) -> Self {
        Self {
            purpose,
            title: title.into(),
            filters: Vec::new(),
            save: false,
            file_name: String::new(),
            start_dir: None,
            multiple: false,
        }
    }

    pub fn save(purpose: &'static str, title: impl Into<String>, file_name: impl Into<String>) -> Self {
        Self { save: true, file_name: file_name.into(), ..Self::open(purpose, title) }
    }

    pub fn filter(mut self, name: impl Into<String>, exts: &[&str]) -> Self {
        self.filters.push((name.into(), exts.iter().map(|e| e.to_string()).collect()));
        self
    }

    pub fn in_dir(mut self, dir: Option<PathBuf>) -> Self {
        self.start_dir = dir;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfirmSpec {
    pub purpose: &'static str,
    pub title: String,
    pub message: String,
    pub confirm_label: String,
    pub destructive: bool,
}

/// A question with several answers, such as Cancel / Send IRIX halt / Force stop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChoiceSpec {
    pub purpose: &'static str,
    pub title: String,
    pub message: String,
    /// The buttons in order; the answer is an index into this list.
    pub choices: Vec<String>,
    /// The answer Esc (or closing the dialog) gives.
    pub dismiss: usize,
    /// The button drawn as the default, if any.
    pub default: Option<usize>,
    /// Buttons drawn as destructive.
    pub destructive: Vec<usize>,
}

/// A one-line text prompt whose rule the controller defines and the view runs
/// as the user types. The controller runs it again on the answer: the view's
/// check is for feedback, the controller's is authoritative.
#[derive(Debug, Clone)]
pub struct TextSpec {
    pub purpose: &'static str,
    pub title: String,
    pub prompt: String,
    pub initial: String,
    pub ok_label: String,
    /// Names the answer may not take, for the validator.
    pub taken: Vec<String>,
    /// Why the controller refused the last answer; shown until the user edits.
    pub error: Option<String>,
    /// `Err("")` disables OK without showing a message.
    pub validator: fn(&TextSpec, &str) -> Result<(), String>,
}

impl TextSpec {
    pub fn validate(&self, text: &str) -> Result<(), String> {
        (self.validator)(self, text)
    }
}

// Spelled out: comparing fn pointers directly trips rustc's
// unpredictable-function-pointer-comparisons lint.
impl PartialEq for TextSpec {
    fn eq(&self, o: &Self) -> bool {
        self.purpose == o.purpose
            && self.title == o.title
            && self.prompt == o.prompt
            && self.initial == o.initial
            && self.ok_label == o.ok_label
            && self.taken == o.taken
            && self.error == o.error
            && self.validator as usize == o.validator as usize
    }
}

impl Eq for TextSpec {}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_empty(_: &TextSpec, s: &str) -> Result<(), String> {
        if s.trim().is_empty() {
            Err(String::new())
        } else {
            Ok(())
        }
    }

    #[test]
    fn text_spec_runs_the_controllers_validator() {
        let spec = TextSpec {
            purpose: "rename",
            title: "Rename".into(),
            prompt: "Name".into(),
            initial: "indy".into(),
            ok_label: "Rename".into(),
            taken: vec![],
            error: None,
            validator: non_empty,
        };
        assert!(spec.validate("octane").is_ok());
        assert_eq!(spec.validate("  "), Err(String::new()));
        assert_eq!(spec.clone(), spec);
    }

    #[test]
    fn enabled_carries_its_reason() {
        assert!(Enabled::when(true, || "x".into()).is_yes());
        assert_eq!(Enabled::when(false, || "running".into()).reason(), Some("running"));
    }
}
