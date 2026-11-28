//! Carry out the controllers' effects with AppKit: alerts, open and save
//! panels, the workspace (reveal, open), the pasteboard. Runs with no
//! controller borrowed, so a modal panel's run loop can tick the app freely.

use std::path::{Path, PathBuf};

use objc2::rc::Retained;
use objc2::MainThreadMarker;
use objc2_app_kit::{
    NSAlert, NSAlertFirstButtonReturn, NSAlertStyle, NSModalResponseOK, NSOpenPanel, NSPasteboard,
    NSPasteboardTypeString, NSSavePanel, NSTextField, NSWorkspace,
};
use objc2_foundation::{NSArray, NSPoint, NSRect, NSSize, NSString, NSURL};
use iris_gui_core::controller::{Answer, ChoiceSpec, ConfirmSpec, Effect, PickSpec, Severity, TextSpec};

use crate::ui::ns;

/// What the shell must do after the effects ran.
#[derive(Default)]
pub struct Outcome {
    pub answers: Vec<Answer>,
    pub toasts: Vec<String>,
    pub quit: bool,
    pub open_settings: Option<&'static str>,
}

thread_local! {
    /// Answers for the next prompts, from `IRIS_APPKIT_ANSWERS` (a test hook
    /// beside `IRIS_APPKIT_SCRIPT`): `yes`, `no`, `text:VALUE`, `cancel`,
    /// `choose:N`, `pick:PATH`, separated by `;`. Prompts past the list show
    /// their real panels.
    static ANSWERS: std::cell::RefCell<Option<std::collections::VecDeque<String>>> = const { std::cell::RefCell::new(None) };
}

fn scripted() -> Option<String> {
    ANSWERS.with(|a| {
        let mut a = a.borrow_mut();
        let q = a.get_or_insert_with(|| {
            std::env::var("IRIS_APPKIT_ANSWERS")
                .map(|s| s.split(';').map(|x| x.to_string()).filter(|x| !x.is_empty()).collect())
                .unwrap_or_default()
        });
        let next = q.pop_front();
        if let Some(n) = &next {
            eprintln!("iris-appkit answer: {n}");
        }
        next
    })
}

/// A prompt's scripted answer, when the hook has one.
fn scripted_answer(e: &Effect) -> Option<Answer> {
    let peek_ok = matches!(e, Effect::Confirm(_) | Effect::AskText(_) | Effect::Choose(_) | Effect::PickFile(_) | Effect::PickFolder(_));
    if !peek_ok {
        return None;
    }
    let a = scripted()?;
    Some(match e {
        Effect::Confirm(s) => Answer::Confirmed { purpose: s.purpose, yes: a == "yes" },
        Effect::AskText(s) => Answer::Text { purpose: s.purpose, text: a.strip_prefix("text:").map(str::to_string) },
        Effect::Choose(s) => Answer::Chose { purpose: s.purpose, index: a.strip_prefix("choose:").and_then(|n| n.parse().ok()).unwrap_or(s.dismiss) },
        Effect::PickFile(s) | Effect::PickFolder(s) => {
            Answer::Picked { purpose: s.purpose, paths: a.strip_prefix("pick:").map(|p| vec![PathBuf::from(p)]).unwrap_or_default() }
        }
        _ => return None,
    })
}

pub fn run(mtm: MainThreadMarker, effects: Vec<Effect>) -> Outcome {
    let mut out = Outcome::default();
    for e in effects {
        if let Some(a) = scripted_answer(&e) {
            out.answers.push(a);
            continue;
        }
        match e {
            Effect::PickFile(spec) => out.answers.push(Answer::Picked { purpose: spec.purpose, paths: pick(mtm, &spec, false) }),
            Effect::PickFolder(spec) => out.answers.push(Answer::Picked { purpose: spec.purpose, paths: pick(mtm, &spec, true) }),
            Effect::Confirm(spec) => out.answers.push(Answer::Confirmed { purpose: spec.purpose, yes: confirm(mtm, &spec) }),
            Effect::AskText(spec) => out.answers.push(Answer::Text { purpose: spec.purpose, text: ask_text(mtm, &spec) }),
            Effect::Choose(spec) => out.answers.push(Answer::Chose { purpose: spec.purpose, index: choose(mtm, &spec) }),
            Effect::Toast(t) => out.toasts.push(t),
            Effect::Alert { severity, title, message } => {
                let a = alert(mtm, &title, &message);
                a.setAlertStyle(match severity {
                    Severity::Info => NSAlertStyle::Informational,
                    Severity::Warning => NSAlertStyle::Warning,
                    Severity::Error => NSAlertStyle::Critical,
                });
                a.addButtonWithTitle(&ns(&t!("common-ok")));
                a.runModal();
            }
            Effect::OpenUrl(url) => {
                if let Some(u) = NSURL::URLWithString(&ns(&url)) {
                    NSWorkspace::sharedWorkspace().openURL(&u);
                }
            }
            Effect::OpenPath(path) => {
                NSWorkspace::sharedWorkspace().openURL(&file_url(&path));
            }
            Effect::RevealPath(path) => {
                let urls = NSArray::from_retained_slice(&[file_url(&path)]);
                NSWorkspace::sharedWorkspace().activateFileViewerSelectingURLs(&urls);
            }
            Effect::CopyToClipboard(text) => {
                let pb = NSPasteboard::generalPasteboard();
                pb.clearContents();
                // SAFETY: AppKit's own pasteboard type constant.
                pb.setString_forType(&ns(&text), unsafe { NSPasteboardTypeString });
            }
            Effect::OpenSettings(page) => out.open_settings = Some(page),
            Effect::Quit => out.quit = true,
        }
    }
    out
}

fn file_url(p: &Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&ns(&p.to_string_lossy()))
}

fn path_of(url: Option<Retained<NSURL>>) -> Option<PathBuf> {
    url.and_then(|u| u.path()).map(|p| PathBuf::from(p.to_string()))
}

#[allow(deprecated)] // allowedFileTypes: still works, and its replacement (UTType) is a framework of its own
fn pick(mtm: MainThreadMarker, spec: &PickSpec, folder: bool) -> Vec<PathBuf> {
    let exts: Vec<Retained<NSString>> =
        spec.filters.iter().flat_map(|(_, e)| e.iter()).map(|e| ns(e.trim_start_matches('.'))).collect();
    if spec.save && !folder {
        let panel = NSSavePanel::savePanel(mtm);
        panel.setMessage(Some(&ns(&spec.title)));
        panel.setCanCreateDirectories(true);
        if !spec.file_name.is_empty() {
            panel.setNameFieldStringValue(&ns(&spec.file_name));
        }
        if let Some(dir) = &spec.start_dir {
            panel.setDirectoryURL(Some(&file_url(dir)));
        }
        return if panel.runModal() == NSModalResponseOK { path_of(panel.URL()).into_iter().collect() } else { Vec::new() };
    }
    let panel = NSOpenPanel::openPanel(mtm);
    panel.setMessage(Some(&ns(&spec.title)));
    panel.setCanChooseFiles(!folder || spec.multiple);
    panel.setCanChooseDirectories(folder);
    panel.setCanCreateDirectories(folder);
    panel.setAllowsMultipleSelection(spec.multiple);
    if !folder && !exts.is_empty() {
        panel.setAllowedFileTypes(Some(&NSArray::from_retained_slice(&exts)));
    }
    if let Some(dir) = &spec.start_dir {
        panel.setDirectoryURL(Some(&file_url(dir)));
    }
    if panel.runModal() != NSModalResponseOK {
        return Vec::new();
    }
    panel.URLs().iter().filter_map(|u| u.path()).map(|p| PathBuf::from(p.to_string())).collect()
}

fn alert(mtm: MainThreadMarker, title: &str, message: &str) -> Retained<NSAlert> {
    let a = NSAlert::new(mtm);
    // Unbundled (cargo run), NSAlert would show the icon of the folder the
    // binary is in; the app's own icon is set at startup.
    if let Some(icon) = objc2_app_kit::NSApplication::sharedApplication(mtm).applicationIconImage() {
        // SAFETY: a valid image.
        unsafe { a.setIcon(Some(&icon)) };
    }
    a.setMessageText(&ns(title));
    a.setInformativeText(&ns(message));
    a
}

fn confirm(mtm: MainThreadMarker, spec: &ConfirmSpec) -> bool {
    let a = alert(mtm, &spec.title, &spec.message);
    a.addButtonWithTitle(&ns(&spec.confirm_label));
    a.addButtonWithTitle(&ns(&t!("common-cancel")));
    if spec.destructive {
        a.setAlertStyle(NSAlertStyle::Critical);
    }
    a.runModal() == NSAlertFirstButtonReturn
}

/// The buttons in the spec's order. AppKit draws the first on the right and
/// makes it the default, so the default choice goes first.
fn choose(mtm: MainThreadMarker, spec: &ChoiceSpec) -> usize {
    let a = alert(mtm, &spec.title, &spec.message);
    let mut order: Vec<usize> = (0..spec.choices.len()).collect();
    if let Some(d) = spec.default {
        order.retain(|i| *i != d);
        order.insert(0, d);
    }
    for &i in &order {
        let b = a.addButtonWithTitle(&ns(&spec.choices[i]));
        if spec.destructive.contains(&i) {
            b.setHasDestructiveAction(true);
        }
    }
    let r = (a.runModal() - NSAlertFirstButtonReturn) as usize;
    order.get(r).copied().unwrap_or(spec.dismiss)
}

/// A text prompt: the controller's rule runs on OK; an error keeps it open.
fn ask_text(mtm: MainThreadMarker, spec: &TextSpec) -> Option<String> {
    let mut error = spec.error.clone();
    let mut text = spec.initial.clone();
    loop {
        let msg = match &error {
            Some(e) if !e.is_empty() => format!("{}\n\n{e}", spec.prompt),
            _ => spec.prompt.clone(),
        };
        let a = alert(mtm, &spec.title, &msg);
        a.addButtonWithTitle(&ns(&spec.ok_label));
        a.addButtonWithTitle(&ns(&t!("common-cancel")));
        let field = NSTextField::initWithFrame(mtm.alloc(), NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(280.0, 24.0)));
        field.setStringValue(&ns(&text));
        a.setAccessoryView(Some(&field));
        a.window().setInitialFirstResponder(Some(&field));
        if a.runModal() != NSAlertFirstButtonReturn {
            return None;
        }
        text = field.stringValue().to_string();
        match spec.validate(&text) {
            Ok(()) => return Some(text.trim().to_string()),
            Err(e) => error = Some(e),
        }
    }
}
