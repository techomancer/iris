//! Carrying out the controllers' [`Effect`]s: alerts, questions, file
//! choosers (through the xdg portal), the clipboard and the host's launchers.
//!
//! Every GTK dialog is asynchronous, unlike AppKit's modal panels: [`run`]
//! returns only what is settled at once, and a question's [`Answer`] reaches
//! the controller later, through `app::answer`. Nothing here blocks, so a
//! test script keeps running while a dialog is up — but it cannot click one,
//! which is what `IRIS_GTK_ANSWERS` is for.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::PathBuf;

use adw::prelude::*;
use gtk::{gio, glib};
use gtk4 as gtk;
use iris_gui_core::controller::{Answer, ChoiceSpec, ConfirmSpec, Effect, PickSpec, Severity, TextSpec};
use libadwaita as adw;

/// What the shell itself must do once the effects have run.
#[derive(Default)]
pub struct Out {
    /// Every guard passed: leave the application.
    pub quit: bool,
    /// Show the machine settings at this page (`ids::settings`).
    pub open_settings: Option<&'static str>,
}

thread_local! {
    /// Answers for the next prompts, from `IRIS_GTK_ANSWERS` (a test hook
    /// beside `IRIS_GTK_SCRIPT`): `yes`, `no`, `cancel`, `text:VALUE`,
    /// `choose:N`, `pick:PATH`, separated by `;`. Prompts past the list show
    /// their real dialogs.
    static ANSWERS: RefCell<Option<VecDeque<String>>> = const { RefCell::new(None) };
    /// The question now on screen, for the script hook's `type:` and
    /// `answer:` steps, which drive the real dialog rather than skipping it.
    static OPEN: RefCell<Option<(adw::AlertDialog, Option<gtk::Entry>)>> = const { RefCell::new(None) };
}

/// Fill the open prompt's text field (a test script's `type:TEXT`).
pub fn script_type(text: &str) {
    OPEN.with(|o| match o.borrow().as_ref().and_then(|(_, e)| e.clone()) {
        Some(entry) => entry.set_text(text),
        None => eprintln!("iris-gtk script: no prompt is asking for text"),
    });
}

/// Press a response on the open question by its id (`ok`, `cancel`, or a
/// choice's index), as a test script's `answer:` step.
pub fn script_answer(response: &str) {
    let dialog = OPEN.with(|o| o.borrow().as_ref().map(|(d, _)| d.clone()));
    match dialog {
        // The bindings have no call for "press this response", so emit the
        // signal the buttons emit, then close as a button would.
        Some(d) if d.is_response_enabled(response) => {
            d.emit_by_name::<()>("response", &[&response]);
            d.close();
        }
        Some(_) => eprintln!("iris-gtk script: the question will not take {response}"),
        None => eprintln!("iris-gtk script: no question is open"),
    }
}

fn scripted() -> Option<String> {
    ANSWERS.with(|a| {
        let mut a = a.borrow_mut();
        let q = a.get_or_insert_with(|| {
            std::env::var("IRIS_GTK_ANSWERS")
                .map(|s| s.split(';').map(str::to_string).filter(|x| !x.is_empty()).collect())
                .unwrap_or_default()
        });
        let next = q.pop_front();
        if let Some(n) = &next {
            eprintln!("iris-gtk answer: {n}");
        }
        next
    })
}

/// A prompt's scripted answer, when the hook has one left.
fn scripted_answer(e: &Effect) -> Option<Answer> {
    if !matches!(
        e,
        Effect::Confirm(_) | Effect::AskText(_) | Effect::Choose(_) | Effect::PickFile(_) | Effect::PickFolder(_)
    ) {
        return None;
    }
    let a = scripted()?;
    Some(match e {
        Effect::Confirm(s) => Answer::Confirmed { purpose: s.purpose, yes: a == "yes" },
        Effect::AskText(s) => Answer::Text { purpose: s.purpose, text: a.strip_prefix("text:").map(str::to_string) },
        Effect::Choose(s) => Answer::Chose {
            purpose: s.purpose,
            index: a.strip_prefix("choose:").and_then(|n| n.parse().ok()).unwrap_or(s.dismiss),
        },
        Effect::PickFile(s) | Effect::PickFolder(s) => Answer::Picked {
            purpose: s.purpose,
            paths: a.strip_prefix("pick:").map(|p| p.split(',').map(PathBuf::from).collect()).unwrap_or_default(),
        },
        _ => return None,
    })
}

pub fn run(fx: Vec<Effect>) -> Out {
    let mut out = Out::default();
    for e in fx {
        if let Some(a) = scripted_answer(&e) {
            crate::app::answer(a);
            continue;
        }
        match e {
            Effect::PickFile(spec) => pick(spec, false),
            Effect::PickFolder(spec) => pick(spec, true),
            Effect::Confirm(spec) => confirm(spec),
            Effect::Choose(spec) => choose(spec),
            Effect::AskText(spec) => ask_text(spec),
            Effect::Alert { severity, title, message } => alert(severity, &title, &message),
            Effect::Toast(text) => crate::app::toast(&text),
            Effect::OpenUrl(url) => {
                gtk::UriLauncher::new(&url).launch(crate::app::window().as_ref(), gio::Cancellable::NONE, |r| {
                    if let Err(e) = r {
                        eprintln!("iris-gtk: opening the link failed: {e}");
                    }
                });
            }
            Effect::OpenPath(path) => {
                gtk::FileLauncher::new(Some(&gio::File::for_path(&path))).launch(
                    crate::app::window().as_ref(),
                    gio::Cancellable::NONE,
                    move |r| {
                        if let Err(e) = r {
                            eprintln!("iris-gtk: opening {} failed: {e}", path.display());
                        }
                    },
                );
            }
            Effect::RevealPath(path) => {
                gtk::FileLauncher::new(Some(&gio::File::for_path(&path))).open_containing_folder(
                    crate::app::window().as_ref(),
                    gio::Cancellable::NONE,
                    move |r| {
                        if let Err(e) = r {
                            eprintln!("iris-gtk: showing {} failed: {e}", path.display());
                        }
                    },
                );
            }
            Effect::CopyToClipboard(text) => {
                if let Some(d) = gtk::gdk::Display::default() {
                    d.clipboard().set_text(&text);
                }
            }
            Effect::OpenSettings(page) => out.open_settings = Some(page),
            Effect::Quit => out.quit = true,
        }
    }
    out
}

/// A dialog with the main window as its parent, shown with no parent when the
/// window is not up yet (a startup effect).
fn present(dialog: &adw::AlertDialog, then: impl Fn(String) + 'static) {
    present_with(dialog, None, then)
}

/// Show `dialog`; `entry` is its text field, when it has one, so the script
/// hook can fill it.
fn present_with(dialog: &adw::AlertDialog, entry: Option<gtk::Entry>, then: impl Fn(String) + 'static) {
    let parent: Option<gtk::Widget> = crate::app::window().map(Cast::upcast);
    OPEN.with(|o| *o.borrow_mut() = Some((dialog.clone(), entry)));
    glib::spawn_future_local(glib::clone!(
        #[strong]
        dialog,
        async move {
            let response = dialog.clone().choose_future(parent.as_ref()).await;
            OPEN.with(|o| {
                let mine = o.borrow().as_ref().is_some_and(|(d, _)| *d == dialog);
                if mine {
                    *o.borrow_mut() = None;
                }
            });
            then(response.to_string());
        }
    ));
}

fn alert(severity: Severity, title: &str, message: &str) {
    let dialog = adw::AlertDialog::new(Some(title), Some(message));
    dialog.add_response("ok", &t!("common-ok"));
    dialog.set_default_response(Some("ok"));
    dialog.set_close_response("ok");
    if severity == Severity::Error {
        dialog.set_response_appearance("ok", adw::ResponseAppearance::Destructive);
    }
    present(&dialog, |_| {});
}

fn confirm(spec: ConfirmSpec) {
    let dialog = adw::AlertDialog::new(Some(&spec.title), Some(&spec.message));
    dialog.add_response("cancel", &t!("common-cancel"));
    dialog.add_response("ok", &spec.confirm_label);
    dialog.set_response_appearance(
        "ok",
        if spec.destructive { adw::ResponseAppearance::Destructive } else { adw::ResponseAppearance::Suggested },
    );
    dialog.set_default_response(Some(if spec.destructive { "cancel" } else { "ok" }));
    dialog.set_close_response("cancel");
    let purpose = spec.purpose;
    present(&dialog, move |r| crate::app::answer(Answer::Confirmed { purpose, yes: r == "ok" }));
}

/// The buttons in the spec's order; the response id is the index, so the
/// answer needs no lookup.
fn choose(spec: ChoiceSpec) {
    let dialog = adw::AlertDialog::new(Some(&spec.title), Some(&spec.message));
    for (i, label) in spec.choices.iter().enumerate() {
        let id = i.to_string();
        dialog.add_response(&id, label);
        if spec.destructive.contains(&i) {
            dialog.set_response_appearance(&id, adw::ResponseAppearance::Destructive);
        } else if spec.default == Some(i) {
            dialog.set_response_appearance(&id, adw::ResponseAppearance::Suggested);
        }
    }
    if let Some(d) = spec.default {
        dialog.set_default_response(Some(&d.to_string()));
    }
    dialog.set_close_response(&spec.dismiss.to_string());
    let (purpose, dismiss) = (spec.purpose, spec.dismiss);
    present(&dialog, move |r| {
        crate::app::answer(Answer::Chose { purpose, index: r.parse().unwrap_or(dismiss) })
    });
}

/// A one-line prompt. The controller's own rule runs as the user types and
/// keeps OK switched off until it passes: a refused answer is simply dropped
/// by the controller, so the view must not let one through.
fn ask_text(spec: TextSpec) {
    let dialog = adw::AlertDialog::new(Some(&spec.title), None);
    dialog.add_response("cancel", &t!("common-cancel"));
    dialog.add_response("ok", &spec.ok_label);
    dialog.set_response_appearance("ok", adw::ResponseAppearance::Suggested);
    dialog.set_default_response(Some("ok"));
    dialog.set_close_response("cancel");

    let body = gtk::Box::new(gtk::Orientation::Vertical, 8);
    let entry = gtk::Entry::new();
    entry.set_text(&spec.initial);
    entry.set_activates_default(true);
    let prompt = gtk::Label::new(Some(&spec.prompt));
    prompt.set_xalign(0.0);
    let error = gtk::Label::new(None);
    error.set_xalign(0.0);
    error.add_css_class("error");
    error.set_wrap(true);
    body.append(&prompt);
    body.append(&entry);
    body.append(&error);
    dialog.set_extra_child(Some(&body));

    let check = {
        let (dialog, error) = (dialog.clone(), error.clone());
        let spec = std::rc::Rc::new(spec.clone());
        move |text: &str| {
            match spec.validate(text) {
                Ok(()) => {
                    dialog.set_response_enabled("ok", true);
                    error.set_visible(false);
                }
                Err(why) => {
                    dialog.set_response_enabled("ok", false);
                    error.set_label(&why);
                    error.set_visible(!why.is_empty());
                }
            }
        }
    };
    check(&entry.text());
    entry.connect_changed(glib::clone!(
        #[strong]
        check,
        move |e| check(&e.text())
    ));

    let purpose = spec.purpose;
    present_with(&dialog, Some(entry.clone()), move |r| {
        let text = (r == "ok").then(|| entry.text().trim().to_string());
        crate::app::answer(Answer::Text { purpose, text });
    });
}

/// A file or folder chooser. `GtkFileDialog` goes through the xdg portal, so
/// it needs no libdbus of our own
/// (clanker/gui/rfd-linux-dlopens-libdbus-bundle-it-in-the-appimage.md).
fn pick(spec: PickSpec, folder: bool) {
    let dialog = gtk::FileDialog::builder().title(&spec.title).modal(true).build();
    if !spec.file_name.is_empty() {
        dialog.set_initial_name(Some(&spec.file_name));
    }
    if let Some(dir) = &spec.start_dir {
        dialog.set_initial_folder(Some(&gio::File::for_path(dir)));
    }
    if !folder && !spec.filters.is_empty() {
        let filters = gio::ListStore::new::<gtk::FileFilter>();
        for (name, exts) in &spec.filters {
            let f = gtk::FileFilter::new();
            f.set_name(Some(name));
            for e in exts {
                f.add_suffix(e.trim_start_matches('.'));
            }
            filters.append(&f);
        }
        dialog.set_filters(Some(&filters));
    }
    let parent = crate::app::window();
    let purpose = spec.purpose;
    let (save, multiple) = (spec.save, spec.multiple);
    glib::spawn_future_local(async move {
        let parent = parent.as_ref();
        let paths = if folder {
            one(dialog.select_folder_future(parent).await)
        } else if save {
            one(dialog.save_future(parent).await)
        } else if multiple {
            match dialog.open_multiple_future(parent).await {
                Ok(list) => (0..list.n_items())
                    .filter_map(|i| list.item(i).and_downcast::<gio::File>())
                    .filter_map(|f| f.path())
                    .collect(),
                Err(e) => {
                    dismissed(&e);
                    Vec::new()
                }
            }
        } else {
            one(dialog.open_future(parent).await)
        };
        crate::app::answer(Answer::Picked { purpose, paths });
    });
}

fn one(r: Result<gio::File, glib::Error>) -> Vec<PathBuf> {
    match r {
        Ok(f) => f.path().into_iter().collect(),
        Err(e) => {
            dismissed(&e);
            Vec::new()
        }
    }
}

/// A cancelled chooser is an answer, not a fault; anything else is worth a line.
fn dismissed(e: &glib::Error) {
    if !e.matches(gtk::DialogError::Dismissed) {
        eprintln!("iris-gtk: the file chooser failed: {e}");
    }
}
