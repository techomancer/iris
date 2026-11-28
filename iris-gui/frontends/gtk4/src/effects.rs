//! Carrying out the controllers' [`Effect`]s.
//!
//! Unlike AppKit's modal panels, every GTK dialog is asynchronous: the ones
//! that ask something are spawned on the main context and their [`Answer`]
//! comes back later as an action, so nothing here blocks and the caller gets
//! only what is settled at once.
//!
//! Slice 1 does the effects that need no dialog; the asking ones arrive with
//! the dialog host (slice 3) and log until then.

use adw::prelude::*;
use gtk4 as gtk;
use iris_gui_core::controller::Effect;
use libadwaita as adw;

/// What the shell itself must do once the effects have run.
#[derive(Default)]
pub struct Out {
    /// Every guard passed: leave the application.
    pub quit: bool,
    /// Show the machine settings at this page (`ids::settings`).
    pub open_settings: Option<&'static str>,
}

pub fn run(fx: Vec<Effect>) -> Out {
    let mut out = Out::default();
    for e in fx {
        match e {
            Effect::Toast(text) => crate::app::toast(&text),
            Effect::OpenUrl(url) => {
                gtk::UriLauncher::new(&url).launch(crate::app::window().as_ref(), gtk::gio::Cancellable::NONE, |r| {
                    if let Err(e) = r {
                        eprintln!("iris-gtk: opening the link failed: {e}");
                    }
                });
            }
            Effect::OpenPath(path) => {
                gtk::FileLauncher::new(Some(&gtk::gio::File::for_path(&path))).launch(
                    crate::app::window().as_ref(),
                    gtk::gio::Cancellable::NONE,
                    move |r| {
                        if let Err(e) = r {
                            eprintln!("iris-gtk: opening {} failed: {e}", path.display());
                        }
                    },
                );
            }
            Effect::RevealPath(path) => {
                gtk::FileLauncher::new(Some(&gtk::gio::File::for_path(&path))).open_containing_folder(
                    crate::app::window().as_ref(),
                    gtk::gio::Cancellable::NONE,
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
            // The dialog host (slice 3).
            Effect::PickFile(s) | Effect::PickFolder(s) => pending(s.purpose, &s.title),
            Effect::Confirm(s) => pending(s.purpose, &s.message),
            Effect::AskText(s) => pending(s.purpose, &s.prompt),
            Effect::Choose(s) => pending(s.purpose, &s.message),
            Effect::Alert { severity, title, message } => {
                eprintln!("iris-gtk: [{severity:?}] {title}: {message}");
            }
        }
    }
    out
}

fn pending(purpose: &str, what: &str) {
    eprintln!("iris-gtk: no dialog host yet, dropping {purpose}: {what}");
}
