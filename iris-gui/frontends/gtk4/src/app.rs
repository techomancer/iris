//! The application: `AdwApplication` and its one app controller, and the loop
//! of action -> effects -> refresh.
//!
//! The controller lives in a thread-local cell, borrowed only to apply an
//! action or tick and to refresh views. Effects run with no borrow held: a
//! dialog spawned from one answers later, through another action.

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::time::{Duration, Instant};

use adw::prelude::*;
use gtk4 as gtk;
use gtk::glib;
use iris_gui_core::controller::app::{AppAction, AppController};
use iris_gui_core::controller::commands::ShellCommand;
use iris_gui_core::controller::{Answer, Controller, Effect, Enabled};
use libadwaita as adw;

use crate::menu;
use crate::wake::GlibWake;
use crate::window::MainWindow;

/// The application id. `GApplication`'s own uniqueness hangs off it: a second
/// `iris-gtk` hands its activation to the first, which presents its window.
/// It is not the egui app's id, so both can be installed side by side.
pub const APP_ID: &str = "io.techomancer.iris.gtk";

struct Shell {
    app: AppController,
    win: MainWindow,
    wake: Arc<GlibWake>,
    /// The controller agreed to quit: the next close goes through.
    may_quit: bool,
}

thread_local! {
    static SHELL: RefCell<Option<Shell>> = const { RefCell::new(None) };
    /// `IRIS_GTK_SCRIPT`, one step at a time (see `script_step`).
    static SCRIPT: RefCell<std::collections::VecDeque<String>> = RefCell::default();
    static APP: RefCell<Option<adw::Application>> = const { RefCell::new(None) };
    /// The shell has been built. Not `SHELL.is_some()`: that cell may be
    /// borrowed when a second launch arrives, and reading it as "no shell"
    /// would start a second controller.
    static STARTED: Cell<bool> = const { Cell::new(false) };
    /// The one pending timed tick: when it fires, and its source.
    static NEXT_TICK: Cell<Option<Instant>> = const { Cell::new(None) };
    static TICK_SOURCE: RefCell<Option<glib::SourceId>> = const { RefCell::new(None) };
}

/// Run `f` with the shell borrowed; `None` when it is busy (a handler that
/// ticks the app underneath) or not built yet.
fn with<R>(f: impl FnOnce(&mut Shell) -> R) -> Option<R> {
    SHELL.with(|s| s.try_borrow_mut().ok().and_then(|mut g| g.as_mut().map(f)))
}

/// Run `f` from an idle callback of its own, once the handler on the stack
/// returned.
pub fn later(f: impl FnOnce() + 'static) {
    glib::source::idle_add_local_once(f);
}

/// The main window, for dialogs and launchers that want a parent.
pub fn window() -> Option<gtk::Window> {
    with(|s| s.win.window.clone().upcast()).or_else(|| {
        // Mid-borrow: ask the application instead.
        APP.with(|a| a.borrow().as_ref().and_then(|app| app.active_window()))
    })
}

pub fn toast(text: &str) {
    let text = text.to_string();
    if with(|s| s.win.toast(&text)).is_none() {
        later(move || toast(&text));
    }
}

/// Run the application until it quits.
pub fn run() -> glib::ExitCode {
    let app = adw::Application::builder().application_id(APP_ID).build();
    APP.with(|a| *a.borrow_mut() = Some(app.clone()));
    app.connect_activate(|app| {
        // A second launch handed its activation to us: show the window.
        if STARTED.with(|f| f.get()) {
            if with(|s| s.win.window.present()).is_none() {
                later(|| {
                    with(|s| s.win.window.present());
                });
            }
            return;
        }
        // Another IRIS front end (the egui app, or a shell under another
        // name) holds the pidfile. Launching would take over its ports and
        // its library, and there is no way to raise another process's window
        // here, so say where it is and leave it alone.
        if let Some(pid) = AppController::other_instance() {
            eprintln!("iris-gtk: IRIS is already running (pid {pid}); bring that window forward");
            app.quit();
            return;
        }
        start(app);
    });
    app.connect_shutdown(|_| {
        with(|s| s.app.shutdown());
        AppController::release_instance();
    });
    app.run()
}

fn start(app: &adw::Application) {
    let wake = Arc::new(GlibWake::default());
    let ctl = AppController::launch(wake.clone());
    // Frames wake the shell like any event: the sink calls the same wake.
    let (head0, head1) = ctl.session().frame_sinks();
    head0.set_wake(wake.clone());
    head1.set_wake(wake.clone());
    menu::window_actions(app);
    let win = MainWindow::new(app);
    win.window.present();
    SHELL.with(|s| *s.borrow_mut() = Some(Shell { app: ctl, win, wake, may_quit: false }));
    STARTED.with(|f| f.set(true));
    refresh();
    tick();
    // A test script: steps separated by `;`, one every few seconds
    // (`IRIS_GTK_SCRIPT_DELAY`, default 3), so a session can drive the shell
    // and look at it unattended. Slice 1 knows `shot:PATH` and `quit`;
    // later slices add command ids and the rest.
    if let Ok(script) = std::env::var("IRIS_GTK_SCRIPT") {
        SCRIPT.with(|q| q.borrow_mut().extend(script.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty())));
        schedule_script_step();
    }
}

fn schedule_script_step() {
    let delay: f64 = std::env::var("IRIS_GTK_SCRIPT_DELAY").ok().and_then(|d| d.parse().ok()).unwrap_or(3.0);
    glib::source::timeout_add_local_once(Duration::from_secs_f64(delay.max(0.0)), script_step);
}

/// Run the next scripted step and schedule the one after.
fn script_step() {
    let Some(step) = SCRIPT.with(|q| q.borrow_mut().pop_front()) else { return };
    eprintln!("iris-gtk script: {step}");
    if let Some(path) = step.strip_prefix("shot:") {
        let path = path.to_string();
        match with(|s| s.win.screenshot(&path)) {
            Some(Ok(())) => eprintln!("iris-gtk script: wrote {path}"),
            Some(Err(e)) => eprintln!("iris-gtk script: no screenshot: {e}"),
            None => eprintln!("iris-gtk script: busy, no screenshot"),
        }
    } else if let Some(text) = step.strip_prefix("type:") {
        crate::effects::script_type(&text.replace("\\n", "\n"));
    } else if let Some(r) = step.strip_prefix("answer:") {
        crate::effects::script_answer(r);
    } else if step == "dump:menu" {
        menu::dump();
    } else if step == "quit" {
        dispatch(AppAction::RequestQuit);
    } else {
        command(&step);
    }
    schedule_script_step();
}

/// A wake: drain the controller, carry out what it asks, redraw.
pub fn tick() {
    let fx = with(|s| {
        s.wake.ticked();
        let fx = s.app.tick();
        // Drained so they do not pile up; the display slice acts on them.
        let _ = s.app.take_session_events();
        fx
    });
    match fx {
        Some(fx) => {
            run_effects(fx);
            refresh();
        }
        // Busy: try again from a callback of its own.
        None => later(tick),
    }
}

/// The timed tick fired: none is pending now.
fn timed_tick() {
    NEXT_TICK.with(|n| n.set(None));
    TICK_SOURCE.with(|s| *s.borrow_mut() = None);
    tick();
}

pub fn refresh() {
    let due = with(|s| {
        s.win.refresh(&s.app);
        let app = &s.app;
        menu::refresh_enabled(|id| app.command_enabled(id) == Enabled::Yes);
        app.wake_due_in()
    })
    .flatten();
    // Nothing wakes a quiet shell, so the autosave pause and the stop after
    // IRIX powers off need a tick of their own
    // (clanker/gui/a-sleeping-shell-needs-a-timed-tick-for-autosave.md).
    // One at a time: a refresh asks again on every event, and a fresh request
    // each time would start another self-renewing chain. Keep the pending one
    // if it comes as soon; else replace it.
    let Some(d) = due else { return };
    let at = Instant::now() + d + Duration::from_millis(50);
    if NEXT_TICK.with(|n| n.get()).is_some_and(|p| p <= at) {
        return;
    }
    if let Some(old) = TICK_SOURCE.with(|s| s.borrow_mut().take()) {
        old.remove();
    }
    let source = glib::source::timeout_add_local_once(at.saturating_duration_since(Instant::now()), timed_tick);
    NEXT_TICK.with(|n| n.set(Some(at)));
    TICK_SOURCE.with(|s| *s.borrow_mut() = Some(source));
}

/// Apply an app action, then carry out its effects with no borrow held.
pub fn dispatch(action: AppAction) {
    let Some(fx) = with(|s| s.app.apply(action)) else {
        return;
    };
    run_effects(fx);
    refresh();
}

pub fn run_effects(fx: Vec<Effect>) {
    if fx.is_empty() {
        return;
    }
    let out = crate::effects::run(fx);
    if let Some(page) = out.open_settings {
        // The settings window arrives with its own slice.
        eprintln!("iris-gtk: no settings window yet (page {page})");
    }
    if out.quit {
        quit();
    }
}

fn quit() {
    with(|s| s.may_quit = true);
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            app.quit();
        }
    });
}

/// The Diagnostics alert's purpose: the shell asked it, so the shell takes
/// the answer (its Copy button).
const DIAGNOSTICS: &str = "shell.diagnostics";

/// A question's answer, from a dialog or the test hook, back to whoever
/// asked. Queued: a dialog answers from inside its own callback, and the
/// scripted answers from inside `run_effects`.
pub fn answer(a: Answer) {
    later(move || {
        if a.purpose() == DIAGNOSTICS {
            if let Answer::Chose { index: 1, .. } = a {
                run_effects(vec![Effect::CopyToClipboard(iris_gui_core::controller::debug::diagnostics_text())]);
            }
            return;
        }
        dispatch(AppAction::Answer(a));
    });
}

/// A menu item, toolbar button or script step with this command id.
pub fn command(id: &str) {
    if let Some(page) = iris_gui_core::controller::help::HelpPage::from_id(id) {
        // The help windows arrive with their own slice.
        eprintln!("iris-gtk: no help window yet ({})", page.id());
        return;
    }
    let Some(routed) = with(|s| s.app.command(id)) else {
        return later({
            let id = id.to_string();
            move || command(&id)
        });
    };
    run_effects(routed.effects);
    if let Some(c) = routed.shell {
        shell_command(c);
    }
    refresh();
}

/// The commands that only change the shell's own views. The ones whose
/// window has not been built yet say so; each arrives with its slice.
fn shell_command(c: ShellCommand) {
    match c {
        ShellCommand::CloseWindow => {
            if let Some(w) = window() {
                w.close();
            }
        }
        ShellCommand::Diagnostics => {
            run_effects(vec![Effect::Choose(iris_gui_core::controller::ChoiceSpec {
                purpose: DIAGNOSTICS,
                title: t!("app-name"),
                message: iris_gui_core::controller::debug::diagnostics_text(),
                choices: vec![t!("common-ok"), t!("common-copy")],
                dismiss: 0,
                default: Some(0),
                destructive: Vec::new(),
            })]);
        }
        ShellCommand::ToggleLibrary
        | ShellCommand::Settings
        | ShellCommand::SavedStates
        | ShellCommand::DiscMenu
        | ShellCommand::NewMachine
        | ShellCommand::About
        | ShellCommand::AppSettings
        | ShellCommand::Benchmark => eprintln!("iris-gtk: {c:?} has no window yet"),
    }
}

/// The window's close button: the controller decides, and may have disks to
/// synchronize before it answers with `Effect::Quit`.
pub fn close_requested() -> glib::Propagation {
    if with(|s| s.may_quit).unwrap_or(true) {
        return glib::Propagation::Proceed;
    }
    later(|| dispatch(AppAction::RequestQuit));
    glib::Propagation::Stop
}
