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
    /// The machine settings, while they are open.
    settings: Option<std::rc::Rc<crate::settings::SettingsDialog>>,
    /// The application preferences, while they are open.
    prefs: Option<std::rc::Rc<crate::prefs::PrefsDialog>>,
    /// The New Machine wizard, while it is open.
    new_machine: Option<std::rc::Rc<crate::new_machine::NewMachineDialog>>,
    /// The disk browser, shown while the controller has one.
    browser: Option<std::rc::Rc<crate::browser::BrowserWindow>>,
    /// The benchmark window; the shell owns its controller, as egui does.
    bench: Option<std::rc::Rc<crate::panels::BenchWindow>>,
    /// The windows that follow a controller of their own.
    serial: Option<std::rc::Rc<crate::panels::SerialWindow>>,
    net: Option<std::rc::Rc<crate::panels::NetWindow>>,
    camera: Option<std::rc::Rc<crate::panels::CameraWindow>>,
    help: Vec<crate::panels::HelpWindow>,
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

/// The application, for the menus and actions a view builds.
pub fn application() -> Option<adw::Application> {
    APP.with(|a| a.borrow().clone())
}

/// Select `name` and show it.
pub fn select_machine(name: String) {
    dispatch(AppAction::Select(name));
}

/// A command id that reports a settings control from a menu:
/// `setting:<control id>#<value>`.
pub const SETTING_PREFIX: &str = "setting:";

/// Open the machine settings, at `page` when one is named.
fn open_settings(page: Option<&str>) {
    let dialog = with(|s| {
        let d = s.settings.get_or_insert_with(crate::settings::SettingsDialog::new).clone();
        d.refresh(&s.app);
        d
    });
    let Some(dialog) = dialog else { return later(move || open_settings(None)) };
    if let Some(page) = page {
        dialog.show_page(page);
    }
    if let Some(window) = window() {
        dialog.dialog.present(Some(&window));
    }
}

/// A settings control reported a new value.
pub fn setting(id: String, value: iris_gui_core::controller::settings::Value) {
    // A text field commits when a redraw takes its focus away, which happens
    // inside the shell's borrow: report it once the redraw is done instead
    // of dropping what was typed.
    if SHELL.with(|s| s.try_borrow_mut().is_err()) {
        return later(move || setting(id, value));
    }
    let action = with(|s| s.settings.as_ref().and_then(|d| d.action(&id, value))).flatten();
    if let Some(a) = action {
        dispatch(AppAction::Settings(a));
    }
}

/// A serial console action.
pub fn dispatch_serial(action: iris_gui_core::controller::serial::SerialAction) {
    later(move || dispatch(AppAction::Serial(action)));
}

/// A networking action.
pub fn dispatch_network(action: iris_gui_core::controller::network::NetworkAction) {
    later(move || dispatch(AppAction::Network(action)));
}

/// The camera test window closed.
pub fn dispatch_camera() {
    later(|| dispatch(AppAction::Camera(iris_gui_core::controller::camera::CameraAction::Close)));
}

/// Show a help page, built now from the selected machine (it carries live
/// values); an open window for the page is rebuilt and brought forward.
fn show_help(page: iris_gui_core::controller::help::HelpPage) {
    let dialog = with(|s| {
        if let Some(i) = s.help.iter().position(|h| h.page == page) {
            s.help.remove(i).dialog.close();
        }
        let doc = s.app.help(page, env!("APP_VERSION"), env!("APP_COMMIT"));
        let w = crate::panels::HelpWindow::new(page, &doc);
        let dialog = w.dialog.clone();
        s.help.push(w);
        dialog
    });
    let Some(dialog) = dialog else { return later(move || show_help(page)) };
    if let Some(window) = window() {
        dialog.present(Some(&window));
    }
}

/// The panels follow their controllers: shown while open, gone after.
fn refresh_panels(s: &mut Shell) {
    fn mirror<W>(slot: &mut Option<std::rc::Rc<W>>, open: bool, make: impl FnOnce() -> std::rc::Rc<W>, dialog: impl Fn(&W) -> adw::Dialog) {
        match (open, slot.is_some()) {
            (true, false) => {
                let w = make();
                let d = dialog(&w);
                *slot = Some(w);
                if let Some(window) = window() {
                    d.present(Some(&window));
                }
            }
            (false, true) => {
                if let Some(w) = slot.take() {
                    dialog(&w).close();
                }
            }
            _ => {}
        }
    }
    mirror(&mut s.serial, s.app.serial().is_open(), crate::panels::SerialWindow::new, |w| w.dialog.clone());
    mirror(&mut s.net, s.app.network().check_open(), crate::panels::NetWindow::new, |w| w.dialog.clone());
    mirror(&mut s.camera, s.app.camera().is_open(), crate::panels::CameraWindow::new, |w| w.dialog.clone());
    if let Some(w) = &s.serial {
        w.refresh(&s.app);
    }
    if let Some(w) = &s.net {
        let check = s.app.net_check();
        w.refresh(check);
    }
    if let Some(w) = &s.camera {
        w.refresh(&s.app);
    }
    if let Some(w) = &s.bench {
        w.refresh(s.app.session().is_running());
    }
}

/// A benchmark action; the shell owns the controller, so it applies here.
pub fn bench(action: iris_gui_core::controller::bench::BenchAction) {
    let fx = with(|s| {
        let running = s.app.session().is_running();
        s.bench.as_ref().map(|w| w.ctl.borrow_mut().apply(action, running))
    })
    .flatten();
    if let Some(fx) = fx {
        run_effects(fx);
        refresh();
    }
}

/// Open the benchmark window.
fn open_bench() {
    let dialog = with(|s| {
        let notify: Arc<dyn iris_gui_core::controller::Notify> = s.wake.clone();
        let w = s.bench.get_or_insert_with(|| crate::panels::BenchWindow::new(notify)).clone();
        w.refresh(s.app.session().is_running());
        w
    });
    let Some(w) = dialog else { return later(open_bench) };
    if let Some(window) = window() {
        w.dialog.present(Some(&window));
    }
}

/// A disk-browser action.
pub fn browser(action: iris_gui_core::controller::disk_browser::BrowserAction) {
    dispatch(AppAction::Browser(action));
}

/// A browser control by its id; Save and Close carry the editor's text first.
pub fn browser_control(id: &str, value: bool) {
    use iris_gui_core::controller::disk_browser::BrowserAction as B;
    let Some(action) = crate::browser::control(id, value) else { return };
    if matches!(action, B::EditorSave | B::EditorClose) {
        let text = with(|s| s.browser.as_ref().map(|w| w.editor_text())).flatten();
        if let Some(text) = text {
            let changed = with(|s| s.app.browser().and_then(|b| b.editor()).is_some_and(|e| e.text != text)).unwrap_or(false);
            if changed {
                dispatch(AppAction::Browser(B::EditorChanged(text)));
            }
        }
    }
    browser(action);
}

/// The browser window's close button.
pub fn close_browser() {
    dispatch(AppAction::CloseBrowser);
}

/// The browser window is shown exactly while the controller has a browser.
fn refresh_browser(s: &mut Shell) {
    match (s.app.browser(), s.browser.clone()) {
        (Some(b), Some(w)) => w.refresh(b),
        (Some(b), None) => {
            let w = crate::browser::BrowserWindow::new();
            w.refresh(b);
            w.window.present();
            s.browser = Some(w);
        }
        (None, Some(w)) => {
            w.window.close();
            s.browser = None;
        }
        (None, None) => {}
    }
}

/// The toolbar's Saved States: restore, rename or delete one.
fn saved_states_menu() {
    use iris_gui_core::controller::ids::{item, toolbar as tb};
    let Some(app) = application() else { return };
    let states = with(|s| s.app.saved_states()).unwrap_or_default();
    let menu = gtk::gio::Menu::new();
    if states.is_empty() {
        let empty = gtk::gio::MenuItem::new(Some(&t!("ov-no-states")), None);
        menu.append_item(&empty);
    }
    for state in &states {
        let section = gtk::gio::Menu::new();
        for (prefix, label) in [
            (item::RESTORE_STATE, t!("ov-restore")),
            (item::RENAME_STATE, t!("ov-rename")),
            (item::DELETE_STATE, t!("ov-delete")),
        ] {
            let it = gtk::gio::MenuItem::new(Some(&label), None);
            it.set_detailed_action(&format!("app.{}", menu::action_for(&app, &item::id(prefix, &state.name))));
            section.append_item(&it);
        }
        menu.append_section(Some(&state.name), &section);
    }
    with(|s| s.win.popup_at(tb::SAVED_STATES, &menu));
}

/// The toolbar's Discs: what can be done with each CD-ROM now.
fn disc_menu() {
    use iris_gui_core::controller::ids::toolbar as tb;
    let Some(app) = application() else { return };
    let groups = with(|s| s.app.disk_menu()).unwrap_or_default();
    let menu = gtk::gio::Menu::new();
    for group in &groups {
        let section = gtk::gio::Menu::new();
        for e in &group.entries {
            let it = gtk::gio::MenuItem::new(Some(&e.label), None);
            it.set_detailed_action(&format!("app.{}", menu::action_for(&app, &e.id)));
            section.append_item(&it);
        }
        menu.append_section(Some(&group.title), &section);
    }
    with(|s| s.win.popup_at(tb::DISC, &menu));
}

/// Open the application preferences.
fn open_prefs() {
    let dialog = with(|s| {
        let d = s.prefs.get_or_insert_with(crate::prefs::PrefsDialog::new).clone();
        d.refresh(&s.app);
        d
    });
    let Some(dialog) = dialog else { return later(open_prefs) };
    if let Some(window) = window() {
        dialog.dialog.present(Some(&window));
    }
}

/// A preferences control reported a new value.
pub fn prefs_control(id: &str, value: iris_gui_core::controller::settings::Value) {
    if let Some(a) = iris_gui_core::controller::preferences::action_for(id, value) {
        dispatch(a);
    }
}

/// Open a fresh New Machine wizard.
fn open_new_machine() {
    let dialog = with(|s| {
        let d = crate::new_machine::NewMachineDialog::new();
        s.new_machine = Some(d.clone());
        d
    });
    let Some(dialog) = dialog else { return later(open_new_machine) };
    if let Some(window) = window() {
        dialog.dialog.present(Some(&window));
    }
}

/// A control in the New Machine wizard reported a value.
pub fn new_machine_control(id: String, value: iris_gui_core::controller::settings::Value) {
    if SHELL.with(|s| s.try_borrow_mut().is_err()) {
        return later(move || new_machine_control(id, value));
    }
    if let Some(d) = with(|s| s.new_machine.clone()).flatten() {
        d.control(&id, value);
    }
}

/// The machine page's tab changed; the shell keeps it across refreshes.
pub fn set_summary_tab(tab: &str) {
    let tab = tab.to_string();
    match with(|s| s.win.set_tab(&tab)) {
        Some(()) => refresh(),
        None => later(move || set_summary_tab(&tab)),
    }
}

/// Input and capture for the display: applied at once, without a full
/// redraw (motion arrives many times a second).
pub fn display(action: iris_gui_core::controller::display::DisplayAction) {
    // GTK reports focus and motion while the shell is changing views
    // (inside its borrow): apply them after, not never.
    if SHELL.with(|s| s.try_borrow_mut().is_err()) {
        return later(move || display(action));
    }
    let lost_focus = matches!(action, iris_gui_core::controller::display::DisplayAction::Focus(false));
    let fx = with(|s| {
        let before = (s.app.display().fullscreen(), s.app.display().detached());
        let fx = s.app.apply(AppAction::Display(action));
        let after = (s.app.display().fullscreen(), s.app.display().detached());
        if before != after {
            s.win.sync_display(&s.app);
        }
        fx
    });
    if let Some(fx) = fx {
        run_effects(fx);
    }
    // Losing focus starts the release grace: refresh, so a tick is due when
    // it runs out (nothing else may wake the shell).
    if lost_focus {
        refresh();
    }
}

/// A library row's own menu, or `None` while the shell is busy.
pub fn machine_menu(name: &str) -> Option<Vec<Vec<iris_gui_core::controller::commands::MenuEntry>>> {
    with(|s| s.app.machine_menu(name))
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
    crate::style::load();
    let wake = Arc::new(GlibWake::default());
    let ctl = AppController::launch(wake.clone());
    // Frames wake the shell like any event: the sink calls the same wake.
    let (head0, head1) = ctl.session().frame_sinks();
    head0.set_wake(wake.clone());
    head1.set_wake(wake.clone());
    menu::window_actions(app);
    let win = MainWindow::new(app);
    win.window.present();
    SHELL.with(|s| *s.borrow_mut() = Some(Shell { app: ctl, win, wake, may_quit: false, settings: None, prefs: None, new_machine: None, bench: None, browser: None, serial: None, net: None, camera: None, help: Vec::new() }));
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
        let text = text.replace("\\n", "\n");
        let in_editor = with(|s| s.browser.as_ref().map(|w| w.set_editor_text(&text))).flatten().is_some();
        if !in_editor {
            crate::effects::script_type(&text);
        }
    } else if let Some(r) = step.strip_prefix("answer:") {
        crate::effects::script_answer(r);
    } else if step == "dump:menu" {
        menu::dump();
    } else if let Some(row) = step.strip_prefix("open:").and_then(|r| r.parse().ok()) {
        browser(iris_gui_core::controller::disk_browser::BrowserAction::Activate(row));
    } else if let Some(rows) = step.strip_prefix("pick:") {
        browser(iris_gui_core::controller::disk_browser::BrowserAction::Select(
            rows.split(',').filter_map(|r| r.parse().ok()).collect(),
        ));
    } else if let Some(path) = step.strip_prefix("drop:") {
        browser(iris_gui_core::controller::disk_browser::BrowserAction::Drop {
            paths: vec![std::path::PathBuf::from(path)],
            into: None,
        });
    } else if step == "dump:browser" {
        with(|s| {
            if let Some(b) = s.app.browser() {
                println!("iris-gtk browser: {:?} read-only: {:?}", b.path(), b.read_only_reason());
                for r in b.rows() {
                    println!("  {} {} {} {}:{}", r.mode, r.size, r.name, r.owner, r.group);
                }
            } else {
                println!("iris-gtk browser: none open");
            }
        });
    } else if let Some((id, v)) = step.strip_prefix("set:").and_then(|s| s.split_once('=')) {
        // set:ID=i:N | b:0/1 | t:TEXT | n:NUMBER, as the control would report.
        use iris_gui_core::controller::settings::Value;
        let value = match v.split_once(':') {
            Some(("i", n)) => n.parse().ok().map(Value::Index),
            Some(("b", b)) => Some(Value::Bool(b == "1")),
            Some(("t", t)) => Some(Value::Text(t.to_string())),
            Some(("n", n)) => n.parse().ok().map(Value::Number),
            _ => None,
        };
        if let Some(value) = value {
            setting(id.to_string(), value);
        }
    } else if let Some(page) = step.strip_prefix("page:") {
        open_settings(Some(page));
    } else if let Some(name) = step.strip_prefix("select:") {
        select_machine(name.to_string());
    } else if let Some(tab) = step.strip_prefix("tab:") {
        set_summary_tab(tab);
    } else if step == "dump:library" {
        with(|s| {
            println!("iris-gtk library:");
            for g in s.app.library() {
                println!("  {} ({})", g.title, if g.collapsed { "collapsed" } else { "open" });
                for r in g.rows {
                    let dot = match &r.dot {
                        Some(iris_gui_core::controller::app::RowDot::Running) => " [running]".to_string(),
                        Some(iris_gui_core::controller::app::RowDot::Problem(p)) => format!(" [problem: {p}]"),
                        None => String::new(),
                    };
                    println!("    {}{} — {}{dot}", r.name, if r.selected { " *" } else { "" }, r.subtitle);
                }
            }
        });
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
        for e in s.app.take_session_events() {
            if e == iris_gui_core::controller::session::SessionEvent::Started {
                s.win.screen.clear();
            }
        }
        if s.app.session().is_running() {
            let (sink, _) = s.app.session().frame_sinks();
            s.win.screen.take(sink);
        }
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
        if let Some(d) = s.settings.clone() {
            d.refresh(&s.app);
        }
        if let Some(d) = s.prefs.clone() {
            d.refresh(&s.app);
        }
        refresh_browser(s);
        refresh_panels(s);
        // The camera preview has no wake of its own: poll it while it runs.
        let bench = s.bench.as_ref().is_some_and(|b| b.ctl.borrow().is_running()).then(|| Duration::from_secs(1));
        let camera = s.app.camera().is_open().then(|| Duration::from_millis(66)).or(bench);
        let app = &s.app;
        menu::refresh_enabled(|id| app.command_enabled(id) == Enabled::Yes);
        match (app.wake_due_in(), camera) {
            (Some(a), Some(c)) => Some(a.min(c)),
            (a, c) => a.or(c),
        }
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
        open_settings(Some(page));
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
        if let Some(d) = with(|s| s.new_machine.clone()).flatten() {
            if d.picked(&a) {
                return;
            }
        }
        if a.purpose().starts_with("bench.") {
            bench(iris_gui_core::controller::bench::BenchAction::Answer(a));
            return;
        }
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
    // The machine capsule's menu: switch to that machine.
    if let Some(name) = id.strip_prefix(crate::toolbar::SELECT_MACHINE) {
        return select_machine(name.to_string());
    }
    // A settings control a menu item stands for (Add Forward's suggestions).
    if let Some(rest) = id.strip_prefix(SETTING_PREFIX) {
        use iris_gui_core::controller::settings::Value;
        let (control, arg) = rest.split_once('#').unwrap_or((rest, ""));
        let value = match arg.parse::<usize>() {
            Ok(i) => Value::Index(i),
            Err(_) => Value::Bool(true),
        };
        return setting(control.to_string(), value);
    }
    if let Some(page) = iris_gui_core::controller::help::HelpPage::from_id(id) {
        return show_help(page);
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
        ShellCommand::ToggleLibrary => {
            with(|s| s.win.toggle_library());
        }
        ShellCommand::Settings => open_settings(None),
        ShellCommand::AppSettings => open_prefs(),
        ShellCommand::NewMachine => open_new_machine(),
        ShellCommand::SavedStates => saved_states_menu(),
        ShellCommand::DiscMenu => disc_menu(),
        ShellCommand::About => show_help(iris_gui_core::controller::help::HelpPage::About),
        ShellCommand::Benchmark => open_bench(),
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
