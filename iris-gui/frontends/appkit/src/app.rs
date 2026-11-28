//! The application: `NSApplication` and its delegate, the one app
//! controller, and the loop of action -> effects -> refresh.
//!
//! The controller lives in a thread-local cell. It is borrowed only to apply
//! an action or tick and to refresh views; effects (modal panels) run with no
//! borrow held, so a panel's run loop may tick the app underneath.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::{NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{define_class, msg_send, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSApplicationTerminateReply};
use objc2_foundation::NSNotification;
use iris_gui_core::controller::app::{AppAction, AppController};
use iris_gui_core::controller::commands::ShellCommand;
use iris_gui_core::controller::{Controller, Effect, Enabled};

use crate::wake::MacWake;
use crate::window::MainWindow;

struct Shell {
    app: AppController,
    win: MainWindow,
    wake: Arc<MacWake>,
    /// The controller agreed to quit: let the next terminate through.
    may_quit: bool,
    settings: Option<crate::settings::SettingsWindow>,
    new_machine: Option<crate::new_machine::NewMachineWindow>,
    browser: Option<crate::browser::BrowserWindow>,
    prefs: Option<crate::panels::PrefsWindow>,
    net: Option<crate::panels::NetWindow>,
    serial: Option<crate::panels::SerialWindow>,
    camera: Option<crate::panels::CameraWindow>,
    bench: Option<crate::panels::BenchWindow>,
    help: Vec<crate::panels::HelpWindow>,
    /// Held while the app works with no window in front (a machine running,
    /// disks synchronizing, a disk converting): without it App Nap slows the
    /// work several times over once the app is in the background.
    busy: Option<Retained<objc2::runtime::ProtocolObject<dyn objc2::runtime::NSObjectProtocol>>>,
}

thread_local! {
    /// Another instance to hand over to, found at launch.
    static HANDOVER: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
    static SHELL: RefCell<Option<Shell>> = const { RefCell::new(None) };
    static DELEGATE: RefCell<Option<Retained<Delegate>>> = const { RefCell::new(None) };
    static LATER: RefCell<VecDeque<Box<dyn FnOnce()>>> = RefCell::default();
    static SCRIPT: RefCell<VecDeque<String>> = RefCell::default();
    /// macOS asked to terminate and waits for `reply_terminate`.
    static TERMINATE_ASKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// When the one pending timed tick fires (`refresh`).
    static NEXT_TICK: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
}

fn mtm() -> MainThreadMarker {
    MainThreadMarker::new().expect("the shell runs on the main thread")
}

/// Run `f` with the shell borrowed; `None` when it is busy (a modal panel
/// ticked from inside a handler) or not built yet.
fn with<R>(f: impl FnOnce(&mut Shell) -> R) -> Option<R> {
    SHELL.with(|s| s.try_borrow_mut().ok().and_then(|mut g| g.as_mut().map(f)))
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISAppDelegate"]
    struct Delegate;

    unsafe impl NSObjectProtocol for Delegate {}

    unsafe impl NSApplicationDelegate for Delegate {
        #[unsafe(method(applicationDidFinishLaunching:))]
        fn did_finish_launching(&self, _note: &NSNotification) {
            if let Some(pid) = HANDOVER.with(|h| h.get()) {
                hand_over(pid);
            }
            start(self.mtm());
        }

        /// Activated (by a second launch handing over, or the app switcher)
        /// with no window showing: show the main window.
        #[unsafe(method(applicationDidBecomeActive:))]
        fn did_become_active(&self, _note: &NSNotification) {
            // Only when no IRIS window shows at all: a main window the user
            // closed stays closed while, say, the serial console is in use.
            let mtm = MainThreadMarker::from(self);
            let any = NSApplication::sharedApplication(mtm).windows().iter().any(|w| w.isVisible() && w.canBecomeMainWindow());
            if !any {
                with(|s| s.win.window.makeKeyAndOrderFront(None));
            }
        }

        /// Cmd+Q and the app menu's Quit come here, not through a window
        /// close: the controller decides (it may fold CHD diffs first and
        /// answer with Effect::Quit later).
        #[unsafe(method(applicationShouldTerminate:))]
        fn should_terminate(&self, _app: &NSApplication) -> NSApplicationTerminateReply {
            if with(|s| s.may_quit).unwrap_or(true) {
                return NSApplicationTerminateReply::TerminateNow;
            }
            // "Later", not "cancel": a logout or restart waits for the answer
            // (`reply_terminate`) instead of being cancelled outright.
            TERMINATE_ASKED.with(|t| t.set(true));
            later(Box::new(|| {
                dispatch(AppAction::RequestQuit);
                // Asked and declined (the quit question's Cancel): no quit is
                // under way and none happened, so say no.
                if TERMINATE_ASKED.with(|t| t.get()) && !with(|s| s.app.quit_pending()).unwrap_or(false) {
                    reply_terminate(false);
                }
            }));
            NSApplicationTerminateReply::TerminateLater
        }

        #[unsafe(method(applicationWillTerminate:))]
        fn will_terminate(&self, _note: &NSNotification) {
            with(|s| s.app.shutdown());
            AppController::release_instance();
        }

        /// The Dock icon clicked with no window showing: bring the main window back.
        #[unsafe(method(applicationShouldHandleReopen:hasVisibleWindows:))]
        fn reopen(&self, _app: &NSApplication, visible: bool) -> bool {
            if !visible {
                with(|s| s.win.window.makeKeyAndOrderFront(None));
            }
            true
        }

        /// The Mac way: closing the window (File > Close, Cmd+W) leaves the app,
        /// and any machine in it, running; Quit (Cmd+Q) ends it, and the
        /// Dock icon brings the window back.
        #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
        fn last_window_closed(&self, _app: &NSApplication) -> bool {
            false
        }
    }
);

/// Run the application until it quits. The app is single instance: when
/// another IRIS front end is running, bring that one forward and exit.
pub fn run() {
    let mtm = mtm();
    // Hand over once launched: only then does macOS make a user-launched app
    // active, and an active app may pass activation on.
    let other = AppController::other_instance();
    HANDOVER.with(|h| h.set(other));
    let ns_app = NSApplication::sharedApplication(mtm);
    ns_app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    let delegate: Retained<Delegate> = {
        let this = mtm.alloc::<Delegate>().set_ivars(());
        // SAFETY: NSObject's plain init.
        unsafe { msg_send![super(this), init] }
    };
    ns_app.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    set_icon(&ns_app);
    DELEGATE.with(|d| *d.borrow_mut() = Some(delegate));
    if other.is_none() {
        crate::menu::build(mtm);
    }
    ns_app.run();
}

/// The app's icon from the repository's `icon.icns`, for a bare binary
/// (`cargo run`), which would otherwise show the generic executable icon. A
/// bundle names its icon in Info.plist and is left alone: setting one here
/// would override the Dock's icon styles (tinted, clear, dark).
fn set_icon(app: &NSApplication) {
    if objc2_foundation::NSBundle::mainBundle().bundlePath().to_string().ends_with(".app") {
        return;
    }
    static ICNS: &[u8] = include_bytes!("../../../assets/icons/icon.icns");
    let data = objc2_foundation::NSData::with_bytes(ICNS);
    match objc2_app_kit::NSImage::initWithData(objc2::AnyThread::alloc(), &data) {
        // SAFETY: a valid image; AppKit keeps it.
        Some(img) => unsafe { app.setApplicationIconImage(Some(&img)) },
        None => eprintln!("iris-appkit: the app icon would not load"),
    }
}

/// Bring the app to the front: `activate` on macOS 14 and later, the older
/// call (deprecated there) on 11 to 13.
fn activate_app(mtm: MainThreadMarker) {
    let app = NSApplication::sharedApplication(mtm);
    // SAFETY: a plain check that the app knows the selector.
    let modern: bool = unsafe { objc2::msg_send![&*app, respondsToSelector: objc2::sel!(activate)] };
    if modern {
        app.activate();
    } else {
        #[allow(deprecated)]
        app.activateIgnoringOtherApps(true);
    }
}

fn start(mtm: MainThreadMarker) {
    let wake = Arc::new(MacWake::default());
    let app = AppController::launch(wake.clone());
    let win = MainWindow::new(mtm);
    // Frames wake the shell like any event: the sink calls the same wake.
    let (head0, head1) = app.session().frame_sinks();
    head0.set_wake(wake.clone());
    head1.set_wake(wake.clone());
    win.window.makeKeyAndOrderFront(None);
    let empty = app.is_empty();
    SHELL.with(|s| *s.borrow_mut() = Some(Shell { app, win, wake, may_quit: false, settings: None, new_machine: None, browser: None, prefs: None, net: None, serial: None, camera: None, bench: None, help: Vec::new(), busy: None }));
    activate_app(mtm);
    refresh();
    tick();
    // A test script: command ids, separated by `;`, one every few seconds
    // (`IRIS_APPKIT_SCRIPT_DELAY`, default 3). "select:NAME" selects a machine,
    // "page:ID" shows a settings section, "set:ID=VALUE" reports a settings control,
    // "open:N" and "pick:N,M" open and select rows of the disk browser,
    // "type:TEXT" fills the browser's editor (\n for a newline), "at:X,Y[,B]"
    // puts the pointer on a guest pixel in Follow mode,
    // "split:W" moves the sidebar's divider to W points,
    // "dump:devices" prints the Devices menu, "dump:browser" the browser's
    // folder, "drop:PATH" drops a host file on it. IRIS_APPKIT_ANSWERS (see
    // effects.rs) answers the prompts these raise.
    if let Ok(script) = std::env::var("IRIS_APPKIT_SCRIPT") {
        SCRIPT.with(|q| q.borrow_mut().extend(script.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty())));
        schedule_script_step();
    }
    if empty && std::env::var_os("IRIS_APPKIT_SCRIPT").is_none() {
        // The first run: offer to make a machine.
        later(Box::new(|| shell_command(ShellCommand::NewMachine)));
    }
}

/// Bring the running instance `pid` forward and exit. Exits directly: the
/// pidfile and the library are the other instance's.
fn hand_over(pid: i32) -> ! {
    use objc2_app_kit::{NSApplicationActivationOptions as O, NSRunningApplication};
    eprintln!("iris-appkit: IRIS is already running (pid {pid}); bringing it forward");
    if let Some(other) = NSRunningApplication::runningApplicationWithProcessIdentifier(pid) {
        let v14 = objc2_foundation::NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(
            objc2_foundation::NSOperatingSystemVersion { majorVersion: 14, minorVersion: 0, patchVersion: 0 },
        );
        if v14 {
            // macOS 14 activates another app only on the active app's word.
            other.activateFromApplication_options(&NSRunningApplication::currentApplication(), O::ActivateAllWindows);
        } else {
            #[allow(deprecated)] // the way below macOS 14
            other.activateWithOptions(O::ActivateIgnoringOtherApps | O::ActivateAllWindows);
        }
    }
    std::process::exit(0)
}

/// Full screen settled: tuck its status bar away.
pub fn tuck_status() {
    with(|s| s.win.tuck_status());
}

/// A toast's few seconds are up: clear it unless another replaced it.
pub fn toast_expired(n: u64) {
    with(|s| s.win.clear_toast(n));
}

/// Run `f` from a main-queue block of its own, once the handler on the stack returned.
pub fn later(f: Box<dyn FnOnce()>) {
    LATER.with(|l| l.borrow_mut().push_back(f));
    dispatch2::DispatchQueue::main().exec_async(|| {
        let next = LATER.with(|l| l.borrow_mut().pop_front());
        if let Some(f) = next {
            f();
        }
    });
}

fn schedule_script_step() {
    let delay: f64 = std::env::var("IRIS_APPKIT_SCRIPT_DELAY").ok().and_then(|d| d.parse().ok()).unwrap_or(3.0);
    let t = crate::ui::target(mtm());
    // SAFETY: the target is never dropped; the selector takes one object argument.
    unsafe {
        let _: () = objc2::msg_send![&*t, performSelector: objc2::sel!(scriptStep:), withObject: std::ptr::null::<objc2::runtime::AnyObject>(), afterDelay: delay];
    }
}

/// Run the next scripted command id and schedule the one after.
pub fn script_step() {
    let Some(step) = SCRIPT.with(|q| q.borrow_mut().pop_front()) else { return };
    eprintln!("iris-appkit script: {step}");
    if let Some(name) = step.strip_prefix("select:") {
        select_machine(name.to_string());
    } else if let Some((id, v)) = step.strip_prefix("set:").and_then(|s| s.split_once('=')) {
        // set:ID=i:N | b:0/1 | t:TEXT | n:NUMBER, as the control with ID would report.
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
    } else if let Some(at) = step.strip_prefix("at:") {
        // at:X,Y[,BUTTONS]: the host pointer over guest pixel (X, Y), as Follow mode sends it.
        let v: Vec<i32> = at.split(',').filter_map(|n| n.trim().parse().ok()).collect();
        if let [x, y, rest @ ..] = v.as_slice() {
            let buttons = rest.first().copied().unwrap_or(0) as u8;
            display(iris_gui_core::controller::display::DisplayAction::PointerAt { x: *x, y: *y, buttons, wheel_lines: 0.0, wheel_points: 0.0 });
        }
    } else if let Some(path) = step.strip_prefix("drop:") {
        browser_drop(vec![std::path::PathBuf::from(path)], None);
    } else if let Some(text) = step.strip_prefix("type:") {
        with(|s| s.browser.as_ref().map(|w| w.set_editor_text(&text.replace("\\n", "\n"))));
    } else if step == "dump:browser" {
        with(|s| {
            if let Some(b) = s.app.browser() {
                eprintln!("browser path: {:?} read-only: {:?}", b.path(), b.read_only_reason());
                for r in b.rows() {
                    eprintln!("  {} {} {} {}:{}", r.mode, r.size, r.name, r.owner, r.group);
                }
            }
        });
    } else if step == "dump:devices" {
        crate::menu::dump_devices(mtm());
    } else if let Some(w) = step.strip_prefix("split:").and_then(|w| w.parse::<f64>().ok()) {
        // The sidebar's divider moved to `w` points, as a drag would; the
        // width it ends at is printed.
        if let Some(now) = with(|s| s.win.drag_sidebar(w)) {
            eprintln!("iris-appkit script: sidebar is {now:.0} pt wide");
        }
    } else if let Some(z) = step.strip_prefix("hover:") {
        // The pointer over a hover zone (`chrome::Zone`), for tests.
        use crate::chrome::Zone;
        let (zone, inside) = match z {
            "library" => (Zone::LibraryEdge, true),
            "panel-out" => (Zone::LibraryPanel, false),
            "status" => (Zone::StatusEdge, true),
            "status-out" => (Zone::StatusBar, false),
            _ => return,
        };
        hover(zone, inside);
    } else if let Some(row) = step.strip_prefix("open:").and_then(|r| r.parse().ok()) {
        browser_activate(row);
    } else if let Some(rows) = step.strip_prefix("pick:") {
        browser_select(rows.split(',').filter_map(|r| r.parse().ok()).collect());
    } else if let Some(page) = step.strip_prefix("page:") {
        setting(format!("settings-page:{page}"), iris_gui_core::controller::settings::Value::Bool(true));
    } else {
        command(&step);
    }
    schedule_script_step();
}

/// A wake: drain the controller, carry out what it asks, present a new
/// frame, redraw.
/// `IRIS_APPKIT_TRACE` set: log ticks and effects to stderr.
fn trace(what: impl FnOnce() -> String) {
    thread_local! {
        static ON: bool = std::env::var_os("IRIS_APPKIT_TRACE").is_some();
    }
    if ON.with(|o| *o) {
        eprintln!("iris-appkit trace: {}", what());
    }
}

pub fn tick() {
    let fx = with(|s| {
        s.wake.ticked();
        let fx = s.app.tick();
        for e in s.app.take_session_events() {
            if e == iris_gui_core::controller::session::SessionEvent::Started {
                s.win.display.clear();
            }
        }
        if s.app.session().is_running() {
            let (sink, _) = s.app.session().frame_sinks();
            if s.win.display.take(sink) {
                s.win.display.draw();
            }
        }
        fx
    });
    match fx {
        Some(fx) => {
            if !fx.is_empty() {
                trace(|| format!("tick effects {fx:?}"));
            }
            run_effects(fx);
            refresh();
        }
        // Busy: try again from a block of its own.
        None => {
            trace(|| "tick busy, again later".into());
            later(Box::new(tick))
        }
    }
}

/// Hold or let go of the App Nap exemption as the work starts and ends.
fn keep_awake(s: &mut Shell) {
    let working = s.app.session().is_running()
        || s.app.session().syncing().is_some()
        || s.app.conversion().is_some()
        || s.app.browser().is_some_and(|b| b.writing());
    let info = objc2_foundation::NSProcessInfo::processInfo();
    match (working, s.busy.take()) {
        (true, None) => {
            let options = objc2_foundation::NSActivityOptions::UserInitiatedAllowingIdleSystemSleep;
            s.busy = Some(info.beginActivityWithOptions_reason(options, &crate::ui::ns(&t!("app-busy-reason"))));
        }
        (true, Some(a)) => s.busy = Some(a),
        // SAFETY: the token beginActivity gave back, ended once.
        (false, Some(a)) => unsafe { info.endActivity(&a) },
        (false, None) => {}
    }
}

pub(crate) fn refresh() {
    let m = mtm();
    let due = with(|s| {
        keep_awake(s);
        s.win.refresh(m, &s.app);
        if let Some(w) = &mut s.settings {
            w.refresh(m, &s.app);
        }
        // The browser window is shown exactly while the controller has a browser.
        match (s.app.browser(), &mut s.browser) {
            (Some(b), Some(w)) => w.refresh(m, b),
            (Some(b), None) => {
                let mut w = crate::browser::BrowserWindow::new(m);
                w.refresh(m, b);
                w.window.makeKeyAndOrderFront(None);
                s.browser = Some(w);
            }
            (None, Some(_)) => {
                if let Some(mut w) = s.browser.take() {
                    w.close();
                }
            }
            (None, None) => {}
        }
        refresh_panels(m, s);
        // The camera preview has no wake of its own, and a benchmark wakes only
        // between kernels: poll them while they run.
        let bench = s.bench.as_ref().is_some_and(|b| b.ctl.is_running()).then(|| std::time::Duration::from_secs(1));
        let camera = s.app.camera().is_open().then(|| std::time::Duration::from_millis(66)).or(bench);
        match (s.app.wake_due_in(), camera) {
            (Some(a), Some(c)) => Some(a.min(c)),
            (a, c) => a.or(c),
        }
    })
    .flatten();
    // Nothing wakes a quiet shell, so the autosave pause and the stop after
    // IRIX powers off need a tick of their own. Each edit asks again; a tick before it is due is harmless.
    // One timed tick at a time: a refresh asks again on every event, and a
    // new request each time would start another self-renewing chain (with
    // the camera test open, thousands a minute, until the UI froze). Keep
    // the one already pending if it comes as soon; else replace it.
    if let Some(d) = due {
        let at = std::time::Instant::now() + d + std::time::Duration::from_millis(50);
        let pending = NEXT_TICK.with(|n| n.get());
        if pending.is_some_and(|p| p <= at) {
            return;
        }
        let t = crate::ui::target(m);
        // SAFETY: the target is never dropped; both selectors take the documented arguments.
        unsafe {
            let null = std::ptr::null::<objc2::runtime::AnyObject>();
            if pending.is_some() {
                let _: () = objc2::msg_send![objc2::class!(NSObject), cancelPreviousPerformRequestsWithTarget: &*t, selector: objc2::sel!(tickNow:), object: null];
            }
            // In every run-loop mode: macOS waits for a terminate answer in
            // the modal-panel mode, and a quit's disk sync still needs ticks.
            let modes = objc2_foundation::NSArray::from_slice(&[objc2_foundation::NSRunLoopCommonModes]);
            let _: () = objc2::msg_send![&*t, performSelector: objc2::sel!(tickNow:), withObject: null, afterDelay: (at - std::time::Instant::now()).as_secs_f64(), inModes: &*modes];
        }
        NEXT_TICK.with(|n| n.set(Some(at)));
    }
}

/// Answer the terminate request macOS is waiting on (a quit, a logout, a
/// restart): `true` lets it go ahead, `false` cancels it.
fn reply_terminate(yes: bool) {
    if TERMINATE_ASKED.with(|t| t.replace(false)) {
        NSApplication::sharedApplication(mtm()).replyToApplicationShouldTerminate(yes);
    }
}

/// The timed tick fired: none is pending now.
pub fn timed_tick() {
    NEXT_TICK.with(|n| n.set(None));
    tick();
}

/// The panels follow their controllers: shown while open, closed after.
fn refresh_panels(m: MainThreadMarker, s: &mut Shell) {
    use crate::panels::*;
    fn mirror<W>(slot: &mut Option<W>, open: bool, make: impl FnOnce() -> W, win: impl Fn(&W) -> Retained<objc2_app_kit::NSWindow>) -> bool {
        match (open, slot.is_some()) {
            (true, false) => {
                let w = make();
                win(&w).makeKeyAndOrderFront(None);
                *slot = Some(w);
                true
            }
            (false, true) => {
                if let Some(w) = slot.take() {
                    // Its close delegate goes with it: clear AppKit's weak
                    // reference first.
                    win(&w).setDelegate(None);
                    win(&w).orderOut(None);
                }
                false
            }
            (open, _) => open,
        }
    }
    if mirror(&mut s.net, s.app.network().check_open(), || NetWindow::new(m), |w| w.window.clone()) {
        let check = s.app.net_check();
        if let Some(w) = &mut s.net {
            w.refresh(m, check);
        }
    }
    if mirror(&mut s.serial, s.app.serial().is_open(), || SerialWindow::new(m), |w| w.window.clone()) {
        if let Some(w) = &mut s.serial {
            w.refresh(&s.app);
        }
    }
    if mirror(&mut s.camera, s.app.camera().is_open(), || CameraWindow::new(m), |w| w.window.clone()) {
        if let Some(w) = &mut s.camera {
            w.refresh(&s.app);
        }
    }
    if let Some(w) = &mut s.prefs {
        w.refresh(m, &s.app);
    }
    let running = s.app.session().is_running();
    if let Some(w) = &mut s.bench {
        w.refresh(running);
    }
}

/// A panel's close button.
pub fn panel_closed(which: crate::panels::Which) {
    use crate::panels::Which;
    use iris_gui_core::controller::{camera::CameraAction, network::NetworkAction, serial::SerialAction};
    match which {
        Which::Prefs => {
            if let Some(w) = with(|s| s.prefs.take()).flatten() {
                w.window.orderOut(None);
            }
        }
        Which::Network => dispatch(AppAction::Network(NetworkAction::CloseCheck)),
        Which::Serial => dispatch(AppAction::Serial(SerialAction::Close)),
        Which::Camera => dispatch(AppAction::Camera(CameraAction::Close)),
        Which::Bench => {
            // A run keeps going until stopped; the window waits for it.
            let busy = with(|s| s.bench.as_ref().is_some_and(|b| b.ctl.is_running())).unwrap_or(false);
            if busy {
                with(|s| s.win.toast(&t!("bench-running")));
            } else if let Some(w) = with(|s| s.bench.take()).flatten() {
                w.window.orderOut(None);
            }
        }
    }
}

/// A control in one of the panels.
fn panel_control(id: &str, value: iris_gui_core::controller::settings::Value) {
    use crate::panels::id as p;
    use iris_gui_core::controller::settings::Value;
    use iris_gui_core::controller::{network::NetworkAction, serial::SerialAction};
    match id {
        p::NET_ADOPT => {
            if let Some(cidr) = with(|s| s.net.as_ref().and_then(|w| w.adopt())).flatten() {
                dispatch(AppAction::Network(NetworkAction::AdoptSubnet(cidr)));
            }
        }
        p::NET_CLOSE => dispatch(AppAction::Network(NetworkAction::CloseCheck)),
        p::SERIAL_INPUT => {
            if let Value::Text(line) = value {
                with(|s| s.serial.as_ref().map(|w| w.input.setStringValue(&crate::ui::ns(""))));
                dispatch(AppAction::Serial(SerialAction::SendLine(line)));
            }
        }
        p::SERIAL_CLEAR => dispatch(AppAction::Serial(SerialAction::Clear)),
        p::SERIAL_CTRL_C => dispatch(AppAction::Serial(SerialAction::SendBytes(vec![3]))),
        _ => {
            let fx = with(|s| {
                let running = s.app.session().is_running();
                let w = s.bench.as_mut()?;
                let a = w.action(id, value)?;
                Some(w.ctl.apply(a, running))
            })
            .flatten();
            if let Some(fx) = fx {
                run_effects(fx);
                refresh();
            }
        }
    }
}

/// A settings control reported a new value.
pub fn setting(id: String, value: iris_gui_core::controller::settings::Value) {
    // A text field commits when a redraw takes its focus away, which happens
    // inside the shell's borrow: report it once the redraw is done instead
    // of dropping it (what was typed would be lost).
    if SHELL.with(|s| s.try_borrow_mut().is_err()) {
        return later(Box::new(move || setting(id, value)));
    }
    if id.starts_with("panel.") {
        return panel_control(&id, value);
    }
    if id == crate::summary::TAB {
        if let iris_gui_core::controller::settings::Value::Index(i) = value {
            with(|s| s.win.set_summary_tab(i));
            refresh();
        }
        return;
    }
    if id.starts_with("prefs.") {
        if let Some(a) = iris_gui_core::controller::preferences::action_for(&id, value) {
            dispatch(a);
        }
        return;
    }
    if id.starts_with("new-machine.") {
        return new_machine_control(id, value);
    }
    if id.starts_with("browser.") {
        let on = matches!(value, iris_gui_core::controller::settings::Value::Bool(true));
        return browser_control(&id, on);
    }
    enum Then {
        Page,
        Act(iris_gui_core::controller::settings::SettingsAction),
    }
    let then = with(|s| {
        let w = s.settings.as_mut()?;
        if let Some(page) = w.page_for(&id) {
            w.show_page(page);
            return Some(Then::Page);
        }
        w.action(&id, value).map(Then::Act)
    })
    .flatten();
    match then {
        Some(Then::Page) => refresh(),
        Some(Then::Act(a)) => dispatch(AppAction::Settings(a)),
        None => {}
    }
}

/// Open a fresh New Machine window.
fn open_new_machine() {
    let m = mtm();
    let w = with(|s| {
        if let Some(old) = s.new_machine.take() {
            old.window.orderOut(None);
        }
        let w = crate::new_machine::NewMachineWindow::new(m);
        let win = w.window.clone();
        s.new_machine = Some(w);
        win
    });
    if let Some(w) = w {
        w.makeKeyAndOrderFront(None);
    }
}

/// A control in the New Machine window reported a value.
fn new_machine_control(id: String, value: iris_gui_core::controller::settings::Value) {
    use crate::new_machine::Next;
    let m = mtm();
    let Some(next) = with(|s| s.new_machine.as_mut().map(|w| w.control(&id, value))).flatten() else { return };
    match next {
        Next::Nothing => {}
        Next::Redraw => {
            with(|s| s.new_machine.as_mut().map(|w| w.redraw(m)));
        }
        Next::Pick(e) => {
            // The picker runs here, with no borrow held; its answer is the window's.
            let out = crate::effects::run(m, vec![e]);
            with(|s| {
                if let Some(w) = s.new_machine.as_mut() {
                    if out.answers.iter().any(|a| w.spec.picked(a)) {
                        w.redraw(m);
                    }
                }
            });
        }
        Next::Close => {
            if let Some(w) = with(|s| s.new_machine.take()).flatten() {
                w.window.orderOut(None);
            }
        }
        Next::Create(action) => {
            if let Some(w) = with(|s| s.new_machine.take()).flatten() {
                w.window.orderOut(None);
            }
            dispatch(action);
        }
    }
}

/// Open the settings window at `page` (a section id), or bring it forward.
fn open_settings(page: Option<&'static str>) {
    let m = mtm();
    let w = with(|s| {
        let w = s.settings.get_or_insert_with(|| {
            crate::settings::SettingsWindow::new(m, page.unwrap_or(iris_gui_core::controller::ids::settings::GENERAL))
        });
        if let Some(p) = page {
            w.show_page(p);
        }
        w.window.clone()
    });
    refresh();
    if let Some(w) = w {
        w.makeKeyAndOrderFront(None);
    }
}

/// Apply an app action, then carry out its effects with no borrow held.
pub fn dispatch(action: AppAction) {
    let Some(fx) = with(|s| s.app.apply(action)) else {
        return;
    };
    run_effects(fx);
    refresh();
}

fn run_effects(fx: Vec<Effect>) {
    if fx.is_empty() {
        return;
    }
    let out = crate::effects::run(mtm(), fx);
    if let Some(t) = out.toasts.last() {
        let t = t.clone();
        with(|s| s.win.toast(&t));
    }
    if let Some(page) = out.open_settings {
        open_settings(Some(page));
    }
    for a in out.answers {
        if a.purpose().starts_with("bench.") {
            let fx = with(|s| {
                let running = s.app.session().is_running();
                s.bench.as_mut().map(|w| w.ctl.apply(iris_gui_core::controller::bench::BenchAction::Answer(a), running))
            })
            .flatten();
            if let Some(fx) = fx {
                run_effects(fx);
            }
        } else if a.purpose() == DIAGNOSTICS {
            if let iris_gui_core::controller::Answer::Chose { index: 1, .. } = a {
                run_effects(vec![Effect::CopyToClipboard(iris_gui_core::controller::debug::diagnostics_text())]);
            }
        } else {
            dispatch(AppAction::Answer(a));
        }
    }
    if out.quit {
        trace(|| "quit".into());
        with(|s| s.may_quit = true);
        if TERMINATE_ASKED.with(|t| t.get()) {
            reply_terminate(true);
        } else {
            NSApplication::sharedApplication(mtm()).terminate(None);
        }
    }
}

/// A browser control: its action, with the editor's text first for Save and Close.
fn browser_control(id: &str, value: bool) {
    use iris_gui_core::controller::disk_browser::{action_for, BrowserAction};
    let Some(action) = action_for(id, value) else { return };
    if matches!(action, BrowserAction::EditorSave | BrowserAction::EditorClose) {
        if let Some(text) = with(|s| s.browser.as_ref().and_then(|w| w.editor_text())).flatten() {
            let changed = with(|s| s.app.browser().and_then(|b| b.editor()).is_some_and(|e| e.text != text)).unwrap_or(false);
            if changed {
                dispatch(AppAction::Browser(BrowserAction::EditorChanged(text)));
            }
        }
    }
    dispatch(AppAction::Browser(action));
}

/// Rows selected in the browser's table.
pub fn browser_select(rows: Vec<usize>) {
    dispatch(AppAction::Browser(iris_gui_core::controller::disk_browser::BrowserAction::Select(rows)));
}

/// Host files dropped on the browser's table (`into`: a folder row).
pub fn browser_drop(paths: Vec<std::path::PathBuf>, into: Option<usize>) {
    dispatch(AppAction::Browser(iris_gui_core::controller::disk_browser::BrowserAction::Drop { paths, into }));
}

/// A row of the browser's table was double-clicked.
pub fn browser_activate(row: usize) {
    dispatch(AppAction::Browser(iris_gui_core::controller::disk_browser::BrowserAction::Activate(row)));
}

/// The browser window's close button.
pub fn close_browser() {
    dispatch(AppAction::CloseBrowser);
}

/// The Diagnostics alert's purpose (its Copy button).
const DIAGNOSTICS: &str = "shell.diagnostics";

/// Show help page `page`, built now from the selected machine (it carries
/// live values); an open window for the page is rebuilt and brought forward.
fn show_help(page: iris_gui_core::controller::help::HelpPage) {
    let m = mtm();
    let w = with(|s| {
        let doc = s.app.help(page, env!("APP_VERSION"), env!("APP_COMMIT"));
        if let Some(i) = s.help.iter().position(|h| h.page == page) {
            let old = s.help.remove(i);
            old.window.orderOut(None);
        }
        let w = crate::panels::HelpWindow::new(m, page, &doc);
        let win = w.window.clone();
        s.help.push(w);
        win
    });
    if let Some(w) = w {
        w.makeKeyAndOrderFront(None);
    }
}

/// A toolbar item, menu item or button with this command id was used.
pub fn command(id: &str) {
    if let Some(url) = id.strip_prefix(crate::panels::LINK) {
        return run_effects(vec![Effect::OpenUrl(url.to_string())]);
    }
    if let Some(page) = iris_gui_core::controller::help::HelpPage::from_id(id) {
        return show_help(page);
    }
    if id.starts_with("browser.") {
        return browser_control(id, false);
    }
    if id.starts_with("panel.") {
        return panel_control(id, iris_gui_core::controller::settings::Value::Bool(false));
    }
    // The machine capsule opens the machine menu; its items select.
    if id == iris_gui_core::controller::ids::toolbar::TITLE {
        if let (Some(groups), Some(at)) = (with(|s| s.app.library()), crate::toolbar::title_view()) {
            crate::menu::machines(mtm(), &groups, &at);
        }
        return;
    }
    if let Some(name) = id.strip_prefix(crate::menu::SELECT_MACHINE) {
        return select_machine(name.to_string());
    }
    // A disk's or CD-ROM's light opens that device's menu.
    if let Some((group, at)) = with(|s| s.app.light_menu(id).zip(s.win.light_view(id))).flatten() {
        crate::menu::pop_device(mtm(), &group, &at);
        return refresh();
    }
    let Some(routed) = with(|s| s.app.command(id)) else { return };
    run_effects(routed.effects);
    if let Some(c) = routed.shell {
        shell_command(c);
    }
    refresh();
}

/// Input and capture for the display: applied at once, without a full
/// redraw (motion arrives many times a second); the view follows capture.
pub fn display(action: iris_gui_core::controller::display::DisplayAction) {
    use iris_gui_core::controller::display::DisplayAction as D;
    // AppKit reports focus and resizes while the shell itself is changing
    // windows (inside its borrow): apply them after, not never.
    if SHELL.with(|s| s.try_borrow_mut().is_err()) {
        return later(Box::new(move || display(action)));
    }
    // Losing focus while captured starts the release grace: refresh, so a
    // tick is due when it runs out (nothing else may wake the shell).
    let lost_focus = matches!(action, D::Focus(false));
    let m = mtm();
    let fx = with(|s| {
        let before = (s.app.display().captured(), s.app.display().fullscreen(), s.app.display().detached());
        let fx = s.app.apply(AppAction::Display(action));
        let after = (s.app.display().captured(), s.app.display().fullscreen(), s.app.display().detached());
        if before != after {
            s.win.sync_display(m, &s.app);
        }
        fx
    });
    if let Some(fx) = fx {
        run_effects(fx);
    }
    if lost_focus {
        refresh();
    }
}

/// A key pressed while the display is not captured: the key map decides.
pub fn shortcut(key: iris_gui_core::controller::keys::Key, mods: iris_gui_core::controller::keys::Mods) {
    let keys = iris_gui_core::controller::keys::KeyMap::default();
    if let Some(cmd) = keys.lookup(key, mods) {
        command(iris_gui_core::controller::commands::id_of(cmd));
    }
}

/// Whether a menu or toolbar item with this id is enabled.
pub fn enabled(id: &str) -> bool {
    with(|s| s.app.command_enabled(id)).map_or(true, |e| e == Enabled::Yes)
}

/// The Devices menu's groups, or `None` while the shell is busy.
pub fn disk_menu() -> Option<Vec<iris_gui_core::controller::commands::MenuGroup>> {
    with(|s| s.app.disk_menu())
}

pub fn machine_menu(name: &str) -> Option<Vec<Vec<iris_gui_core::controller::commands::MenuEntry>>> {
    with(|s| s.app.machine_menu(name))
}

/// The pointer entered or left one of the window's hover zones.
pub fn hover(zone: crate::chrome::Zone, inside: bool) {
    let _ = with(|s| s.win.hover(zone, inside));
}

/// The display refits its scroll view (a window was resized).
pub fn display_fit() {
    let _ = with(|s| s.win.display.fit());
}

/// The display's window came back on screen: draw the picture it holds
/// (a still guest sends no new frame).
pub fn display_redraw() {
    let _ = with(|s| s.win.display.draw());
}

pub fn select_machine(name: String) {
    dispatch(AppAction::Select(name));
}

/// The commands that only change the shell's own views.
fn shell_command(c: ShellCommand) {
    match c {
        ShellCommand::ToggleLibrary => {
            with(|s| s.win.toggle_library());
        }
        ShellCommand::CloseWindow => {
            // The front window: the settings window as well as the main one.
            // With no key window (the app in the background), the frontmost
            // visible one; never the main window by default.
            let app = NSApplication::sharedApplication(mtm());
            let front = app.keyWindow().or_else(|| app.orderedWindows().iter().find(|w| w.isVisible()));
            if let Some(w) = front {
                w.performClose(None);
            }
        }
        ShellCommand::About => {
            let doc = with(|s| s.app.help(iris_gui_core::controller::help::HelpPage::About, env!("APP_VERSION"), env!("APP_COMMIT")));
            if let Some(doc) = doc {
                crate::panels::show_about(mtm(), &doc, env!("APP_VERSION"), env!("APP_COMMIT"));
            }
        }
        ShellCommand::Settings => open_settings(None),
        ShellCommand::NewMachine => open_new_machine(),
        ShellCommand::DiscMenu => {
            let groups = with(|s| s.app.disk_menu()).unwrap_or_default();
            crate::menu::discs(mtm(), &groups);
        }
        ShellCommand::SavedStates => {
            let states = with(|s| s.app.saved_states()).unwrap_or_default();
            crate::menu::saved_states(mtm(), &states);
        }
        ShellCommand::AppSettings => {
            let m = mtm();
            let w = with(|s| s.prefs.get_or_insert_with(|| crate::panels::PrefsWindow::new(m)).window.clone());
            refresh();
            if let Some(w) = w {
                w.makeKeyAndOrderFront(None);
            }
        }
        ShellCommand::Benchmark => {
            let m = mtm();
            let w = with(|s| {
                let notify: Arc<dyn iris_gui_core::controller::Notify> = s.wake.clone();
                s.bench.get_or_insert_with(|| crate::panels::BenchWindow::new(m, notify)).window.clone()
            });
            refresh();
            if let Some(w) = w {
                w.makeKeyAndOrderFront(None);
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
    }
}
