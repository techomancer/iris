//! The smaller windows. Each mirrors a controller: Preferences
//! (`controller::preferences`), Check Networking (`app.network()`), the
//! serial console (`app.serial()`), the camera test (`app.camera()`) and the
//! benchmark (a `BenchController` the shell owns, as egui does) and the
//! blank-image form (`app.create_disk()`). A window is shown while its
//! controller is open; its close button asks the controller.

use objc2::rc::Retained;
use objc2::runtime::{NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{define_class, msg_send, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{
    NSBackingStoreType, NSBitmapImageRep, NSButton, NSColor, NSFont, NSImage, NSImageView, NSLayoutAttribute,
    NSProgressIndicator, NSProgressIndicatorStyle, NSStackView, NSTextField, NSTextView, NSView, NSWindow,
    NSWindowDelegate, NSWindowStyleMask,
};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};
use iris_gui_core::controller::app::AppController;
use iris_gui_core::controller::bench::{BenchAction, BenchController};
use iris_gui_core::controller::network::{CheckLine, NetCheck};
use iris_gui_core::controller::Enabled;

use crate::settings::{checkbox, field_view, push, stack, view, wire};
use crate::ui::{button, heading, label, ns, secondary, set_id};

/// Ids of the panels' controls (all reported through `app::setting`).
pub mod id {
    pub const NET_ADOPT: &str = "panel.net-adopt";
    pub const NET_CLOSE: &str = "panel.net-close";
    pub const SERIAL_INPUT: &str = "panel.serial-input";
    pub const SERIAL_CLEAR: &str = "panel.serial-clear";
    pub const SERIAL_CTRL_C: &str = "panel.serial-ctrl-c";
    pub const BENCH_CPU: &str = "panel.bench-cpu";
    pub const BENCH_QUICK: &str = "panel.bench-quick";
    pub const BENCH_START: &str = "panel.bench-start";
    pub const BENCH_SAVE: &str = "panel.bench-save";
    pub const CREATE_DISK_CREATE: &str = "panel.create-disk-create";
    pub const CREATE_DISK_CANCEL: &str = "panel.create-disk-cancel";
}

/// Which window a close button belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Which {
    Prefs,
    Network,
    Serial,
    Camera,
    Bench,
    CreateDisk,
}

pub struct CloseIvars {
    which: Which,
}

define_class!(
    /// A panel's close button: the shell asks the panel's controller.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISPanelDelegate"]
    #[ivars = CloseIvars]
    pub struct CloseAsk;

    unsafe impl NSObjectProtocol for CloseAsk {}

    unsafe impl NSWindowDelegate for CloseAsk {
        #[unsafe(method(windowShouldClose:))]
        fn should_close(&self, _w: &NSWindow) -> bool {
            let which = self.ivars().which;
            crate::app::later(Box::new(move || crate::app::panel_closed(which)));
            false
        }
    }
);

/// A titled, closable panel window whose close button reports `which`.
fn panel(mtm: MainThreadMarker, title: &str, size: NSSize, which: Which) -> (Retained<NSWindow>, Retained<CloseAsk>) {
    let style = NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Miniaturizable
        | NSWindowStyleMask::Resizable;
    // SAFETY: a standard window, kept by the shell while it is open.
    let w = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            mtm.alloc(),
            NSRect::new(NSPoint::new(220.0, 220.0), size),
            style,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    // SAFETY: the shell keeps the window.
    unsafe { w.setReleasedWhenClosed(false) };
    w.setTitle(&ns(title));
    let d: Retained<CloseAsk> = {
        let this = mtm.alloc::<CloseAsk>().set_ivars(CloseIvars { which });
        // SAFETY: NSObject's plain init.
        unsafe { msg_send![super(this), init] }
    };
    w.setDelegate(Some(ProtocolObject::from_ref(&*d)));
    w.center();
    (w, d)
}

/// Put `page` in `w`, pinned to its edges.
fn fill(w: &NSWindow, page: &NSView) {
    let root = NSView::new(MainThreadMarker::from(w));
    page.setTranslatesAutoresizingMaskIntoConstraints(false);
    root.addSubview(page);
    page.leadingAnchor().constraintEqualToAnchor(&root.leadingAnchor()).setActive(true);
    page.trailingAnchor().constraintEqualToAnchor(&root.trailingAnchor()).setActive(true);
    page.topAnchor().constraintEqualToAnchor(&root.topAnchor()).setActive(true);
    page.bottomAnchor().constraintEqualToAnchor(&root.bottomAnchor()).setActive(true);
    w.setContentView(Some(&root));
}

fn insets(s: &NSStackView, n: f64) {
    s.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: n, left: n, bottom: n, right: n });
}

/// A read-only, selectable, monospaced text view in a scroll view.
fn text_area(mtm: MainThreadMarker, id: &str) -> (Retained<objc2_app_kit::NSScrollView>, Retained<NSTextView>) {
    let scroll = NSTextView::scrollableTextView(mtm);
    let tv = scroll.documentView().and_then(|d| d.downcast::<NSTextView>().ok()).expect("a scrollable text view");
    tv.setEditable(false);
    tv.setSelectable(true);
    tv.setRichText(false);
    tv.setFont(Some(&NSFont::userFixedPitchFontOfSize(12.0).unwrap_or_else(|| NSFont::systemFontOfSize(12.0))));
    set_id(&tv, id);
    (scroll, tv)
}

/// Whether `tv` is scrolled to its end (or has nothing to scroll).
fn at_end(tv: &NSTextView) -> bool {
    let Some(scroll) = tv.enclosingScrollView() else { return true };
    let visible = scroll.contentView().documentVisibleRect();
    let bottom = visible.origin.y + visible.size.height;
    bottom >= tv.frame().size.height - 4.0
}

fn set_text(tv: &NSTextView, text: &str, follow: bool) {
    tv.setString(&NSString::from_str(text));
    if follow {
        let len = tv.string().length();
        tv.scrollRangeToVisible(objc2_foundation::NSRange::new(len, 0));
    }
}

// ---- Preferences -------------------------------------------------------------

/// The preferences: a tab per group of fields, chosen in the toolbar as in
/// other Mac apps' settings windows.
pub struct PrefsWindow {
    pub window: Retained<NSWindow>,
    _d: Retained<CloseAsk>,
    tabs: Retained<objc2_app_kit::NSTabViewController>,
    /// Each tab's view controller and view, refilled when its fields change.
    pages: Vec<(Retained<objc2_app_kit::NSViewController>, Retained<NSView>)>,
    last: Option<Vec<iris_gui_core::controller::preferences::Tab>>,
}

impl PrefsWindow {
    pub fn new(mtm: MainThreadMarker) -> Self {
        let (window, d) = panel(mtm, &t!("prefs-window-title"), NSSize::new(520.0, 240.0), Which::Prefs);
        window.setStyleMask(window.styleMask() & !NSWindowStyleMask::Resizable);
        window.setToolbarStyle(objc2_app_kit::NSWindowToolbarStyle::Preference);
        let tabs = objc2_app_kit::NSTabViewController::new(mtm);
        tabs.setTabStyle(objc2_app_kit::NSTabViewControllerTabStyle::Toolbar);
        Self { window, _d: d, tabs, pages: Vec::new(), last: None }
    }

    pub fn refresh(&mut self, mtm: MainThreadMarker, app: &AppController) {
        let now = iris_gui_core::controller::preferences::tabs(app.prefs());
        if self.last.as_ref() == Some(&now) {
            return;
        }
        self.window.makeFirstResponder(None);
        if self.pages.len() != now.len() {
            // A different set of tabs: start again rather than add to the old.
            for item in self.tabs.tabViewItems().iter() {
                self.tabs.removeTabViewItem(&item);
            }
            self.pages.clear();
            for tab in &now {
                let page = NSView::new(mtm);
                page.widthAnchor().constraintEqualToConstant(520.0).setActive(true);
                let vc = objc2_app_kit::NSViewController::new(mtm);
                vc.setView(&page);
                vc.setTitle(Some(&ns(&tab.title)));
                let item = objc2_app_kit::NSTabViewItem::tabViewItemWithViewController(&vc);
                item.setLabel(&ns(&tab.title));
                if let Some(img) = crate::glyphs::icon(tab.icon, &tab.title) {
                    item.setImage(Some(&img));
                }
                self.tabs.addTabViewItem(&item);
                self.pages.push((vc, page));
            }
            self.window.setContentViewController(Some(&self.tabs));
        }
        for (tab, (vc, page)) in now.iter().zip(&self.pages) {
            if self.last.as_ref().and_then(|l| l.iter().find(|t| t.id == tab.id)) == Some(tab) {
                continue;
            }
            for v in page.subviews().iter() {
                v.removeFromSuperview();
            }
            let mut rows: Vec<Retained<NSView>> = Vec::new();
            let mut toml = None;
            for f in &tab.fields {
                rows.push(field_view(mtm, f, &mut toml));
            }
            let body = stack(mtm, &rows, true, 14.0);
            insets(&body, 22.0);
            body.setTranslatesAutoresizingMaskIntoConstraints(false);
            page.addSubview(&body);
            body.leadingAnchor().constraintEqualToAnchor(&page.leadingAnchor()).setActive(true);
            body.trailingAnchor().constraintLessThanOrEqualToAnchor(&page.trailingAnchor()).setActive(true);
            body.topAnchor().constraintEqualToAnchor(&page.topAnchor()).setActive(true);
            body.bottomAnchor().constraintEqualToAnchor(&page.bottomAnchor()).setActive(true);
            // The window takes each tab's own height as it is chosen.
            page.layoutSubtreeIfNeeded();
            vc.setPreferredContentSize(page.fittingSize());
        }
        self.last = Some(now);
    }
}

// ---- Check Networking --------------------------------------------------------

pub struct NetWindow {
    pub window: Retained<NSWindow>,
    _d: Retained<CloseAsk>,
    last: Option<NetCheck>,
}

impl NetWindow {
    pub fn new(mtm: MainThreadMarker) -> Self {
        let (window, d) = panel(mtm, &t!("window-net-title"), NSSize::new(620.0, 420.0), Which::Network);
        Self { window, _d: d, last: None }
    }

    pub fn refresh(&mut self, mtm: MainThreadMarker, check: NetCheck) {
        if self.last.as_ref() == Some(&check) {
            return;
        }
        let mut rows: Vec<Retained<NSView>> = Vec::new();
        for line in &check.lines {
            let l: Retained<NSTextField> = match line {
                CheckLine::Good(t) => {
                    let l = heading(mtm, t, 15.0);
                    l.setTextColor(Some(&NSColor::systemGreenColor()));
                    l
                }
                CheckLine::Bad(t) => {
                    let l = heading(mtm, t, 15.0);
                    l.setTextColor(Some(&NSColor::systemRedColor()));
                    l
                }
                CheckLine::Text(t) => label(mtm, t),
                CheckLine::Weak(t) => secondary(mtm, t),
                CheckLine::Strong(t) => heading(mtm, t, 13.0),
                CheckLine::Code(t) => {
                    // Selectable, so the command can be copied.
                    let l = label(mtm, t);
                    l.setSelectable(true);
                    l.setFont(Some(&NSFont::userFixedPitchFontOfSize(12.0).unwrap_or_else(|| NSFont::systemFontOfSize(12.0))));
                    l
                }
            };
            l.setLineBreakMode(objc2_app_kit::NSLineBreakMode::ByWordWrapping);
            l.setPreferredMaxLayoutWidth(560.0);
            rows.push(view(&l));
        }
        let mut buttons: Vec<Retained<NSView>> = Vec::new();
        if let Some(cidr) = &check.adopt {
            let b = push(mtm, id::NET_ADOPT, &t!("window-net-use-guest-subnet", network = cidr.as_str()));
            buttons.push(view(&b));
        }
        buttons.push(view(&push(mtm, id::NET_CLOSE, &t!("common-close"))));
        rows.push(view(&stack(mtm, &buttons, false, 8.0)));
        let page = stack(mtm, &rows, true, 10.0);
        insets(&page, 22.0);
        fill(&self.window, &page);
        self.last = Some(check);
    }

    /// The subnet the Use button offers.
    pub fn adopt(&self) -> Option<String> {
        self.last.as_ref().and_then(|c| c.adopt.clone())
    }
}

// ---- Serial console ----------------------------------------------------------

pub struct SerialWindow {
    pub window: Retained<NSWindow>,
    _d: Retained<CloseAsk>,
    text: Retained<NSTextView>,
    status: Retained<NSTextField>,
    pub input: Retained<NSTextField>,
    seq: Option<u64>,
    /// The text the view holds now, so new output is appended to it.
    shown: String,
}

impl SerialWindow {
    pub fn new(mtm: MainThreadMarker) -> Self {
        let (window, d) = panel(mtm, &t!("window-serial-title"), NSSize::new(760.0, 520.0), Which::Serial);
        let (scroll, text) = text_area(mtm, "serial.text");
        let status = secondary(mtm, "");
        let input = NSTextField::textFieldWithString(&ns(""), mtm);
        input.setPlaceholderString(Some(&ns(&t!("window-serial-input-hint"))));
        input.setFont(Some(&NSFont::userFixedPitchFontOfSize(12.0).unwrap_or_else(|| NSFont::systemFontOfSize(12.0))));
        wire(&input, id::SERIAL_INPUT);
        input.setContentHuggingPriority_forOrientation(1.0, objc2_app_kit::NSLayoutConstraintOrientation::Horizontal);
        let ctrl_c = push(mtm, id::SERIAL_CTRL_C, "Ctrl-C");
        let clear = push(mtm, id::SERIAL_CLEAR, &t!("window-serial-clear"));
        let bar = stack(mtm, &[view(&input), view(&ctrl_c), view(&clear)], false, 8.0);
        let page = stack(mtm, &[view(&scroll), view(&bar), view(&status)], true, 8.0);
        page.setAlignment(NSLayoutAttribute::Width);
        insets(&page, 12.0);
        fill(&window, &page);
        window.setInitialFirstResponder(Some(&input));
        Self { window, _d: d, text, status, input, seq: None, shown: String::new() }
    }

    pub fn refresh(&mut self, app: &AppController) {
        let Some(v) = app.serial().view() else { return };
        if self.seq == Some(v.seq) {
            return;
        }
        // New output is appended, keeping the reader's selection, and the
        // view follows it only when it was at the end already (scrolled up
        // to read back, it stays put). A trimmed or cleared buffer is set
        // whole.
        let at_end = at_end(&self.text);
        match v.text.strip_prefix(self.shown.as_str()) {
            Some(more) if !self.shown.is_empty() => {
                if !more.is_empty() {
                    // SAFETY: a plain getter; the storage lives with the view.
                    if let Some(store) = unsafe { self.text.textStorage() } {
                        let end = store.length();
                        store.replaceCharactersInRange_withString(objc2_foundation::NSRange::new(end, 0), &NSString::from_str(more));
                    }
                }
            }
            _ => self.text.setString(&NSString::from_str(&v.text)),
        }
        if at_end {
            let len = self.text.string().length();
            self.text.scrollRangeToVisible(objc2_foundation::NSRange::new(len, 0));
        }
        self.shown = v.text.clone();
        self.status.setStringValue(&ns(&v.status));
        self.status.setTextColor(Some(&*if v.error { NSColor::systemRedColor() } else { NSColor::secondaryLabelColor() }));
        self.seq = Some(v.seq);
    }
}

// ---- Camera test -------------------------------------------------------------

pub struct CameraWindow {
    pub window: Retained<NSWindow>,
    _d: Retained<CloseAsk>,
    image: Retained<NSImageView>,
    status: Retained<NSTextField>,
    seq: u64,
}

impl CameraWindow {
    pub fn new(mtm: MainThreadMarker) -> Self {
        let (window, d) = panel(mtm, &t!("window-camera-title"), NSSize::new(680.0, 560.0), Which::Camera);
        let image = NSImageView::new(mtm);
        image.setImageScaling(objc2_app_kit::NSImageScaling::ScaleProportionallyUpOrDown);
        image.widthAnchor().constraintGreaterThanOrEqualToConstant(320.0).setActive(true);
        image.heightAnchor().constraintGreaterThanOrEqualToConstant(240.0).setActive(true);
        image.setContentHuggingPriority_forOrientation(1.0, objc2_app_kit::NSLayoutConstraintOrientation::Vertical);
        set_id(&image, "camera.preview");
        let status = secondary(mtm, &t!("window-camera-starting"));
        status.setLineBreakMode(objc2_app_kit::NSLineBreakMode::ByWordWrapping);
        status.setPreferredMaxLayoutWidth(620.0);
        let page = stack(mtm, &[view(&image), view(&status)], true, 8.0);
        page.setAlignment(NSLayoutAttribute::Width);
        insets(&page, 12.0);
        fill(&window, &page);
        Self { window, _d: d, image, status, seq: 0 }
    }

    pub fn refresh(&mut self, app: &AppController) {
        if let Some((text, error)) = app.camera().status() {
            let text = if error { format!("{text}\n{}", t!("window-camera-error-detail")) } else { text };
            if self.status.stringValue().to_string() != text {
                self.status.setStringValue(&ns(&text));
                self.status.setTextColor(Some(&*if error { NSColor::systemRedColor() } else { NSColor::secondaryLabelColor() }));
            }
        }
        if let Some((w, h, rgba)) = app.camera().take_new_frame(&mut self.seq) {
            if let Some(img) = rgba_image(MainThreadMarker::from(&*self.window), w, h, &rgba) {
                self.image.setImage(Some(&img));
            }
        }
    }
}

/// An NSImage holding a copy of `rgba` (`w` x `h`, 8 bits per channel).
fn rgba_image(_mtm: MainThreadMarker, w: u32, h: u32, rgba: &[u8]) -> Option<Retained<NSImage>> {
    if rgba.len() < (w * h * 4) as usize || w == 0 || h == 0 {
        return None;
    }
    // SAFETY: null planes make the rep allocate its own buffer, which is then
    // filled from `rgba` (at least w*h*4 bytes, checked above).
    unsafe {
        let rep = NSBitmapImageRep::initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel(
            objc2::AnyThread::alloc(),
            std::ptr::null_mut(),
            w as isize,
            h as isize,
            8,
            4,
            true,
            false,
            objc2_app_kit::NSDeviceRGBColorSpace,
            (w * 4) as isize,
            32,
        )?;
        let dst = rep.bitmapData();
        if dst.is_null() {
            return None;
        }
        std::ptr::copy_nonoverlapping(rgba.as_ptr(), dst, (w * h * 4) as usize);
        let img = NSImage::initWithSize(objc2::AnyThread::alloc(), NSSize::new(w as f64, h as f64));
        img.addRepresentation(&rep);
        Some(img)
    }
}

// ---- Benchmark ---------------------------------------------------------------

pub struct BenchWindow {
    pub window: Retained<NSWindow>,
    _d: Retained<CloseAsk>,
    pub ctl: BenchController,
    cpu: Retained<objc2_app_kit::NSPopUpButton>,
    quick: Retained<NSButton>,
    start: Retained<NSButton>,
    save: Retained<NSButton>,
    progress: Retained<NSProgressIndicator>,
    line: Retained<NSTextField>,
    text: Retained<NSTextView>,
    shown: usize,
}

impl BenchWindow {
    pub fn new(mtm: MainThreadMarker, notify: std::sync::Arc<dyn iris_gui_core::controller::Notify>) -> Self {
        let (window, d) = panel(mtm, &t!("bench-heading"), NSSize::new(760.0, 560.0), Which::Bench);
        let ctl = BenchController::new(notify);
        let (names, sel) = ctl.cpus();
        // The shared popup: its items carry the indexes the controller takes.
        let cpu = crate::settings::popup(mtm, id::BENCH_CPU, &names, sel);
        cpu.setToolTip(Some(&ns(&t!("bench-cpu-tip"))));
        let quick = checkbox(mtm, id::BENCH_QUICK, &t!("bench-quick"), ctl.quick());
        quick.setToolTip(Some(&ns(&t!("bench-quick-tip"))));
        let run = if cfg!(target_os = "macos") { t!("bench-run-mac") } else { t!("bench-run-pc") };
        let start = button(mtm, &run, id::BENCH_START);
        let save = button(mtm, &t!("bench-save-report"), id::BENCH_SAVE);
        let top = stack(mtm, &[view(&cpu), view(&quick), view(&start), view(&save)], false, 10.0);
        let progress = NSProgressIndicator::new(mtm);
        progress.setStyle(NSProgressIndicatorStyle::Bar);
        progress.setIndeterminate(false);
        progress.widthAnchor().constraintEqualToConstant(260.0).setActive(true);
        let line = secondary(mtm, "");
        let mid = stack(mtm, &[view(&progress), view(&line)], false, 10.0);
        let (scroll, text) = text_area(mtm, "bench.text");
        let page = stack(mtm, &[view(&top), view(&mid), view(&scroll)], true, 10.0);
        page.setAlignment(NSLayoutAttribute::Width);
        insets(&page, 14.0);
        fill(&window, &page);
        Self { window, _d: d, ctl, cpu, quick, start, save, progress, line, text, shown: usize::MAX }
    }

    pub fn refresh(&mut self, machine_running: bool) {
        self.ctl.tick();
        let running = self.ctl.is_running();
        let run = if cfg!(target_os = "macos") { t!("bench-run-mac") } else { t!("bench-run-pc") };
        self.start.setTitle(&ns(&if running { t!("bench-stop") } else { run }));
        let can = running || self.ctl.can_start(machine_running) == Enabled::Yes;
        self.start.setEnabled(can);
        if let Enabled::No(why) = self.ctl.can_start(machine_running) {
            if !running {
                self.start.setToolTip(Some(&ns(&why)));
            }
        }
        self.cpu.setEnabled(!running);
        self.quick.setEnabled(!running);
        self.save.setEnabled(!running && matches!(self.ctl.last(), Some(Ok(_))));
        match self.ctl.progress() {
            Some(p) => {
                self.progress.setHidden(false);
                self.progress.setDoubleValue(p.fraction as f64 * 100.0);
                self.line.setStringValue(&ns(&format!("{}   {}", p.line, p.time)));
            }
            None => {
                self.progress.setHidden(true);
                let done = match self.ctl.last() {
                    Some(Ok(r)) => BenchController::report_text(r).lines().next().unwrap_or_default().to_string(),
                    Some(Err(e)) => t!("bench-failed", error = e.as_str()),
                    None => t!("bench-run-tip", secs = if self.ctl.quick() { 35 } else { 60 }),
                };
                self.line.setStringValue(&ns(&done));
            }
        }
        let lines = self.ctl.lines();
        let text = match (running, self.ctl.last()) {
            (false, Some(Ok(r))) => BenchController::report_text(r),
            _ => lines.join("\n"),
        };
        if text.len() != self.shown {
            set_text(&self.text, &text, running);
            self.shown = text.len();
        }
    }

    /// The action a bench control means.
    pub fn action(&self, which: &str, value: iris_gui_core::controller::settings::Value) -> Option<BenchAction> {
        use iris_gui_core::controller::settings::Value;
        Some(match (which, value) {
            (id::BENCH_CPU, Value::Index(i)) => BenchAction::SetCpu(i),
            (id::BENCH_QUICK, Value::Bool(b)) => BenchAction::SetQuick(b),
            (id::BENCH_START, _) => {
                if self.ctl.is_running() {
                    BenchAction::Stop
                } else {
                    BenchAction::Start
                }
            }
            (id::BENCH_SAVE, _) => BenchAction::SaveReport,
            _ => return None,
        })
    }
}

// ---- Blank disk image --------------------------------------------------------

/// The blank-image form: the controller's fields (drawn as the settings
/// draw theirs, reporting `create-disk.*` ids), a progress bar with Cancel
/// while the image is written, and Create.
pub struct CreateDiskWindow {
    pub window: Retained<NSWindow>,
    _d: Retained<CloseAsk>,
    /// Holds the form; replaced when the fields change.
    body: Retained<NSStackView>,
    create: Retained<NSButton>,
    cancel: Retained<NSButton>,
    progress: Retained<NSProgressIndicator>,
    line: Retained<NSTextField>,
    shown: Vec<iris_gui_core::controller::settings::Field>,
}

impl CreateDiskWindow {
    pub fn new(mtm: MainThreadMarker, name: &str) -> Self {
        let title = format!("{} — {name}", t!("create-disk-title"));
        let (window, d) = panel(mtm, &title, NSSize::new(600.0, 220.0), Which::CreateDisk);
        let body = stack(mtm, &[], true, 10.0);
        let progress = NSProgressIndicator::new(mtm);
        progress.setStyle(NSProgressIndicatorStyle::Bar);
        progress.setIndeterminate(false);
        progress.widthAnchor().constraintEqualToConstant(260.0).setActive(true);
        let line = secondary(mtm, "");
        let cancel = button(mtm, &t!("common-cancel"), id::CREATE_DISK_CANCEL);
        let create = button(mtm, &t!("create-disk-create"), id::CREATE_DISK_CREATE);
        create.setKeyEquivalent(&ns("\r"));
        let spacer = NSView::new(mtm);
        spacer.setContentHuggingPriority_forOrientation(1.0, objc2_app_kit::NSLayoutConstraintOrientation::Horizontal);
        let bottom = stack(mtm, &[view(&progress), view(&line), view(&spacer), view(&cancel), view(&create)], false, 10.0);
        let page = stack(mtm, &[view(&body), view(&bottom)], true, 16.0);
        page.setAlignment(NSLayoutAttribute::Width);
        insets(&page, 18.0);
        fill(&window, &page);
        Self { window, _d: d, body, create, cancel, progress, line, shown: Vec::new() }
    }

    pub fn refresh(&mut self, mtm: MainThreadMarker, c: &iris_gui_core::controller::create_disk::CreateDiskController) {
        let fields = c.fields();
        if fields != self.shown {
            for v in self.body.arrangedSubviews().iter() {
                self.body.removeArrangedSubview(&v);
                v.removeFromSuperview();
            }
            let mut toml = None;
            for f in &fields {
                self.body.addArrangedSubview(&field_view(mtm, f, &mut toml));
            }
            self.shown = fields;
        }
        let can = c.can_create();
        self.create.setEnabled(can.is_yes());
        self.create.setToolTip(can.reason().map(ns).as_deref());
        match c.job() {
            Some(job) => {
                self.progress.setHidden(false);
                self.cancel.setHidden(false);
                match job.progress.fraction() {
                    Some(f) => {
                        self.progress.setIndeterminate(false);
                        self.progress.setDoubleValue(f as f64 * 100.0);
                    }
                    None => {
                        self.progress.setIndeterminate(true);
                        // SAFETY: a plain AppKit call on the main thread.
                        unsafe { self.progress.startAnimation(None) };
                    }
                }
                self.line.setStringValue(&ns(&job.label));
            }
            None => {
                self.progress.setHidden(true);
                self.cancel.setHidden(true);
                self.line.setStringValue(&ns(""));
            }
        }
    }
}

// ---- Help pages ----------------------------------------------------------------

pub struct HelpWindow {
    pub window: Retained<NSWindow>,
    pub page: iris_gui_core::controller::help::HelpPage,
}

/// The id prefix of a link button; the URL follows.
pub const LINK: &str = "link:";

impl HelpWindow {
    /// A window showing `doc`; closing it simply hides it.
    pub fn new(mtm: MainThreadMarker, page: iris_gui_core::controller::help::HelpPage, doc: &iris_gui_core::controller::help::HelpDoc) -> Self {
        use iris_gui_core::controller::help::Line;
        let style = NSWindowStyleMask::Titled | NSWindowStyleMask::Closable | NSWindowStyleMask::Resizable;
        // SAFETY: a standard window, kept by the shell.
        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                mtm.alloc(),
                NSRect::new(NSPoint::new(260.0, 200.0), NSSize::new(640.0, 560.0)),
                style,
                NSBackingStoreType::Buffered,
                false,
            )
        };
        // SAFETY: the shell keeps the window.
        unsafe { window.setReleasedWhenClosed(false) };
        window.setTitle(&ns(&doc.title));
        let mono = || NSFont::userFixedPitchFontOfSize(12.0).unwrap_or_else(|| NSFont::systemFontOfSize(12.0));
        let wrap = |text: &str, color: Option<Retained<NSColor>>| {
            let l = NSTextField::wrappingLabelWithString(&ns(text), mtm);
            l.setSelectable(true);
            l.setPreferredMaxLayoutWidth(580.0);
            if let Some(c) = color {
                l.setTextColor(Some(&c));
            }
            l
        };
        let mut rows: Vec<Retained<NSView>> = Vec::new();
        for line in &doc.lines {
            let v: Retained<NSView> = match line {
                Line::Heading(t) => view(&heading(mtm, t, 16.0)),
                Line::Strong(t) => view(&heading(mtm, t, 13.0)),
                Line::Text(t) => view(&wrap(t, None)),
                Line::Weak(t) => view(&wrap(t, Some(NSColor::secondaryLabelColor()))),
                Line::Warn(t) => view(&wrap(t, Some(NSColor::systemOrangeColor()))),
                Line::Error(t) => view(&wrap(t, Some(NSColor::systemRedColor()))),
                Line::Code(t) => {
                    let l = wrap(t, None);
                    l.setFont(Some(&mono()));
                    view(&l)
                }
                Line::Link { label, url } => {
                    let b = button(mtm, label, &format!("{LINK}{url}"));
                    b.setBordered(false);
                    b.setContentTintColor(Some(&NSColor::linkColor()));
                    b.setToolTip(Some(&ns(url)));
                    view(&b)
                }
                Line::Rule => {
                    let b = objc2_app_kit::NSBox::new(mtm);
                    b.setBoxType(objc2_app_kit::NSBoxType::Separator);
                    b.widthAnchor().constraintEqualToConstant(580.0).setActive(true);
                    view(&b)
                }
            };
            rows.push(v);
        }
        let page_view = stack(mtm, &rows, true, 8.0);
        insets(&page_view, 20.0);
        let scroll = objc2_app_kit::NSScrollView::initWithFrame(mtm.alloc(), NSRect::ZERO);
        scroll.setHasVerticalScroller(true);
        scroll.setDrawsBackground(false);
        let doc_view = crate::ui::flipped(mtm);
        doc_view.setTranslatesAutoresizingMaskIntoConstraints(false);
        page_view.setTranslatesAutoresizingMaskIntoConstraints(false);
        doc_view.addSubview(&page_view);
        scroll.setDocumentView(Some(&doc_view));
        let clip = scroll.contentView();
        doc_view.topAnchor().constraintEqualToAnchor(&clip.topAnchor()).setActive(true);
        doc_view.leadingAnchor().constraintEqualToAnchor(&clip.leadingAnchor()).setActive(true);
        doc_view.widthAnchor().constraintEqualToAnchor(&clip.widthAnchor()).setActive(true);
        page_view.topAnchor().constraintEqualToAnchor(&doc_view.topAnchor()).setActive(true);
        page_view.leadingAnchor().constraintEqualToAnchor(&doc_view.leadingAnchor()).setActive(true);
        doc_view.heightAnchor().constraintEqualToAnchor(&page_view.heightAnchor()).setActive(true);
        fill(&window, &scroll);
        set_id(&scroll, page.id());
        window.center();
        Self { window, page }
    }
}

// ---- About -----------------------------------------------------------------------

/// The standard About panel: the icon, the name and version above, the rest
/// of the About page (authors, source, build features) in its small credits
/// box, which scrolls.
pub fn show_about(mtm: MainThreadMarker, doc: &iris_gui_core::controller::help::HelpDoc, version: &str, commit: &str) {
    use iris_gui_core::controller::help::Line;
    use objc2_foundation::{NSAttributedString, NSDictionary, NSMutableAttributedString, NSRange};
    let credits = NSMutableAttributedString::new();
    let centre = objc2_app_kit::NSMutableParagraphStyle::new();
    centre.setAlignment(objc2_app_kit::NSTextAlignment::Center);
    centre.setParagraphSpacing(2.0);
    let add = |text: &str, font: Retained<NSFont>, color: Retained<NSColor>, link: Option<&str>| {
        let start = credits.length();
        credits.appendAttributedString(&NSAttributedString::initWithString(<NSAttributedString as objc2::AnyThread>::alloc(), &ns(&format!("{text}\n"))));
        let range = NSRange::new(start, credits.length() - start);
        // SAFETY: each value has the type its attribute key expects.
        unsafe {
            credits.addAttribute_value_range(objc2_app_kit::NSFontAttributeName, &font, range);
            credits.addAttribute_value_range(objc2_app_kit::NSForegroundColorAttributeName, &color, range);
            credits.addAttribute_value_range(objc2_app_kit::NSParagraphStyleAttributeName, &centre, range);
            if let Some(url) = link.and_then(|u| objc2_foundation::NSURL::URLWithString(&ns(u))) {
                credits.addAttribute_value_range(objc2_app_kit::NSLinkAttributeName, &url, range);
            }
        }
    };
    let small = || NSFont::systemFontOfSize(11.0);
    let bold = || NSFont::boldSystemFontOfSize(11.0);
    // The page's heading and version line are the panel's own name and version.
    for line in doc.lines.iter().skip(2) {
        match line {
            Line::Heading(t) | Line::Strong(t) => add(t, bold(), NSColor::labelColor(), None),
            Line::Text(t) | Line::Code(t) => add(t, small(), NSColor::labelColor(), None),
            Line::Weak(t) => add(t, small(), NSColor::secondaryLabelColor(), None),
            Line::Warn(t) => add(t, small(), NSColor::systemOrangeColor(), None),
            Line::Error(t) => add(t, small(), NSColor::systemRedColor(), None),
            Line::Link { label, url } => add(label, small(), NSColor::linkColor(), Some(url)),
            Line::Rule => add("", small(), NSColor::labelColor(), None),
        }
    }
    let name = ns(&t!("app-name"));
    let ver = ns(version);
    let build = ns(commit);
    let app = objc2_app_kit::NSApplication::sharedApplication(mtm);
    // The app's icon as the shell set it (an unbundled build has no
    // bundle icon for the panel to find).
    let icon = app.applicationIconImage();
    // SAFETY: the keys are AppKit's constant strings.
    let mut keys: Vec<&NSString> = unsafe {
        vec![
            objc2_app_kit::NSAboutPanelOptionApplicationName,
            objc2_app_kit::NSAboutPanelOptionApplicationVersion,
            objc2_app_kit::NSAboutPanelOptionVersion,
            objc2_app_kit::NSAboutPanelOptionCredits,
        ]
    };
    let mut values: Vec<&objc2::runtime::AnyObject> = vec![&name, &ver, &build, &credits];
    if let Some(icon) = icon.as_deref() {
        // SAFETY: as above.
        keys.push(unsafe { objc2_app_kit::NSAboutPanelOptionApplicationIcon });
        values.push(icon);
    }
    let options = NSDictionary::from_slices(&keys, &values);
    // SAFETY: every key gets the type the About panel documents for it.
    unsafe { app.orderFrontStandardAboutPanelWithOptions(&options) };
}
