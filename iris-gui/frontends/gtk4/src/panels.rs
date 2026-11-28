//! The windows that follow a controller of their own: the serial console,
//! Check Networking, the camera test, and the Help pages.
//!
//! Each is an `AdwDialog` over the main window, shown exactly while its
//! controller says it is open, so closing one is an action like any other.

use std::cell::{Cell, RefCell};

use adw::prelude::*;
use gtk::{gdk, glib};
use gtk4 as gtk;
use iris_gui_core::controller::app::AppController;
use iris_gui_core::controller::help::{HelpDoc, HelpPage, Line};
use iris_gui_core::controller::network::{CheckLine, NetCheck, NetworkAction};
use iris_gui_core::controller::serial::SerialAction;
use libadwaita as adw;

/// A dialog with a header bar and a scrolling body.
fn shell_dialog(title: &str, width: i32, height: i32) -> (adw::Dialog, adw::ToolbarView, gtk::Box) {
    let header = adw::HeaderBar::new();
    header.set_title_widget(Some(&adw::WindowTitle::new(title, "")));
    let body = gtk::Box::new(gtk::Orientation::Vertical, 12);
    body.set_margin_top(12);
    body.set_margin_bottom(12);
    body.set_margin_start(12);
    body.set_margin_end(12);
    let scroll = gtk::ScrolledWindow::builder().hscrollbar_policy(gtk::PolicyType::Never).vexpand(true).child(&body).build();
    let view = adw::ToolbarView::new();
    view.add_top_bar(&header);
    view.set_content(Some(&scroll));
    let dialog = adw::Dialog::builder().title(title).content_width(width).content_height(height).build();
    dialog.set_child(Some(&view));
    (dialog, view, body)
}

// ---- the serial console ----------------------------------------------------

pub struct SerialWindow {
    pub dialog: adw::Dialog,
    text: gtk::TextView,
    status: gtk::Label,
    entry: gtk::Entry,
    seq: Cell<u64>,
}

impl SerialWindow {
    pub fn new() -> std::rc::Rc<Self> {
        let (dialog, view, body) = shell_dialog(&t!("tb-serial-console"), 760, 520);
        let text = gtk::TextView::builder().editable(false).monospace(true).wrap_mode(gtk::WrapMode::WordChar).build();
        let scroll = gtk::ScrolledWindow::builder().vexpand(true).child(&text).build();
        scroll.add_css_class("card");
        scroll.set_size_request(-1, 320);
        body.append(&scroll);

        let status = gtk::Label::new(None);
        status.set_xalign(0.0);
        status.add_css_class("dim-label");
        body.append(&status);

        let entry = gtk::Entry::builder().placeholder_text(t!("window-serial-send")).hexpand(true).build();
        let row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        row.append(&entry);
        let ctrl_c = gtk::Button::with_label("Ctrl+C");
        ctrl_c.connect_clicked(|_| crate::app::dispatch_serial(SerialAction::SendBytes(vec![3])));
        row.append(&ctrl_c);
        let clear = gtk::Button::with_label(&t!("window-serial-clear"));
        clear.connect_clicked(|_| crate::app::dispatch_serial(SerialAction::Clear));
        row.append(&clear);
        view.add_bottom_bar(&row);

        let me = std::rc::Rc::new(Self { dialog, text, status, entry, seq: Cell::new(u64::MAX) });
        me.entry.connect_activate(|e| {
            let line = e.text().to_string();
            e.set_text("");
            crate::app::dispatch_serial(SerialAction::SendLine(line));
        });
        me.dialog.connect_closed(|_| crate::app::dispatch_serial(SerialAction::Close));
        me
    }

    pub fn refresh(&self, app: &AppController) {
        let Some(view) = app.serial().view() else { return };
        if self.seq.replace(view.seq) != view.seq {
            let buffer = self.text.buffer();
            buffer.set_text(&view.text);
            // Follow the tail, as a console does.
            let end = buffer.end_iter();
            self.text.scroll_to_iter(&mut end.clone(), 0.0, false, 0.0, 1.0);
        }
        self.status.set_label(&view.status);
        if view.error {
            self.status.add_css_class("error");
        } else {
            self.status.remove_css_class("error");
        }
        self.entry.set_sensitive(view.connected);
    }
}

// ---- Check Networking ------------------------------------------------------

pub struct NetWindow {
    pub dialog: adw::Dialog,
    body: gtk::Box,
    shown: RefCell<Option<NetCheck>>,
}

impl NetWindow {
    pub fn new() -> std::rc::Rc<Self> {
        let (dialog, _view, body) = shell_dialog(&t!("mb-check-network"), 640, 560);
        let me = std::rc::Rc::new(Self { dialog, body, shown: RefCell::default() });
        me.dialog.connect_closed(|_| crate::app::dispatch_network(NetworkAction::CloseCheck));
        me
    }

    pub fn refresh(&self, check: NetCheck) {
        if self.shown.borrow().as_ref() == Some(&check) {
            return;
        }
        while let Some(child) = self.body.first_child() {
            self.body.remove(&child);
        }
        for line in &check.lines {
            self.body.append(&check_line(line));
        }
        if let Some(cidr) = &check.adopt {
            let button = gtk::Button::with_label(&t!("window-net-use-guest-subnet", network = cidr.as_str()));
            button.add_css_class("suggested-action");
            let cidr = cidr.clone();
            button.connect_clicked(move |_| crate::app::dispatch_network(NetworkAction::AdoptSubnet(cidr.clone())));
            self.body.append(&button);
        }
        *self.shown.borrow_mut() = Some(check);
    }
}

fn check_line(line: &CheckLine) -> gtk::Widget {
    let (text, class) = match line {
        CheckLine::Good(t) => (t, Some("success")),
        CheckLine::Bad(t) => (t, Some("error")),
        CheckLine::Text(t) => (t, None),
        CheckLine::Weak(t) => (t, Some("dim-label")),
        CheckLine::Code(t) => (t, Some("monospace")),
        CheckLine::Strong(t) => (t, Some("heading")),
    };
    let label = gtk::Label::new(Some(text));
    label.set_xalign(0.0);
    label.set_wrap(true);
    label.set_selectable(matches!(line, CheckLine::Code(_)));
    if let Some(class) = class {
        label.add_css_class(class);
    }
    label.upcast()
}

// ---- the camera test -------------------------------------------------------

pub struct CameraWindow {
    pub dialog: adw::Dialog,
    picture: gtk::Picture,
    status: gtk::Label,
    seq: Cell<u64>,
}

impl CameraWindow {
    pub fn new() -> std::rc::Rc<Self> {
        let (dialog, _view, body) = shell_dialog(&t!("mb-camera-test"), 680, 560);
        let picture = gtk::Picture::builder().vexpand(true).content_fit(gtk::ContentFit::Contain).build();
        picture.add_css_class("iris-screen");
        picture.set_size_request(-1, 360);
        body.append(&picture);
        let status = gtk::Label::new(None);
        status.set_wrap(true);
        status.set_xalign(0.0);
        body.append(&status);
        let me = std::rc::Rc::new(Self { dialog, picture, status, seq: Cell::new(0) });
        me.dialog.connect_closed(|_| crate::app::dispatch_camera());
        me
    }

    pub fn refresh(&self, app: &AppController) {
        if let Some((status, bad)) = app.camera().status() {
            self.status.set_label(&status);
            if bad {
                self.status.add_css_class("error");
            } else {
                self.status.remove_css_class("error");
            }
        }
        let mut seq = self.seq.get();
        if let Some((w, h, rgba)) = app.camera().take_new_frame(&mut seq) {
            self.seq.set(seq);
            let bytes = glib::Bytes::from_owned(rgba);
            let texture = gdk::MemoryTexture::new(w as i32, h as i32, gdk::MemoryFormat::R8g8b8a8, &bytes, w as usize * 4);
            self.picture.set_paintable(Some(&texture));
        }
    }
}

// ---- Help ------------------------------------------------------------------

pub struct HelpWindow {
    pub dialog: adw::Dialog,
    pub page: HelpPage,
}

impl HelpWindow {
    pub fn new(page: HelpPage, doc: &HelpDoc) -> Self {
        let (dialog, _view, body) = shell_dialog(&doc.title, 680, 620);
        for line in &doc.lines {
            body.append(&help_line(line));
        }
        Self { dialog, page }
    }
}

fn help_line(line: &Line) -> gtk::Widget {
    match line {
        Line::Rule => gtk::Separator::new(gtk::Orientation::Horizontal).upcast(),
        Line::Link { label, url } => {
            let button = gtk::LinkButton::with_label(url, label);
            button.set_halign(gtk::Align::Start);
            button.upcast()
        }
        other => {
            let (text, class) = match other {
                Line::Heading(t) => (t, Some("title-4")),
                Line::Strong(t) => (t, Some("heading")),
                Line::Text(t) => (t, None),
                Line::Weak(t) => (t, Some("dim-label")),
                Line::Warn(t) => (t, Some("warning")),
                Line::Error(t) => (t, Some("error")),
                Line::Code(t) => (t, Some("monospace")),
                Line::Rule | Line::Link { .. } => unreachable!("handled above"),
            };
            let label = gtk::Label::new(Some(text));
            label.set_xalign(0.0);
            label.set_wrap(true);
            label.set_selectable(true);
            if let Some(class) = class {
                label.add_css_class(class);
            }
            if matches!(other, Line::Code(_)) {
                label.add_css_class("iris-code");
            }
            label.upcast()
        }
    }
}

// ---- the benchmark ---------------------------------------------------------

use iris_gui_core::controller::bench::{BenchAction, BenchController};
use iris_gui_core::controller::Notify;

pub struct BenchWindow {
    pub dialog: adw::Dialog,
    pub ctl: RefCell<BenchController>,
    start: gtk::Button,
    quick: gtk::CheckButton,
    cpu: gtk::DropDown,
    progress: gtk::ProgressBar,
    status: gtk::Label,
    text: gtk::TextView,
    save: gtk::Button,
    /// The finished run's figures, replaced when a run ends.
    results: gtk::Box,
    shown: RefCell<Option<iris_gui_core::controller::bench::Results>>,
}

impl BenchWindow {
    pub fn new(notify: std::sync::Arc<dyn Notify>) -> std::rc::Rc<Self> {
        let (dialog, _view, body) = shell_dialog(&t!("bench-heading"), 720, 620);

        let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        let ctl = BenchController::new(notify);
        let (cpus, selected) = ctl.cpus();
        let model = gtk::StringList::new(&[]);
        for c in &cpus {
            model.append(c);
        }
        let cpu = gtk::DropDown::builder().model(&model).selected(selected as u32).tooltip_text(t!("bench-cpu-tip")).build();
        row.append(&cpu);
        let quick = gtk::CheckButton::with_label(&t!("bench-quick"));
        quick.set_active(ctl.quick());
        quick.set_tooltip_text(Some(&t!("bench-quick-tip")));
        row.append(&quick);
        let start = gtk::Button::with_label(&t!("bench-run-pc"));
        start.add_css_class("suggested-action");
        start.set_hexpand(true);
        start.set_halign(gtk::Align::End);
        row.append(&start);
        body.append(&row);

        let progress = gtk::ProgressBar::new();
        progress.set_show_text(true);
        progress.set_visible(false);
        body.append(&progress);
        let status = gtk::Label::new(None);
        status.set_xalign(0.0);
        status.set_wrap(true);
        body.append(&status);

        let results = gtk::Box::new(gtk::Orientation::Vertical, 10);
        body.append(&results);

        let text = gtk::TextView::builder().editable(false).monospace(true).build();
        let scroll = gtk::ScrolledWindow::builder().vexpand(true).child(&text).build();
        scroll.add_css_class("card");
        scroll.set_size_request(-1, 220);
        body.append(&scroll);

        let save = gtk::Button::with_label(&t!("bench-save-report"));
        save.set_halign(gtk::Align::End);
        save.set_sensitive(false);
        save.connect_clicked(|_| crate::app::bench(BenchAction::SaveReport));
        body.append(&save);

        let me = std::rc::Rc::new(Self {
            dialog,
            ctl: RefCell::new(ctl),
            start,
            quick,
            cpu,
            progress,
            status,
            text,
            save,
            results,
            shown: RefCell::default(),
        });
        let w = std::rc::Rc::downgrade(&me);
        me.start.connect_clicked(move |_| {
            if let Some(me) = w.upgrade() {
                let running = me.ctl.borrow().is_running();
                crate::app::bench(if running { BenchAction::Stop } else { BenchAction::Start });
            }
        });
        me.quick.connect_toggled(|b| crate::app::bench(BenchAction::SetQuick(b.is_active())));
        me.cpu.connect_selected_notify(|d| crate::app::bench(BenchAction::SetCpu(d.selected() as usize)));
        me
    }

    pub fn refresh(&self, machine_running: bool) {
        // A finished run moves into `last` only when the controller is
        // ticked; without this there are never any figures and Save report
        // never lights up.
        self.ctl.borrow_mut().tick();
        // The figures a finished run produced, from the controller.
        let results = self.ctl.borrow_mut().results();
        if *self.shown.borrow() != results {
            while let Some(child) = self.results.first_child() {
                self.results.remove(&child);
            }
            if let Some(r) = &results {
                self.results.append(&rows_group(None, &r.headline, true));
                if !r.categories.is_empty() {
                    self.results.append(&rows_group(Some(&t!("bench-results")), &r.categories, false));
                }
                for line in &r.caveats {
                    let label = gtk::Label::new(Some(line));
                    label.set_xalign(0.0);
                    label.set_wrap(true);
                    label.add_css_class("dim-label");
                    self.results.append(&label);
                }
                if !r.machine.is_empty() {
                    let group = rows_group(Some(&t!("bench-machine")), &r.machine, false);
                    self.results.append(&group);
                }
            }
            *self.shown.borrow_mut() = results;
        }
        let ctl = self.ctl.borrow();
        self.save.set_sensitive(!ctl.is_running() && matches!(ctl.last(), Some(Ok(_))));
        let running = ctl.is_running();
        self.start.set_label(&if running { t!("bench-stop") } else { t!("bench-run-pc") });
        let can = ctl.can_start(machine_running);
        self.start.set_sensitive(running || can.is_yes());
        self.start.set_tooltip_text(can.reason());
        self.quick.set_sensitive(!running);
        self.cpu.set_sensitive(!running);
        match ctl.progress() {
            Some(p) => {
                self.progress.set_visible(true);
                self.progress.set_fraction(p.fraction as f64);
                self.progress.set_text(Some(&p.line));
                self.status.set_label(&p.time);
            }
            None => {
                self.progress.set_visible(false);
                self.status.set_label(match ctl.last() {
                    Some(Err(e)) => e,
                    _ => "",
                });
            }
        }
        let lines = ctl.lines().join("\n");
        let buffer = self.text.buffer();
        if buffer.text(&buffer.start_iter(), &buffer.end_iter(), false) != lines {
            buffer.set_text(&lines);
            let end = buffer.end_iter();
            self.text.scroll_to_iter(&mut end.clone(), 0.0, false, 0.0, 1.0);
        }
    }
}

/// A titled group of result rows: the label, the value, and a comparison.
fn rows_group(title: Option<&str>, rows: &[iris_gui_core::controller::bench::ResultRow], big: bool) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    if let Some(title) = title {
        group.set_title(title);
    }
    for r in rows {
        let row = adw::ActionRow::builder().title(&r.label).build();
        if !r.note.is_empty() {
            row.set_subtitle(&r.note);
        }
        let value = gtk::Label::new(Some(&r.value));
        value.add_css_class("numeric");
        if big {
            value.add_css_class("title-4");
        } else {
            value.add_css_class("monospace");
        }
        value.set_valign(gtk::Align::Center);
        row.add_suffix(&value);
        if let Some(tip) = &r.tip {
            row.set_tooltip_text(Some(tip));
        }
        group.add(&row);
    }
    group
}
