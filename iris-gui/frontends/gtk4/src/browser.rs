//! The disk browser (D23): the device's partitions on the left, the chosen
//! one's files on the right, a path bar above them and the controller's
//! tools in the header. Everything it can do is a `BrowserAction`; what can
//! be done now, and why not, is the controller's `enabled`.
//!
//! A window of its own, shown exactly while the app controller has a
//! browser, so closing it is an action like any other.

use std::cell::{Cell, RefCell};

use adw::prelude::*;
use gtk::{gio, glib};
use gtk4 as gtk;
use iris_gui_core::controller::disk_browser::{action_for, ids, tools, BrowserAction, DiskBrowserController, Editor, Row};
use libadwaita as adw;

use crate::icons;

pub struct BrowserWindow {
    pub window: adw::Window,
    title: adw::WindowTitle,
    parts: gtk::ListBox,
    path_bar: gtk::Box,
    list: gtk::ListBox,
    note: gtk::Label,
    progress: gtk::ProgressBar,
    editor_box: gtk::Box,
    editor_text: gtk::TextView,
    editor_title: gtk::Label,
    toolbar: gtk::Box,
    /// What is drawn now, so an unchanged tick redraws nothing.
    shown_parts: RefCell<Vec<String>>,
    shown_rows: RefCell<Vec<Row>>,
    shown_path: RefCell<Vec<String>>,
    shown_editor: RefCell<Option<Editor>>,
    syncing: Cell<bool>,
}

impl BrowserWindow {
    pub fn new() -> std::rc::Rc<Self> {
        let title = adw::WindowTitle::new("", "");
        let header = adw::HeaderBar::new();
        header.set_title_widget(Some(&title));

        // The controller's tools, in its order, with its gaps.
        let toolbar = gtk::Box::new(gtk::Orientation::Horizontal, 4);
        toolbar.set_margin_start(8);
        toolbar.set_margin_end(8);
        toolbar.set_margin_top(6);
        toolbar.set_margin_bottom(6);

        let parts = gtk::ListBox::new();
        parts.add_css_class("navigation-sidebar");
        let parts_scroll = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vexpand(true)
            .width_request(240)
            .child(&parts)
            .build();

        let path_bar = gtk::Box::new(gtk::Orientation::Horizontal, 2);
        path_bar.set_margin_start(8);
        path_bar.set_margin_end(8);
        path_bar.set_margin_top(4);
        path_bar.set_margin_bottom(4);

        let list = gtk::ListBox::new();
        list.set_selection_mode(gtk::SelectionMode::Multiple);
        list.add_css_class("boxed-list");
        let list_scroll = gtk::ScrolledWindow::builder().vexpand(true).child(&list).build();
        list_scroll.set_margin_start(8);
        list_scroll.set_margin_end(8);

        let note = gtk::Label::new(None);
        note.set_wrap(true);
        note.set_xalign(0.0);
        note.set_margin_start(10);
        note.add_css_class("dim-label");

        let progress = gtk::ProgressBar::new();
        progress.set_visible(false);
        progress.set_margin_start(10);
        progress.set_margin_end(10);

        let editor_title = gtk::Label::new(None);
        editor_title.set_xalign(0.0);
        editor_title.add_css_class("heading");
        let editor_text = gtk::TextView::builder().monospace(true).build();
        let editor_scroll = gtk::ScrolledWindow::builder().vexpand(true).child(&editor_text).build();
        editor_scroll.add_css_class("card");
        let editor_buttons = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        editor_buttons.set_halign(gtk::Align::End);
        for (id, label, suggested) in [
            (ids::EDITOR_SAVE, t!("browser-editor-save"), true),
            (ids::EDITOR_CLOSE, t!("browser-editor-close"), false),
        ] {
            let b = gtk::Button::with_label(&label);
            if suggested {
                b.add_css_class("suggested-action");
            }
            b.set_widget_name(id);
            b.connect_clicked(move |_| crate::app::browser_control(id, false));
            editor_buttons.append(&b);
        }
        let editor_box = gtk::Box::new(gtk::Orientation::Vertical, 6);
        editor_box.set_margin_start(8);
        editor_box.set_margin_end(8);
        editor_box.set_margin_bottom(8);
        editor_box.set_visible(false);
        editor_box.append(&editor_title);
        editor_box.append(&editor_scroll);
        editor_box.append(&editor_buttons);

        let right = gtk::Box::new(gtk::Orientation::Vertical, 4);
        right.append(&path_bar);
        right.append(&list_scroll);
        right.append(&note);
        right.append(&progress);
        right.append(&editor_box);

        let split = adw::OverlaySplitView::builder()
            .sidebar(&parts_scroll)
            .content(&right)
            .min_sidebar_width(200.0)
            .max_sidebar_width(320.0)
            .build();

        let view = adw::ToolbarView::new();
        view.add_top_bar(&header);
        view.add_top_bar(&toolbar);
        view.set_content(Some(&split));

        let window = adw::Window::builder().default_width(980).default_height(680).content(&view).build();
        window.set_widget_name(ids::WINDOW);
        window.connect_close_request(|_| {
            crate::app::close_browser();
            glib::Propagation::Proceed
        });

        let me = std::rc::Rc::new(Self {
            window,
            title,
            parts,
            path_bar,
            list,
            note,
            progress,
            editor_box,
            editor_text,
            editor_title,
            toolbar,
            shown_parts: RefCell::default(),
            shown_rows: RefCell::default(),
            shown_path: RefCell::default(),
            shown_editor: RefCell::default(),
            syncing: Cell::new(false),
        });

        me.parts.connect_row_selected(glib::clone!(
            #[weak]
            me,
            move |_, row| {
                if me.syncing.get() {
                    return;
                }
                if let Some(row) = row {
                    let i = row.index() as usize;
                    crate::app::later(move || crate::app::browser(BrowserAction::SelectPartition(i)));
                }
            }
        ));
        me.list.connect_selected_rows_changed(glib::clone!(
            #[weak]
            me,
            move |list| {
                if me.syncing.get() {
                    return;
                }
                let rows: Vec<usize> = list.selected_rows().iter().map(|r| r.index() as usize).collect();
                crate::app::later(move || crate::app::browser(BrowserAction::Select(rows.clone())));
            }
        ));
        me.list.connect_row_activated(glib::clone!(
            #[weak]
            me,
            move |_, row| {
                let _ = &me;
                let i = row.index() as usize;
                crate::app::later(move || crate::app::browser(BrowserAction::Activate(i)));
            }
        ));

        // Files dragged from the host's file manager land in the folder shown.
        let drop = gtk::DropTarget::new(gio::File::static_type(), gtk::gdk::DragAction::COPY);
        drop.connect_drop(|_, value, _, _| {
            if let Ok(file) = value.get::<gio::File>() {
                if let Some(path) = file.path() {
                    crate::app::later(move || crate::app::browser(BrowserAction::Drop { paths: vec![path.clone()], into: None }));
                    return true;
                }
            }
            false
        });
        me.list.add_controller(drop);

        me
    }

    /// Bring the window in line with the controller.
    pub fn refresh(self: &std::rc::Rc<Self>, b: &DiskBrowserController) {
        self.syncing.set(true);
        self.title.set_title(&b.title());
        self.title.set_subtitle(&b.target().machine);

        // The tools: the controller says which can be used now and why not.
        if self.toolbar.first_child().is_none() {
            for tool in tools() {
                match tool {
                    Some(tool) => {
                        let button = gtk::Button::builder()
                            .child(&icons::image(tool.icon, 16, &tool.label))
                            .tooltip_text(&tool.label)
                            .build();
                        button.add_css_class("flat");
                        button.set_widget_name(tool.id);
                        let id = tool.id;
                        button.connect_clicked(move |_| crate::app::browser_control(id, false));
                        self.toolbar.append(&button);
                    }
                    None => {
                        let gap = gtk::Separator::new(gtk::Orientation::Vertical);
                        gap.set_margin_start(4);
                        gap.set_margin_end(4);
                        self.toolbar.append(&gap);
                    }
                }
            }
        }
        for tool in tools().into_iter().flatten() {
            if let Some(w) = self.tool_widget(tool.id) {
                let enabled = b.enabled(&tool.action);
                w.set_sensitive(enabled.is_yes());
                w.set_tooltip_text(Some(enabled.reason().unwrap_or(&tool.label)));
            }
        }

        let parts: Vec<String> = b.partitions().iter().map(|p| format!("{}|{}|{}", p.name, p.detail, p.browsable)).collect();
        if *self.shown_parts.borrow() != parts {
            while let Some(child) = self.parts.first_child() {
                self.parts.remove(&child);
            }
            for p in b.partitions() {
                let row = adw::ActionRow::builder().title(&p.name).subtitle(&p.detail).build();
                row.set_subtitle_lines(0);
                row.set_sensitive(p.browsable);
                let holder = gtk::ListBoxRow::builder().child(&row).selectable(p.browsable).build();
                holder.set_widget_name(&format!("{}{}", ids::PART, p.index));
                self.parts.append(&holder);
            }
            *self.shown_parts.borrow_mut() = parts;
        }
        if let Some(i) = b.selected_partition() {
            if self.parts.selected_row().map(|r| r.index() as usize) != Some(i) {
                if let Some(row) = self.parts.row_at_index(i as i32) {
                    self.parts.select_row(Some(&row));
                }
            }
        }

        let path = b.path();
        if *self.shown_path.borrow() != path {
            while let Some(child) = self.path_bar.first_child() {
                self.path_bar.remove(&child);
            }
            for (depth, part) in path.iter().enumerate() {
                if depth > 0 {
                    self.path_bar.append(&gtk::Label::new(Some("/")));
                }
                let button = gtk::Button::with_label(part);
                button.add_css_class("flat");
                let id = format!("{}{}", ids::PATH, depth);
                button.set_widget_name(&id);
                button.connect_clicked(move |_| crate::app::browser_control(&id, false));
                self.path_bar.append(&button);
            }
            *self.shown_path.borrow_mut() = path;
        }

        let rows = b.rows();
        if *self.shown_rows.borrow() != rows {
            while let Some(child) = self.list.first_child() {
                self.list.remove(&child);
            }
            for r in &rows {
                let row = adw::ActionRow::builder().title(glib::markup_escape_text(&r.name).as_str()).build();
                row.set_subtitle(&format!("{}  {}  {}:{}  {}", r.mode, r.size, r.owner, r.group, r.date));
                row.add_prefix(&icons::image(r.icon, 16, ""));
                row.set_activatable(true);
                self.list.append(&row);
            }
            *self.shown_rows.borrow_mut() = rows;
        }
        let want: Vec<usize> = b.selection().to_vec();
        let have: Vec<usize> = self.list.selected_rows().iter().map(|r| r.index() as usize).collect();
        if want != have {
            self.list.unselect_all();
            for i in &want {
                if let Some(row) = self.list.row_at_index(*i as i32) {
                    self.list.select_row(Some(&row));
                }
            }
        }

        match (b.read_only_reason(), b.error()) {
            (_, Some(e)) => {
                self.note.set_label(e);
                self.note.add_css_class("error");
                self.note.set_visible(true);
            }
            (Some(why), None) => {
                self.note.set_label(&why);
                self.note.remove_css_class("error");
                self.note.set_visible(true);
            }
            (None, None) => self.note.set_visible(false),
        }

        match b.job() {
            Some(job) => {
                self.progress.set_visible(true);
                match job.progress.fraction() {
                    Some(f) => self.progress.set_fraction(f as f64),
                    None => self.progress.pulse(),
                }
                self.progress.set_text(Some(&format!("{} {}", job.label, job.phase)));
                self.progress.set_show_text(true);
            }
            None => self.progress.set_visible(false),
        }

        let editor = b.editor().cloned();
        if *self.shown_editor.borrow() != editor {
            match &editor {
                Some(e) => {
                    self.editor_box.set_visible(true);
                    let endings = if e.read_only { t!("browser-editor-read-only") } else { e.line_endings.to_string() };
                    self.editor_title.set_label(&format!("{}  ·  {endings}", e.name));
                    if self.editor_text.buffer().text(&self.editor_text.buffer().start_iter(), &self.editor_text.buffer().end_iter(), false) != e.text {
                        self.editor_text.buffer().set_text(&e.text);
                    }
                    self.editor_text.set_editable(!e.read_only);
                }
                None => self.editor_box.set_visible(false),
            }
            *self.shown_editor.borrow_mut() = editor;
        }
        self.syncing.set(false);
    }

    fn tool_widget(&self, id: &str) -> Option<gtk::Widget> {
        let mut child = self.toolbar.first_child();
        while let Some(w) = child {
            if w.widget_name() == id {
                return Some(w);
            }
            child = w.next_sibling();
        }
        None
    }

    /// The editor's text, for Save.
    pub fn editor_text(&self) -> String {
        let buffer = self.editor_text.buffer();
        buffer.text(&buffer.start_iter(), &buffer.end_iter(), false).to_string()
    }

    /// Fill the editor (a test script's `type:` step).
    pub fn set_editor_text(&self, text: &str) {
        self.editor_text.buffer().set_text(text);
    }
}

/// The action a control's id means, with the editor's text where Save needs it.
pub fn control(id: &str, value: bool) -> Option<BrowserAction> {
    action_for(id, value)
}
