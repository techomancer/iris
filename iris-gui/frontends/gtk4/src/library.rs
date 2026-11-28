//! The machine library on the left: a `GtkListBox` in the sidebar, over the
//! app controller's groups (the user's folders first, then families). A row
//! has the machine's picture, its name, the model line, a dot only when it is
//! running or has a problem, and its own Start button.

use std::cell::RefCell;

use adw::prelude::*;
use gtk::{gio, glib};
use gtk4 as gtk;
use iris_gui_core::controller::app::{LibraryGroup, LibraryRow, RowDot};
use iris_gui_core::controller::icons::Icon;
use iris_gui_core::controller::commands::menu as m;
use iris_gui_core::controller::ids;
use libadwaita as adw;

use crate::icons;

pub struct Library {
    pub root: gtk::Widget,
    list: gtk::ListBox,
    /// What the list shows now, so a refresh that changes nothing does not
    /// rebuild it (and lose the keyboard focus).
    shown: RefCell<Vec<LibraryGroup>>,
    /// Row index -> machine name, for selection and the context menu.
    rows: RefCell<Vec<Option<String>>>,
    /// Set while the shell moves the selection itself, so it is not reported.
    syncing: std::cell::Cell<bool>,
}

impl Library {
    pub fn new(app: &adw::Application) -> std::rc::Rc<Self> {
        let list = gtk::ListBox::new();
        list.set_selection_mode(gtk::SelectionMode::Single);
        // The GNOME sidebar look: inset rounded selections, no frame.
        list.add_css_class("navigation-sidebar");
        list.set_widget_name(ids::library::LIST);

        let scroll = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vexpand(true)
            .child(&list)
            .build();

        // The sidebar's own header: the title, and New Machine / New Folder.
        let add = gio::Menu::new();
        add.append_item(&menu_item(app, &t!("mb-new-machine"), m::NEW_MACHINE));
        add.append_item(&menu_item(app, &t!("mb-new-folder"), m::NEW_FOLDER));
        let add_button = gtk::MenuButton::builder()
            .icon_name("list-add-symbolic")
            .tooltip_text(t!("mb-new-machine"))
            .menu_model(&add)
            .build();
        add_button.set_widget_name(ids::library::NEW_MACHINE);
        let header = adw::HeaderBar::builder().show_end_title_buttons(false).build();
        header.set_title_widget(Some(&adw::WindowTitle::new(&t!("library-title"), "")));
        header.pack_end(&add_button);

        let view = adw::ToolbarView::new();
        view.add_top_bar(&header);
        view.set_content(Some(&scroll));

        let me = std::rc::Rc::new(Self {
            root: view.upcast(),
            list: list.clone(),
            shown: RefCell::default(),
            rows: RefCell::default(),
            syncing: std::cell::Cell::new(false),
        });

        list.connect_row_selected(glib::clone!(
            #[weak]
            me,
            move |_, row| {
                if me.syncing.get() {
                    return;
                }
                let Some(name) = row.and_then(|r| me.rows.borrow().get(r.index() as usize).cloned().flatten()) else {
                    return;
                };
                // After this handler: selecting changes the window's contents.
                crate::app::later(move || crate::app::select_machine(name));
            }
        ));

        // Right-click (or a long press on a touch screen) for what you can do
        // with the machine under the pointer.
        let click = gtk::GestureClick::new();
        click.set_button(gtk::gdk::BUTTON_SECONDARY);
        click.connect_pressed(glib::clone!(
            #[weak]
            me,
            move |g, _, x, y| {
                let Some(row) = me.list.row_at_y(y as i32) else { return };
                let Some(name) = me.rows.borrow().get(row.index() as usize).cloned().flatten() else { return };
                g.set_state(gtk::EventSequenceState::Claimed);
                me.row_menu(&row, &name, x, y);
            }
        ));
        list.add_controller(click);
        me
    }

    /// The row's own menu, from `AppController::machine_menu`.
    fn row_menu(&self, row: &gtk::ListBoxRow, name: &str, x: f64, y: f64) {
        let Some(sections) = crate::app::machine_menu(name) else { return };
        let Some(app) = crate::app::application() else { return };
        let model = gio::Menu::new();
        for section in &sections {
            let s = gio::Menu::new();
            for e in section {
                s.append_item(&menu_item(&app, &e.label, &e.id));
            }
            model.append_section(None, &s);
        }
        let popover = gtk::PopoverMenu::from_model(Some(&model));
        popover.set_parent(&self.list);
        popover.set_has_arrow(false);
        popover.set_halign(gtk::Align::Start);
        let _ = (row, x);
        popover.set_pointing_to(Some(&gtk::gdk::Rectangle::new(x as i32, y as i32, 1, 1)));
        // The popover is the list's child until it closes.
        popover.connect_closed(|p| p.unparent());
        popover.popup();
    }

    /// Show `groups`. Rebuilds only when something changed, so the selection
    /// and the scroll position survive a tick.
    pub fn refresh(&self, groups: Vec<LibraryGroup>) {
        if *self.shown.borrow() == groups {
            self.sync_selection(&groups);
            return;
        }
        let app = crate::app::application();
        while let Some(child) = self.list.first_child() {
            self.list.remove(&child);
        }
        let mut rows = Vec::new();
        for g in &groups {
            self.list.append(&group_row(&g.title));
            rows.push(None);
            if g.collapsed {
                continue;
            }
            for r in &g.rows {
                self.list.append(&machine_row(app.as_ref(), r));
                rows.push(Some(r.name.clone()));
            }
        }
        *self.rows.borrow_mut() = rows;
        *self.shown.borrow_mut() = groups.clone();
        self.sync_selection(&groups);
    }

    fn sync_selection(&self, groups: &[LibraryGroup]) {
        let selected = groups.iter().flat_map(|g| &g.rows).find(|r| r.selected).map(|r| r.name.clone());
        let want = selected.and_then(|name| self.rows.borrow().iter().position(|r| r.as_deref() == Some(&*name)));
        let have = self.list.selected_row().map(|r| r.index() as usize);
        if want == have {
            return;
        }
        self.syncing.set(true);
        match want.and_then(|i| self.list.row_at_index(i as i32)) {
            Some(row) => self.list.select_row(Some(&row)),
            None => self.list.unselect_all(),
        }
        self.syncing.set(false);
    }
}

fn menu_item(app: &adw::Application, label: &str, id: &str) -> gio::MenuItem {
    let it = gio::MenuItem::new(Some(label), None);
    it.set_detailed_action(&format!("app.{}", crate::menu::action_for(app, id)));
    it
}

fn group_row(title: &str) -> gtk::ListBoxRow {
    let label = gtk::Label::new(Some(&title.to_uppercase()));
    label.set_xalign(0.0);
    label.add_css_class("dim-label");
    label.add_css_class("heading");
    label.set_margin_top(10);
    label.set_margin_bottom(2);
    label.set_margin_start(6);
    let row = gtk::ListBoxRow::builder().child(&label).selectable(false).activatable(false).build();
    row.add_css_class("iris-group");
    row
}

fn machine_row(app: Option<&adw::Application>, m: &LibraryRow) -> gtk::ListBoxRow {
    let picture = icons::machine(m.icon, 32);
    picture.set_valign(gtk::Align::Center);
    picture.add_css_class("dim-label");

    let name = gtk::Label::new(Some(&m.name));
    name.set_xalign(0.0);
    name.set_ellipsize(gtk::pango::EllipsizeMode::End);
    let sub = gtk::Label::new(Some(&m.subtitle));
    sub.set_xalign(0.0);
    sub.set_ellipsize(gtk::pango::EllipsizeMode::End);
    sub.add_css_class("dim-label");
    sub.add_css_class("caption");

    let text = gtk::Box::new(gtk::Orientation::Vertical, 0);
    text.set_hexpand(true);
    text.set_valign(gtk::Align::Center);
    text.append(&name);
    text.append(&sub);

    let row_box = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    row_box.set_margin_top(4);
    row_box.set_margin_bottom(4);
    row_box.append(&picture);
    row_box.append(&text);

    match &m.dot {
        Some(RowDot::Running) => row_box.append(&dot("success", &t!("status-running"))),
        Some(RowDot::Problem(why)) => row_box.append(&dot("warning", why)),
        // A machine that is not running has its own Start, as UTM does.
        None => {
            if let Some(app) = app {
                let id = ids::item::id(ids::item::START_MACHINE, &m.name);
                let tip = t!("tb-start-machine", name = m.name.as_str());
                let start = gtk::Button::builder()
                    .child(&icons::image(Icon::PowerOn, 16, &tip))
                    .tooltip_text(&tip)
                    .valign(gtk::Align::Center)
                    .action_name(format!("app.{}", crate::menu::action_for(app, &id)))
                    .build();
                start.add_css_class("flat");
                start.add_css_class("circular");
                start.set_widget_name(&id);
                row_box.append(&start);
            }
        }
    }

    let row = gtk::ListBoxRow::builder().child(&row_box).build();
    row.set_widget_name(&format!("library.row.{}", m.name));
    row.set_tooltip_text(Some(&m.subtitle));
    row
}

fn dot(class: &str, tip: &str) -> gtk::Widget {
    let label = gtk::Label::new(Some("\u{25CF}"));
    label.add_css_class(class);
    label.set_valign(gtk::Align::Center);
    label.set_tooltip_text(Some(tip));
    label.upcast()
}
