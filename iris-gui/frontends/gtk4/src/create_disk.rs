//! The blank-image form: the controller's fields, a progress bar while the
//! image is written, and Create. Shown exactly while the app controller has
//! one, like the disk browser.

use std::cell::RefCell;

use adw::prelude::*;
use gtk4 as gtk;
use iris_gui_core::controller::create_disk::{CreateDiskAction, CreateDiskController};
use libadwaita as adw;

use crate::settings::Form;

pub struct CreateDiskDialog {
    pub dialog: adw::Dialog,
    /// Holds the group; a redraw replaces it, because an
    /// AdwPreferencesGroup cannot be emptied (its `first_child` is its own
    /// box, which `remove` refuses — clearing it in place never ends).
    body: gtk::Box,
    group: RefCell<adw::PreferencesGroup>,
    create: gtk::Button,
    cancel: gtk::Button,
    progress: gtk::ProgressBar,
    form: std::rc::Rc<Form>,
    /// The fields as drawn, so an unchanged tick redraws nothing.
    shown: RefCell<Vec<iris_gui_core::controller::settings::Field>>,
}

impl CreateDiskDialog {
    pub fn new(title: &str) -> std::rc::Rc<Self> {
        let header = adw::HeaderBar::new();
        header.set_title_widget(Some(&adw::WindowTitle::new(&t!("create-disk-title"), title)));
        let create = gtk::Button::with_label(&t!("create-disk-create"));
        create.add_css_class("suggested-action");
        header.pack_end(&create);
        let cancel = gtk::Button::with_label(&t!("common-cancel"));
        cancel.set_visible(false);
        header.pack_start(&cancel);

        let group = adw::PreferencesGroup::new();
        let progress = gtk::ProgressBar::new();
        progress.set_show_text(true);
        progress.set_visible(false);
        let body = gtk::Box::new(gtk::Orientation::Vertical, 12);
        body.set_margin_top(12);
        body.set_margin_bottom(12);
        body.set_margin_start(12);
        body.set_margin_end(12);
        body.append(&group);
        body.append(&progress);

        let view = adw::ToolbarView::new();
        view.add_top_bar(&header);
        view.set_content(Some(&body));
        let dialog = adw::Dialog::builder().title(t!("create-disk-title")).content_width(560).build();
        dialog.set_child(Some(&view));

        let me = std::rc::Rc::new(Self {
            dialog,
            body: body.clone(),
            group: RefCell::new(group),
            create,
            cancel,
            progress,
            form: Form::new(|id, value| crate::app::create_disk_control(id, value)),
            shown: RefCell::default(),
        });
        me.create.connect_clicked(|_| crate::app::create_disk(CreateDiskAction::Create));
        me.cancel.connect_clicked(|_| crate::app::create_disk(CreateDiskAction::Cancel));
        me.dialog.connect_closed(|_| crate::app::create_disk(CreateDiskAction::Close));
        me
    }

    pub fn refresh(self: &std::rc::Rc<Self>, c: &CreateDiskController) {
        let fields = c.fields();
        if *self.shown.borrow() != fields {
            let fresh = adw::PreferencesGroup::new();
            self.form.fill(&fresh, &fields);
            let old = self.group.replace(fresh.clone());
            self.body.remove(&old);
            self.body.prepend(&fresh);
            *self.shown.borrow_mut() = fields;
        }
        let can = c.can_create();
        self.create.set_sensitive(can.is_yes());
        self.create.set_tooltip_text(can.reason());
        match c.job() {
            Some(job) => {
                self.progress.set_visible(true);
                self.cancel.set_visible(true);
                match job.progress.fraction() {
                    Some(f) => self.progress.set_fraction(f as f64),
                    None => self.progress.pulse(),
                }
                self.progress.set_text(Some(&job.label));
            }
            None => {
                self.progress.set_visible(false);
                self.cancel.set_visible(false);
            }
        }
    }
}
