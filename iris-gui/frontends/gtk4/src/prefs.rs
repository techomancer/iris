//! The application preferences: the controller's tabs of fields, drawn by
//! the same [`Form`](crate::settings::Form) the machine settings use.

use std::cell::RefCell;

use adw::prelude::*;
use iris_gui_core::controller::app::AppController;
use iris_gui_core::controller::preferences;
use libadwaita as adw;

use crate::settings::{icon_name, Form};

pub struct PrefsDialog {
    pub dialog: adw::PreferencesDialog,
    form: std::rc::Rc<Form>,
    shown: RefCell<Vec<preferences::Tab>>,
    pages: RefCell<Vec<adw::PreferencesPage>>,
}

impl PrefsDialog {
    pub fn new() -> std::rc::Rc<Self> {
        let dialog = adw::PreferencesDialog::builder().title(t!("prefs-window-title")).build();
        std::rc::Rc::new(Self {
            dialog,
            form: Form::new(|id, value| crate::app::prefs_control(&id, value)),
            shown: RefCell::default(),
            pages: RefCell::default(),
        })
    }

    pub fn refresh(&self, app: &AppController) {
        let tabs = preferences::tabs(app.prefs());
        if *self.shown.borrow() == tabs {
            return;
        }
        let open = self.dialog.visible_page().and_then(|p| p.name()).map(|n| n.to_string());
        for page in self.pages.borrow_mut().drain(..) {
            self.dialog.remove(&page);
        }
        for tab in &tabs {
            let page = adw::PreferencesPage::builder().title(&tab.title).name(tab.id).build();
            page.set_icon_name(Some(icon_name(tab.icon)));
            let group = adw::PreferencesGroup::new();
            self.form.fill(&group, &tab.fields);
            page.add(&group);
            self.dialog.add(&page);
            self.pages.borrow_mut().push(page);
        }
        if let Some(name) = open {
            self.dialog.set_visible_page_name(&name);
        }
        *self.shown.borrow_mut() = tabs;
    }
}
