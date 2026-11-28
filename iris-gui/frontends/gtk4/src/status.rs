//! The status bar under the machine: on the left the run state, the
//! executor and a pending CPU change; on the right one clickable light per
//! configured disk, CD-ROM, the network, audio and the rest, then the speed.
//!
//! Every light's id, icon, state and tooltip come from the session
//! controller; the shell picks the colour and opens what the light stands
//! for (a disk's own menu, or the command behind it).

use std::cell::RefCell;

use adw::prelude::*;
use gtk::{gio, glib};
use gtk4 as gtk;
use iris_gui_core::controller::app::AppController;
use iris_gui_core::controller::ids;
use iris_gui_core::controller::session::{DeviceLight, LightState};
use libadwaita as adw;

use crate::icons;

pub struct StatusBar {
    pub root: gtk::Widget,
    run: gtk::Label,
    executor: gtk::Label,
    pending: gtk::Label,
    unsaved: gtk::Label,
    speed: gtk::Label,
    lights_box: gtk::Box,
    /// What the lights showed last, so they are rebuilt only on a change.
    shown: RefCell<Vec<DeviceLight>>,
    /// While disks synchronize after a stop.
    sync: gtk::ProgressBar,
}

impl StatusBar {
    pub fn new() -> Self {
        let run = chip(ids::status::RUN_STATE);
        run.add_css_class("heading");
        let executor = chip(ids::status::EXECUTOR);
        executor.add_css_class("dim-label");
        let pending = chip("status.cpu-pending");
        pending.add_css_class("warning");
        let unsaved = chip("status.unsaved");
        unsaved.add_css_class("dim-label");
        let speed = chip(ids::status::MIPS);
        speed.add_css_class("dim-label");
        speed.add_css_class("numeric");

        let sync = gtk::ProgressBar::new();
        sync.set_valign(gtk::Align::Center);
        sync.set_width_request(120);
        sync.set_visible(false);

        let lights_box = gtk::Box::new(gtk::Orientation::Horizontal, 2);
        lights_box.set_halign(gtk::Align::End);

        let root = gtk::Box::new(gtk::Orientation::Horizontal, 10);
        root.set_margin_start(12);
        root.set_margin_end(10);
        root.set_margin_top(4);
        root.set_margin_bottom(4);
        for w in [&run, &executor, &pending, &unsaved] {
            root.append(w);
        }
        root.append(&sync);
        let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        spacer.set_hexpand(true);
        root.append(&spacer);
        root.append(&lights_box);
        root.append(&speed);

        Self { root: root.upcast(), run, executor, pending, unsaved, speed, lights_box, shown: RefCell::default(), sync }
    }

    pub fn refresh(&self, app: &AppController) {
        let bar = app.status_bar();
        self.run.set_label(&bar.run_label);
        self.executor.set_label(&bar.executor);
        set_or_hide(&self.pending, bar.cpu_pending.as_deref());
        set_or_hide(&self.unsaved, bar.unsaved.then(|| t!("welcome-unsaved")).as_deref());
        let speed = match (&bar.mips, &bar.hz) {
            (Some(m), Some(h)) => format!("{m} · {h}"),
            (Some(m), None) => m.clone(),
            (None, Some(h)) => h.clone(),
            (None, None) => String::new(),
        };
        set_or_hide(&self.speed, (!speed.is_empty()).then_some(speed.as_str()));

        match app.session().syncing() {
            Some(p) => {
                self.sync.set_visible(true);
                self.sync.set_fraction(p.overall() as f64);
                self.sync.set_tooltip_text(Some(&t!("status-syncing")));
            }
            None => self.sync.set_visible(false),
        }

        if *self.shown.borrow() != bar.lights {
            while let Some(child) = self.lights_box.first_child() {
                self.lights_box.remove(&child);
            }
            for l in &bar.lights {
                self.lights_box.append(&light(l, app));
            }
            *self.shown.borrow_mut() = bar.lights;
        }
    }
}

fn chip(id: &str) -> gtk::Label {
    let l = gtk::Label::new(None);
    l.set_widget_name(id);
    l
}

fn set_or_hide(label: &gtk::Label, text: Option<&str>) {
    match text {
        Some(t) if !t.is_empty() => {
            label.set_label(t);
            label.set_visible(true);
        }
        _ => label.set_visible(false),
    }
}

/// One light: the device's icon in the colour of what it is doing, with a
/// badge for a copy-on-write disk's dirty sectors.
fn light(l: &DeviceLight, app: &AppController) -> gtk::Widget {
    let image = icons::image(l.icon, 16, &l.tooltip);
    // Idle takes the bar's own colour, so it gets no class at all (GTK
    // refuses an empty one).
    if let Some(class) = match l.state {
        LightState::Off => Some("dim-label"),
        LightState::Idle => None,
        LightState::Reading | LightState::Active => Some("success"),
        LightState::Writing => Some("warning"),
        LightState::Fault => Some("error"),
    } {
        image.add_css_class(class);
    }
    let child: gtk::Widget = match &l.badge {
        Some(badge) => {
            let b = gtk::Box::new(gtk::Orientation::Horizontal, 3);
            b.append(&image);
            let count = gtk::Label::new(Some(badge));
            count.add_css_class("caption");
            count.add_css_class("dim-label");
            b.append(&count);
            b.upcast()
        }
        None => image.upcast(),
    };

    // A disk or CD-ROM opens its own menu; the others run one command.
    let gtk_app = crate::app::application();
    match (app.light_menu(&l.id), gtk_app) {
        (Some(group), Some(gtk_app)) => {
            let menu = gio::Menu::new();
            for e in &group.entries {
                let item = gio::MenuItem::new(Some(&e.label), None);
                item.set_detailed_action(&format!("app.{}", crate::menu::action_for(&gtk_app, &e.id)));
                menu.append_item(&item);
            }
            let b = gtk::MenuButton::builder().child(&child).tooltip_text(&l.tooltip).menu_model(&menu).build();
            b.add_css_class("flat");
            b.set_widget_name(&l.id);
            b.upcast()
        }
        _ => {
            let b = gtk::Button::builder().child(&child).tooltip_text(&l.tooltip).build();
            b.add_css_class("flat");
            b.set_widget_name(&l.id);
            let id = l.id.clone();
            b.connect_clicked(glib::clone!(move |_| {
                let id = id.clone();
                crate::app::later(move || crate::app::command(&id));
            }));
            b.upcast()
        }
    }
}
