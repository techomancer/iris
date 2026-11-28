//! The header bar's items, from `AppController::toolbar_layout`: what you do
//! to a stopped machine on the left of the gap, what you do while it runs on
//! the right. The controller decides which items there are, what each means
//! (an icon) and whether it can be used now; the shell only draws them.
//!
//! A graceful power action carries its forced version in a menu
//! (`commands::toolbar_menu`), which is what `AdwSplitButton` is.

use adw::prelude::*;
use gtk::gio;
use gtk4 as gtk;
use iris_gui_core::controller::app::AppController;
use iris_gui_core::controller::commands::{self, menu as m};
use iris_gui_core::controller::ids::toolbar as tb;
use iris_gui_core::controller::keys::KeyMap;
use libadwaita as adw;

use crate::icons;

/// The machine capsule's menu id prefix: select that machine.
pub const SELECT_MACHINE: &str = "gtk.select:";

pub struct Toolbar {
    header: adw::HeaderBar,
    /// What `toolbar_layout` said last, so the bar is rebuilt only on a change.
    layout: Vec<&'static str>,
    /// The items this shell packed, to take off again.
    items: Vec<gtk::Widget>,
    /// The machine capsule, which is also the window title.
    capsule: gtk::MenuButton,
    picture: gtk::Box,
    title: adw::WindowTitle,
}

impl Toolbar {
    pub fn new(header: &adw::HeaderBar) -> Self {
        let title = adw::WindowTitle::new(&t!("app-name"), "");
        let picture = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        picture.append(&title);
        let capsule = gtk::MenuButton::builder().child(&picture).build();
        capsule.set_widget_name(tb::TITLE);
        capsule.add_css_class("flat");
        capsule.set_tooltip_text(Some(&t!("tb-switch-machine")));
        header.set_title_widget(Some(&capsule));
        Self { header: header.clone(), layout: Vec::new(), items: Vec::new(), capsule, picture, title }
    }

    /// Draw the bar for what the controller holds now.
    pub fn refresh(&mut self, app: &AppController) {
        let o = app.overview();
        match (&o, app.selected()) {
            (Some(o), Some(name)) => {
                self.title.set_title(name);
                self.title.set_subtitle(o.state.as_deref().unwrap_or(&o.spec));
                self.set_picture(Some(o.icon));
            }
            _ => {
                self.title.set_title(&t!("app-name"));
                self.title.set_subtitle("");
                self.set_picture(None);
            }
        }
        self.capsule.set_menu_model(Some(&self.machines_menu(app)));

        let layout = app.toolbar_layout();
        if layout != self.layout {
            for w in self.items.drain(..) {
                self.header.remove(&w);
            }
            // Everything after the gap, packed from the right edge inwards.
            let after_gap: Vec<_> = layout.iter().skip_while(|id| **id != tb::GAP).skip(1).copied().collect();
            for id in after_gap.iter().rev() {
                let w = self.item(id);
                self.header.pack_end(&w);
                self.items.push(w);
            }
            self.layout = layout;
        }
    }

    /// The machine's front view beside the title, as the Mac app has it.
    fn set_picture(&self, icon: Option<iris_gui_core::controller::profile::MachineIcon>) {
        while let Some(child) = self.picture.first_child() {
            if child.downcast_ref::<adw::WindowTitle>().is_some() {
                break;
            }
            self.picture.remove(&child);
        }
        if let Some(icon) = icon {
            let image = icons::machine(icon, 22);
            image.add_css_class("dim-label");
            self.picture.prepend(&image);
        }
    }

    /// Every machine, to switch to, and New Machine under them.
    fn machines_menu(&self, app: &AppController) -> gio::Menu {
        let menu = gio::Menu::new();
        let Some(gtk_app) = crate::app::application() else { return menu };
        for group in app.library() {
            if group.rows.is_empty() {
                continue;
            }
            let section = gio::Menu::new();
            for row in &group.rows {
                let item = gio::MenuItem::new(Some(&row.name), None);
                let id = format!("{SELECT_MACHINE}{}", row.name);
                item.set_detailed_action(&format!("app.{}", crate::menu::action_for(&gtk_app, &id)));
                section.append_item(&item);
            }
            menu.append_section(Some(&group.title), &section);
        }
        let last = gio::Menu::new();
        let item = gio::MenuItem::new(Some(&t!("mb-new-machine")), None);
        item.set_detailed_action(&format!("app.{}", crate::menu::action_for(&gtk_app, m::NEW_MACHINE)));
        last.append_item(&item);
        menu.append_section(None, &last);
        menu
    }

    fn item(&self, id: &str) -> gtk::Widget {
        let keys = KeyMap::default();
        let tip = commands::tip(id, &keys);
        let (icon, hover) = match &tip {
            Some(t) => (t.icon, t.hover_text()),
            None => (iris_gui_core::controller::icons::Icon::Info, id.to_string()),
        };
        let Some(app) = crate::app::application() else { return gtk::Box::new(gtk::Orientation::Horizontal, 0).upcast() };
        let action = format!("app.{}", crate::menu::action_for(&app, id));
        let child = icons::image(icon, 16, &hover);
        let extra = commands::toolbar_menu(id);
        let widget: gtk::Widget = if extra.is_empty() {
            let b = gtk::Button::builder().child(&child).tooltip_text(&hover).action_name(&action).build();
            if id == tb::POWER_ON {
                b.add_css_class("suggested-action");
            }
            b.upcast()
        } else {
            // Shut Down with Power Off in its menu; Restart with Reset.
            let menu = gio::Menu::new();
            for e in &extra {
                let item = gio::MenuItem::new(Some(&e.label), None);
                item.set_detailed_action(&format!("app.{}", crate::menu::action_for(&app, &e.id)));
                menu.append_item(&item);
            }
            let b = adw::SplitButton::builder().child(&child).tooltip_text(&hover).action_name(&action).menu_model(&menu).build();
            b.upcast()
        };
        widget.set_widget_name(id);
        widget
    }
}
