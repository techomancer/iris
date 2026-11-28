//! The machine page: the last screen with a large Start button over it, then
//! Details (the hardware, each row opening what it stands for) or Saved
//! States (a timeline), and the warnings above both.
//!
//! Built fresh from `AppController::overview` whenever it changes, and
//! nothing else: the page holds no state of its own but the chosen tab.

use adw::prelude::*;
use gtk::gdk;
use gtk4 as gtk;
use iris_gui_core::controller::icons::Icon;
use iris_gui_core::controller::ids::{item, toolbar as tb};
use iris_gui_core::controller::overview::{HardwareRow, Overview};
use iris_gui_core::controller::session::{LightTarget, SavedState};
use iris_gui_core::controller::Severity;
use libadwaita as adw;

use crate::icons;

/// Which tab the page shows; the shell keeps it between refreshes.
pub const TAB_DETAILS: &str = "details";
pub const TAB_STATES: &str = "states";

/// Build the page for `o`, or the empty state when no machine is selected.
pub fn build(app: Option<&adw::Application>, o: Option<&Overview>, tab: &str, empty: bool) -> gtk::Widget {
    let Some(o) = o else {
        let page = adw::StatusPage::builder().title(t!("toast-no-machine")).icon_name("computer-symbolic").build();
        if empty {
            page.set_description(Some(&t!("ov-no-machine")));
        }
        return page.upcast();
    };

    let page = gtk::Box::new(gtk::Orientation::Vertical, 18);
    page.set_margin_top(18);
    page.set_margin_bottom(18);

    if !o.warnings.is_empty() {
        let group = adw::PreferencesGroup::new();
        for (severity, text) in &o.warnings {
            let row = adw::ActionRow::builder().title(glib_escape(text)).build();
            row.set_title_lines(0);
            let icon = match severity {
                Severity::Info => Icon::Info,
                Severity::Warning => Icon::Warning,
                Severity::Error => Icon::Error,
            };
            let image = icons::image(icon, 16, "");
            image.add_css_class(match severity {
                Severity::Info => "accent",
                Severity::Warning => "warning",
                Severity::Error => "error",
            });
            row.add_prefix(&image);
            group.add(&row);
        }
        page.append(&group);
    }

    page.append(&screen(app, o));

    let stack = gtk::Stack::builder().transition_type(gtk::StackTransitionType::Crossfade).build();
    stack.add_titled(&details(app, o), Some(TAB_DETAILS), &t!("ov-details"));
    stack.add_titled(&states(app, o), Some(TAB_STATES), &t!("ov-saved-states"));
    stack.set_visible_child_name(tab);
    // The page is rebuilt whenever the overview changes, so the shell has to
    // remember which tab was showing.
    stack.connect_visible_child_name_notify(|s| {
        if let Some(name) = s.visible_child_name() {
            crate::app::set_summary_tab(&name);
        }
    });
    let switcher = gtk::StackSwitcher::builder().stack(&stack).halign(gtk::Align::Center).build();
    switcher.set_widget_name(TAB_SWITCHER);
    page.append(&switcher);
    page.append(&stack);

    let clamp = adw::Clamp::builder().maximum_size(760).tightening_threshold(600).child(&page).build();
    gtk::ScrolledWindow::builder().hscrollbar_policy(gtk::PolicyType::Never).vexpand(true).child(&clamp).build().upcast()
}

/// The tab switcher's widget id, so a test script can find it.
pub const TAB_SWITCHER: &str = "summary.tab";

/// The machine's last screen, with Power On over it; with no screen yet,
/// the machine's own picture and the button under it (over the picture it
/// would hide a wide case such as the Indigo2 altogether).
fn screen(app: Option<&adw::Application>, o: &Overview) -> gtk::Widget {
    let backdrop = gtk::Box::new(gtk::Orientation::Vertical, 18);
    backdrop.add_css_class("iris-screen");
    backdrop.set_height_request(260);
    backdrop.set_overflow(gtk::Overflow::Hidden);

    let texture = o.preview.as_ref().and_then(|p| gdk::Texture::from_filename(&p.path).ok());
    let button = (app, &o.state).0.filter(|_| o.state.is_none()).map(|app| {
        // Start belongs to a stopped machine; a running one shows its display.
        let button = gtk::Button::builder()
            .label(t!("tb-power-on"))
            .halign(gtk::Align::Center)
            .valign(gtk::Align::Center)
            .action_name(format!("app.{}", crate::menu::action_for(app, tb::POWER_ON)))
            .build();
        button.add_css_class("pill");
        button.add_css_class("suggested-action");
        button.set_widget_name(tb::POWER_ON);
        button
    });

    match texture {
        Some(texture) => {
            let picture = gtk::Picture::for_paintable(&texture);
            picture.set_content_fit(gtk::ContentFit::Contain);
            picture.set_vexpand(true);
            backdrop.append(&picture);
            let frame = gtk::Overlay::new();
            frame.set_child(Some(&backdrop));
            if let Some(button) = button {
                frame.add_overlay(&button);
            }
            frame.upcast()
        }
        None => {
            let art = icons::machine(o.icon, 110);
            art.set_halign(gtk::Align::Center);
            art.set_valign(gtk::Align::End);
            art.set_vexpand(true);
            art.add_css_class("iris-screen-art");
            backdrop.append(&art);
            match button {
                Some(button) => {
                    button.set_margin_bottom(18);
                    button.set_vexpand(true);
                    button.set_valign(gtk::Align::Start);
                    backdrop.append(&button);
                }
                None => art.set_valign(gtk::Align::Center),
            }
            backdrop.upcast()
        }
    }
}

fn details(app: Option<&adw::Application>, o: &Overview) -> gtk::Widget {
    let group = adw::PreferencesGroup::new();
    for r in &o.rows {
        group.add(&hardware_row(app, r));
    }
    let box_ = gtk::Box::new(gtk::Orientation::Vertical, 12);
    box_.append(&group);
    if let Some(app) = app {
        let edit = gtk::Button::builder()
            .label(t!("ov-edit-settings"))
            .halign(gtk::Align::Center)
            .action_name(format!("app.{}", crate::menu::action_for(app, tb::MACHINE_SETTINGS)))
            .build();
        edit.set_widget_name(tb::MACHINE_SETTINGS);
        box_.append(&edit);
    }
    box_.upcast()
}

fn hardware_row(app: Option<&adw::Application>, r: &HardwareRow) -> adw::ActionRow {
    let row = adw::ActionRow::builder().title(glib_escape(&r.label)).subtitle(glib_escape(&r.value)).build();
    row.set_subtitle_lines(0);
    row.add_prefix(&icons::image(r.icon, 16, ""));
    if !r.note.is_empty() {
        let note = gtk::Label::new(Some(&r.note));
        note.add_css_class("dim-label");
        note.add_css_class("caption");
        row.add_suffix(&note);
    }
    let Some(app) = app else { return row };
    if r.browsable {
        if let Some(LightTarget::Disk { key } | LightTarget::CdRom { key }) = &r.target {
            row.add_suffix(&icon_button(app, &item::id(item::BROWSE_DISK, key), Icon::BrowseFiles, &t!("ov-browse")));
        }
    }
    if r.convertible {
        if let Some(LightTarget::Disk { key }) = &r.target {
            let convert = gtk::Button::builder()
                .label(t!("ov-convert"))
                .tooltip_text(t!("convert-menu-tip"))
                .valign(gtk::Align::Center)
                .action_name(format!("app.{}", crate::menu::action_for(app, &item::id(item::CONVERT_DISK, key))))
                .build();
            row.add_suffix(&convert);
            let keep = gtk::Button::builder()
                .label(t!("convert-keep"))
                .tooltip_text(t!("convert-keep-tip"))
                .valign(gtk::Align::Center)
                .action_name(format!("app.{}", crate::menu::action_for(app, &item::id(item::DECLINE_CONVERT, key))))
                .build();
            keep.add_css_class("flat");
            row.add_suffix(&keep);
        }
    }
    // The row itself opens what it stands for: the device's settings.
    if r.target.is_some() {
        row.set_activatable(true);
        row.set_action_name(Some(&format!("app.{}", crate::menu::action_for(app, tb::MACHINE_SETTINGS))));
    }
    row
}

fn states(app: Option<&adw::Application>, o: &Overview) -> gtk::Widget {
    let group = adw::PreferencesGroup::new();
    // The current state: what Start runs. Saving needs the machine running.
    let now = adw::ActionRow::builder()
        .title(t!("ov-current-state"))
        .subtitle(format!("{}\n{}", t!("ov-not-running"), t!("ov-save-while-running")))
        .build();
    now.set_subtitle_lines(0);
    now.add_prefix(&thumbnail(o.preview.as_ref(), Some(o.icon)));
    group.add(&now);

    if o.states.is_empty() {
        let empty = adw::ActionRow::builder().title(t!("ov-no-states")).build();
        empty.add_css_class("dim-label");
        group.add(&empty);
    }
    for s in &o.states {
        group.add(&state_row(app, s));
    }
    group.upcast()
}

fn state_row(app: Option<&adw::Application>, s: &SavedState) -> adw::ActionRow {
    let when = s.saved.map(format_time).unwrap_or_default();
    let row = adw::ActionRow::builder()
        .title(glib_escape(&s.name))
        .subtitle(format!("{} · {}", t!("ov-saved-at", when = when.as_str()), human(s.bytes)))
        .build();
    row.add_prefix(&thumbnail(s.picture.as_ref(), None));
    let Some(app) = app else { return row };
    for (prefix, icon, tip) in [
        (item::RESTORE_STATE, Icon::Restore, t!("ov-restore")),
        (item::RENAME_STATE, Icon::Rename, t!("ov-rename")),
        (item::DELETE_STATE, Icon::Delete, t!("ov-delete")),
    ] {
        row.add_suffix(&icon_button(app, &item::id(prefix, &s.name), icon, &tip));
    }
    row
}

fn icon_button(app: &adw::Application, id: &str, icon: Icon, tip: &str) -> gtk::Button {
    let b = gtk::Button::builder()
        .child(&icons::image(icon, 16, tip))
        .tooltip_text(tip)
        .valign(gtk::Align::Center)
        .action_name(format!("app.{}", crate::menu::action_for(app, id)))
        .build();
    b.add_css_class("flat");
    b.set_widget_name(id);
    b
}

/// A small picture of a screen, or the machine's own art when there is none.
fn thumbnail(picture: Option<&iris_gui_core::controller::session::Picture>, fallback: Option<iris_gui_core::controller::profile::MachineIcon>) -> gtk::Widget {
    let holder = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    holder.set_size_request(96, 72);
    holder.set_valign(gtk::Align::Center);
    holder.set_halign(gtk::Align::Start);
    holder.set_hexpand(false);
    holder.add_css_class("iris-thumb");
    holder.set_overflow(gtk::Overflow::Hidden);
    match picture.and_then(|p| gdk::Texture::from_filename(&p.path).ok()) {
        Some(texture) => {
            let pic = gtk::Picture::for_paintable(&texture);
            pic.set_content_fit(gtk::ContentFit::Contain);
            pic.set_size_request(96, 72);
            holder.append(&pic);
        }
        None => {
            let art = match fallback {
                Some(icon) => icons::machine(icon, 40),
                None => icons::image(Icon::SavedStates, 24, ""),
            };
            art.set_halign(gtk::Align::Center);
            art.set_size_request(96, 72);
            art.add_css_class("dim-label");
            holder.append(&art);
        }
    }
    holder.upcast()
}

/// Pango markup is off for these labels, but an ampersand in a path would
/// still be read as markup by the rows that do use it.
fn glib_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;")
}

fn human(n: u64) -> String {
    if n >= 1 << 30 {
        format!("{:.1} GB", n as f64 / (1u64 << 30) as f64)
    } else if n >= 1 << 20 {
        format!("{:.1} MB", n as f64 / (1u64 << 20) as f64)
    } else {
        format!("{} KB", n >> 10)
    }
}

/// A date in the user's locale, through GLib.
fn format_time(secs: u64) -> String {
    gtk::glib::DateTime::from_unix_local(secs as i64)
        .and_then(|d| d.format("%x %H:%M"))
        .map(|s| s.to_string())
        .unwrap_or_default()
}
