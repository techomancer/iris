//! The application menu (D7), and the action behind every command id.
//!
//! GNOME has no menu bar: the same items live in the header bar's primary
//! menu, laid out as `gio::Menu` sections. Each item runs one `GSimpleAction`
//! of its own, so GTK can grey it out from the controller's
//! `command_enabled` — a single parameterised action would have one
//! enabled state for every command.
//!
//! Action names are sanitised command ids (GAction names take no `.` or `:`);
//! nothing but GTK sees them, and [`Actions::id_of`] maps back.

use std::cell::RefCell;
use std::collections::HashMap;

use adw::prelude::*;
use gtk::{gio, glib};
use gtk4 as gtk;
use iris_gui_core::controller::commands::{id_of, menu as m};
use iris_gui_core::controller::ids::toolbar as tb;
use iris_gui_core::controller::keys::{Chord, Command, Key, KeyMap};
use libadwaita as adw;

/// The prefix every command action lives under (`app.<name>`).
const PREFIX: &str = "app";

thread_local! {
    /// command id -> action name, and the action itself.
    static ACTIONS: RefCell<HashMap<String, (String, gio::SimpleAction)>> = RefCell::default();
    /// The primary menu, for `dump`.
    static MENU: RefCell<Option<gio::Menu>> = const { RefCell::new(None) };
}

/// The action name for `id`, registering the action the first time.
pub fn action_for(app: &adw::Application, id: &str) -> String {
    if let Some((name, _)) = ACTIONS.with(|a| a.borrow().get(id).cloned()) {
        return name;
    }
    let mut name: String = id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect();
    if ACTIONS.with(|a| a.borrow().values().any(|(n, _)| *n == name)) {
        name.push_str(&format!("-{}", ACTIONS.with(|a| a.borrow().len())));
    }
    let action = gio::SimpleAction::new(&name, None);
    let owned = id.to_string();
    action.connect_activate(move |_, _| crate::app::command(&owned));
    app.add_action(&action);
    ACTIONS.with(|a| a.borrow_mut().insert(id.to_string(), (name.clone(), action)));
    // A shortcut for the command this id stands for.
    if let Some(cmd) = Command::ALL.iter().copied().find(|c| id_of(*c) == id) {
        if let Some(accel) = KeyMap::default().chord(cmd).and_then(accelerator) {
            app.set_accels_for_action(&format!("{PREFIX}.{name}"), &[&accel]);
        }
    }
    name
}

/// Grey out every registered command the controller will not take now.
pub fn refresh_enabled(enabled: impl Fn(&str) -> bool) {
    ACTIONS.with(|a| {
        for (id, (_, action)) in a.borrow().iter() {
            action.set_enabled(enabled(id));
        }
    });
}

/// The chord as GTK writes an accelerator, or `None` for one this platform
/// does not take.
fn accelerator(chord: Chord) -> Option<String> {
    let mut s = String::new();
    if chord.primary {
        s.push_str("<Primary>");
    }
    if chord.alt {
        s.push_str("<Alt>");
    }
    if chord.shift {
        s.push_str("<Shift>");
    }
    s.push_str(&match chord.key {
        Key::F(n) => format!("F{n}"),
        Key::Char(',') => "comma".into(),
        Key::Char(c) => c.to_ascii_lowercase().to_string(),
        Key::Escape => "Escape".into(),
        Key::Enter => "Return".into(),
        Key::Space => "space".into(),
        Key::Tab => "Tab".into(),
        Key::Delete => "Delete".into(),
        Key::Backspace => "BackSpace".into(),
        Key::Up => "Up".into(),
        Key::Down => "Down".into(),
        Key::Left => "Left".into(),
        Key::Right => "Right".into(),
        Key::PageUp => "Page_Up".into(),
        Key::PageDown => "Page_Down".into(),
        Key::Home => "Home".into(),
        Key::End => "End".into(),
    });
    gtk::accelerator_parse(&s).map(|_| s)
}

/// One item: the label, and the action its command id runs.
fn item(app: &adw::Application, label: &str, id: &str) -> gio::MenuItem {
    let it = gio::MenuItem::new(Some(label), None);
    it.set_detailed_action(&format!("{PREFIX}.{}", action_for(app, id)));
    it
}

fn section(app: &adw::Application, items: &[(String, &str)]) -> gio::Menu {
    let menu = gio::Menu::new();
    for (label, id) in items {
        menu.append_item(&item(app, label, id));
    }
    menu
}

/// The primary menu: every command the shell has, in the order the Mac's menu
/// bar has them, as sections and submenus.
pub fn primary(app: &adw::Application) -> gio::Menu {
    let menu = gio::Menu::new();

    let machine = gio::Menu::new();
    machine.append_section(
        None,
        &section(
            app,
            &[
                (t!("tb-power-on"), tb::POWER_ON),
                (t!("tb-shut-down"), tb::SHUT_DOWN_GUEST),
                (t!("tb-power-off"), tb::POWER_OFF),
                (t!("tb-restart"), tb::RESTART),
                (t!("tb-reset"), tb::RESET),
            ],
        ),
    );
    machine.append_section(
        None,
        &section(
            app,
            &[
                (t!("tb-save-state"), tb::SAVE_STATE),
                (t!("tb-saved-states"), tb::SAVED_STATES),
                (t!("tb-screenshot"), tb::SCREENSHOT),
            ],
        ),
    );
    let send = section(
        app,
        &[("F11".to_string(), m::SEND_F11), ("Ctrl+Alt+Del".to_string(), m::SEND_CAD), ("Esc".to_string(), m::SEND_ESC)],
    );
    machine.append_section(None, &section(app, &[(t!("mb-capture"), m::CAPTURE)]));
    machine.append_submenu(Some(&t!("mb-send-key")), &send);
    machine.append_section(None, &section(app, &[(t!("tb-settings"), tb::MACHINE_SETTINGS)]));

    let view = gio::Menu::new();
    view.append_section(None, &section(app, &[(t!("tb-show-library"), tb::SHOW_LIBRARY)]));
    view.append_section(
        None,
        &section(
            app,
            &[
                (t!("prefs-display-actual"), m::VIEW_ACTUAL),
                (t!("mb-view-fit"), m::VIEW_FIT),
                (t!("mb-view-stretch"), m::VIEW_STRETCH),
            ],
        ),
    );
    view.append_section(
        None,
        &section(app, &[(t!("tb-full-screen"), tb::FULL_SCREEN), (t!("tb-detach"), tb::DETACH_DISPLAY)]),
    );
    view.append_section(None, &section(app, &[(t!("tb-serial-console"), tb::SERIAL_CONSOLE)]));

    let file = gio::Menu::new();
    file.append_section(
        None,
        &section(app, &[(t!("mb-new-machine"), m::NEW_MACHINE), (t!("mb-new-folder"), m::NEW_FOLDER)]),
    );
    file.append_section(
        None,
        &section(
            app,
            &[(t!("mb-duplicate"), m::DUPLICATE), (t!("mb-rename"), m::RENAME), (t!("mb-delete"), m::DELETE)],
        ),
    );
    file.append_section(
        None,
        &section(app, &[(t!("mb-show-in-files"), m::SHOW_IN_FINDER), (t!("mb-open-config"), m::OPEN_CONFIG)]),
    );

    let tools = section(
        app,
        &[
            (t!("mb-benchmark"), m::BENCHMARK),
            (t!("mb-camera-test"), m::CAMERA_TEST),
            (t!("mb-diagnostics"), m::DIAGNOSTICS),
            (t!("mb-reset-nvram"), m::RESET_NVRAM),
        ],
    );

    let help = gio::Menu::new();
    help.append_item(&item(app, &t!("mb-guide"), m::GUIDE));
    let pages = gio::Menu::new();
    for p in iris_gui_core::controller::help::HelpPage::all() {
        pages.append_item(&item(app, &p.menu_title(), p.id()));
    }
    help.append_section(None, &pages);
    help.append_section(None, &section(app, &[(t!("mb-report-issue"), m::REPORT_ISSUE)]));

    let name = t!("app-name");
    menu.append_submenu(Some(&t!("mb-machine")), &machine);
    menu.append_submenu(Some(&t!("mb-view")), &view);
    menu.append_submenu(Some(&t!("mb-file")), &file);
    menu.append_submenu(Some(&t!("mb-tools")), &tools);
    menu.append_section(
        None,
        &section(
            app,
            &[(t!("mb-app-settings"), m::APP_SETTINGS), (t!("mb-about", name = name.as_str()), m::ABOUT)],
        ),
    );
    menu.append_submenu(Some(&t!("mb-help")), &help);
    menu.append_section(None, &section(app, &[(t!("mb-quit", name = name.as_str()), m::QUIT)]));
    MENU.with(|m| *m.borrow_mut() = Some(menu.clone()));
    menu
}

/// Items the window also needs as plain actions, with no menu entry of their
/// own: Close, so Ctrl+W works.
pub fn window_actions(app: &adw::Application) {
    action_for(app, m::CLOSE);
}

/// Print the menu as it stands — every item, its command id and whether the
/// controller will take it now — for a test script's `dump:menu` step. A
/// pop-up menu is its own surface and no window screenshot catches it.
pub fn dump() {
    fn walk(menu: &gio::MenuModel, depth: usize) {
        for i in 0..menu.n_items() {
            let label = menu
                .item_attribute_value(i, gio::MENU_ATTRIBUTE_LABEL, Some(glib::VariantTy::STRING))
                .and_then(|v| v.get::<String>())
                .unwrap_or_default();
            let action = menu
                .item_attribute_value(i, gio::MENU_ATTRIBUTE_ACTION, Some(glib::VariantTy::STRING))
                .and_then(|v| v.get::<String>())
                .unwrap_or_default();
            let name = action.strip_prefix("app.").unwrap_or(&action);
            let (id, on) = ACTIONS.with(|a| {
                a.borrow()
                    .iter()
                    .find(|(_, (n, _))| *n == name)
                    .map(|(id, (_, act))| (id.clone(), act.is_enabled()))
                    .unwrap_or_default()
            });
            if !label.is_empty() || !id.is_empty() {
                let mark = if id.is_empty() { ' ' } else if on { '+' } else { '-' };
                println!("{:indent$}{mark} {label} [{id}]", "", indent = depth * 2);
            }
            for link in [gio::MENU_LINK_SECTION, gio::MENU_LINK_SUBMENU] {
                if let Some(sub) = menu.item_link(i, link) {
                    walk(&sub, depth + 1);
                }
            }
        }
    }
    MENU.with(|m| {
        if let Some(menu) = m.borrow().as_ref() {
            println!("iris-gtk menu:");
            walk(menu.upcast_ref(), 0);
        }
    });
}
