//! The menu bar (D7, plan "Menu bar (macOS)"). Every item of ours carries a
//! command id, goes to the action target and greys out with the controller's
//! `command_enabled`; shortcuts come from the controller's key map. The
//! standard items (Edit, Window) go to the first responder.

use objc2::rc::Retained;
use objc2::runtime::Sel;
use objc2::runtime::{NSObject, NSObjectProtocol};
use objc2::{define_class, sel, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSApplication, NSEventModifierFlags, NSMenu, NSMenuDelegate, NSMenuItem, NSUserInterfaceItemIdentification};
use iris_gui_core::controller::commands::{id_of, menu as m};
use iris_gui_core::controller::ids::toolbar as tb;
use iris_gui_core::controller::keys::{Command, Key, KeyMap};

use crate::ui::{ns, target};

fn system(mtm: MainThreadMarker, title: &str, action: Sel, key: &str) -> Retained<NSMenuItem> {
    // SAFETY: a plain menu item; its action goes up the responder chain.
    unsafe { NSMenuItem::initWithTitle_action_keyEquivalent(mtm.alloc(), &ns(title), Some(action), &ns(key)) }
}

/// Check the menu bar's item for `on` and uncheck the items for `off`.
pub fn check(on: &str, off: &[&str]) {
    fn walk(menu: &NSMenu, on: &str, off: &[&str]) {
        for item in menu.itemArray().iter() {
            if let Some(id) = item.identifier().map(|i| i.to_string()) {
                if id == on {
                    item.setState(objc2_app_kit::NSControlStateValueOn);
                } else if off.contains(&id.as_str()) {
                    item.setState(objc2_app_kit::NSControlStateValueOff);
                }
            }
            if let Some(sub) = item.submenu() {
                walk(&sub, on, off);
            }
        }
    }
    let mtm = MainThreadMarker::new().expect("main thread");
    if let Some(bar) = NSApplication::sharedApplication(mtm).mainMenu() {
        walk(&bar, on, off);
    }
}

/// Give the menu bar's item for `id` a new title (anywhere in the menus).
pub fn retitle(id: &str, title: &str) {
    fn walk(menu: &NSMenu, id: &str, title: &str) {
        for item in menu.itemArray().iter() {
            if item.identifier().is_some_and(|i| i.to_string() == id) {
                item.setTitle(&ns(title));
            }
            if let Some(sub) = item.submenu() {
                walk(&sub, id, title);
            }
        }
    }
    let mtm = MainThreadMarker::new().expect("main thread");
    if let Some(bar) = NSApplication::sharedApplication(mtm).mainMenu() {
        walk(&bar, id, title);
    }
}

/// A menu item that runs the command `id` (a context menu's entry).
pub fn command_item(mtm: MainThreadMarker, title: &str, id: &str) -> Retained<NSMenuItem> {
    ours(mtm, &KeyMap::default(), title, id)
}

/// An item of ours: reports `id`, with the key map's shortcut for it.
fn ours(mtm: MainThreadMarker, keys: &KeyMap, title: &str, id: &str) -> Retained<NSMenuItem> {
    let it = system(mtm, title, sel!(act:), "");
    let t = target(mtm);
    // SAFETY: the target is never dropped.
    unsafe { it.setTarget(Some(&t)) };
    it.setIdentifier(Some(&ns(id)));
    if let Some(cmd) = Command::ALL.iter().copied().find(|c| id_of(*c) == id) {
        if let Some(chord) = keys.chord(cmd) {
            let key = match chord.key {
                Key::Char(c) => c.to_ascii_lowercase().to_string(),
                // F-keys are handled by the key map in the window, not as equivalents.
                _ => String::new(),
            };
            if !key.is_empty() {
                let mut mods = NSEventModifierFlags::empty();
                if chord.primary {
                    mods |= NSEventModifierFlags::Command;
                }
                if chord.mac_ctrl {
                    mods |= NSEventModifierFlags::Control;
                }
                if chord.alt {
                    mods |= NSEventModifierFlags::Option;
                }
                if chord.shift {
                    mods |= NSEventModifierFlags::Shift;
                }
                it.setKeyEquivalent(&ns(&key));
                it.setKeyEquivalentModifierMask(mods);
            }
        }
    }
    it
}

fn submenu(mtm: MainThreadMarker, bar: &NSMenu, title: &str, items: Vec<Option<Retained<NSMenuItem>>>) -> Retained<NSMenu> {
    let menu = NSMenu::initWithTitle(mtm.alloc(), &ns(title));
    for it in items {
        match it {
            Some(it) => menu.addItem(&it),
            None => menu.addItem(&NSMenuItem::separatorItem(mtm)),
        }
    }
    let host = NSMenuItem::new(mtm);
    host.setSubmenu(Some(&menu));
    bar.addItem(&host);
    menu
}

pub fn build(mtm: MainThreadMarker) {
    let keys = KeyMap::default();
    let o = |title: String, id: &str| Some(ours(mtm, &keys, &title, id));
    let bar = NSMenu::new(mtm);

    let app_name = t!("app-name");
    submenu(
        mtm,
        &bar,
        &app_name,
        vec![
            o(t!("mb-about", name = app_name.as_str()), m::ABOUT),
            None,
            o(t!("mb-app-settings"), m::APP_SETTINGS),
            None,
            Some(system(mtm, &t!("mb-hide", name = app_name.as_str()), sel!(hide:), "h")),
            None,
            o(t!("mb-quit", name = app_name.as_str()), m::QUIT),
        ],
    );
    submenu(
        mtm,
        &bar,
        &t!("mb-file"),
        vec![
            o(t!("mb-new-machine"), m::NEW_MACHINE),
            o(t!("mb-new-folder"), m::NEW_FOLDER),
            None,
            o(t!("mb-duplicate"), m::DUPLICATE),
            o(t!("mb-rename"), m::RENAME),
            o(t!("mb-delete"), m::DELETE),
            None,
            o(t!("mb-show-in-finder"), m::SHOW_IN_FINDER),
            o(t!("mb-open-config"), m::OPEN_CONFIG),
            None,
            o(t!("mb-close"), m::CLOSE),
        ],
    );
    submenu(
        mtm,
        &bar,
        &t!("mb-edit"),
        vec![
            Some(system(mtm, &t!("mb-undo"), sel!(undo:), "z")),
            None,
            Some(system(mtm, &t!("mb-cut"), sel!(cut:), "x")),
            Some(system(mtm, &t!("mb-copy"), sel!(copy:), "c")),
            Some(system(mtm, &t!("mb-paste"), sel!(paste:), "v")),
            Some(system(mtm, &t!("mb-select-all"), sel!(selectAll:), "a")),
        ],
    );
    submenu(
        mtm,
        &bar,
        &t!("mb-view"),
        vec![
            o(t!("tb-show-library"), tb::SHOW_LIBRARY),
            None,
            o(t!("prefs-display-actual"), m::VIEW_ACTUAL),
            o(t!("mb-view-fit"), m::VIEW_FIT),
            o(t!("mb-view-stretch"), m::VIEW_STRETCH),
            None,
            o(t!("tb-full-screen"), tb::FULL_SCREEN),
            o(t!("tb-detach"), tb::DETACH_DISPLAY),
            None,
            o(t!("tb-serial-console"), tb::SERIAL_CONSOLE),
        ],
    );
    let send = NSMenu::initWithTitle(mtm.alloc(), &ns(&t!("mb-send-key")));
    for (title, id) in [("F11", m::SEND_F11), ("Ctrl+Alt+Del", m::SEND_CAD), ("Esc", m::SEND_ESC)] {
        send.addItem(&ours(mtm, &keys, title, id));
    }
    let send_host = NSMenuItem::new(mtm);
    send_host.setTitle(&ns(&t!("mb-send-key")));
    send_host.setSubmenu(Some(&send));
    submenu(
        mtm,
        &bar,
        &t!("mb-machine"),
        vec![
            o(t!("tb-power-on"), tb::POWER_ON),
            o(t!("tb-shut-down"), tb::SHUT_DOWN_GUEST),
            o(t!("tb-power-off"), tb::POWER_OFF),
            o(t!("tb-restart"), tb::RESTART),
            o(t!("tb-reset"), tb::RESET),
            None,
            o(t!("tb-save-state"), tb::SAVE_STATE),
            o(t!("tb-saved-states"), tb::SAVED_STATES),
            o(t!("tb-screenshot"), tb::SCREENSHOT),
            None,
            o(t!("mb-capture"), m::CAPTURE),
            Some(send_host),
            None,
            o(t!("tb-settings"), tb::MACHINE_SETTINGS),
        ],
    );
    // Devices: one submenu per disk and CD-ROM, rebuilt each time it opens.
    let devices = submenu(mtm, &bar, &t!("mb-devices"), Vec::new());
    let delegate = DevicesDelegate::new(mtm);
    devices.setDelegate(Some(objc2::runtime::ProtocolObject::from_ref(&*delegate)));
    DEVICES.with(|d| *d.borrow_mut() = Some(delegate));
    fill_devices(mtm, &devices, &[]);
    submenu(
        mtm,
        &bar,
        &t!("mb-tools"),
        vec![
            o(t!("mb-benchmark"), m::BENCHMARK),
            o(t!("mb-camera-test"), m::CAMERA_TEST),
            o(t!("mb-diagnostics"), m::DIAGNOSTICS),
            None,
            o(t!("mb-reset-nvram"), m::RESET_NVRAM),
        ],
    );
    let window = submenu(
        mtm,
        &bar,
        &t!("mb-window"),
        vec![
            Some(system(mtm, &t!("mb-minimize"), sel!(performMiniaturize:), "m")),
            Some(system(mtm, &t!("mb-zoom"), sel!(performZoom:), "")),
        ],
    );
    let mut help = vec![o(t!("mb-guide"), m::GUIDE), None];
    for p in iris_gui_core::controller::help::HelpPage::all() {
        help.push(o(p.menu_title(), p.id()));
    }
    help.extend([None, o(t!("mb-report-issue"), m::REPORT_ISSUE)]);
    submenu(mtm, &bar, &t!("mb-help"), help);

    let app = NSApplication::sharedApplication(mtm);
    app.setMainMenu(Some(&bar));
    app.setWindowsMenu(Some(&window));
}

thread_local! {
    static DEVICES: std::cell::RefCell<Option<Retained<DevicesDelegate>>> = const { std::cell::RefCell::new(None) };
}

define_class!(
    /// Rebuilds the Devices menu from `AppController::disk_menu` as it opens.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISDevicesMenuDelegate"]
    pub struct DevicesDelegate;

    unsafe impl NSObjectProtocol for DevicesDelegate {}

    unsafe impl NSMenuDelegate for DevicesDelegate {
        #[unsafe(method(menuNeedsUpdate:))]
        fn needs_update(&self, menu: &NSMenu) {
            if let Some(groups) = crate::app::disk_menu() {
                fill_devices(self.mtm(), menu, &groups);
            }
        }
    }
);

impl DevicesDelegate {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = mtm.alloc::<Self>().set_ivars(());
        // SAFETY: NSObject's plain init.
        unsafe { objc2::msg_send![super(this), init] }
    }
}

/// The Devices menu: the machine's disks and CD-ROMs, then the commands for all.
fn fill_devices(mtm: MainThreadMarker, menu: &NSMenu, groups: &[iris_gui_core::controller::commands::MenuGroup]) {
    let keys = KeyMap::default();
    menu.removeAllItems();
    for g in groups {
        let sub = NSMenu::initWithTitle(mtm.alloc(), &ns(&g.title));
        for e in &g.entries {
            let it = ours(mtm, &keys, &e.label, &e.id);
            if let Some(tip) = &e.tooltip {
                it.setToolTip(Some(&ns(tip)));
            }
            sub.addItem(&it);
        }
        let host = system(mtm, &g.title, sel!(act:), "");
        // SAFETY: a parent item; the submenu holds the actions.
        unsafe { host.setAction(None) };
        if let Some(img) = crate::glyphs::icon(g.icon, &g.title) {
            host.setImage(Some(&img));
        }
        host.setSubmenu(Some(&sub));
        menu.addItem(&host);
    }
    if !groups.is_empty() {
        menu.addItem(&NSMenuItem::separatorItem(mtm));
    }
    for (title, id) in [
        (t!("tb-insert-disc"), tb::INSERT_DISC),
        (t!("tb-browse-files"), tb::BROWSE_FILES),
        (t!("mb-sync-disks"), m::SYNC_DISKS),
        (t!("mb-check-network"), m::CHECK_NETWORK),
    ] {
        menu.addItem(&ours(mtm, &keys, &title, id));
    }
    // The App Sandbox reaches only folders the user granted: a disk folder
    // granted up front lets every disk in it be changed and folded.
    if cfg!(feature = "appstore") {
        let it = ours(mtm, &keys, &t!("menu-file-grant-folder"), m::GRANT_FOLDER);
        it.setToolTip(Some(&ns(&t!("menu-file-grant-folder-tip"))));
        menu.addItem(&it);
    }
}

/// The machine capsule's menu: the library's machines by family (the
/// selected one ticked), then New Machine. Opens under the capsule.
pub fn machines(mtm: MainThreadMarker, groups: &[iris_gui_core::controller::app::LibraryGroup], at: &objc2_app_kit::NSView) {
    let keys = KeyMap::default();
    let menu = NSMenu::initWithTitle(mtm.alloc(), &ns(&t!("tb-switch-machine")));
    for g in groups.iter().filter(|g| !g.rows.is_empty()) {
        if menu.numberOfItems() > 0 {
            menu.addItem(&NSMenuItem::separatorItem(mtm));
        }
        let head = NSMenuItem::new(mtm);
        head.setTitle(&ns(&g.title));
        head.setEnabled(false);
        menu.addItem(&head);
        for r in &g.rows {
            let it = ours(mtm, &keys, &r.name, &format!("{SELECT_MACHINE}{}", r.name));
            it.setIndentationLevel(1);
            if r.selected {
                it.setState(objc2_app_kit::NSControlStateValueOn);
            }
            if let Some(img) = crate::glyphs::machine_image(r.icon, 16.0, &r.subtitle) {
                it.setImage(Some(&img));
            }
            it.setToolTip(Some(&ns(&r.subtitle)));
            menu.addItem(&it);
        }
    }
    menu.addItem(&NSMenuItem::separatorItem(mtm));
    menu.addItem(&ours(mtm, &keys, &t!("mb-new-machine"), tb::NEW_MACHINE));
    let h = at.bounds().size.height;
    let y = if at.isFlipped() { h + 4.0 } else { -4.0 };
    menu.popUpMenuPositioningItem_atLocation_inView(None, objc2_foundation::NSPoint::new(0.0, y), Some(at));
}

/// The id prefix of the machine menu's items; the machine's name follows.
pub const SELECT_MACHINE: &str = "shell.select-machine:";

/// A device's menu, popped up at `at` (a status-bar light), as VMware's
/// device icons do.
pub fn pop_device(mtm: MainThreadMarker, group: &iris_gui_core::controller::commands::MenuGroup, at: &objc2_app_kit::NSView) {
    let keys = KeyMap::default();
    let menu = NSMenu::initWithTitle(mtm.alloc(), &ns(&group.title));
    let head = NSMenuItem::new(mtm);
    head.setTitle(&ns(&group.title));
    head.setEnabled(false);
    if let Some(img) = crate::glyphs::icon(group.icon, &group.title) {
        head.setImage(Some(&img));
    }
    menu.addItem(&head);
    menu.addItem(&NSMenuItem::separatorItem(mtm));
    for e in &group.entries {
        let it = ours(mtm, &keys, &e.label, &e.id);
        if let Some(tip) = &e.tooltip {
            it.setToolTip(Some(&ns(tip)));
        }
        menu.addItem(&it);
    }
    // Above the light: the status bar is the window's bottom edge.
    let h = at.bounds().size.height;
    let y = if at.isFlipped() { -4.0 } else { h + 4.0 };
    menu.popUpMenuPositioningItem_atLocation_inView(None, objc2_foundation::NSPoint::new(0.0, y), Some(at));
}

/// The saved-states menu the toolbar's Saved States opens: each state with
/// Restore, Rename and Delete (per-item ids, so the controller decides), or
/// a note when there are none. Pops up under the pointer.
pub fn saved_states(mtm: MainThreadMarker, states: &[iris_gui_core::controller::session::SavedState]) {
    use iris_gui_core::controller::ids::item;
    let keys = KeyMap::default();
    let menu = NSMenu::initWithTitle(mtm.alloc(), &ns(&t!("ov-saved-states")));
    if states.is_empty() {
        let none = system(mtm, &t!("ov-no-states"), sel!(act:), "");
        none.setEnabled(false);
        // SAFETY: an item with no target, shown disabled.
        unsafe { none.setAction(None) };
        menu.addItem(&none);
    }
    for s in states {
        let sub = NSMenu::initWithTitle(mtm.alloc(), &ns(&s.name));
        for (title, prefix) in [
            (t!("ov-restore"), item::RESTORE_STATE),
            (t!("ov-rename"), item::RENAME_STATE),
            (t!("ov-delete"), item::DELETE_STATE),
        ] {
            sub.addItem(&ours(mtm, &keys, &title, &item::id(prefix, &s.name)));
        }
        let it = system(mtm, &s.name, sel!(act:), "");
        // SAFETY: a parent item; the submenu holds the actions.
        unsafe { it.setAction(None) };
        it.setSubmenu(Some(&sub));
        menu.addItem(&it);
    }
    let at = objc2_app_kit::NSEvent::mouseLocation();
    menu.popUpMenuPositioningItem_atLocation_inView(None, at, None);
}

/// For the script hook (`dump:devices`): rebuild the Devices menu and print
/// each item, indented by depth, with whether the controller enables it.
pub fn dump_devices(mtm: MainThreadMarker) {
    let Some(bar) = NSApplication::sharedApplication(mtm).mainMenu() else { return };
    let title = t!("mb-devices");
    let Some(menu) = bar.itemArray().iter().find_map(|i| i.submenu().filter(|m| m.title().to_string() == title)) else { return };
    if let Some(groups) = crate::app::disk_menu() {
        fill_devices(mtm, &menu, &groups);
    }
    fn walk(menu: &NSMenu, depth: usize) {
        for it in menu.itemArray().iter() {
            if it.isSeparatorItem() {
                eprintln!("{}---", "  ".repeat(depth));
                continue;
            }
            let id = it.identifier().map(|i| i.to_string()).unwrap_or_default();
            let on = id.is_empty() || crate::app::enabled(&id);
            eprintln!("{}{} [{}]{}", "  ".repeat(depth), it.title(), id, if on { "" } else { " (disabled)" });
            if let Some(sub) = it.submenu() {
                walk(&sub, depth + 1);
            }
        }
    }
    walk(&menu, 0);
}

/// The toolbar's Discs menu: each CD-ROM's entries (insert, change, eject,
/// mount), under the pointer. Disks are left to the Devices menu.
pub fn discs(mtm: MainThreadMarker, groups: &[iris_gui_core::controller::commands::MenuGroup]) {
    use iris_gui_core::controller::icons::Icon;
    let keys = KeyMap::default();
    let menu = NSMenu::initWithTitle(mtm.alloc(), &ns(&t!("tb-disc")));
    for g in groups.iter().filter(|g| g.icon == Icon::CdRom) {
        let head = system(mtm, &g.title, sel!(act:), "");
        // SAFETY: a heading with no action, shown disabled.
        unsafe { head.setAction(None) };
        head.setEnabled(false);
        menu.addItem(&head);
        for e in &g.entries {
            let it = ours(mtm, &keys, &e.label, &e.id);
            it.setIndentationLevel(1);
            if let Some(tip) = &e.tooltip {
                it.setToolTip(Some(&ns(tip)));
            }
            menu.addItem(&it);
        }
    }
    let at = objc2_app_kit::NSEvent::mouseLocation();
    menu.popUpMenuPositioningItem_atLocation_inView(None, at, None);
}
