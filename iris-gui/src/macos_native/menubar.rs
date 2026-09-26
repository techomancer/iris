//! The system menu bar: [`super::menus`] rendered as a real `NSMenu`.
//!
//! An `NSMenuItem` can't carry a Rust closure, so each item gets a *tag*, an
//! index into [`ACTIONS`] (the action table for the menu as last built). A click
//! resolves the tag and queues the [`Action`]. The app drains the queue on its
//! next frame and applies it. By then the menu has closed, which is also what
//! lets a menu item open a file dialog.
//!
//! Everything here runs on the main thread. AppKit requires it, and the click
//! callback comes from the menu's own tracking loop, which runs inside winit's
//! event loop on that same thread.
//!
//! winit is on objc2 0.5 / objc2-app-kit 0.2 while this crate uses 0.6 / 0.3.
//! Two versions in one graph are fine: both bind the same runtime. Just never
//! hand a *typed* object from one to the other.

use super::menus::{Action, Item, Menu};
use objc2::rc::Retained;
use objc2::runtime::{NSObject, NSObjectProtocol};
use objc2::{define_class, msg_send, sel, AnyThread, MainThreadMarker};
use objc2_app_kit::{
    NSApplication, NSControlStateValueOff, NSControlStateValueOn, NSEventModifierFlags, NSMenu,
    NSMenuItem,
};
use objc2_foundation::NSString;
use parking_lot::Mutex;
use std::sync::OnceLock;

/// Actions of the menu as last built, indexed by an item's tag.
static ACTIONS: Mutex<Vec<Action>> = Mutex::new(Vec::new());
/// Actions the user has picked but the app hasn't applied yet.
static PICKED: Mutex<Vec<Action>> = Mutex::new(Vec::new());
/// The egui context, so a menu click can wake a window that is otherwise idle.
static CTX: OnceLock<eframe::egui::Context> = OnceLock::new();

// The object every menu item targets. Ivar-less: all the state it needs is in
// the statics above, which is simpler than threading a pointer through AppKit
// and just as correct — there is only ever one menu bar.
define_class!(
    // SAFETY: NSObject has no subclassing requirements, and MenuTarget has no
    // Drop implementation.
    #[unsafe(super(NSObject))]
    #[name = "IrisMenuTarget"]
    struct MenuTarget;

    impl MenuTarget {
        #[unsafe(method(irisMenuAction:))]
        fn iris_menu_action(&self, sender: &NSMenuItem) {
            let tag = sender.tag();
            if tag >= 0 {
                if let Some(action) = ACTIONS.lock().get(tag as usize).cloned() {
                    PICKED.lock().push(action);
                }
            }
            // The main window may be idle (nothing running, no animation), and
            // the menu click is not an egui event, so ask for a frame in which
            // the queued action can be applied.
            if let Some(ctx) = CTX.get() {
                ctx.request_repaint();
            }
        }
    }

    unsafe impl NSObjectProtocol for MenuTarget {}
);

/// The one target instance, kept alive for the process's lifetime (menu items
/// hold an unretained pointer to their target).
static TARGET: Mutex<Option<Retained<MenuTarget>>> = Mutex::new(None);

/// Keep tool windows separate from the emulator, including in fullscreen.
/// AppKit otherwise groups a newly opened configuration window into the
/// fullscreen window's tab group, replacing its requested content size.
pub fn disable_automatic_window_tabbing() {
    if let Some(mtm) = MainThreadMarker::new() {
        objc2_app_kit::NSWindow::setAllowsAutomaticWindowTabbing(false, mtm);
    }
}

/// Remember the egui context so a menu click can request a repaint.
pub fn install(ctx: &eframe::egui::Context) {
    let _ = CTX.set(ctx.clone());
}

/// Everything the user has picked since the last call.
pub fn take_actions() -> Vec<Action> {
    std::mem::take(&mut *PICKED.lock())
}

/// Replace the system menu bar with `menus`.
///
/// Rebuilds from scratch — cheap at the handful of times a second the app
/// actually calls it (only when the model changes), and much less error-prone
/// than diffing a live `NSMenu`. Safe to call at any point *between* menu
/// interactions: while a menu is open, AppKit's tracking loop blocks the event
/// loop this is called from, so a rebuild can never land mid-click.
pub fn rebuild(menus: &[Menu]) {
    let Some(mtm) = MainThreadMarker::new() else { return };
    let app = NSApplication::sharedApplication(mtm);

    let mut actions = Vec::new();
    let target = target(mtm);

    let bar = NSMenu::new(mtm);
    bar.setAutoenablesItems(false);

    // The application menu. Its title is ignored by AppKit (the app's name from
    // the bundle is used), but the item must exist and be first.
    let app_item = NSMenuItem::new(mtm);
    let app_menu = NSMenu::new(mtm);
    app_menu.setAutoenablesItems(false);
    add_item(&app_menu, mtm, &target, &mut actions, "About IRIS", Some(Action::About), true, false, None);
    app_menu.addItem(&NSMenuItem::separatorItem(mtm));
    let services = NSMenu::new(mtm);
    let services_item = plain_item(mtm, "Services");
    services_item.setSubmenu(Some(&services));
    app_menu.addItem(&services_item);
    app.setServicesMenu(Some(&services));
    app_menu.addItem(&NSMenuItem::separatorItem(mtm));
    // Standard responder-chain actions: target `nil` sends them up to NSApp.
    system_item(&app_menu, mtm, "Hide IRIS", sel!(hide:), Some(("h", false)));
    system_item(&app_menu, mtm, "Hide Others", sel!(hideOtherApplications:), None);
    system_item(&app_menu, mtm, "Show All", sel!(unhideAllApplications:), None);
    app_menu.addItem(&NSMenuItem::separatorItem(mtm));
    // Our own Quit rather than `terminate:`, so quitting through the menu goes
    // through the app's close handling (which folds pending CHD changes back
    // into their disks) instead of tearing the process down underneath it.
    add_item(&app_menu, mtm, &target, &mut actions, "Quit IRIS", Some(Action::Quit), true, false, Some(("q", false)));
    app_item.setSubmenu(Some(&app_menu));
    bar.addItem(&app_item);

    for menu in menus {
        let item = plain_item(mtm, &menu.title);
        let sub = NSMenu::new(mtm);
        sub.setAutoenablesItems(false);
        // NSMenu takes its *title* from the menu, not the item, for the bar.
        sub.setTitle(&NSString::from_str(&menu.title));
        build_items(&sub, mtm, &target, &mut actions, &menu.items);
        item.setSubmenu(Some(&sub));
        bar.addItem(&item);
    }

    *ACTIONS.lock() = actions;
    app.setMainMenu(Some(&bar));
}

fn target(mtm: MainThreadMarker) -> Retained<MenuTarget> {
    let _ = mtm;
    let mut slot = TARGET.lock();
    if slot.is_none() {
        let this = MenuTarget::alloc().set_ivars(());
        let obj: Retained<MenuTarget> = unsafe { msg_send![super(this), init] };
        *slot = Some(obj);
    }
    slot.as_ref().unwrap().clone()
}

fn build_items(
    menu: &NSMenu,
    mtm: MainThreadMarker,
    target: &Retained<MenuTarget>,
    actions: &mut Vec<Action>,
    items: &[Item],
) {
    for item in items {
        match item {
            Item::Separator => menu.addItem(&NSMenuItem::separatorItem(mtm)),
            Item::Info(text) => {
                // A disabled item with no action: the menu's way of saying
                // something without offering to do anything.
                let it = plain_item(mtm, text);
                it.setEnabled(false);
                menu.addItem(&it);
            }
            Item::Sub { label, items } => {
                let it = plain_item(mtm, label);
                let sub = NSMenu::new(mtm);
                sub.setAutoenablesItems(false);
                sub.setTitle(&NSString::from_str(label));
                build_items(&sub, mtm, target, actions, items);
                it.setSubmenu(Some(&sub));
                menu.addItem(&it);
            }
            Item::Action { label, action, enabled, checked, accel } => {
                let key = accel.map(|a| (a.key.to_string(), a.shift));
                add_item(
                    menu,
                    mtm,
                    target,
                    actions,
                    label,
                    Some(action.clone()),
                    *enabled,
                    *checked,
                    key.as_ref().map(|(k, shift)| (k.as_str(), *shift)),
                );
            }
        }
    }
}

/// A menu item with no action attached yet (a submenu holder, or a label).
fn plain_item(mtm: MainThreadMarker, title: &str) -> Retained<NSMenuItem> {
    let item = NSMenuItem::new(mtm);
    item.setTitle(&NSString::from_str(title));
    item
}

#[allow(clippy::too_many_arguments)]
fn add_item(
    menu: &NSMenu,
    mtm: MainThreadMarker,
    target: &Retained<MenuTarget>,
    actions: &mut Vec<Action>,
    title: &str,
    action: Option<Action>,
    enabled: bool,
    checked: bool,
    key: Option<(&str, bool)>,
) {
    let item = plain_item(mtm, title);
    if let Some(action) = action {
        item.setTag(actions.len() as isize);
        actions.push(action);
        // SAFETY: `target` outlives the menu (it is kept in a static), and
        // `irisMenuAction:` is defined on it with exactly this signature.
        unsafe {
            item.setTarget(Some(target));
            item.setAction(Some(sel!(irisMenuAction:)));
        }
    }
    item.setEnabled(enabled);
    item.setState(if checked { NSControlStateValueOn } else { NSControlStateValueOff });
    if let Some((k, shift)) = key {
        item.setKeyEquivalent(&NSString::from_str(k));
        let mut mask = NSEventModifierFlags::Command;
        if shift {
            mask |= NSEventModifierFlags::Shift;
        }
        item.setKeyEquivalentModifierMask(mask);
    }
    menu.addItem(&item);
}

/// An item wired to a standard AppKit selector, dispatched up the responder
/// chain (`target: nil`) the way the system menus do it.
fn system_item(
    menu: &NSMenu,
    mtm: MainThreadMarker,
    title: &str,
    sel: objc2::runtime::Sel,
    key: Option<(&str, bool)>,
) {
    let item = plain_item(mtm, title);
    // SAFETY: these are AppKit's own selectors, taking a single sender.
    unsafe {
        item.setTarget(None);
        item.setAction(Some(sel));
    }
    if let Some((k, shift)) = key {
        item.setKeyEquivalent(&NSString::from_str(k));
        let mut mask = NSEventModifierFlags::Command;
        if shift {
            mask |= NSEventModifierFlags::Shift;
        }
        item.setKeyEquivalentModifierMask(mask);
    }
    menu.addItem(&item);
}
