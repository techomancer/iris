//! The main window's toolbar (plan, "The window"): icon buttons with
//! tooltips, enabled from the controller through `validateToolbarItem:`.

use objc2::rc::Retained;
use objc2::runtime::{NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{define_class, msg_send, sel, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSToolbar, NSToolbarDelegate, NSToolbarDisplayMode, NSToolbarFlexibleSpaceItemIdentifier, NSToolbarItem};
use objc2_foundation::{NSArray, NSCopying, NSString};
use iris_gui_core::controller::commands;
use iris_gui_core::controller::keys::KeyMap;
use iris_gui_core::controller::ids::toolbar as tb;

use crate::glyphs;
use crate::ui::{ns, target};

/// Every item the toolbar can show; the controller picks which, by state
/// (`AppController::toolbar_layout`).
const ITEMS: &[&str] = &[
    tb::SHOW_LIBRARY,
    tb::POWER_ON,
    tb::SHUT_DOWN_GUEST,
    tb::POWER_OFF,
    tb::RESTART,
    tb::RESET,
    tb::SAVE_STATE,
    tb::SAVED_STATES,
    tb::SCREENSHOT,
    tb::INSERT_DISC,
    tb::DISC,
    tb::CAPTURE,
    tb::BROWSE_FILES,
    tb::DUPLICATE,
    tb::DELETE,
    tb::DETACH_DISPLAY,
    tb::FULL_SCREEN,
    tb::SERIAL_CONSOLE,
    tb::MACHINE_SETTINGS,
    tb::GAP,
    tb::TITLE,
];

/// The machine capsule's parts: the name, the model and the picture.
struct Title {
    name: Retained<objc2_app_kit::NSTextField>,
    model: Retained<objc2_app_kit::NSTextField>,
    picture: Retained<objc2_app_kit::NSImageView>,
    icon: std::cell::Cell<Option<iris_gui_core::controller::profile::MachineIcon>>,
}

thread_local! {
    static TITLE: std::cell::RefCell<Option<Title>> = const { std::cell::RefCell::new(None) };
}

/// Show `name`, `model` and the machine's picture in the capsule.
pub fn set_title(name: &str, model: &str, icon: Option<iris_gui_core::controller::profile::MachineIcon>) {
    TITLE.with(|t| {
        if let Some(t) = &*t.borrow() {
            if t.name.stringValue().to_string() != name {
                t.name.setStringValue(&ns(name));
            }
            if t.model.stringValue().to_string() != model {
                t.model.setStringValue(&ns(model));
            }
            if t.icon.get() != icon {
                t.icon.set(icon);
                t.picture.setImage(icon.and_then(|i| glyphs::machine_image(i, 22.0, model)).as_deref());
            }
        }
    });
}

fn identifier(id: &str) -> Retained<NSString> {
    if id == tb::GAP {
        // SAFETY: a constant AppKit defines.
        unsafe { NSToolbarFlexibleSpaceItemIdentifier }.copy()
    } else {
        ns(id)
    }
}

/// AppKit's separator that follows the sidebar's edge: what is left of it
/// sits over the sidebar (the library button), the rest over the machine's
/// pane, and the sidebar gets its own section of glass, as in Finder.
fn sidebar_separator() -> Retained<NSString> {
    // SAFETY: a constant AppKit defines.
    unsafe { objc2_app_kit::NSToolbarSidebarTrackingSeparatorItemIdentifier }.copy()
}

/// The ids left to right, with the sidebar separator after the library button.
fn with_separator(layout: &[&str]) -> Vec<Retained<NSString>> {
    let mut v = Vec::new();
    for id in layout {
        v.push(identifier(id));
        if *id == tb::SHOW_LIBRARY {
            v.push(sidebar_separator());
        }
    }
    v
}

/// Show `layout` (ids left to right), replacing what the toolbar shows.
pub fn set_layout(toolbar: &NSToolbar, layout: &[&str]) {
    for i in (0..toolbar.items().len()).rev() {
        toolbar.removeItemAtIndex(i as isize);
    }
    for (i, id) in with_separator(layout).iter().enumerate() {
        toolbar.insertItemWithItemIdentifier_atIndex(id, i as isize);
    }
}

fn make_item(mtm: MainThreadMarker, id: &NSString) -> Option<Retained<NSToolbarItem>> {
    let key = id.to_string();
    if key == tb::TITLE {
        // The machine capsule, where Finder has back and forward: the
        // machine's picture and its name over its model. It only shows which
        // machine this is; the library switches machines.
        use objc2_app_kit::{NSFont, NSImageView, NSLayoutAttribute, NSStackView, NSUserInterfaceLayoutOrientation as O};
        let view = |v: &objc2_app_kit::NSView| -> Retained<objc2_app_kit::NSView> { objc2::Message::retain(v) };
        let name = crate::ui::label(mtm, "");
        name.setFont(Some(&NSFont::boldSystemFontOfSize(12.0)));
        let model = crate::ui::secondary(mtm, "");
        model.setFont(Some(&NSFont::systemFontOfSize(10.5)));
        let lines = NSStackView::stackViewWithViews(&NSArray::from_retained_slice(&[view(&name), view(&model)]), mtm);
        lines.setOrientation(O::Vertical);
        lines.setAlignment(NSLayoutAttribute::Leading);
        lines.setSpacing(0.0);
        let picture = NSImageView::new(mtm);
        picture.setContentTintColor(Some(&objc2_app_kit::NSColor::secondaryLabelColor()));
        picture.widthAnchor().constraintEqualToConstant(26.0).setActive(true);
        let row = NSStackView::stackViewWithViews(&NSArray::from_retained_slice(&[view(&picture), view(&lines)]), mtm);
        row.setOrientation(O::Horizontal);
        row.setAlignment(NSLayoutAttribute::CenterY);
        row.setSpacing(7.0);
        // As much room after the text as before the picture: the capsule is
        // as wide as what it says.
        row.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: 0.0, left: 10.0, bottom: 0.0, right: 12.0 });
        row.heightAnchor().constraintEqualToConstant(36.0).setActive(true);
        let item = NSToolbarItem::initWithItemIdentifier(mtm.alloc(), id);
        item.setView(Some(&row));
        // In the toolbar's own glass, as the buttons on the right are.
        item.setBordered(true);
        item.setAutovalidates(false);
        item.setLabel(&ns(&t!("tb-machine")));
        TITLE.with(|t| {
            *t.borrow_mut() = Some(Title { name, model, picture, icon: std::cell::Cell::new(None) })
        });
        return Some(item);
    }
    let tip = commands::tip(&key, &KeyMap::default())?;
    let (icon, tooltip) = (tip.icon, tip.hover_text());
    // A graceful power action with its forced version in a menu beside it
    // (Shut Down / Power Off, Restart / Reset), as VMware has them.
    let forced = commands::toolbar_menu(&key);
    let item: Retained<NSToolbarItem> = if forced.is_empty() {
        NSToolbarItem::initWithItemIdentifier(mtm.alloc(), id)
    } else {
        let m = objc2_app_kit::NSMenuToolbarItem::initWithItemIdentifier(mtm.alloc(), id);
        let menu = objc2_app_kit::NSMenu::new(mtm);
        for e in &forced {
            let it = crate::menu::command_item(mtm, &e.label, &e.id);
            if let Some(tip) = &e.tooltip {
                it.setToolTip(Some(&ns(tip)));
            }
            menu.addItem(&it);
        }
        m.setMenu(&menu);
        m.setShowsIndicator(true);
        Retained::into_super(m)
    };
    // The library button needs no placement of its own: the sidebar
    // separator after it keeps it over the sidebar, beside the window buttons.
    item.setLabel(&ns(&tooltip));
    item.setPaletteLabel(&ns(&tooltip));
    item.setToolTip(Some(&ns(&tooltip)));
    match glyphs::icon(icon, &tooltip) {
        Some(image) => item.setImage(Some(&image)),
        None => item.setLabel(&ns(&tooltip)),
    }
    let t = target(mtm);
    // SAFETY: the target is never dropped.
    unsafe {
        item.setTarget(Some(&t));
        item.setAction(Some(sel!(act:)));
    }
    Some(item)
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISToolbarDelegate"]
    pub struct Delegate;

    unsafe impl NSObjectProtocol for Delegate {}

    unsafe impl NSToolbarDelegate for Delegate {
        #[unsafe(method_id(toolbar:itemForItemIdentifier:willBeInsertedIntoToolbar:))]
        fn item(&self, _toolbar: &NSToolbar, id: &NSString, _insert: bool) -> Option<Retained<NSToolbarItem>> {
            make_item(self.mtm(), id)
        }

        #[unsafe(method_id(toolbarDefaultItemIdentifiers:))]
        fn defaults(&self, _toolbar: &NSToolbar) -> Retained<NSArray<NSString>> {
            default_ids()
        }

        #[unsafe(method_id(toolbarAllowedItemIdentifiers:))]
        fn allowed(&self, _toolbar: &NSToolbar) -> Retained<NSArray<NSString>> {
            all_ids()
        }
    }
);

fn all_ids() -> Retained<NSArray<NSString>> {
    let mut v: Vec<_> = ITEMS.iter().map(|id| identifier(id)).collect();
    v.push(sidebar_separator());
    NSArray::from_retained_slice(&v)
}

/// What the toolbar starts with: the stopped layout (the window sets the
/// real one on its first refresh).
fn default_ids() -> Retained<NSArray<NSString>> {
    let v = with_separator(&[tb::SHOW_LIBRARY, tb::TITLE, tb::GAP, tb::POWER_ON]);
    NSArray::from_retained_slice(&v)
}

/// The toolbar and its delegate, which the caller keeps (the toolbar holds it weakly).
pub fn build(mtm: MainThreadMarker) -> (Retained<NSToolbar>, Retained<Delegate>) {
    let delegate: Retained<Delegate> = {
        let this = mtm.alloc::<Delegate>().set_ivars(());
        // SAFETY: NSObject's plain init.
        unsafe { msg_send![super(this), init] }
    };
    let toolbar = NSToolbar::initWithIdentifier(mtm.alloc(), &ns("iris.main"));
    toolbar.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    toolbar.setDisplayMode(NSToolbarDisplayMode::IconOnly);
    toolbar.setAllowsUserCustomization(false);
    (toolbar, delegate)
}
