//! Small AppKit helpers and the one action target every control reports to.

use std::cell::OnceCell;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol};
use objc2::{define_class, msg_send, sel, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{
    NSAccessibility, NSButton, NSColor, NSFont, NSMenuItem, NSTextField, NSToolbarItem, NSUserInterfaceItemIdentification, NSView,
};
use objc2_foundation::NSString;

pub fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(&s.replace('\0', ""))
}

define_class!(
    /// The target of every control's, toolbar item's and menu item's action.
    /// The sender's identifier is the controller's widget id.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISTarget"]
    pub struct Target;

    unsafe impl NSObjectProtocol for Target {}

    impl Target {
        #[unsafe(method(act:))]
        fn act(&self, sender: Option<&AnyObject>) {
            if let Some(id) = sender.and_then(sender_id) {
                crate::app::command(&id);
            }
        }

        /// A settings control changed (see `settings.rs`).
        #[unsafe(method(setting:))]
        fn setting(&self, sender: Option<&AnyObject>) {
            if let Some((id, value)) = sender.and_then(crate::settings::value_of) {
                crate::app::setting(id, value);
            }
        }

        /// The machine settings' memory slider (see `settings::memory_slider`).
        #[unsafe(method(memorySlider:))]
        fn memory_slider(&self, sender: Option<&AnyObject>) {
            if let Some((id, value)) = sender.and_then(crate::settings::memory_slider) {
                crate::app::setting(id, value);
            }
        }

        /// A timed tick (the autosave pause).
        #[unsafe(method(tickNow:))]
        fn tick_now(&self, _sender: Option<&AnyObject>) {
            crate::app::timed_tick();
        }

        /// One step of `IRIS_APPKIT_SCRIPT` (see `app::script`).
        #[unsafe(method(scriptStep:))]
        fn script_step(&self, _sender: Option<&AnyObject>) {
            crate::app::script_step();
        }

        #[unsafe(method(validateMenuItem:))]
        fn validate_menu_item(&self, item: &NSMenuItem) -> bool {
            item.identifier().map_or(true, |id| crate::app::enabled(&id.to_string()))
        }

        #[unsafe(method(validateToolbarItem:))]
        fn validate_toolbar_item(&self, item: &NSToolbarItem) -> bool {
            crate::app::enabled(&item.itemIdentifier().to_string())
        }
    }
);

define_class!(
    /// A container whose y axis grows down, so a scroll view's short page
    /// sits at the top.
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISFlippedView"]
    pub struct Flipped;

    unsafe impl NSObjectProtocol for Flipped {}

    impl Flipped {
        #[unsafe(method(isFlipped))]
        fn is_flipped(&self) -> bool {
            true
        }

        /// A click on empty space ends text editing, as in System Settings:
        /// the field commits and loses its focus ring.
        #[unsafe(method(mouseDown:))]
        fn mouse_down(&self, _e: &objc2_app_kit::NSEvent) {
            if let Some(w) = self.window() {
                w.makeFirstResponder(None);
            }
        }
    }
);

/// A new flipped container.
pub fn flipped(mtm: MainThreadMarker) -> Retained<NSView> {
    let this = mtm.alloc::<Flipped>().set_ivars(());
    // SAFETY: NSView's designated initialiser.
    let v: Retained<Flipped> = unsafe { msg_send![super(this), initWithFrame: objc2_foundation::NSRect::ZERO] };
    Retained::into_super(v)
}

thread_local! {
    static TARGET: OnceCell<Retained<Target>> = const { OnceCell::new() };
}

pub fn target(mtm: MainThreadMarker) -> Retained<Target> {
    TARGET.with(|t| {
        t.get_or_init(|| {
            let this = mtm.alloc::<Target>().set_ivars(());
            // SAFETY: NSObject's plain init.
            unsafe { msg_send![super(this), init] }
        })
        .clone()
    })
}

fn sender_id(sender: &AnyObject) -> Option<String> {
    if let Some(item) = sender.downcast_ref::<NSToolbarItem>() {
        return Some(item.itemIdentifier().to_string());
    }
    if let Some(item) = sender.downcast_ref::<NSMenuItem>() {
        return item.identifier().map(|i| i.to_string());
    }
    let view = sender.downcast_ref::<NSView>()?;
    view.identifier().map(|s| s.to_string())
}

/// Give `view` its controller id, as identifier and accessibility identifier.
pub fn set_id(view: &NSView, id: &str) {
    view.setIdentifier(Some(&ns(id)));
    view.setAccessibilityIdentifier(Some(&ns(id)));
}

/// A push button that reports `id` to the target.
pub fn button(mtm: MainThreadMarker, title: &str, id: &str) -> Retained<NSButton> {
    let t = target(mtm);
    // SAFETY: the target outlives every control (it is never dropped).
    let b = unsafe { NSButton::buttonWithTitle_target_action(&ns(title), Some(&t), Some(sel!(act:)), mtm) };
    set_id(&b, id);
    b
}

/// A non-editable label.
pub fn label(mtm: MainThreadMarker, text: &str) -> Retained<NSTextField> {
    let l = NSTextField::labelWithString(&ns(text), mtm);
    l.setSelectable(false);
    l
}

/// A label in a bold font of `size` points.
pub fn heading(mtm: MainThreadMarker, text: &str, size: f64) -> Retained<NSTextField> {
    let l = label(mtm, text);
    l.setFont(Some(&NSFont::boldSystemFontOfSize(size)));
    l
}

/// A secondary label (smaller, the secondary text colour).
pub fn secondary(mtm: MainThreadMarker, text: &str) -> Retained<NSTextField> {
    let l = label(mtm, text);
    l.setTextColor(Some(&NSColor::secondaryLabelColor()));
    l
}

/// Whether this Mac has SF Symbols (macOS 11 and later).
pub fn has_symbols() -> bool {
    objc2_foundation::NSProcessInfo::processInfo()
        .isOperatingSystemAtLeastVersion(objc2_foundation::NSOperatingSystemVersion {
            majorVersion: 11,
            minorVersion: 0,
            patchVersion: 0,
        })
}

/// For the `dump:controls` script step: every visible window's title, then
/// each control under it that has an id, with what it shows.
pub fn dump_controls(mtm: MainThreadMarker) {
    // Under a view whose id ends in `.results`, every label's text too.
    fn walk(v: &objc2_app_kit::NSView, out: &mut Vec<String>, labels: bool) {
        let id = objc2_app_kit::NSUserInterfaceItemIdentification::identifier(v).map(|i| i.to_string());
        let labels = labels || id.as_deref().is_some_and(|i| i.ends_with(".results"));
        if labels && id.is_none() {
            if let Some(t) = v.downcast_ref::<NSTextField>() {
                let tip = t.toolTip().or_else(|| unsafe { t.superview() }.and_then(|p| p.toolTip()));
                out.push(format!("    {:?}{}", t.stringValue().to_string(), tip.map(|x| format!("  (tip: {x})")).unwrap_or_default()));
            }
        }
        if let Some(id) = id.filter(|i| !i.starts_with('_')) {
            let shown = if let Some(b) = v.downcast_ref::<objc2_app_kit::NSButton>() {
                format!("button {:?} state={} enabled={}", b.title().to_string(), b.state(), b.isEnabled())
            } else if let Some(p) = v.downcast_ref::<objc2_app_kit::NSPopUpButton>() {
                format!("popup {:?}", p.titleOfSelectedItem().map(|t| t.to_string()).unwrap_or_default())
            } else if let Some(t) = v.downcast_ref::<NSTextField>() {
                format!("text {:?} enabled={}", t.stringValue().to_string(), t.isEnabled())
            } else {
                String::from("view")
            };
            out.push(format!("  {id}: {shown}"));
        }
        for sub in v.subviews().iter() {
            walk(&sub, out, labels);
        }
    }
    let app = objc2_app_kit::NSApplication::sharedApplication(mtm);
    for w in app.windows().iter() {
        if !w.isVisible() {
            continue;
        }
        let mut out = vec![format!("window {:?}", w.title().to_string())];
        if let Some(v) = w.contentView() {
            walk(&v, &mut out, false);
        }
        eprintln!("{}", out.join("\n"));
    }
}
