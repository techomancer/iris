//! The window's chrome that comes and goes: hover zones along the edges that
//! bring back a hidden library (as a floating panel, as Arc does) or, in full
//! screen, the status bar.

use std::cell::Cell;

use objc2::rc::Retained;
use objc2::runtime::NSObjectProtocol;
use objc2::{define_class, msg_send, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{
    NSEvent, NSTrackingArea, NSTrackingAreaOptions, NSView,
};
use objc2_foundation::{NSPoint, NSRect};

/// What a hover zone stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    /// The left edge: show the hidden library as a floating panel.
    LibraryEdge,
    /// The floating library panel: leaving it puts the library away.
    LibraryPanel,
    /// The bottom edge, in full screen: show the status bar.
    StatusEdge,
    /// The status bar shown in full screen: leaving it hides it again.
    StatusBar,
}

define_class!(
    /// A view that reports the pointer entering and leaving it.
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISHoverZone"]
    #[ivars = Cell<Zone>]
    pub struct HoverZone;

    unsafe impl NSObjectProtocol for HoverZone {}

    impl HoverZone {
        #[unsafe(method(mouseEntered:))]
        fn entered(&self, _e: &NSEvent) {
            let z = self.ivars().get();
            crate::app::later(Box::new(move || crate::app::hover(z, true)));
        }

        #[unsafe(method(mouseExited:))]
        fn exited(&self, _e: &NSEvent) {
            let z = self.ivars().get();
            crate::app::later(Box::new(move || crate::app::hover(z, false)));
        }

        /// The edge zones only watch the pointer: clicks go through to the
        /// display or page under them.
        #[unsafe(method_id(hitTest:))]
        fn hit_test(&self, point: NSPoint) -> Option<Retained<NSView>> {
            match self.ivars().get() {
                Zone::LibraryEdge | Zone::StatusEdge => None,
                // SAFETY: NSView's own hit test.
                _ => unsafe { msg_send![super(self), hitTest: point] },
            }
        }
    }
);

impl HoverZone {
    pub fn new(mtm: MainThreadMarker, zone: Zone) -> Retained<Self> {
        let this = mtm.alloc::<Self>().set_ivars(Cell::new(zone));
        let v: Retained<Self> = unsafe { msg_send![super(this), initWithFrame: NSRect::ZERO] };
        // Follows the view's bounds, while the window is key.
        let opts = NSTrackingAreaOptions::MouseEnteredAndExited
            | NSTrackingAreaOptions::ActiveInKeyWindow
            | NSTrackingAreaOptions::InVisibleRect;
        // SAFETY: the view owns the area and outlives it.
        let area = unsafe {
            NSTrackingArea::initWithRect_options_owner_userInfo(mtm.alloc(), NSRect::ZERO, opts, Some(&v), None)
        };
        v.addTrackingArea(&area);
        v.setTranslatesAutoresizingMaskIntoConstraints(false);
        v
    }
}
