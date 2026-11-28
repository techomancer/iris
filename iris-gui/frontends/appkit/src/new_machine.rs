//! The New Machine window: a wizard over the spec's steps
//! (`NewMachineSpec::steps`: the machine type, name and hardware, disks, a
//! summary), each drawn with the settings renderer, with Back, Continue and
//! at the end Create. The spec lives here until Create hands
//! `NewMachineSpec::create()` to the app controller.

use objc2::rc::Retained;
use objc2::MainThreadMarker;
use objc2_app_kit::{NSBackingStoreType, NSView, NSWindow, NSWindowStyleMask};
use objc2_foundation::{NSPoint, NSRect, NSSize};
use iris_gui_core::controller::app::AppAction;
use iris_gui_core::controller::new_machine::{field, NewMachineSpec};
use iris_gui_core::controller::settings::{control_id, part, Value};
use iris_gui_core::controller::{Effect, Enabled};

use crate::settings::{field_view, push, stack, view};
use crate::ui::{heading, ns, secondary, set_id};

/// The ids of the Cancel, Back and Continue buttons.
const CANCEL: &str = "new-machine.cancel";
const BACK: &str = "new-machine.back";
const NEXT: &str = "new-machine.next";

pub struct NewMachineWindow {
    pub window: Retained<NSWindow>,
    pub spec: NewMachineSpec,
    holder: Retained<NSView>,
    /// The wizard's step being shown.
    step: usize,
}

/// What a control in the window asks the shell to do.
pub enum Next {
    Redraw,
    Pick(Effect),
    Create(AppAction),
    Close,
    Nothing,
}

impl NewMachineWindow {
    pub fn new(mtm: MainThreadMarker) -> Self {
        let style = NSWindowStyleMask::Titled | NSWindowStyleMask::Closable;
        // SAFETY: a standard window, kept by the shell while it is open.
        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                mtm.alloc(),
                NSRect::new(NSPoint::new(240.0, 240.0), NSSize::new(640.0, 520.0)),
                style,
                NSBackingStoreType::Buffered,
                false,
            )
        };
        // SAFETY: the shell keeps the window.
        unsafe { window.setReleasedWhenClosed(false) };
        window.setTitle(&ns(&t!("dialog-new-machine-title")));
        let holder = NSView::new(mtm);
        window.setContentView(Some(&holder));
        set_id(&holder, "new-machine.window");
        let mut w = Self { window, spec: NewMachineSpec::default(), holder, step: 0 };
        w.redraw(mtm);
        w.window.center();
        w
    }

    /// Rebuild the form from the spec.
    pub fn redraw(&mut self, mtm: MainThreadMarker) {
        self.window.makeFirstResponder(None);
        for v in self.holder.subviews().iter() {
            v.removeFromSuperview();
        }
        let steps = self.spec.steps();
        let last = steps.len().saturating_sub(1);
        self.step = self.step.min(last);
        let (title, fields) = &steps[self.step];
        let mut rows: Vec<Retained<NSView>> = vec![view(&heading(mtm, &t!("dialog-new-machine-title"), 20.0))];
        rows.push(view(&secondary(mtm, &format!("{} \u{2014} {}", t!("wizard-step-of", n = self.step + 1, total = steps.len()), title))));
        if self.step == 0 {
            // The machine as it looks, for the type chosen.
            let icon = iris_gui_core::controller::profile::traits(self.spec.profile).icon;
            let pic = objc2_app_kit::NSImageView::new(mtm);
            if let Some(img) = crate::glyphs::machine_image(icon, 96.0, self.spec.profile.label()) {
                pic.setImage(Some(&img));
            }
            pic.setContentTintColor(Some(&objc2_app_kit::NSColor::secondaryLabelColor()));
            rows.push(view(&pic));
        }
        if self.step == last {
            // What will be made, line by line.
            for (label, value) in self.spec.summary() {
                let l = secondary(mtm, &label);
                l.setAlignment(objc2_app_kit::NSTextAlignment::Right);
                l.widthAnchor().constraintEqualToConstant(150.0).setActive(true);
                let v = crate::ui::label(mtm, &value);
                v.setLineBreakMode(objc2_app_kit::NSLineBreakMode::ByTruncatingMiddle);
                rows.push(view(&stack(mtm, &[view(&l), view(&v)], false, 12.0)));
            }
        } else {
            let mut toml = None;
            for f in fields {
                rows.push(field_view(mtm, f, &mut toml));
            }
        }
        let cancel = push(mtm, CANCEL, &t!("common-cancel"));
        cancel.setKeyEquivalent(&ns("\u{1b}"));
        let spacer = NSView::new(mtm);
        spacer.widthAnchor().constraintGreaterThanOrEqualToConstant(200.0).setActive(true);
        let mut buttons = vec![spacer, view(&cancel)];
        if self.step > 0 {
            buttons.push(view(&push(mtm, BACK, &t!("wizard-back"))));
        }
        let (next, allowed) = if self.step == last {
            (push(mtm, field::CREATE, &t!("dialog-new-machine-create")), self.spec.can_create())
        } else {
            (push(mtm, NEXT, &t!("wizard-continue")), self.spec.can_continue(self.step))
        };
        next.setKeyEquivalent(&ns("\r"));
        if let Enabled::No(why) = allowed {
            next.setEnabled(false);
            next.setToolTip(Some(&ns(&why)));
        }
        buttons.push(view(&next));
        rows.push(view(&stack(mtm, &buttons, false, 8.0)));
        let page = stack(mtm, &rows, true, 14.0);
        page.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: 20.0, left: 24.0, bottom: 20.0, right: 24.0 });
        page.setTranslatesAutoresizingMaskIntoConstraints(false);
        self.holder.addSubview(&page);
        page.topAnchor().constraintEqualToAnchor(&self.holder.topAnchor()).setActive(true);
        page.leadingAnchor().constraintEqualToAnchor(&self.holder.leadingAnchor()).setActive(true);
        page.trailingAnchor().constraintEqualToAnchor(&self.holder.trailingAnchor()).setActive(true);
        page.bottomAnchor().constraintEqualToAnchor(&self.holder.bottomAnchor()).setActive(true);
    }

    /// A control in the window reported `value`.
    pub fn control(&mut self, id: &str, value: Value) -> Next {
        if id == CANCEL {
            return Next::Close;
        }
        if id == BACK {
            self.step = self.step.saturating_sub(1);
            return Next::Redraw;
        }
        if id == NEXT {
            if let Enabled::Yes = self.spec.can_continue(self.step) {
                self.step += 1;
                return Next::Redraw;
            }
            return Next::Nothing;
        }
        if id == field::CREATE {
            return match self.spec.can_create() {
                Enabled::Yes => Next::Create(self.spec.create()),
                Enabled::No(_) => Next::Nothing,
            };
        }
        for f in [field::DISK, field::CDROM] {
            if id == control_id(f, None, Some(part::BROWSE)) {
                return self.spec.pick(f).map_or(Next::Nothing, Next::Pick);
            }
        }
        if self.spec.apply(id, value) {
            Next::Redraw
        } else {
            Next::Nothing
        }
    }
}
