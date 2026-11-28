//! The machine settings window: the sections down the left, the selected
//! section's fields on the right, drawn from `AppController::settings_sections`.
//!
//! Each control's identifier is `settings::control_id(..)`. A control reports
//! to the target's `setting:` with its value; the window turns that into a
//! `SettingsAction` with `settings::action_for` and the shell dispatches it.
//! Edits apply at once (the app autosaves), so there is no OK or Cancel.

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{sel, MainThreadMarker, Message};
use objc2_app_kit::{
    NSBackingStoreType, NSBezelStyle, NSButton, NSButtonType, NSColor, NSControl, NSControlStateValueOn, NSFont,
    NSImageView, NSLayoutAttribute, NSPopUpButton, NSScrollView, NSStackView, NSStepper, NSTextField, NSTextView,
    NSUserInterfaceLayoutOrientation, NSView, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{NSArray, NSPoint, NSRect, NSSize};
use iris_gui_core::controller::app::AppController;
use iris_gui_core::controller::settings::{
    action_for, control_id, disk_kind_index, disk_kind_labels, field as fid, part, DiskRow, Field, FieldKind, ForwardRow,
    PickKind, Section, SettingsAction, Value,
};
use iris_gui_core::controller::{Enabled, Severity};

use crate::glyphs;
use crate::ui::{heading, label, ns, secondary, set_id, target};

/// The id prefix of the section list's buttons.
const PAGE: &str = "settings-page:";
const LABEL_WIDTH: f64 = 150.0;

/// The memory slider's part id (its value is an index, not megabytes).
const MEMORY_SLIDER: &str = "slider";

thread_local! {
    /// Each memory field shown now, by its field id (the settings page's and
    /// the wizard's can be open together): the stops its slider steps
    /// through and the box that shows the slider's amount as it moves.
    static MEMORY: std::cell::RefCell<Vec<(&'static str, Vec<u32>, Retained<NSTextField>)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The memory slider moved: show its amount in the box, and once it is let
/// go (or moved by the keyboard) the amount to set, under the field's id
/// (the settings page's or the New Machine wizard's), as the box would report.
pub fn memory_slider(sender: &AnyObject) -> Option<(String, Value)> {
    let s = sender.downcast_ref::<objc2_app_kit::NSSlider>()?;
    let i = s.doubleValue().round().max(0.0) as usize;
    // The slider's own id names its field (`control_id(field, None, "slider")`).
    let own = objc2_app_kit::NSUserInterfaceItemIdentification::identifier(&**s)?.to_string();
    let (id, mb) = MEMORY.with(|m| {
        let m = m.borrow();
        let (id, stops, field) = m.iter().find(|(id, _, _)| control_id(id, None, Some(MEMORY_SLIDER)) == own)?;
        let mb = *stops.get(i.min(stops.len().saturating_sub(1)))?;
        field.setStringValue(&ns(&mb.to_string()));
        Some((*id, mb))
    })?;
    // While the pointer drags the knob, only the box changes: setting the
    // amount redraws the page, which would end the drag.
    let mtm = MainThreadMarker::from(s);
    let dragging = objc2_app_kit::NSApplication::sharedApplication(mtm).currentEvent().is_some_and(|e| {
        matches!(e.r#type(), objc2_app_kit::NSEventType::LeftMouseDragged | objc2_app_kit::NSEventType::LeftMouseDown)
    });
    (!dragging).then(|| (id.to_string(), Value::Number(mb as f64)))
}

pub struct SettingsWindow {
    pub window: Retained<NSWindow>,
    list: Retained<NSStackView>,
    content: Retained<NSScrollView>,
    page: &'static str,
    /// What the views show now: rebuilt only when this changes.
    last: Option<(Option<String>, &'static str, Vec<Section>)>,
    toml: Option<Retained<NSTextView>>,
}

pub fn view<T: Message + AsRef<NSView>>(v: &Retained<T>) -> Retained<NSView> {
    let r: &NSView = (**v).as_ref();
    r.retain()
}

pub fn stack(mtm: MainThreadMarker, views: &[Retained<NSView>], vertical: bool, spacing: f64) -> Retained<NSStackView> {
    let s = NSStackView::stackViewWithViews(&NSArray::from_retained_slice(views), mtm);
    s.setOrientation(if vertical {
        NSUserInterfaceLayoutOrientation::Vertical
    } else {
        NSUserInterfaceLayoutOrientation::Horizontal
    });
    s.setSpacing(spacing);
    s.setAlignment(if vertical { NSLayoutAttribute::Leading } else { NSLayoutAttribute::CenterY });
    s
}

/// Make `c` report its value to `setting:` under `id`.
pub fn wire(c: &NSControl, id: &str) {
    let t = target(MainThreadMarker::from(c));
    // SAFETY: the target is never dropped; `setting:` takes the sender.
    unsafe {
        c.setTarget(Some(&t));
        c.setAction(Some(sel!(setting:)));
    }
    set_id(c, id);
}

pub(crate) fn popup(mtm: MainThreadMarker, id: &str, options: &[String], selected: usize) -> Retained<NSPopUpButton> {
    grouped_popup(mtm, id, &[(String::new(), options.to_vec())], selected)
}

/// A popup whose options come in titled groups (an empty title: none). Each
/// option's tag is its index among the options, which `value_of` reports.
fn grouped_popup(mtm: MainThreadMarker, id: &str, groups: &[(String, Vec<String>)], selected: usize) -> Retained<NSPopUpButton> {
    let p = NSPopUpButton::initWithFrame_pullsDown(mtm.alloc(), NSRect::ZERO, false);
    p.setAutoenablesItems(false);
    let menu = p.menu().expect("a popup has a menu");
    let mut index = 0isize;
    for (title, options) in groups {
        if !title.is_empty() {
            if menu.numberOfItems() > 0 {
                menu.addItem(&objc2_app_kit::NSMenuItem::separatorItem(mtm));
            }
            let head = objc2_app_kit::NSMenuItem::new(mtm);
            head.setTitle(&ns(title));
            head.setEnabled(false);
            head.setTag(-1);
            menu.addItem(&head);
        }
        for o in options {
            // Added through the menu: AppKit's addItemWithTitle merges equal titles.
            let it = objc2_app_kit::NSMenuItem::new(mtm);
            it.setTitle(&ns(o));
            it.setTag(index);
            if !title.is_empty() {
                it.setIndentationLevel(1);
            }
            menu.addItem(&it);
            index += 1;
        }
    }
    p.selectItemWithTag(selected as isize);
    wire(&p, id);
    p
}

pub fn checkbox(mtm: MainThreadMarker, id: &str, title: &str, on: bool) -> Retained<NSButton> {
    let b = NSButton::new(mtm);
    b.setButtonType(NSButtonType::Switch);
    b.setTitle(&ns(title));
    b.setState(if on { NSControlStateValueOn } else { 0 });
    wire(&b, id);
    b
}

pub fn push(mtm: MainThreadMarker, id: &str, title: &str) -> Retained<NSButton> {
    let b = NSButton::new(mtm);
    b.setBezelStyle(NSBezelStyle::Push);
    b.setTitle(&ns(title));
    wire(&b, id);
    b
}

fn text_field(mtm: MainThreadMarker, id: &str, value: &str, placeholder: &str, width: f64) -> Retained<NSTextField> {
    let f = NSTextField::textFieldWithString(&ns(value), mtm);
    f.setPlaceholderString(Some(&ns(placeholder)));
    // Commit on Return and when focus leaves the field.
    if let Some(cell) = f.cell() {
        cell.setSendsActionOnEndEditing(true);
    }
    f.widthAnchor().constraintEqualToConstant(width).setActive(true);
    wire(&f, id);
    f
}

/// A text box for a number: typed, checked by a number formatter, reported
/// as `Value::Number` by `value_of`.
fn number_field(mtm: MainThreadMarker, id: &str, value: f64, min: f64, max: f64, fraction: bool) -> Retained<NSTextField> {
    let f = objc2_foundation::NSNumberFormatter::new();
    f.setNumberStyle(objc2_foundation::NSNumberFormatterStyle::DecimalStyle);
    f.setUsesGroupingSeparator(false);
    f.setAllowsFloats(fraction);
    // No minimum or maximum on the formatter: it would refuse an
    // out-of-range entry and hold the focus in the box. The controller
    // clamps what is typed.
    let _ = (min, max);
    let t = NSTextField::textFieldWithString(&ns(&if fraction { format!("{value}") } else { format!("{}", value.round() as i64) }), mtm);
    t.setFormatter(Some(&f));
    // Commit on Return and when focus leaves the box.
    if let Some(cell) = t.cell() {
        cell.setSendsActionOnEndEditing(true);
    }
    t.setAlignment(objc2_app_kit::NSTextAlignment::Right);
    t.widthAnchor().constraintEqualToConstant(80.0).setActive(true);
    wire(&t, id);
    t
}

/// The id and value a settings control reports.
pub fn value_of(sender: &AnyObject) -> Option<(String, Value)> {
    let v = sender.downcast_ref::<NSView>()?;
    let id = objc2_app_kit::NSUserInterfaceItemIdentification::identifier(v)?.to_string();
    let value = if let Some(s) = sender.downcast_ref::<objc2_app_kit::NSSlider>() {
        Value::Number(s.doubleValue())
    } else if let Some(t) = sender.downcast_ref::<NSTextField>().filter(|t| t.formatter().is_some()) {
        Value::Number(t.doubleValue())
    } else if let Some(s) = sender.downcast_ref::<objc2_app_kit::NSSegmentedControl>() {
        Value::Index(s.selectedSegment().max(0) as usize)
    } else if let Some(p) = sender.downcast_ref::<NSPopUpButton>() {
        Value::Index(p.selectedItem().map_or(0, |i| i.tag()).max(0) as usize)
    } else if let Some(b) = sender.downcast_ref::<NSButton>() {
        Value::Bool(b.state() == NSControlStateValueOn)
    } else if let Some(s) = sender.downcast_ref::<NSStepper>() {
        Value::Number(s.doubleValue())
    } else if let Some(t) = sender.downcast_ref::<NSTextField>() {
        Value::Text(t.stringValue().to_string())
    } else {
        return None;
    };
    Some((id, value))
}

impl SettingsWindow {
    pub fn new(mtm: MainThreadMarker, page: &'static str) -> Self {
        let style = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Closable
            | NSWindowStyleMask::Miniaturizable
            | NSWindowStyleMask::Resizable;
        // SAFETY: a standard window, kept by the shell while it is open.
        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                mtm.alloc(),
                NSRect::new(NSPoint::new(200.0, 200.0), NSSize::new(820.0, 560.0)),
                style,
                NSBackingStoreType::Buffered,
                false,
            )
        };
        // SAFETY: the shell keeps the window.
        unsafe { window.setReleasedWhenClosed(false) };
        window.setMinSize(NSSize::new(640.0, 400.0));
        set_id(&window.contentView().expect("a window has a content view"), "settings.window");

        let list = stack(mtm, &[], true, 2.0);
        list.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: 16.0, left: 10.0, bottom: 16.0, right: 10.0 });
        list.setAlignment(NSLayoutAttribute::Width);
        let content = NSScrollView::initWithFrame(mtm.alloc(), NSRect::ZERO);
        content.setHasVerticalScroller(true);
        content.setDrawsBackground(false);

        let side = objc2_app_kit::NSVisualEffectView::new(mtm);
        side.setMaterial(objc2_app_kit::NSVisualEffectMaterial::Sidebar);
        let root = NSView::new(mtm);
        for v in [view(&side), view(&list), view(&content)] {
            v.setTranslatesAutoresizingMaskIntoConstraints(false);
            root.addSubview(&v);
        }
        for v in [view(&side), view(&list)] {
            v.leadingAnchor().constraintEqualToAnchor(&root.leadingAnchor()).setActive(true);
            v.topAnchor().constraintEqualToAnchor(&root.topAnchor()).setActive(true);
            v.widthAnchor().constraintEqualToConstant(200.0).setActive(true);
        }
        side.bottomAnchor().constraintEqualToAnchor(&root.bottomAnchor()).setActive(true);
        content.leadingAnchor().constraintEqualToAnchor(&side.trailingAnchor()).setActive(true);
        content.trailingAnchor().constraintEqualToAnchor(&root.trailingAnchor()).setActive(true);
        content.topAnchor().constraintEqualToAnchor(&root.topAnchor()).setActive(true);
        content.bottomAnchor().constraintEqualToAnchor(&root.bottomAnchor()).setActive(true);
        window.setContentView(Some(&root));
        window.center();
        window.setFrameAutosaveName(&ns("IRISSettingsWindow"));
        Self { window, list, content, page, last: None, toml: None }
    }

    /// Show `page` (a section id); the next refresh draws it.
    pub fn show_page(&mut self, page: &'static str) {
        self.page = page;
    }

    /// The section-list button `id` was used: `Some(page)` when it named one.
    pub fn page_for(&self, id: &str) -> Option<&'static str> {
        let want = id.strip_prefix(PAGE)?;
        self.last.as_ref()?.2.iter().find(|s| s.id == want).map(|s| s.id)
    }

    /// The action a control's report means, from what the window shows.
    pub fn action(&self, id: &str, value: Value) -> Option<SettingsAction> {
        let sections = &self.last.as_ref()?.2;
        let fields = sections.iter().flat_map(|s| s.fields.iter());
        let mut disk_keys: Vec<u8> = Vec::new();
        let mut forwards: Vec<ForwardRow> = Vec::new();
        let mut sizes: Vec<u32> = Vec::new();
        for f in fields {
            match &f.kind {
                FieldKind::DiskTable(rows) => disk_keys = rows.iter().map(|r| r.key).collect(),
                FieldKind::PortForwardTable(rows) => forwards = rows.clone(),
                FieldKind::BankGrid { sizes: s, .. } => sizes = s.clone(),
                _ => {}
            }
        }
        // A changer disc's Remove: an empty path takes it out.
        if let Some(slot) = id.strip_suffix(CLEAR_DISC) {
            return action_for(slot, Value::Text(String::new()), &disk_keys, &forwards);
        }
        let value = match value {
            // A bank's popup lists the sizes; the action takes megabytes.
            Value::Index(i) if id.starts_with(&format!("{}/", fid::BANKS)) => Value::Number(*sizes.get(i)? as f64),
            // Apply sends the text view's text.
            _ if id == control_id(fid::TOML, None, Some(part::APPLY)) => {
                Value::Text(self.toml.as_ref()?.string().to_string())
            }
            // A port forward's protocol popup: TCP first.
            v => v,
        };
        action_for(id, value, &disk_keys, &forwards)
    }

    /// Bring the window in line with the controller.
    pub fn refresh(&mut self, mtm: MainThreadMarker, app: &AppController) {
        let name = app.selected().map(str::to_string);
        let sections = if name.is_some() { app.settings_sections() } else { Vec::new() };
        if !sections.iter().any(|s| s.id == self.page) {
            if let Some(s) = sections.first() {
                self.page = s.id;
            }
        }
        let now = (name.clone(), self.page, sections);
        if self.last.as_ref() == Some(&now) {
            return;
        }
        let title = match &name {
            Some(n) => t!("settings-window-title", name = n.as_str()),
            None => t!("tb-settings"),
        };
        self.window.setTitle(&ns(&title));
        // End any edit first: its commit arrives while the shell is busy and is
        // reported after this redraw (`app::setting`).
        self.window.makeFirstResponder(None);
        self.build_list(mtm, &now.2);
        let page = match now.2.iter().find(|s| s.id == self.page) {
            Some(s) => self.build_section(mtm, s),
            None => view(&secondary(mtm, &t!("settings-no-machine"))),
        };
        // The same page redrawn after a change keeps its scroll position.
        let same_page = self.last.as_ref().is_some_and(|l| l.1 == now.1);
        let scrolled = self.content.contentView().bounds().origin;
        let doc = crate::ui::flipped(mtm);
        doc.setTranslatesAutoresizingMaskIntoConstraints(false);
        page.setTranslatesAutoresizingMaskIntoConstraints(false);
        doc.addSubview(&page);
        self.content.setDocumentView(Some(&doc));
        let clip = self.content.contentView();
        doc.topAnchor().constraintEqualToAnchor(&clip.topAnchor()).setActive(true);
        doc.leadingAnchor().constraintEqualToAnchor(&clip.leadingAnchor()).setActive(true);
        doc.widthAnchor().constraintEqualToAnchor(&clip.widthAnchor()).setActive(true);
        page.topAnchor().constraintEqualToAnchor(&doc.topAnchor()).setActive(true);
        page.leadingAnchor().constraintEqualToAnchor(&doc.leadingAnchor()).setActive(true);
        page.trailingAnchor().constraintLessThanOrEqualToAnchor(&doc.trailingAnchor()).setActive(true);
        // At least the visible height, so a click below the fields lands on
        // the page (and ends editing) rather than on the scroll view.
        // The page keeps its own height (pinned at the top, nothing pulls its
        // bottom), so its rows are never spread apart; the document is the
        // taller of the page and the visible area.
        doc.heightAnchor().constraintGreaterThanOrEqualToAnchor(&page.heightAnchor()).setActive(true);
        doc.heightAnchor().constraintGreaterThanOrEqualToAnchor(&clip.heightAnchor()).setActive(true);
        let snug = doc.heightAnchor().constraintEqualToAnchor(&clip.heightAnchor());
        snug.setPriority(250.0);
        snug.setActive(true);
        if same_page {
            self.content.layoutSubtreeIfNeeded();
            clip.scrollToPoint(scrolled);
            self.content.reflectScrolledClipView(&clip);
        }
        self.last = Some(now);
    }

    fn build_list(&self, mtm: MainThreadMarker, sections: &[Section]) {
        for v in self.list.arrangedSubviews().iter() {
            self.list.removeArrangedSubview(&v);
            v.removeFromSuperview();
        }
        for s in sections {
            let b = NSButton::new(mtm);
            b.setButtonType(NSButtonType::PushOnPushOff);
            b.setBezelStyle(NSBezelStyle::AccessoryBar);
            // The selected row keeps its highlight; the others show one under the pointer.
            b.setShowsBorderOnlyWhileMouseInside(s.id != self.page);
            b.setState(if s.id == self.page { NSControlStateValueOn } else { 0 });
            b.setAlignment(objc2_app_kit::NSTextAlignment::Left);
            b.setTitle(&ns(&s.title));
            b.setToolTip(Some(&ns(&s.summary)));
            if let Some(img) = glyphs::icon(s.icon, &s.title) {
                b.setImage(Some(&img));
                b.setImagePosition(objc2_app_kit::NSCellImagePosition::ImageLeading);
            }
            wire(&b, &format!("{PAGE}{}", s.id));
            self.list.addArrangedSubview(&b);
            // Every row as wide as the sidebar, so titles line up on the left.
            b.widthAnchor().constraintEqualToConstant(180.0).setActive(true);
            b.heightAnchor().constraintEqualToConstant(24.0).setActive(true);
        }
    }

    fn build_section(&mut self, mtm: MainThreadMarker, s: &Section) -> Retained<NSView> {
        self.toml = None;
        // The title alone: the fields say what the summary would.
        let mut rows: Vec<Retained<NSView>> = vec![view(&heading(mtm, &s.title, 20.0))];
        for f in &s.fields {
            rows.push(field_view(mtm, f, &mut self.toml));
        }
        let page = stack(mtm, &rows, true, 14.0);
        page.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: 20.0, left: 24.0, bottom: 24.0, right: 24.0 });
        view(&page)
    }
}

/// One field, drawn by kind. The TOML editor's text view lands in `toml`, for Apply.
pub fn field_view(mtm: MainThreadMarker, f: &Field, toml: &mut Option<Retained<NSTextView>>) -> Retained<NSView> {
    let id = f.id;
    let control: Retained<NSView> = match &f.kind {
        FieldKind::Choice { options, selected } => view(&popup(mtm, id, options, *selected)),
        FieldKind::GroupedChoice { groups, selected } => view(&grouped_popup(mtm, id, groups, *selected)),
        // The checkbox carries its own label, as on the Mac; the label
        // column stays empty.
        FieldKind::Toggle(on) => view(&checkbox(mtm, id, &f.label, *on)),
        FieldKind::Text { value, placeholder, error } => {
            let t = text_field(mtm, id, value, placeholder, 260.0);
            match error {
                Some(e) => {
                    let err = label(mtm, e);
                    err.setTextColor(Some(&NSColor::systemRedColor()));
                    view(&stack(mtm, &[view(&t), view(&err)], true, 4.0))
                }
                None => view(&t),
            }
        }
        FieldKind::Number { value, min, max, step, unit } => {
            // A box to type in (the controller clamps), not arrows to click.
            let t = number_field(mtm, id, *value, *min, *max, *step < 1.0);
            let mut parts = vec![view(&t)];
            if !unit.is_empty() {
                parts.push(view(&secondary(mtm, unit)));
            }
            view(&stack(mtm, &parts, false, 6.0))
        }
        FieldKind::Memory { mb, totals, stops } => {
            // A slider with a notch at each stop (powers of two; its value is
            // the stop's index), and a box to type any amount the machine
            // can hold. The box follows the slider as it moves; the amount is
            // set when the slider is let go. The controller snaps a typed amount.
            let lo = *totals.first().unwrap_or(mb) as f64;
            let hi = *totals.last().unwrap_or(mb) as f64;
            let at = stops.iter().position(|t| t >= mb).unwrap_or(stops.len().saturating_sub(1));
            let last = stops.len().saturating_sub(1).max(1) as f64;
            // SAFETY: the target is never dropped; `memorySlider:` takes the sender.
            let s = unsafe {
                objc2_app_kit::NSSlider::sliderWithValue_minValue_maxValue_target_action(
                    at as f64,
                    0.0,
                    last,
                    Some(&target(mtm)),
                    Some(sel!(memorySlider:)),
                    mtm,
                )
            };
            s.setContinuous(true);
            // Notches while they can be told apart; snapping either way.
            if stops.len() <= 48 {
                s.setNumberOfTickMarks(stops.len() as isize);
                s.setAllowsTickMarkValuesOnly(true);
            }
            s.widthAnchor().constraintEqualToConstant(260.0).setActive(true);
            set_id(&s, &control_id(id, None, Some(MEMORY_SLIDER)));
            let t = number_field(mtm, id, *mb as f64, lo, hi, false);
            MEMORY.with(|m| {
                let mut m = m.borrow_mut();
                m.retain(|(i, _, _)| *i != id);
                m.push((id, stops.clone(), t.clone()));
            });
            view(&stack(mtm, &[view(&s), view(&t), view(&secondary(mtm, "MB"))], false, 8.0))
        }
        FieldKind::Path { value, pick } => {
            let t = text_field(mtm, id, value, "", 320.0);
            let title = match pick {
                PickKind::File { .. } | PickKind::Folder => t!("common-browse"),
            };
            let b = push(mtm, &control_id(id, None, Some(part::BROWSE)), &title);
            view(&stack(mtm, &[view(&t), view(&b)], false, 6.0))
        }
        FieldKind::BankGrid { banks, sizes, summary } => {
            let labels: Vec<String> =
                sizes.iter().map(|mb| if *mb == 0 { t!("common-none") } else { format!("{mb} MB") }).collect();
            let mut cells: Vec<Retained<NSView>> = Vec::new();
            for (i, mb) in banks.iter().enumerate() {
                let sel = sizes.iter().position(|s| s == mb).unwrap_or(0);
                let p = popup(mtm, &control_id(id, Some(i), None), &labels, sel);
                cells.push(view(&p));
            }
            // Two banks to a line keeps eight banks readable.
            let mut lines: Vec<Retained<NSView>> =
                cells.chunks(2).map(|c| view(&stack(mtm, c, false, 8.0))).collect();
            lines.push(view(&secondary(mtm, summary)));
            view(&stack(mtm, &lines, true, 6.0))
        }
        FieldKind::DiskTable(rows) => disk_table(mtm, id, rows),
        FieldKind::PortForwardTable(rows) => forward_table(mtm, id, rows),
        FieldKind::Toml { text } => {
            let scroll = NSTextView::scrollableTextView(mtm);
            scroll.setHasVerticalScroller(true);
            scroll.setBorderType(objc2_app_kit::NSBorderType::BezelBorder);
            scroll.widthAnchor().constraintEqualToConstant(540.0).setActive(true);
            scroll.heightAnchor().constraintEqualToConstant(320.0).setActive(true);
            let doc = scroll.documentView();
            if let Some(tv) = doc.as_ref().and_then(|d| d.downcast_ref::<NSTextView>()) {
                tv.setString(&ns(text));
                tv.setFont(Some(&NSFont::userFixedPitchFontOfSize(12.0).unwrap_or_else(|| NSFont::systemFontOfSize(12.0))));
                tv.setRichText(false);
                tv.setAutomaticQuoteSubstitutionEnabled(false);
                tv.setAutomaticDashSubstitutionEnabled(false);
                set_id(tv, id);
                *toml = Some(tv.retain());
            }
            let apply = push(mtm, &control_id(id, None, Some(part::APPLY)), &t!("common-apply"));
            view(&stack(mtm, &[view(&scroll), view(&apply)], true, 8.0))
        }
        FieldKind::Note(sev, text) => {
            let mut parts: Vec<Retained<NSView>> = Vec::new();
            let (icon, color) = match sev {
                Severity::Info => (iris_gui_core::controller::icons::Icon::Info, NSColor::secondaryLabelColor()),
                Severity::Warning => (iris_gui_core::controller::icons::Icon::Warning, NSColor::systemOrangeColor()),
                Severity::Error => (iris_gui_core::controller::icons::Icon::Error, NSColor::systemRedColor()),
            };
            if let Some(img) = glyphs::icon(icon, text) {
                let iv = NSImageView::imageViewWithImage(&img, mtm);
                iv.setContentTintColor(Some(&color));
                parts.push(view(&iv));
            }
            let l = label(mtm, text);
            l.setLineBreakMode(objc2_app_kit::NSLineBreakMode::ByWordWrapping);
            l.setPreferredMaxLayoutWidth(480.0);
            parts.push(view(&l));
            return view(&stack(mtm, &parts, false, 6.0));
        }
    };
    if let Enabled::No(reason) = &f.enabled {
        disable(&control, reason);
    }
    // Help is a tooltip on the label and the control (a disabled control
    // keeps its reason), not a line of grey text under it.
    if let Some(h) = &f.help {
        if control.toolTip().is_none() {
            control.setToolTip(Some(&ns(h)));
        }
    }
    let mut right: Vec<Retained<NSView>> = vec![control];
    // Tables and the TOML editor are wide: their label goes above them.
    if matches!(f.kind, FieldKind::Toml { .. } | FieldKind::DiskTable(_) | FieldKind::PortForwardTable(_)) {
        right.insert(0, view(&heading(mtm, &f.label, 13.0)));
        return view(&stack(mtm, &right, true, 8.0));
    }
    let name = label(mtm, if matches!(f.kind, FieldKind::Toggle(_)) { "" } else { &f.label });
    if let Some(h) = &f.help {
        name.setToolTip(Some(&ns(h)));
    }
    name.setAlignment(objc2_app_kit::NSTextAlignment::Right);
    // A long label wraps onto a second line rather than being cut off.
    name.setLineBreakMode(objc2_app_kit::NSLineBreakMode::ByWordWrapping);
    name.setPreferredMaxLayoutWidth(LABEL_WIDTH);
    if let Some(cell) = name.cell() {
        cell.setWraps(true);
        cell.setTruncatesLastVisibleLine(false);
    }
    name.widthAnchor().constraintEqualToConstant(LABEL_WIDTH).setActive(true);
    let line = stack(mtm, &[view(&name), view(&stack(mtm, &right, true, 4.0))], false, 12.0);
    line.setAlignment(NSLayoutAttribute::FirstBaseline);
    view(&line)
}

/// A changer disc's Remove button: its slot's id with this suffix, which
/// [`SettingsWindow::action`] turns into an empty path (out of the changer).
const CLEAR_DISC: &str = "#clear";

fn disk_table(mtm: MainThreadMarker, id: &'static str, rows: &[DiskRow]) -> Retained<NSView> {
    let kinds = disk_kind_labels();
    let mut lines: Vec<Retained<NSView>> = Vec::new();
    let indent = || {
        let v = NSView::new(mtm);
        v.widthAnchor().constraintEqualToConstant(56.0).setActive(true);
        v
    };
    let tip = |v: &NSView, t: String| v.setToolTip(Some(&ns(&t)));
    for (r, d) in rows.iter().enumerate() {
        let at = |p: &str| control_id(id, Some(r), Some(p));
        let name = secondary(mtm, &format!("SCSI {}", d.name));
        name.widthAnchor().constraintEqualToConstant(56.0).setActive(true);
        let kind = popup(mtm, &at(part::KIND), &kinds, disk_kind_index(d));
        let remove = push(mtm, &at(part::REMOVE), &t!("config-remove"));
        // The first line: what the target is. Copy-on-write only for what
        // IRIX writes to an image: not a CD-ROM, not a DaynaPort.
        let mut first = vec![view(&name), view(&kind)];
        if d.can_overlay() && !d.is_daynaport() {
            let overlay = checkbox(mtm, &at(part::OVERLAY), &t!("settings-overlay"), d.overlay);
            tip(&overlay, t!("settings-overlay-tip"));
            first.push(view(&overlay));
        }
        first.push(view(&remove));
        if d.scratch {
            first.insert(2, view(&secondary(mtm, &t!("settings-scratch"))));
        }
        let mut line = vec![view(&stack(mtm, &first, false, 6.0))];
        if d.is_daynaport() {
            // An Ethernet card on the bus: its address and its own NAT
            // network, both optional.
            let mac = text_field(mtm, &at(part::MAC), &d.mac, "", 150.0);
            tip(&mac, t!("config-scsi-mac-tip"));
            let subnet = text_field(mtm, &at(part::SUBNET), &d.subnet, "192.168.10.0/24", 140.0);
            tip(&subnet, t!("config-scsi-subnet-tip"));
            line.push(view(&stack(
                mtm,
                &[indent(), view(&secondary(mtm, &t!("config-scsi-mac"))), view(&mac)],
                false,
                6.0,
            )));
            line.push(view(&stack(
                mtm,
                &[indent(), view(&secondary(mtm, &t!("config-scsi-subnet"))), view(&subnet)],
                false,
                6.0,
            )));
        } else {
            let path = text_field(mtm, &at(part::PATH), &d.path, "", 360.0);
            let pick = push(mtm, &at(part::PICK), &t!("settings-choose"));
            let mut second = vec![indent(), view(&path), view(&pick)];
            if d.is_disk() {
                second.push(view(&push(mtm, &at(part::NEW_DISK), &t!("scsi-create-blank"))));
            }
            line.push(view(&stack(mtm, &second, false, 6.0)));
        }
        if d.is_disk() {
            // A scratch volume is made empty at every start, so it has a
            // size instead of an image.
            let scratch = checkbox(mtm, &at(part::SCRATCH), &t!("config-scsi-scratch"), d.scratch);
            let mut third = vec![indent(), view(&scratch)];
            if d.scratch {
                let size = number_field(mtm, &at(part::SCRATCH_MB), d.scratch_mb as f64, 1.0, 1_048_576.0, false);
                third.push(view(&size));
                third.push(view(&secondary(mtm, &t!("create-disk-size-unit"))));
            }
            let cache = checkbox(mtm, &at(part::WRITE_CACHE), &t!("config-scsi-write-cache"), d.write_cache);
            tip(&cache, t!("config-scsi-write-cache-tip"));
            third.push(view(&cache));
            line.push(view(&stack(mtm, &third, false, 6.0)));
        }
        if d.is_cdrom() {
            // The changer: the discs behind the one loaded now, each a slot
            // that reports `disc:<n>` (a path, or Choose for a picker).
            for (n, disc) in d.discs.iter().enumerate() {
                let slot = format!("{}{n}", part::DISC);
                let label = secondary(mtm, &t!("config-scsi-changer-slot", n = n + 1));
                label.widthAnchor().constraintEqualToConstant(56.0).setActive(true);
                let field = text_field(mtm, &at(&slot), disc, "", 300.0);
                let pick = push(mtm, &at(&slot), &t!("settings-choose"));
                let drop = push(mtm, &format!("{}{CLEAR_DISC}", at(&slot)), &t!("config-remove"));
                tip(&drop, t!("config-scsi-disc-remove-tip"));
                line.push(view(&stack(mtm, &[view(&label), view(&field), view(&pick), view(&drop)], false, 6.0)));
            }
            let add = push(mtm, &at(part::ADD_DISC), &t!("config-scsi-add-disc"));
            tip(&add, t!("config-scsi-changer"));
            line.push(view(&stack(mtm, &[indent(), view(&add)], false, 6.0)));
        }
        if let Some(p) = &d.problem {
            let l = label(mtm, p);
            l.setTextColor(Some(&NSColor::systemOrangeColor()));
            line.push(view(&l));
        }
        lines.push(view(&stack(mtm, &line, true, 4.0)));
    }
    let add = stack(
        mtm,
        &[
            view(&push(mtm, &control_id(id, None, Some(part::ADD_DISK)), &t!("settings-add-disk"))),
            view(&push(mtm, &control_id(id, None, Some(part::ADD_CDROM)), &t!("settings-add-cdrom"))),
        ],
        false,
        8.0,
    );
    lines.push(view(&add));
    view(&stack(mtm, &lines, true, 8.0))
}

fn forward_table(mtm: MainThreadMarker, id: &'static str, rows: &[ForwardRow]) -> Retained<NSView> {
    let protos = [t!("config-forward-tcp"), t!("config-forward-udp")];
    let mut lines: Vec<Retained<NSView>> = Vec::new();
    for (r, f) in rows.iter().enumerate() {
        let proto = popup(mtm, &control_id(id, Some(r), Some(part::TCP)), &protos, if f.tcp { 0 } else { 1 });
        let host = text_field(mtm, &control_id(id, Some(r), Some(part::HOST)), &f.host_port.to_string(), "", 64.0);
        let guest = text_field(mtm, &control_id(id, Some(r), Some(part::GUEST)), &f.guest_port.to_string(), "", 64.0);
        let any = checkbox(mtm, &control_id(id, Some(r), Some(part::ANY)), &t!("config-forward-bind-any"), f.any_interface);
        let remove = push(mtm, &control_id(id, Some(r), Some(part::REMOVE)), &t!("config-remove"));
        lines.push(view(&stack(
            mtm,
            &[
                view(&proto),
                view(&secondary(mtm, &t!("config-forward-host"))),
                view(&host),
                view(&secondary(mtm, &t!("config-forward-guest"))),
                view(&guest),
                view(&any),
                view(&remove),
            ],
            false,
            6.0,
        )));
    }
    lines.push(view(&forward_add_menu(mtm, id, rows)));
    view(&stack(mtm, &lines, true, 8.0))
}

/// "Add Forward": a pull-down of the suggested forwards (a service already
/// forwarded is greyed out, saying so), then Custom. Each item's tag is its
/// index in `forward_presets`, Custom's the one past them.
fn forward_add_menu(mtm: MainThreadMarker, id: &'static str, rows: &[ForwardRow]) -> Retained<NSPopUpButton> {
    let p = NSPopUpButton::initWithFrame_pullsDown(mtm.alloc(), NSRect::ZERO, true);
    p.setAutoenablesItems(false);
    let menu = p.menu().expect("a popup has a menu");
    // A pull-down's first item is its title.
    let title = objc2_app_kit::NSMenuItem::new(mtm);
    title.setTitle(&ns(&t!("config-forward-add")));
    title.setTag(-1);
    menu.addItem(&title);
    let presets = iris_gui_core::controller::settings::forward_menu(rows);
    for (i, (preset, enabled)) in presets.iter().enumerate() {
        let it = objc2_app_kit::NSMenuItem::new(mtm);
        it.setTitle(&ns(&preset.label));
        it.setTag(i as isize);
        match enabled {
            Enabled::Yes => it.setToolTip(Some(&ns(&preset.tip))),
            Enabled::No(why) => {
                it.setEnabled(false);
                it.setToolTip(Some(&ns(why)));
            }
        }
        menu.addItem(&it);
    }
    menu.addItem(&objc2_app_kit::NSMenuItem::separatorItem(mtm));
    let custom = objc2_app_kit::NSMenuItem::new(mtm);
    custom.setTitle(&ns(&t!("config-forward-custom")));
    custom.setTag(presets.len() as isize);
    menu.addItem(&custom);
    wire(&p, &control_id(id, None, Some(part::ADD)));
    p
}

/// Disable every control under `v`, saying why in its tooltip.
fn disable(v: &NSView, reason: &str) {
    if let Some(c) = v.downcast_ref::<NSControl>() {
        c.setEnabled(false);
    }
    v.setToolTip(Some(&ns(reason)));
    for sub in v.subviews().iter() {
        disable(&sub, reason);
    }
}
