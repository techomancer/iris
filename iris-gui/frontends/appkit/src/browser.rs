//! The disk browser window, a view of `AppController::browser()`: the
//! partitions on the left, the path bar and the tools above the file table,
//! the read-only reason, the running job with its progress and Cancel, and a
//! text editor window. It is shown exactly while the controller has a
//! browser; closing it asks the controller to close the browser.
//!
//! Every control carries a `disk_browser::ids` id; `app::browser_control`
//! turns it into a `BrowserAction` with `disk_browser::action_for`.

use std::cell::{Cell, RefCell};

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{define_class, msg_send, sel, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{
    NSBackingStoreType, NSBezelStyle, NSButton, NSColor, NSControlStateValueOn, NSControlTextEditingDelegate, NSFont,
    NSImageCell, NSLayoutAttribute, NSProgressIndicator, NSProgressIndicatorStyle, NSScrollView, NSStackView,
    NSDragOperation, NSDraggingInfo, NSTableColumn, NSTableView, NSTableViewDataSource, NSTableViewDelegate,
    NSTableViewDropOperation, NSTextField, NSTextView, NSView,
    NSWindow, NSWindowDelegate, NSWindowStyleMask,
};
use objc2_foundation::{NSIndexSet, NSMutableIndexSet, NSNotification, NSPoint, NSRect, NSSize, NSString};
use iris_gui_core::controller::disk_browser::{columns, ids, tools, DiskBrowserController, PartRow, Row};
use iris_gui_core::controller::Enabled;

use crate::settings::{checkbox, stack, view};
use crate::ui::{button, label, ns, secondary, set_id};

/// Column widths by id; the titles come from `disk_browser::columns`.
fn column_width(id: &str) -> f64 {
    match id {
        "name" => 260.0,
        "date" => 150.0,
        "mode" => 100.0,
        _ => 80.0,
    }
}

pub struct TableIvars {
    rows: RefCell<Vec<Row>>,
    /// Set while the shell changes the selection itself, so the delegate
    /// does not report it back.
    syncing: Cell<bool>,
}

define_class!(
    /// The file table's data source and delegate: it holds its own copy of
    /// the rows, so AppKit can ask for cells while the shell is busy.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISBrowserTable"]
    #[ivars = TableIvars]
    pub struct TableData;

    unsafe impl NSObjectProtocol for TableData {}

    unsafe impl NSTableViewDataSource for TableData {
        #[unsafe(method(numberOfRowsInTableView:))]
        fn number_of_rows(&self, _table: &NSTableView) -> isize {
            self.ivars().rows.borrow().len() as isize
        }

        #[unsafe(method(tableView:objectValueForTableColumn:row:))]
        fn object_value(&self, _table: &NSTableView, column: Option<&NSTableColumn>, row: isize) -> *mut AnyObject {
            let rows = self.ivars().rows.borrow();
            let (Some(col), Some(r)) = (column, rows.get(row as usize)) else { return std::ptr::null_mut() };
            let id = col.identifier().to_string();
            let obj: Option<Retained<AnyObject>> = match id.as_str() {
                "icon" => crate::glyphs::icon(r.icon, &r.name).map(|i| Retained::into_super(Retained::into_super(i))),
                _ => {
                    let text = match id.as_str() {
                        "name" => r.name.as_str(),
                        "size" => r.size.as_str(),
                        "mode" => r.mode.as_str(),
                        "owner" => r.owner.as_str(),
                        "group" => r.group.as_str(),
                        "date" => r.date.as_str(),
                        _ => "",
                    };
                    Some(Retained::into_super(Retained::into_super(NSString::from_str(text))))
                }
            };
            // AppKit retains what it keeps; hand it an autoreleased object.
            obj.map_or(std::ptr::null_mut(), Retained::autorelease_return)
        }

        /// Host files dragged over the table: copy them in. Over a folder
        /// row they go into that folder; elsewhere into the open folder.
        #[unsafe(method(tableView:validateDrop:proposedRow:proposedDropOperation:))]
        fn validate_drop(
            &self,
            table: &NSTableView,
            info: &ProtocolObject<dyn NSDraggingInfo>,
            row: isize,
            op: NSTableViewDropOperation,
        ) -> NSDragOperation {
            if dropped_paths(info).is_empty() {
                return NSDragOperation::None;
            }
            let on_folder = op == NSTableViewDropOperation::On
                && self.ivars().rows.borrow().get(row as usize).is_some_and(|r| r.is_dir);
            if !on_folder {
                // Retarget to the whole table: the open folder.
                table.setDropRow_dropOperation(-1, NSTableViewDropOperation::On);
            }
            NSDragOperation::Copy
        }

        #[unsafe(method(tableView:acceptDrop:row:dropOperation:))]
        fn accept_drop(
            &self,
            _table: &NSTableView,
            info: &ProtocolObject<dyn NSDraggingInfo>,
            row: isize,
            op: NSTableViewDropOperation,
        ) -> objc2::runtime::Bool {
            let paths = dropped_paths(info);
            if paths.is_empty() {
                return objc2::runtime::Bool::NO;
            }
            let into = (row >= 0 && op == NSTableViewDropOperation::On
                && self.ivars().rows.borrow().get(row as usize).is_some_and(|r| r.is_dir))
                .then_some(row as usize);
            crate::app::later(Box::new(move || crate::app::browser_drop(paths, into)));
            objc2::runtime::Bool::YES
        }
    }

    unsafe impl NSControlTextEditingDelegate for TableData {}

    unsafe impl NSTableViewDelegate for TableData {
        #[unsafe(method(tableViewSelectionDidChange:))]
        fn selection_changed(&self, note: &NSNotification) {
            if self.ivars().syncing.get() {
                return;
            }
            let Some(table) = note.object().and_then(|o| o.downcast::<NSTableView>().ok()) else { return };
            let picked = indices(&table.selectedRowIndexes());
            crate::app::later(Box::new(move || crate::app::browser_select(picked)));
        }
    }

    impl TableData {
        #[unsafe(method(doubleClicked:))]
        fn double_clicked(&self, sender: Option<&AnyObject>) {
            let Some(table) = sender.and_then(|s| s.downcast_ref::<NSTableView>()) else { return };
            let row = table.clickedRow();
            if row >= 0 {
                let row = row as usize;
                crate::app::later(Box::new(move || crate::app::browser_activate(row)));
            }
        }
    }
);

define_class!(
    /// Closing the browser window asks the controller, which may refuse
    /// while a job writes; the window follows `app.browser()`.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISBrowserWindowDelegate"]
    pub struct CloseDelegate;

    unsafe impl NSObjectProtocol for CloseDelegate {}

    unsafe impl NSWindowDelegate for CloseDelegate {
        #[unsafe(method(windowShouldClose:))]
        fn should_close(&self, _w: &NSWindow) -> bool {
            crate::app::later(Box::new(crate::app::close_browser));
            false
        }
    }
);

/// The host files on a drag's pasteboard.
fn dropped_paths(info: &ProtocolObject<dyn NSDraggingInfo>) -> Vec<std::path::PathBuf> {
    let pb = info.draggingPasteboard();
    let Some(items) = pb.pasteboardItems() else { return Vec::new() };
    items
        .iter()
        // SAFETY: AppKit's own pasteboard type constant.
        .filter_map(|it| it.stringForType(unsafe { objc2_app_kit::NSPasteboardTypeFileURL }))
        .filter_map(|s| objc2_foundation::NSURL::URLWithString(&s))
        .filter_map(|u| u.path())
        .map(|p| std::path::PathBuf::from(p.to_string()))
        .collect()
}

fn indices(set: &NSIndexSet) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = set.firstIndex();
    while i != objc2_foundation::NSNotFound as usize {
        out.push(i);
        i = set.indexGreaterThanIndex(i);
    }
    out
}

struct EditorWindow {
    window: Retained<NSWindow>,
    text: Retained<NSTextView>,
    save: Retained<NSButton>,
    info: Retained<NSTextField>,
    name: String,
}

pub struct BrowserWindow {
    pub window: Retained<NSWindow>,
    _delegate: Retained<CloseDelegate>,
    parts: Retained<NSStackView>,
    path: Retained<NSStackView>,
    tools: Vec<(&'static str, iris_gui_core::controller::disk_browser::BrowserAction, Retained<NSButton>)>,
    endings: Retained<NSButton>,
    replay: Retained<NSButton>,
    notice: Retained<NSTextField>,
    edit_anyway: Retained<NSButton>,
    table: Retained<NSTableView>,
    data: Retained<TableData>,
    empty: Retained<NSTextField>,
    job: Retained<NSStackView>,
    job_label: Retained<NSTextField>,
    progress: Retained<NSProgressIndicator>,
    cancel: Retained<NSButton>,
    editor: Option<EditorWindow>,
    last_parts: Option<(Vec<PartRow>, Option<usize>)>,
    last_path: Vec<String>,
    /// The controller's selection as last put on the table: it is written
    /// again only when the controller changes it, so a tick between a click
    /// and its report cannot undo the click.
    last_selection: Vec<usize>,
}

fn new_window(mtm: MainThreadMarker, size: NSSize, style: NSWindowStyleMask) -> Retained<NSWindow> {
    // SAFETY: a standard window, kept by the shell while the browser is open.
    let w = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            mtm.alloc(),
            NSRect::new(NSPoint::new(180.0, 180.0), size),
            style,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    // SAFETY: the shell keeps the window.
    unsafe { w.setReleasedWhenClosed(false) };
    w
}

fn pin(v: &NSView, to: &NSView) {
    v.setTranslatesAutoresizingMaskIntoConstraints(false);
    v.leadingAnchor().constraintEqualToAnchor(&to.leadingAnchor()).setActive(true);
    v.trailingAnchor().constraintEqualToAnchor(&to.trailingAnchor()).setActive(true);
    v.topAnchor().constraintEqualToAnchor(&to.topAnchor()).setActive(true);
    v.bottomAnchor().constraintEqualToAnchor(&to.bottomAnchor()).setActive(true);
}

fn clear(s: &NSStackView) {
    for v in s.arrangedSubviews().iter() {
        s.removeArrangedSubview(&v);
        v.removeFromSuperview();
    }
}

fn tool_button(mtm: MainThreadMarker, id: &str, title: &str, icon: iris_gui_core::controller::icons::Icon) -> Retained<NSButton> {
    let b = button(mtm, title, id);
    b.setBezelStyle(NSBezelStyle::Push);
    b.setToolTip(Some(&ns(title)));
    if let Some(img) = crate::glyphs::icon(icon, title) {
        b.setImage(Some(&img));
        b.setTitle(&ns(""));
    }
    b
}

impl BrowserWindow {
    pub fn new(mtm: MainThreadMarker) -> Self {
        let style = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Closable
            | NSWindowStyleMask::Miniaturizable
            | NSWindowStyleMask::Resizable;
        let window = new_window(mtm, NSSize::new(1000.0, 620.0), style);
        window.setMinSize(NSSize::new(720.0, 400.0));
        let delegate: Retained<CloseDelegate> = {
            let this = mtm.alloc::<CloseDelegate>().set_ivars(());
            // SAFETY: NSObject's plain init.
            unsafe { msg_send![super(this), init] }
        };
        window.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));

        // Left: the partitions.
        let parts = stack(mtm, &[], true, 2.0);
        parts.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: 14.0, left: 10.0, bottom: 14.0, right: 10.0 });
        let side = objc2_app_kit::NSVisualEffectView::new(mtm);
        side.setMaterial(objc2_app_kit::NSVisualEffectMaterial::Sidebar);

        // Top: the tools, then the path bar.
        let mut tool_views: Vec<Retained<NSView>> = Vec::new();
        let mut tool_buttons = Vec::new();
        for t in tools() {
            match t {
                Some(t) => {
                    let b = tool_button(mtm, t.id, &t.label, t.icon);
                    tool_views.push(view(&b));
                    tool_buttons.push((t.id, t.action, b));
                }
                None => {
                    let gap = NSView::new(mtm);
                    gap.widthAnchor().constraintEqualToConstant(10.0).setActive(true);
                    tool_views.push(gap);
                }
            }
        }
        let tools_row = stack(mtm, &tool_views, false, 4.0);
        let path = stack(mtm, &[], false, 2.0);
        // The path on the left, the import option on the right.
        let endings = checkbox(mtm, ids::UNIX_LINE_ENDINGS, &t!("browser-unix-line-endings"), false);
        let path_spacer = NSView::new(mtm);
        path_spacer.setContentHuggingPriority_forOrientation(1.0, objc2_app_kit::NSLayoutConstraintOrientation::Horizontal);
        let path_row = stack(mtm, &[view(&path), path_spacer, view(&endings)], false, 12.0);

        // The read-only reason, with Edit Anyway when the controller offers it.
        let notice = label(mtm, "");
        notice.setTextColor(Some(&NSColor::secondaryLabelColor()));
        notice.setLineBreakMode(objc2_app_kit::NSLineBreakMode::ByWordWrapping);
        notice.setPreferredMaxLayoutWidth(600.0);
        let edit_anyway = button(mtm, &t!("browser-edit-anyway"), ids::EDIT_ANYWAY);
        let replay = button(mtm, &t!("browser-replay"), ids::REPLAY_LOG);
        let notice_row = stack(mtm, &[view(&notice), view(&edit_anyway), view(&replay)], false, 8.0);

        // The table.
        let table = NSTableView::initWithFrame(mtm.alloc(), NSRect::ZERO);
        let icon = NSTableColumn::initWithIdentifier(mtm.alloc(), &ns("icon"));
        icon.setTitle(&ns(""));
        icon.setWidth(22.0);
        // SAFETY: an image cell for an image column; the column keeps it.
        unsafe { icon.setDataCell(&NSImageCell::new(mtm)) };
        icon.setResizingMask(objc2_app_kit::NSTableColumnResizingOptions::NoResizing);
        table.addTableColumn(&icon);
        for (id, title) in columns() {
            let c = NSTableColumn::initWithIdentifier(mtm.alloc(), &ns(id));
            c.setTitle(&ns(&title));
            c.setWidth(column_width(id));
            table.addTableColumn(&c);
        }
        table.setAllowsMultipleSelection(true);
        // SAFETY: AppKit's own pasteboard type constant.
        table.registerForDraggedTypes(&objc2_foundation::NSArray::from_slice(&[unsafe { objc2_app_kit::NSPasteboardTypeFileURL }]));
        table.setUsesAlternatingRowBackgroundColors(true);
        set_id(&table, ids::TABLE);
        let data: Retained<TableData> = {
            let this = mtm.alloc::<TableData>().set_ivars(TableIvars { rows: RefCell::default(), syncing: Cell::new(false) });
            // SAFETY: NSObject's plain init.
            unsafe { msg_send![super(this), init] }
        };
        // SAFETY: the window keeps `data` as long as the table.
        unsafe {
            table.setDataSource(Some(ProtocolObject::from_ref(&*data)));
            table.setDelegate(Some(ProtocolObject::from_ref(&*data)));
            table.setTarget(Some(&data));
            table.setDoubleAction(Some(sel!(doubleClicked:)));
        }
        let scroll = NSScrollView::initWithFrame(mtm.alloc(), NSRect::ZERO);
        scroll.setDocumentView(Some(&table));
        scroll.setHasVerticalScroller(true);
        scroll.setHasHorizontalScroller(true);
        let empty = secondary(mtm, &t!("browser-empty"));

        // The job: what it does, its progress, Cancel.
        let job_label = label(mtm, "");
        let progress = NSProgressIndicator::new(mtm);
        progress.setStyle(NSProgressIndicatorStyle::Bar);
        progress.widthAnchor().constraintEqualToConstant(240.0).setActive(true);
        let cancel = button(mtm, &t!("common-cancel"), ids::CANCEL);
        let job = stack(mtm, &[view(&job_label), view(&progress), view(&cancel)], false, 10.0);

        let right = stack(mtm, &[view(&tools_row), view(&path_row), view(&notice_row), view(&scroll), view(&job)], true, 8.0);
        right.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: 12.0, left: 12.0, bottom: 12.0, right: 12.0 });
        right.setAlignment(NSLayoutAttribute::Leading);
        scroll.leadingAnchor().constraintEqualToAnchor(&right.leadingAnchor()).setActive(true);
        // Fill the width (inside the insets) and take the spare height.
        scroll.trailingAnchor().constraintEqualToAnchor_constant(&right.trailingAnchor(), -12.0).setActive(true);
        path_row.trailingAnchor().constraintEqualToAnchor_constant(&right.trailingAnchor(), -12.0).setActive(true);
        scroll.setContentHuggingPriority_forOrientation(1.0, objc2_app_kit::NSLayoutConstraintOrientation::Vertical);
        scroll.heightAnchor().constraintGreaterThanOrEqualToConstant(200.0).setActive(true);

        let root = NSView::new(mtm);
        for v in [view(&side), view(&parts), view(&right), view(&empty)] {
            v.setTranslatesAutoresizingMaskIntoConstraints(false);
            root.addSubview(&v);
        }
        for v in [view(&side), view(&parts)] {
            v.leadingAnchor().constraintEqualToAnchor(&root.leadingAnchor()).setActive(true);
            v.topAnchor().constraintEqualToAnchor(&root.topAnchor()).setActive(true);
            v.widthAnchor().constraintEqualToConstant(200.0).setActive(true);
        }
        side.bottomAnchor().constraintEqualToAnchor(&root.bottomAnchor()).setActive(true);
        right.leadingAnchor().constraintEqualToAnchor(&side.trailingAnchor()).setActive(true);
        right.trailingAnchor().constraintEqualToAnchor(&root.trailingAnchor()).setActive(true);
        right.topAnchor().constraintEqualToAnchor(&root.topAnchor()).setActive(true);
        right.bottomAnchor().constraintEqualToAnchor(&root.bottomAnchor()).setActive(true);
        empty.centerXAnchor().constraintEqualToAnchor(&scroll.centerXAnchor()).setActive(true);
        empty.centerYAnchor().constraintEqualToAnchor(&scroll.centerYAnchor()).setActive(true);
        window.setContentView(Some(&root));
        set_id(&root, ids::WINDOW);
        window.center();
        window.setFrameAutosaveName(&ns("IRISBrowserWindow"));
        Self {
            window,
            _delegate: delegate,
            parts,
            path,
            tools: tool_buttons,
            endings,
            replay,
            notice,
            edit_anyway,
            table,
            data,
            empty,
            job,
            job_label,
            progress,
            cancel,
            editor: None,
            last_parts: None,
            last_path: Vec::new(),
            last_selection: Vec::new(),
        }
    }

    /// Replace the editor's text, as typing would (the script hook).
    pub fn set_editor_text(&self, text: &str) {
        if let Some(e) = &self.editor {
            e.text.setString(&ns(text));
        }
    }

    /// The editor's text, for Save and Close.
    pub fn editor_text(&self) -> Option<String> {
        self.editor.as_ref().map(|e| e.text.string().to_string())
    }

    /// Close the editor window and this one (the controller has no browser).
    pub fn close(&mut self) {
        if let Some(e) = self.editor.take() {
            e.window.orderOut(None);
        }
        self.window.orderOut(None);
        // AppKit holds these weakly and may outlive us (an accessibility
        // client, an autoreleased window): it must not reach them freed.
        // SAFETY: clearing a weak reference.
        unsafe {
            self.table.setDataSource(None);
            self.table.setDelegate(None);
            self.table.setTarget(None);
        }
        self.window.setDelegate(None);
    }

    /// Bring the window in line with the browser.
    pub fn refresh(&mut self, mtm: MainThreadMarker, b: &DiskBrowserController) {
        let title = b.title();
        if self.window.title().to_string() != title {
            self.window.setTitle(&ns(&title));
        }
        // Partitions.
        let parts = (b.partitions(), b.selected_partition());
        if self.last_parts.as_ref() != Some(&parts) {
            clear(&self.parts);
            for (pos, p) in parts.0.iter().enumerate() {
                let bt = button(mtm, &p.name, &format!("{}{pos}", ids::PART));
                bt.setBezelStyle(NSBezelStyle::AccessoryBar);
                bt.setButtonType(objc2_app_kit::NSButtonType::PushOnPushOff);
                let on = parts.1 == Some(pos);
                bt.setState(if on { NSControlStateValueOn } else { 0 });
                bt.setShowsBorderOnlyWhileMouseInside(!on);
                bt.setAlignment(objc2_app_kit::NSTextAlignment::Left);
                bt.setEnabled(p.browsable);
                let detail = secondary(mtm, &p.detail);
                detail.setFont(Some(&NSFont::systemFontOfSize(11.0)));
                detail.setToolTip(Some(&ns(&p.detail)));
                let cell = stack(mtm, &[view(&bt), view(&detail)], true, 0.0);
                bt.widthAnchor().constraintEqualToConstant(180.0).setActive(true);
                self.parts.addArrangedSubview(&cell);
            }
            if let Some(e) = b.error() {
                let l = label(mtm, e);
                l.setTextColor(Some(&NSColor::systemRedColor()));
                l.setLineBreakMode(objc2_app_kit::NSLineBreakMode::ByWordWrapping);
                l.setPreferredMaxLayoutWidth(176.0);
                self.parts.addArrangedSubview(&l);
            }
            self.last_parts = Some(parts);
        }
        // Path bar.
        let path = b.path();
        if path != self.last_path {
            clear(&self.path);
            for (depth, name) in path.iter().enumerate() {
                if depth > 0 {
                    self.path.addArrangedSubview(&secondary(mtm, "\u{203a}"));
                }
                let bt = button(mtm, name, &format!("{}{depth}", ids::PATH));
                bt.setBordered(false);
                self.path.addArrangedSubview(&bt);
            }
            self.last_path = path;
        }
        // Tools.
        for (_, action, bt) in &self.tools {
            let e = b.enabled(action);
            bt.setEnabled(e == Enabled::Yes);
            if let Enabled::No(why) = &e {
                if !why.is_empty() {
                    bt.setToolTip(Some(&ns(why)));
                }
            }
        }
        self.endings.setState(if b.unix_line_endings() { NSControlStateValueOn } else { 0 });
        // Read-only reason.
        let reason = b.read_only_reason().unwrap_or_default();
        if self.notice.stringValue().to_string() != reason {
            self.notice.setStringValue(&ns(&reason));
        }
        self.notice.setHidden(reason.is_empty());
        let anyway = b.enabled(&iris_gui_core::controller::disk_browser::BrowserAction::EditAnyway) == Enabled::Yes;
        self.edit_anyway.setHidden(!anyway);
        let replay = b.enabled(&iris_gui_core::controller::disk_browser::BrowserAction::ReplayLog) == Enabled::Yes;
        self.replay.setHidden(!replay);
        // Rows: reload only when they changed; then the selection.
        let rows = b.rows();
        let iv = self.data.ivars();
        let reloaded = *iv.rows.borrow() != rows;
        if reloaded {
            *iv.rows.borrow_mut() = rows;
            iv.syncing.set(true);
            self.table.reloadData();
            iv.syncing.set(false);
        }
        self.empty.setHidden(!iv.rows.borrow().is_empty() || b.selected_partition().is_none());
        let want = b.selection().to_vec();
        if (reloaded || want != self.last_selection) && indices(&self.table.selectedRowIndexes()) != want {
            let set = NSMutableIndexSet::new();
            for i in &want {
                set.addIndex(*i);
            }
            iv.syncing.set(true);
            self.table.selectRowIndexes_byExtendingSelection(&set, false);
            iv.syncing.set(false);
        }
        self.last_selection = want;
        // The job.
        match b.job() {
            Some(j) if j.is_running() => {
                self.job.setHidden(false);
                let text = if j.phase.is_empty() { j.label.clone() } else { format!("{}: {}", j.label, j.phase) };
                self.job_label.setStringValue(&ns(&text));
                match j.progress.fraction() {
                    Some(f) => {
                        self.progress.setIndeterminate(false);
                        self.progress.setDoubleValue(f as f64 * 100.0);
                    }
                    None => {
                        self.progress.setIndeterminate(true);
                        // SAFETY: the sender argument is unused.
                        unsafe { self.progress.startAnimation(None) };
                    }
                }
                self.cancel.setEnabled(j.can_cancel && !j.cancel_requested);
            }
            _ => {
                // SAFETY: the sender argument is unused.
                unsafe { self.progress.stopAnimation(None) };
                self.job.setHidden(true);
            }
        }
        self.refresh_editor(mtm, b);
    }

    fn refresh_editor(&mut self, mtm: MainThreadMarker, b: &DiskBrowserController) {
        match (b.editor(), &mut self.editor) {
            (None, Some(_)) => {
                if let Some(e) = self.editor.take() {
                    e.window.orderOut(None);
                }
            }
            (Some(ed), None) => {
                let w = new_window(mtm, NSSize::new(720.0, 560.0), NSWindowStyleMask::Titled | NSWindowStyleMask::Resizable);
                let scroll = NSTextView::scrollableTextView(mtm);
                let text = scroll
                    .documentView()
                    .and_then(|d| d.downcast::<NSTextView>().ok())
                    .expect("a scrollable text view holds a text view");
                text.setString(&ns(&ed.text));
                text.setFont(Some(&NSFont::userFixedPitchFontOfSize(12.0).unwrap_or_else(|| NSFont::systemFontOfSize(12.0))));
                text.setRichText(false);
                text.setAutomaticQuoteSubstitutionEnabled(false);
                text.setAutomaticDashSubstitutionEnabled(false);
                text.setAutomaticSpellingCorrectionEnabled(false);
                set_id(&text, ids::EDITOR_TEXT);
                let info = secondary(mtm, "");
                let close = button(mtm, &t!("browser-editor-close"), ids::EDITOR_CLOSE);
                close.setKeyEquivalent(&ns("\u{1b}"));
                let save = button(mtm, &t!("browser-editor-save"), ids::EDITOR_SAVE);
                save.setKeyEquivalent(&ns("s"));
                save.setKeyEquivalentModifierMask(objc2_app_kit::NSEventModifierFlags::Command);
                let spacer = NSView::new(mtm);
                spacer.setContentHuggingPriority_forOrientation(1.0, objc2_app_kit::NSLayoutConstraintOrientation::Horizontal);
                let bar = stack(mtm, &[view(&info), spacer, view(&close), view(&save)], false, 8.0);
                let page = stack(mtm, &[view(&scroll), view(&bar)], true, 8.0);
                page.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: 10.0, left: 10.0, bottom: 10.0, right: 10.0 });
                page.setAlignment(NSLayoutAttribute::Width);
                let root = NSView::new(mtm);
                root.addSubview(&page);
                pin(&page, &root);
                w.setContentView(Some(&root));
                w.center();
                w.makeKeyAndOrderFront(None);
                self.editor = Some(EditorWindow { window: w, text, save, info, name: String::new() });
                self.refresh_editor(mtm, b);
            }
            (Some(ed), Some(e)) => {
                let title = if ed.dirty { format!("{} \u{2022}", ed.name) } else { ed.name.clone() };
                if e.name != title {
                    e.window.setTitle(&ns(&title));
                    e.name = title;
                }
                let mut info = ed.line_endings.to_string();
                if ed.read_only {
                    info.push_str(&format!("   {}", t!("browser-editor-read-only")));
                }
                e.info.setStringValue(&ns(&info));
                e.text.setEditable(!ed.read_only);
                let save = b.enabled(&iris_gui_core::controller::disk_browser::BrowserAction::EditorSave);
                e.save.setEnabled(save == Enabled::Yes && !ed.read_only);
            }
            (None, None) => {}
        }
    }
}
