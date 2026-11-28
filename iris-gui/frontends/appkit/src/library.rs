//! The machine library on the left: a view-based `NSTableView` over the app
//! controller's groups (user folders first, then families). A row has the
//! machine's picture, its name, the model line, and a dot only when it is
//! running or has a problem.

use std::cell::{Cell, RefCell};

use objc2::rc::Retained;
use objc2::runtime::{NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{define_class, msg_send, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{
    NSColor, NSControlTextEditingDelegate, NSFont, NSMenu, NSMenuDelegate, NSMenuItem, NSImageView, NSLayoutConstraintOrientation, NSScrollView, NSStackView, NSStackViewDistribution,
    NSTableColumn, NSTableView, NSTableViewDataSource, NSTableViewDelegate, NSTableViewSelectionHighlightStyle,
    NSTableViewStyle, NSTextField, NSUserInterfaceLayoutOrientation, NSView,
};
use objc2_foundation::{NSInteger, NSNotification, NSPoint, NSRect, NSSize};
use iris_gui_core::controller::app::{LibraryGroup, LibraryRow, RowDot};
use iris_gui_core::controller::ids;

use crate::glyphs;
use crate::ui::{label, ns, secondary, set_id};

#[derive(Clone)]
enum Item {
    Group(String),
    Machine(LibraryRow),
}

#[derive(Default)]
pub struct State {
    items: RefCell<Vec<Item>>,
    /// Set while the shell sets the selection itself, so it is not reported back.
    syncing: Cell<bool>,
    /// The table, for the row a right-click landed on.
    table: RefCell<Option<Retained<NSTableView>>>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISLibrary"]
    #[ivars = State]
    pub struct Source;

    unsafe impl NSObjectProtocol for Source {}

    unsafe impl NSTableViewDataSource for Source {
        #[unsafe(method(numberOfRowsInTableView:))]
        fn rows(&self, _t: &NSTableView) -> NSInteger {
            self.ivars().items.borrow().len() as NSInteger
        }
    }

    unsafe impl NSControlTextEditingDelegate for Source {}

    unsafe impl NSMenuDelegate for Source {
        /// A right-click on a machine: its menu, from `AppController::machine_menu`.
        #[unsafe(method(menuNeedsUpdate:))]
        fn menu_needs_update(&self, menu: &NSMenu) {
            menu.removeAllItems();
            let Some(table) = self.ivars().table.borrow().clone() else { return };
            let row = table.clickedRow();
            if row < 0 {
                return;
            }
            let name = match self.ivars().items.borrow().get(row as usize) {
                Some(Item::Machine(m)) => m.name.clone(),
                _ => return,
            };
            let Some(sections) = crate::app::machine_menu(&name) else { return };
            for (i, section) in sections.iter().enumerate() {
                if i > 0 {
                    menu.addItem(&NSMenuItem::separatorItem(self.mtm()));
                }
                for e in section {
                    menu.addItem(&crate::menu::command_item(self.mtm(), &e.label, &e.id));
                }
            }
        }
    }

    unsafe impl NSTableViewDelegate for Source {
        #[unsafe(method_id(tableView:viewForTableColumn:row:))]
        fn view(&self, _t: &NSTableView, _c: Option<&NSTableColumn>, row: NSInteger) -> Option<Retained<NSView>> {
            let item = self.ivars().items.borrow().get(row as usize).cloned();
            item.map(|i| cell(self.mtm(), &i))
        }

        #[unsafe(method(tableView:isGroupRow:))]
        fn is_group(&self, _t: &NSTableView, row: NSInteger) -> bool {
            matches!(self.ivars().items.borrow().get(row as usize), Some(Item::Group(_)))
        }

        #[unsafe(method(tableView:shouldSelectRow:))]
        fn should_select(&self, _t: &NSTableView, row: NSInteger) -> bool {
            matches!(self.ivars().items.borrow().get(row as usize), Some(Item::Machine(_)))
        }

        #[unsafe(method(tableView:heightOfRow:))]
        fn height(&self, _t: &NSTableView, row: NSInteger) -> f64 {
            match self.ivars().items.borrow().get(row as usize) {
                Some(Item::Group(_)) => 26.0,
                _ => 46.0,
            }
        }

        #[unsafe(method(tableViewSelectionDidChange:))]
        fn selection_changed(&self, note: &NSNotification) {
            if self.ivars().syncing.get() {
                return;
            }
            let Some(table) = note.object().and_then(|o| o.downcast::<NSTableView>().ok()) else { return };
            let row = table.selectedRow();
            if row < 0 {
                return;
            }
            let name = match self.ivars().items.borrow().get(row as usize) {
                Some(Item::Machine(m)) => m.name.clone(),
                _ => return,
            };
            // After this handler: selecting changes the window's contents.
            crate::app::later(Box::new(move || crate::app::select_machine(name)));
        }
    }
);

fn cell(mtm: MainThreadMarker, item: &Item) -> Retained<NSView> {
    match item {
        Item::Group(title) => {
            let l = secondary(mtm, &title.to_uppercase());
            l.setFont(Some(&NSFont::boldSystemFontOfSize(11.0)));
            let g = NSStackView::stackViewWithViews(
                &objc2_foundation::NSArray::from_retained_slice(&[Retained::into_super(Retained::into_super(l))]),
                mtm,
            );
            g.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: 6.0, left: 12.0, bottom: 2.0, right: 8.0 });
            Retained::into_super(g)
        }
        Item::Machine(m) => {
            let image = NSImageView::new(mtm);
            if let Some(img) = glyphs::machine_image(m.icon, 32.0, &m.subtitle) {
                image.setImage(Some(&img));
            }
            image.setContentTintColor(Some(&NSColor::secondaryLabelColor()));
            let name = label(mtm, &m.name);
            name.setFont(Some(&NSFont::systemFontOfSize(13.0)));
            let sub = secondary(mtm, &m.subtitle);
            sub.setFont(Some(&NSFont::systemFontOfSize(11.0)));
            // A narrow sidebar cuts the name and model short with "…" rather
            // than holding the sidebar open.
            for l in [&name, &sub] {
                l.setLineBreakMode(objc2_app_kit::NSLineBreakMode::ByTruncatingTail);
                l.setContentCompressionResistancePriority_forOrientation(500.0, NSLayoutConstraintOrientation::Horizontal);
            }
            let text = NSStackView::stackViewWithViews(
                &objc2_foundation::NSArray::from_retained_slice(&[
                    Retained::into_super(Retained::into_super(name)),
                    Retained::into_super(Retained::into_super(sub)),
                ]),
                mtm,
            );
            text.setOrientation(NSUserInterfaceLayoutOrientation::Vertical);
            text.setAlignment(objc2_app_kit::NSLayoutAttribute::Leading);
            text.setSpacing(1.0);
            text.setClippingResistancePriority_forOrientation(500.0, NSLayoutConstraintOrientation::Horizontal);
            let mut views: Vec<Retained<NSView>> =
                vec![Retained::into_super(Retained::into_super(image)), Retained::into_super(text)];
            let spacer = NSView::new(mtm);
            spacer.setContentHuggingPriority_forOrientation(1.0, NSLayoutConstraintOrientation::Horizontal);
            views.push(spacer);
            if let Some(dot) = &m.dot {
                let d: Retained<NSTextField> = label(mtm, "\u{25CF}");
                let (color, tip) = match dot {
                    RowDot::Running => (NSColor::systemGreenColor(), t!("status-running")),
                    RowDot::Problem(p) => (NSColor::systemOrangeColor(), p.clone()),
                };
                d.setTextColor(Some(&color));
                d.setToolTip(Some(&ns(&tip)));
                views.push(Retained::into_super(Retained::into_super(d)));
            }
            // Its own Start, as UTM has; a running machine shows its dot instead.
            if !matches!(m.dot, Some(RowDot::Running)) {
                let id = iris_gui_core::controller::ids::item::id(iris_gui_core::controller::ids::item::START_MACHINE, &m.name);
                let tip = t!("tb-start-machine", name = m.name.as_str());
                let b = crate::ui::button(mtm, "", &id);
                b.setBordered(false);
                if let Some(img) = glyphs::symbol("play.circle.fill", &tip) {
                    b.setImage(Some(&img));
                    b.setSymbolConfiguration(Some(&objc2_app_kit::NSImageSymbolConfiguration::configurationWithPointSize_weight(20.0, 0.0)));
                } else {
                    b.setTitle(&ns("\u{25B6}"));
                }
                b.setContentTintColor(Some(&NSColor::labelColor()));
                b.setToolTip(Some(&ns(&tip)));
                views.push(Retained::into_super(Retained::into_super(b)));
            }
            let row = NSStackView::stackViewWithViews(&objc2_foundation::NSArray::from_retained_slice(&views), mtm);
            row.setOrientation(NSUserInterfaceLayoutOrientation::Horizontal);
            row.setSpacing(8.0);
            row.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: 4.0, left: 12.0, bottom: 4.0, right: 8.0 });
            row.setDistribution(NSStackViewDistribution::Fill);
            row.setHuggingPriority_forOrientation(250.0, NSLayoutConstraintOrientation::Horizontal);
            set_id(&row, &format!("library.row.{}", m.name));
            Retained::into_super(row)
        }
    }
}

/// The library pane.
pub struct Library {
    pub scroll: Retained<NSScrollView>,
    table: Retained<NSTableView>,
    source: Retained<Source>,
}

impl Library {
    pub fn new(mtm: MainThreadMarker) -> Self {
        let source: Retained<Source> = {
            let this = mtm.alloc::<Source>().set_ivars(State::default());
            // SAFETY: NSObject's plain init.
            unsafe { msg_send![super(this), init] }
        };
        let table = NSTableView::initWithFrame(mtm.alloc(), NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(240.0, 400.0)));
        let column = NSTableColumn::initWithIdentifier(mtm.alloc(), &ns("machine"));
        column.setWidth(230.0);
        column.setResizingMask(objc2_app_kit::NSTableColumnResizingOptions::AutoresizingMask);
        table.addTableColumn(&column);
        // The one column is as wide as the sidebar, so a row's ▶ and dot sit
        // at its trailing edge however wide the sidebar is.
        table.setColumnAutoresizingStyle(objc2_app_kit::NSTableViewColumnAutoresizingStyle::FirstColumnOnlyAutoresizingStyle);
        column.setMaxWidth(10_000.0);
        table.setIntercellSpacing(NSSize::new(0.0, 2.0));
        table.setHeaderView(None);
        // See-through, so the sidebar's own material shows behind the rows.
        table.setBackgroundColor(&NSColor::clearColor());
        // The sidebar's own look: rounded selections inset from the edges, as
        // Finder's (the source-list style; before macOS 11 the old highlight).
        if crate::ui::has_symbols() {
            table.setStyle(NSTableViewStyle::SourceList);
        } else {
            #[allow(deprecated)]
            table.setSelectionHighlightStyle(NSTableViewSelectionHighlightStyle::SourceList);
        }
        // Group titles scroll with the list, on the sidebar's tone (a floating
        // title draws a strip of its own).
        table.setFloatsGroupRows(false);
        // SAFETY: the source lives as long as the table (both are kept in `Library`).
        unsafe {
            table.setDataSource(Some(ProtocolObject::from_ref(&*source)));
            table.setDelegate(Some(ProtocolObject::from_ref(&*source)));
        }
        set_id(&table, ids::library::LIST);
        // Right-click a machine for what you can do with it.
        let menu = NSMenu::new(mtm);
        menu.setDelegate(Some(ProtocolObject::from_ref(&*source)));
        // SAFETY: the menu is kept by the table; its delegate by `Library`.
        unsafe { table.setMenu(Some(&menu)) };
        *source.ivars().table.borrow_mut() = Some(table.clone());
        let scroll = NSScrollView::initWithFrame(mtm.alloc(), NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(240.0, 400.0)));
        table.setAutoresizingMask(objc2_app_kit::NSAutoresizingMaskOptions::ViewWidthSizable);
        scroll.setDocumentView(Some(&table));
        table.sizeLastColumnToFit();
        scroll.setHasVerticalScroller(true);
        scroll.setDrawsBackground(false);
        Self { scroll, table, source }
    }

    /// Show `groups`, with `selected` highlighted. Reloads only on a change.
    pub fn refresh(&self, groups: &[LibraryGroup]) {
        let mut items = Vec::new();
        let mut selected = None;
        for g in groups {
            items.push(Item::Group(g.title.clone()));
            if g.collapsed {
                continue;
            }
            for r in &g.rows {
                if r.selected {
                    selected = Some(items.len());
                }
                items.push(Item::Machine(r.clone()));
            }
        }
        let st = self.source.ivars();
        let changed = {
            let old = st.items.borrow();
            old.len() != items.len()
                || old.iter().zip(&items).any(|(a, b)| match (a, b) {
                    (Item::Group(x), Item::Group(y)) => x != y,
                    (Item::Machine(x), Item::Machine(y)) => x != y,
                    _ => true,
                })
        };
        if changed {
            *st.items.borrow_mut() = items;
            st.syncing.set(true);
            self.table.reloadData();
            st.syncing.set(false);
        }
        match selected {
            Some(row) if self.table.selectedRow() != row as NSInteger => {
                st.syncing.set(true);
                let set = objc2_foundation::NSIndexSet::indexSetWithIndex(row);
                self.table.selectRowIndexes_byExtendingSelection(&set, false);
                st.syncing.set(false);
            }
            // No machine selected (deleted, or its group collapsed): no
            // highlight left on whatever row now sits there.
            None if self.table.selectedRow() >= 0 => {
                st.syncing.set(true);
                // SAFETY: the standard action; the sender may be nil.
                unsafe { self.table.deselectAll(None) };
                st.syncing.set(false);
            }
            _ => {}
        }
    }
}
