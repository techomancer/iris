//! The main window (D3): toolbar, the library on the left, the selected
//! machine on the right (its summary when stopped), the status bar under it.

use objc2::rc::Retained;
use objc2::Message;
use objc2::runtime::{NSObject, NSObjectProtocol};
use objc2::{define_class, MainThreadOnly};
use objc2_app_kit::NSWindowDelegate;
use objc2::MainThreadMarker;
use objc2_app_kit::{
    NSBackingStoreType, NSButton, NSColor, NSAnimatablePropertyContainer, NSFont, NSScrollView, NSStackView, NSTextField,
    NSUserInterfaceLayoutOrientation, NSView, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{NSArray, NSPoint, NSRect, NSSize};
use iris_gui_core::controller::app::AppController;
use iris_gui_core::controller::ids;
use iris_gui_core::controller::overview::Overview;
use iris_gui_core::controller::session::{DeviceLight, LightState};

use crate::library::Library;
use crate::toolbar;
use crate::ui::{label, ns, secondary, set_id};

pub struct MainWindow {
    pub window: Retained<NSWindow>,
    /// The library as a sidebar beside the machine's pane (AppKit's
    /// sidebar: its material, collapse animation and resizing rules).
    _split_vc: Retained<objc2_app_kit::NSSplitViewController>,
    sidebar: Retained<objc2_app_kit::NSSplitViewItem>,
    /// The sidebar's view; the library (its list and the New Machine button
    /// under it) lives in it, or in the floating panel while that is open.
    sidebar_view: Retained<NSView>,
    library: Library,
    library_body: Retained<NSView>,
    content: Retained<NSScrollView>,
    /// Holds the summary or the display, whichever the run state calls for.
    pane: Retained<NSView>,
    pub display: Retained<crate::display::DisplayView>,
    /// Holds the display: black around a smaller picture, scroll bars
    /// around a larger one (Actual Size).
    display_scroll: Retained<NSScrollView>,
    /// The display's leading and top edges: at the pane's safe area (clear
    /// of the toolbar and a floating sidebar), or in full screen at the
    /// pane's own edges, so a sidebar or toolbar shown there floats over the
    /// picture instead of moving it.
    display_safe: Vec<Retained<objc2_app_kit::NSLayoutConstraint>>,
    display_edge: Vec<Retained<objc2_app_kit::NSLayoutConstraint>>,
    detached: Option<Retained<NSWindow>>,
    /// In the main window while the display is in its own window.
    detached_note: Retained<NSStackView>,
    _detached_delegate: Retained<DetachedDelegate>,
    _window_delegate: Retained<WindowDelegate>,
    status: Retained<NSTextField>,
    lights: Retained<NSStackView>,
    light_buttons: Vec<Retained<NSButton>>,
    last_lights: Vec<DeviceLight>,
    toast: Retained<NSTextField>,
    /// While disks synchronize (a CHD fold after a stop): how far along.
    sync_bar: Retained<objc2_app_kit::NSProgressIndicator>,
    sync_label: Retained<NSTextField>,
    /// Cancels a disk conversion; shown while one runs.
    sync_cancel: Retained<NSButton>,
    _toolbar_delegate: Retained<toolbar::Delegate>,
    toolbar: Retained<objc2_app_kit::NSToolbar>,
    toolbar_layout: Vec<&'static str>,
    /// The machine page's tab: 0 Details, 1 Saved States.
    pub summary_tab: usize,
    /// Chrome that comes and goes (`chrome`): the split view's delegate, the
    /// hover zones along the left and bottom edges, the floating library
    /// panel, and the status bar's holder.
    edge_left: Retained<crate::chrome::HoverZone>,
    edge_bottom: Retained<crate::chrome::HoverZone>,
    panel: Retained<crate::chrome::HoverZone>,
    panel_body: Retained<NSView>,
    panel_open: bool,
    /// The panel's leading edge: slid in from beyond the window's edge.
    panel_leading: Retained<objc2_app_kit::NSLayoutConstraint>,
    /// Counts the panel's openings, so a close that finishes after its
    /// slide does not undo a newer opening.
    panel_seq: u64,
    _status_zone: Retained<crate::chrome::HoverZone>,
    split_to_bar: Retained<objc2_app_kit::NSLayoutConstraint>,
    split_to_root: Retained<objc2_app_kit::NSLayoutConstraint>,
    /// The status bar's bottom edge: 0 shows it; in full screen it slides
    /// down to leave a sliver showing.
    status_bottom: Retained<objc2_app_kit::NSLayoutConstraint>,
    /// The pointer is over the status bar (full screen keeps it up then).
    status_hovered: bool,
    /// Full screen's chrome is in force (the library only as the floating
    /// panel, the status bar on hover).
    full_chrome: bool,
    last: Option<Option<Overview>>,
    /// The sidebar's collapsed state as last applied to the chrome.
    library_shown_as: Option<bool>,
}

/// The status bar's height, and how much of it shows when it is tucked away
/// in full screen.
const STATUS_H: f64 = 26.0;
const STATUS_PEEK: f64 = 3.0;
/// How far up from the bottom edge the pointer brings the status bar back in
/// full screen: before it reaches the edge, where the Dock comes up.
const STATUS_REACH: f64 = 18.0;
/// The chrome's animations, as long as the sidebar's.
const SLIDE: f64 = 0.25;
/// The library's width the first time, and the floating panel's.
const SIDEBAR_W: f64 = 220.0;
/// The floating panel's distance from the window's edges.
const PANEL_INSET: f64 = 8.0;

impl MainWindow {
    pub fn new(mtm: MainThreadMarker) -> Self {
        let frame = NSRect::new(NSPoint::new(120.0, 120.0), NSSize::new(1180.0, 760.0));
        // The content runs under the title bar, as Finder's: the sidebar then
        // takes the full height in AppKit's glass, and the machine's pane
        // keeps below the toolbar (its safe area).
        let style = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Closable
            | NSWindowStyleMask::Miniaturizable
            | NSWindowStyleMask::Resizable
            | NSWindowStyleMask::FullSizeContentView;
        // SAFETY: a standard window; it is kept until the app quits.
        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(mtm.alloc(), frame, style, NSBackingStoreType::Buffered, false)
        };
        // SAFETY: we keep the window alive ourselves.
        unsafe { window.setReleasedWhenClosed(false) };
        window.setTitle(&ns(&t!("app-name")));
        window.setTitleVisibility(objc2_app_kit::NSWindowTitleVisibility::Hidden);
        // No title-bar band: the library's card runs up under the window
        // buttons, as Finder's sidebar does.
        window.setTitlebarAppearsTransparent(true);
        // Small enough for Size Window to Display to fit a 640x480 guest.
        window.setMinSize(NSSize::new(600.0, 400.0));
        let (tb, delegate) = toolbar::build(mtm);
        window.setToolbar(Some(&tb));

        let library = Library::new(mtm);
        let content = NSScrollView::initWithFrame(mtm.alloc(), NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(900.0, 700.0)));
        content.setHasVerticalScroller(true);
        content.setDrawsBackground(false);
        let display = crate::display::DisplayView::new(mtm);
        let display_scroll = NSScrollView::initWithFrame(mtm.alloc(), NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(900.0, 700.0)));
        display_scroll.setHasVerticalScroller(true);
        display_scroll.setHasHorizontalScroller(true);
        display_scroll.setAutohidesScrollers(true);
        // A clip view that refits the display whenever its own size changes
        // (a window resized from any edge, the sidebar sliding, scroll bars
        // coming and going), not only when the window says it resized. Set
        // first: the background below is the clip view's.
        display_scroll.setContentView(&crate::display::clip(mtm));
        display_scroll.setDrawsBackground(true);
        display_scroll.setBackgroundColor(&objc2_app_kit::NSColor::blackColor());
        display.setTranslatesAutoresizingMaskIntoConstraints(true);
        display_scroll.setDocumentView(Some(&display));
        let pane = NSView::new(mtm);
        content.setTranslatesAutoresizingMaskIntoConstraints(false);
        pane.addSubview(&content);
        pin_safe(&content, &pane);
        display_scroll.setTranslatesAutoresizingMaskIntoConstraints(false);
        pane.addSubview(&display_scroll);
        let (display_safe, display_edge) = pin_display(&display_scroll, &pane);
        display_scroll.setHidden(true);
        // While detached: a line saying where the display went, and a button
        // that brings it back.
        let note_text = secondary(mtm, &t!("display-detached-note"));
        let back = crate::ui::button(mtm, &t!("tb-attach"), ids::toolbar::DETACH_DISPLAY);
        let detached_note = NSStackView::stackViewWithViews(
            &NSArray::from_retained_slice(&[as_view(&note_text), Retained::into_super(Retained::into_super(back))]),
            mtm,
        );
        detached_note.setOrientation(NSUserInterfaceLayoutOrientation::Vertical);
        detached_note.setSpacing(12.0);
        detached_note.setTranslatesAutoresizingMaskIntoConstraints(false);
        detached_note.setHidden(true);
        pane.addSubview(&detached_note);
        detached_note.centerXAnchor().constraintEqualToAnchor(&pane.centerXAnchor()).setActive(true);
        detached_note.centerYAnchor().constraintEqualToAnchor(&pane.centerYAnchor()).setActive(true);
        // The sidebar and the pane, each in a view controller of its own.
        let sidebar_view = NSView::new(mtm);
        library.scroll.setTranslatesAutoresizingMaskIntoConstraints(false);
        // The list starts under the toolbar, not behind it.
        library.scroll.setAutomaticallyAdjustsContentInsets(false);
        library.scroll.setContentInsets(objc2_foundation::NSEdgeInsets { top: 0.0, left: 0.0, bottom: 0.0, right: 0.0 });
        // The sidebar as Finder's: one lighter tone from the window buttons
        // down, and a hairline down its edge against the machine's pane
        // (AppKit's own glass is faint over a plain window).
        let fill = objc2_app_kit::NSBox::new(mtm);
        fill.setBoxType(objc2_app_kit::NSBoxType::Custom);
        fill.setTitlePosition(objc2_app_kit::NSTitlePosition::NoTitle);
        fill.setBorderWidth(0.0);
        fill.setCornerRadius(0.0);
        fill.setFillColor(&NSColor::quaternaryLabelColor());
        fill.setTranslatesAutoresizingMaskIntoConstraints(false);
        sidebar_view.addSubview(&fill);
        pin(&fill, &sidebar_view);
        let edge = objc2_app_kit::NSBox::new(mtm);
        edge.setBoxType(objc2_app_kit::NSBoxType::Custom);
        edge.setTitlePosition(objc2_app_kit::NSTitlePosition::NoTitle);
        edge.setBorderWidth(0.0);
        // As plain as the window's own rim on the other side (the separator
        // colour, at a tenth, all but vanished on the sidebar's tone).
        edge.setFillColor(&NSColor::tertiaryLabelColor());
        edge.setTranslatesAutoresizingMaskIntoConstraints(false);
        sidebar_view.addSubview(&edge);
        edge.trailingAnchor().constraintEqualToAnchor(&sidebar_view.trailingAnchor()).setActive(true);
        edge.topAnchor().constraintEqualToAnchor(&sidebar_view.topAnchor()).setActive(true);
        edge.bottomAnchor().constraintEqualToAnchor(&sidebar_view.bottomAnchor()).setActive(true);
        edge.widthAnchor().constraintEqualToConstant(1.0).setActive(true);
        // The library: its list, and New Machine under it.
        let library_body = NSView::new(mtm);
        library_body.setTranslatesAutoresizingMaskIntoConstraints(false);
        library_body.addSubview(&library.scroll);
        let new_machine = crate::ui::button(mtm, &t!("tb-new-machine"), iris_gui_core::controller::commands::menu::NEW_MACHINE);
        new_machine.setBordered(false);
        if let Some(img) = crate::glyphs::symbol("plus", &t!("tb-new-machine")) {
            new_machine.setImage(Some(&img));
            new_machine.setImagePosition(objc2_app_kit::NSCellImagePosition::ImageLeading);
            new_machine.setSymbolConfiguration(Some(&objc2_app_kit::NSImageSymbolConfiguration::configurationWithPointSize_weight(12.0, 0.0)));
        }
        new_machine.setContentTintColor(Some(&NSColor::secondaryLabelColor()));
        new_machine.setFont(Some(&NSFont::systemFontOfSize(12.0)));
        new_machine.setToolTip(Some(&ns(&t!("tb-new-machine"))));
        new_machine.setTranslatesAutoresizingMaskIntoConstraints(false);
        library_body.addSubview(&new_machine);
        library.scroll.leadingAnchor().constraintEqualToAnchor(&library_body.leadingAnchor()).setActive(true);
        library.scroll.trailingAnchor().constraintEqualToAnchor(&library_body.trailingAnchor()).setActive(true);
        library.scroll.topAnchor().constraintEqualToAnchor(&library_body.topAnchor()).setActive(true);
        library.scroll.bottomAnchor().constraintEqualToAnchor_constant(&new_machine.topAnchor(), -4.0).setActive(true);
        new_machine.leadingAnchor().constraintEqualToAnchor_constant(&library_body.leadingAnchor(), 14.0).setActive(true);
        new_machine.trailingAnchor().constraintLessThanOrEqualToAnchor_constant(&library_body.trailingAnchor(), -8.0).setActive(true);
        new_machine.bottomAnchor().constraintEqualToAnchor_constant(&library_body.bottomAnchor(), -10.0).setActive(true);
        new_machine.heightAnchor().constraintEqualToConstant(22.0).setActive(true);
        sidebar_view.addSubview(&library_body);
        place_on_card(&library_body, &sidebar_view);
        let side_vc = objc2_app_kit::NSViewController::new(mtm);
        side_vc.setView(&sidebar_view);
        // The machine's column: the pane (the page or the display), the
        // status bar under it, and the hover zones and floating library.
        let column = NSView::new(mtm);
        pane.setTranslatesAutoresizingMaskIntoConstraints(false);
        column.addSubview(&pane);
        let main_vc = objc2_app_kit::NSViewController::new(mtm);
        main_vc.setView(&column);
        let sidebar = objc2_app_kit::NSSplitViewItem::sidebarWithViewController(&side_vc);
        sidebar.setMinimumThickness(150.0);
        sidebar.setMaximumThickness(360.0);
        sidebar.setCanCollapse(true);
        // The card runs up under the window buttons: no line across it.
        sidebar.setTitlebarSeparatorStyle(objc2_app_kit::NSTitlebarSeparatorStyle::None);
        let main_item = objc2_app_kit::NSSplitViewItem::splitViewItemWithViewController(&main_vc);
        main_item.setMinimumThickness(480.0);
        // macOS 26 and later: the sidebar floats over the machine's column in
        // its own glass, inset as Finder's, and the column's safe area makes
        // room for it (what the column holds keeps to its safe area).
        let floats: bool = unsafe { objc2::msg_send![&*main_item, respondsToSelector: objc2::sel!(setAutomaticallyAdjustsSafeAreaInsets:)] };
        if floats {
            main_item.setAutomaticallyAdjustsSafeAreaInsets(true);
        }
        let split_vc = objc2_app_kit::NSSplitViewController::new(mtm);
        split_vc.addSplitViewItem(&sidebar);
        split_vc.addSplitViewItem(&main_item);
        // A new name with the narrower default, so a width saved from the
        // old default (saved whether or not anyone dragged it) goes.
        split_vc.splitView().setAutosaveName(Some(&ns(SPLIT_AUTOSAVE)));
        // A hairline between the sidebar and the machine's pane.
        split_vc.splitView().setDividerStyle(objc2_app_kit::NSSplitViewDividerStyle::Thin);

        let status = label(mtm, "");
        status.setFont(Some(&NSFont::systemFontOfSize(12.0)));
        set_id(&status, ids::status::RUN_STATE);
        let toast = secondary(mtm, "");
        toast.setFont(Some(&NSFont::systemFontOfSize(12.0)));
        set_id(&toast, ids::status::TOAST);
        let sync_bar = objc2_app_kit::NSProgressIndicator::new(mtm);
        sync_bar.setStyle(objc2_app_kit::NSProgressIndicatorStyle::Bar);
        sync_bar.setIndeterminate(false);
        sync_bar.setMinValue(0.0);
        sync_bar.setMaxValue(100.0);
        sync_bar.setControlSize(objc2_app_kit::NSControlSize::Small);
        sync_bar.widthAnchor().constraintEqualToConstant(160.0).setActive(true);
        sync_bar.setHidden(true);
        let sync_label = secondary(mtm, "");
        sync_label.setFont(Some(&NSFont::monospacedDigitSystemFontOfSize_weight(12.0, 0.0)));
        sync_label.setHidden(true);
        let sync_cancel = crate::ui::button(mtm, &t!("convert-cancel"), iris_gui_core::controller::convert::CANCEL_ID);
        sync_cancel.setControlSize(objc2_app_kit::NSControlSize::Small);
        sync_cancel.setHidden(true);
        let lights = NSStackView::new(mtm);
        lights.setOrientation(NSUserInterfaceLayoutOrientation::Horizontal);
        lights.setSpacing(6.0);
        let spacer = NSView::new(mtm);
        spacer.setContentHuggingPriority_forOrientation(1.0, objc2_app_kit::NSLayoutConstraintOrientation::Horizontal);
        let bar = NSStackView::stackViewWithViews(
            &NSArray::from_retained_slice(&[
                as_view(&status),
                as_view(&toast),
                Retained::into_super(sync_bar.clone()),
                as_view(&sync_label),
                Retained::into_super(Retained::into_super(sync_cancel.clone())),
                spacer,
                Retained::into_super(lights.clone()),
            ]),
            mtm,
        );
        bar.setOrientation(NSUserInterfaceLayoutOrientation::Horizontal);
        bar.setSpacing(12.0);
        bar.setEdgeInsets(objc2_foundation::NSEdgeInsets { top: 4.0, left: 12.0, bottom: 4.0, right: 12.0 });
        bar.heightAnchor().constraintEqualToConstant(STATUS_H).setActive(true);

        // The status bar sits on its own backing (it floats over the display
        // in full screen), in a hover zone that hides it again there.
        let bar_bg = objc2_app_kit::NSVisualEffectView::new(mtm);
        // The title bar's material and a hairline above: window chrome, set
        // apart from the page over it, as Xcode's and Finder's bars are.
        bar_bg.setMaterial(objc2_app_kit::NSVisualEffectMaterial::Titlebar);
        bar_bg.setBlendingMode(objc2_app_kit::NSVisualEffectBlendingMode::WithinWindow);
        bar_bg.setState(objc2_app_kit::NSVisualEffectState::FollowsWindowActiveState);
        let status_zone = crate::chrome::HoverZone::new(mtm, crate::chrome::Zone::StatusBar);
        {
            let (bar_v, bg_v, zone_v): (&NSView, &NSView, &NSView) = (&bar, &bar_bg, &status_zone);
            for (child, parent) in [(bar_v, bg_v), (bg_v, zone_v)] {
                child.setTranslatesAutoresizingMaskIntoConstraints(false);
                parent.addSubview(child);
                pin(child, parent);
            }
        }
        let hairline = objc2_app_kit::NSBox::new(mtm);
        hairline.setBoxType(objc2_app_kit::NSBoxType::Separator);
        hairline.setTranslatesAutoresizingMaskIntoConstraints(false);
        status_zone.addSubview(&hairline);
        hairline.leadingAnchor().constraintEqualToAnchor(&status_zone.leadingAnchor()).setActive(true);
        hairline.trailingAnchor().constraintEqualToAnchor(&status_zone.trailingAnchor()).setActive(true);
        hairline.topAnchor().constraintEqualToAnchor(&status_zone.topAnchor()).setActive(true);

        // The pane fills the column above the status bar (over it in full
        // screen, where the bar floats).
        let root = column.clone();
        status_zone.setTranslatesAutoresizingMaskIntoConstraints(false);
        root.addSubview(&status_zone);
        pane.leadingAnchor().constraintEqualToAnchor(&root.leadingAnchor()).setActive(true);
        pane.trailingAnchor().constraintEqualToAnchor(&root.trailingAnchor()).setActive(true);
        // The status bar keeps clear of a floating sidebar.
        status_zone.leadingAnchor().constraintEqualToAnchor(&root.safeAreaLayoutGuide().leadingAnchor()).setActive(true);
        status_zone.trailingAnchor().constraintEqualToAnchor(&root.trailingAnchor()).setActive(true);
        pane.topAnchor().constraintEqualToAnchor(&root.topAnchor()).setActive(true);
        let split_to_bar = pane.bottomAnchor().constraintEqualToAnchor(&status_zone.topAnchor());
        split_to_bar.setActive(true);
        let split_to_root = pane.bottomAnchor().constraintEqualToAnchor(&root.bottomAnchor());
        let status_bottom = status_zone.bottomAnchor().constraintEqualToAnchor(&root.bottomAnchor());
        status_bottom.setActive(true);

        // Hover zones: the left edge brings back a hidden library, the bottom
        // edge (in full screen) the status bar.
        let edge_left = crate::chrome::HoverZone::new(mtm, crate::chrome::Zone::LibraryEdge);
        let edge_bottom = crate::chrome::HoverZone::new(mtm, crate::chrome::Zone::StatusEdge);
        root.addSubview(&edge_left);
        root.addSubview(&edge_bottom);
        edge_left.leadingAnchor().constraintEqualToAnchor(&root.leadingAnchor()).setActive(true);
        edge_left.topAnchor().constraintEqualToAnchor(&root.topAnchor()).setActive(true);
        edge_left.bottomAnchor().constraintEqualToAnchor(&root.bottomAnchor()).setActive(true);
        edge_left.widthAnchor().constraintEqualToConstant(6.0).setActive(true);
        edge_bottom.leadingAnchor().constraintEqualToAnchor(&root.leadingAnchor()).setActive(true);
        edge_bottom.trailingAnchor().constraintEqualToAnchor(&root.trailingAnchor()).setActive(true);
        edge_bottom.bottomAnchor().constraintEqualToAnchor(&root.bottomAnchor()).setActive(true);
        edge_bottom.heightAnchor().constraintEqualToConstant(STATUS_REACH).setActive(true);
        edge_left.setHidden(true);
        edge_bottom.setHidden(true);

        // The floating library: a rounded sidebar panel over the content.
        let panel = crate::chrome::HoverZone::new(mtm, crate::chrome::Zone::LibraryPanel);
        let panel_fx = objc2_app_kit::NSVisualEffectView::new(mtm);
        panel_fx.setMaterial(objc2_app_kit::NSVisualEffectMaterial::Sidebar);
        panel_fx.setBlendingMode(objc2_app_kit::NSVisualEffectBlendingMode::WithinWindow);
        panel_fx.setState(objc2_app_kit::NSVisualEffectState::Active);
        panel_fx.setWantsLayer(true);
        if let Some(layer) = panel_fx.layer() {
            layer.setCornerRadius(12.0);
            layer.setMasksToBounds(true);
        }
        panel_fx.setTranslatesAutoresizingMaskIntoConstraints(false);
        panel.addSubview(&panel_fx);
        pin(&panel_fx, &panel);
        root.addSubview(&panel);
        // Out beyond the window's edge until it slides in.
        let panel_leading = panel.leadingAnchor().constraintEqualToAnchor_constant(&root.leadingAnchor(), -(SIDEBAR_W + PANEL_INSET));
        panel_leading.setActive(true);
        panel.topAnchor().constraintEqualToAnchor_constant(&root.safeAreaLayoutGuide().topAnchor(), PANEL_INSET).setActive(true);
        panel.bottomAnchor().constraintEqualToAnchor_constant(&root.bottomAnchor(), -PANEL_INSET).setActive(true);
        panel.widthAnchor().constraintEqualToConstant(SIDEBAR_W).setActive(true);
        panel.setHidden(true);
        // The split view controller is the window's content, so the sidebar
        // gets AppKit's full-height glass, as Finder's does.
        let frame = window.frame();
        window.setContentViewController(Some(&split_vc));
        window.setFrame_display(frame, false);
        // Full screen, as the green button offers.
        window.setCollectionBehavior(window.collectionBehavior() | objc2_app_kit::NSWindowCollectionBehavior::FullScreenPrimary);
        // SIDEBAR_W wide the first time; after that, as the user left it.
        let split_view = split_vc.splitView();
        let defaults = objc2_foundation::NSUserDefaults::standardUserDefaults();
        if defaults.objectForKey(&ns(&format!("NSSplitView Subview Frames {SPLIT_AUTOSAVE}"))).is_none() {
            split_view.setPosition_ofDividerAtIndex(SIDEBAR_W, 0);
        }
        defaults.removeObjectForKey(&ns("NSSplitView Subview Frames IRISLibrarySplit"));
        // The library always starts shown: the autosave keeps the width,
        // but a collapsed state restored from it disagrees with the item.
        sidebar.setCollapsed(false);
        window.center();
        window.setFrameAutosaveName(&ns("IRISMainWindow"));
        window.setAcceptsMouseMovedEvents(true);
        let window_delegate = WindowDelegate::new(mtm);
        window.setDelegate(Some(objc2::runtime::ProtocolObject::from_ref(&*window_delegate)));
        Self {
            window,
            _split_vc: split_vc,
            sidebar,
            sidebar_view,
            library,
            library_body,
            content,
            pane,
            display,
            display_scroll,
            display_safe,
            display_edge,
            detached: None,
            detached_note,
            _detached_delegate: DetachedDelegate::new(mtm),
            _window_delegate: window_delegate,
            status,
            lights,
            light_buttons: Vec::new(),
            last_lights: Vec::new(),
            toast,
            sync_bar,
            sync_label,
            sync_cancel,
            _toolbar_delegate: delegate,
            toolbar: tb.clone(),
            toolbar_layout: Vec::new(),
            summary_tab: 0,
            edge_left,
            edge_bottom,
            panel,
            panel_body: Retained::into_super(panel_fx),
            panel_open: false,
            panel_leading,
            panel_seq: 0,
            _status_zone: status_zone,
            split_to_bar,
            split_to_root,
            status_bottom,
            status_hovered: false,
            full_chrome: false,
            last: None,
            library_shown_as: None,
        }
    }

    /// The display follows the controller: shown while a machine runs (in
    /// its own window when detached), capture, full screen and scaling.
    pub fn sync_display(&mut self, mtm: MainThreadMarker, app: &AppController) {
        let running = app.session().is_running();
        let d = app.display();
        let display_view: &NSView = &self.display_scroll;
        // Detached: the display moves to a window of its own.
        if running && d.detached() && self.detached.is_none() {
            let name = app.selected().unwrap_or_default().to_string();
            let w = new_display_window(mtm, &name);
            w.setDelegate(Some(objc2::runtime::ProtocolObject::from_ref(&*self._detached_delegate)));
            display_view.removeFromSuperview();
            // A window's content view is sized by the window, not by
            // constraints: without this it stays zero-sized and shows nothing.
            display_view.setTranslatesAutoresizingMaskIntoConstraints(true);
            display_view.setAutoresizingMask(
                objc2_app_kit::NSAutoresizingMaskOptions::ViewWidthSizable | objc2_app_kit::NSAutoresizingMaskOptions::ViewHeightSizable,
            );
            w.setContentView(Some(display_view));
            w.makeKeyAndOrderFront(None);
            w.makeFirstResponder(Some(&*self.display));
            self.detached = Some(w);
        } else if (!running || !d.detached()) && self.detached.is_some() {
            // A full-screen display window leaves full screen first (closing
            // it there would leave an empty Space): its exit notification
            // brings us back here to finish.
            if let Some(w) = &self.detached {
                if w.styleMask().contains(NSWindowStyleMask::FullScreen) {
                    if !FULL_SCREEN_MOVING.with(|m| m.get()) {
                        w.toggleFullScreen(None);
                    }
                    return;
                }
            }
            if let Some(w) = self.detached.take() {
                w.setDelegate(None);
                display_view.removeFromSuperview();
                w.orderOut(None);
            }
            display_view.setTranslatesAutoresizingMaskIntoConstraints(false);
            self.pane.addSubview(display_view);
            let (safe, edge) = pin_display(display_view, &self.pane);
            self.display_safe = safe;
            self.display_edge = edge;
            self.full_chrome_edges();
            if running {
                self.window.makeKeyAndOrderFront(None);
                self.window.makeFirstResponder(Some(&*self.display));
            }
        }
        let show_display = running && self.detached.is_none();
        let away = running && self.detached.is_some();
        // Shown whenever the machine runs, in whichever window holds it.
        if self.display_scroll.isHidden() == running {
            self.display_scroll.setHidden(!running);
            if show_display {
                self.window.makeFirstResponder(Some(&self.display));
            }
        }
        self.content.setHidden(show_display || away);
        self.detached_note.setHidden(!away);
        self.display.set_captured(running && d.captured());
        self.display.set_follow(running && iris_gui_core::controller::display::follows_pointer(app.prefs()));
        let sizing = iris_gui_core::controller::display::sizing(app.prefs());
        let filter = match app.prefs().display_filter {
            iris_gui_core::controller::display::DisplayFilter::Auto => None,
            iris_gui_core::controller::display::DisplayFilter::Nearest => Some(true),
            iris_gui_core::controller::display::DisplayFilter::Linear => Some(false),
        };
        self.display.set_sizing(sizing, filter);
        self.display.fit();
        {
            use iris_gui_core::controller::commands::menu as m;
            crate::menu::check(app.display_size_item(), &[m::VIEW_ACTUAL, m::VIEW_FIT, m::VIEW_STRETCH]);
        }
        self.display.set_overlay(mtm, app.display_overlay().as_deref());
        self.set_full_chrome(d.fullscreen() && self.detached.is_none());
        // Full screen: the window that shows the display. Never while a
        // window is still moving in or out of full screen: its style mask
        // changes before the animation ends, and a toggle then turns the
        // green button's exit straight back into full screen.
        let w = self.detached.clone().unwrap_or_else(|| self.window.clone());
        let is_full = w.styleMask().contains(NSWindowStyleMask::FullScreen);
        if is_full != d.fullscreen() && !FULL_SCREEN_MOVING.with(|m| m.get()) {
            w.toggleFullScreen(None);
        }
    }

    /// Move the sidebar's divider to `w` points (for tests: as a drag
    /// would) and report the width it settles at.
    pub fn drag_sidebar(&self, w: f64) -> f64 {
        let split = self._split_vc.splitView();
        split.setPosition_ofDividerAtIndex(w, 0);
        split.layoutSubtreeIfNeeded();
        self.sidebar_view.frame().size.width
    }

    /// Resize the main window to `w` x `h` points, keeping its top-left
    /// corner (for tests: as a drag on an edge would).
    pub fn resize_for_test(&self, w: f64, h: f64) {
        let mut f = self.window.frame();
        let top = f.origin.y + f.size.height;
        f.size = NSSize::new(w, h);
        f.origin.y = top - h;
        self.window.setFrame_display(f, true);
    }

    /// In full screen the library is the floating panel; in a window, the
    /// sidebar.
    fn library_hidden(&self) -> bool {
        if self.full_chrome {
            !self.panel_open
        } else {
            self.sidebar.isCollapsed()
        }
    }

    /// Show or hide the library, sliding as a sidebar does. The window
    /// keeps its size; the machine's pane takes the room.
    pub fn toggle_library(&mut self) {
        if self.full_chrome {
            if self.panel_open {
                self.close_panel();
            } else {
                self.open_panel();
            }
            self.sync_library_chrome();
            return;
        }
        self.close_panel();
        // The split view controller's own toggle: it animates both ways
        // (setting the item's collapsed state through its animator slid it
        // closed but snapped it open).
        // SAFETY: the standard action; the sender may be nil.
        unsafe { self._split_vc.toggleSidebar(None) };
        self.sync_library_chrome();
    }

    /// One tone down the whole sidebar, as Finder's: the library's scroll
    /// view puts a sidebar material of its own behind a source-list table,
    /// darker than the sidebar's fill, so the strip under the window buttons
    /// showed as a band. Hide that material (it is the scroll view's own
    /// subview; AppKit adds it again when the table reloads).
    fn hide_sidebar_band(&self) {
        for v in self.library.scroll.subviews().iter() {
            if v.downcast_ref::<objc2_app_kit::NSVisualEffectView>().is_some() && !v.isHidden() {
                v.setHidden(true);
            }
        }
    }

    /// The edge zone and the button's label follow the sidebar, however it
    /// was collapsed (the button, the menu, or dragging the divider).
    fn sync_library_chrome(&mut self) {
        self.hide_sidebar_band();
        let hidden = self.library_hidden();
        if self.library_shown_as == Some(hidden) {
            return;
        }
        self.library_shown_as = Some(hidden);
        self.edge_left.setHidden(!hidden);
        self.label_library_toggle();
    }

    /// The pointer entered or left a hover zone.
    pub fn hover(&mut self, zone: crate::chrome::Zone, inside: bool) {
        use crate::chrome::Zone;
        match (zone, inside) {
            (Zone::LibraryEdge, true) if self.library_hidden() => self.open_panel(),
            // The panel slides in under a pointer resting at the edge, which
            // reports it entering and leaving: leaving counts only once the
            // pointer is clear of the panel's strip.
            (Zone::LibraryPanel, false) if self.panel_open => {
                if self.pointer_by_panel() {
                    // Looked at again shortly: the pointer may leave the
                    // strip without crossing the panel.
                    if let Ok(when) = dispatch2::DispatchTime::try_from(std::time::Duration::from_millis(400)) {
                        let _ = dispatch2::DispatchQueue::main().after(when, || crate::app::hover(Zone::LibraryPanel, false));
                    }
                } else {
                    self.close_panel();
                }
            }
            (Zone::StatusEdge, true) if self.full_chrome => {
                self.status_hovered = true;
                self.slide_status(true);
            }
            (Zone::StatusBar, true) => self.status_hovered = true,
            (Zone::StatusBar, false) => {
                self.status_hovered = false;
                if self.full_chrome {
                    self.slide_status(false);
                }
            }
            _ => {}
        }
    }

    /// The pointer is over the strip the open panel takes: from the
    /// window's edge to the panel's far side, as tall as the panel.
    fn pointer_by_panel(&self) -> bool {
        // SAFETY: a plain getter, on the main thread.
        let Some(root) = (unsafe { self.panel.superview() }) else { return false };
        let p = root.convertPoint_fromView(self.window.mouseLocationOutsideOfEventStream(), None);
        let (b, f) = (root.bounds(), self.panel.frame());
        p.x >= b.origin.x - 1.0
            && p.x < b.origin.x + PANEL_INSET + SIDEBAR_W
            && p.y >= f.origin.y
            && p.y <= f.origin.y + f.size.height
    }

    /// Slide the status bar up into view or down to a sliver (full screen).
    fn slide_status(&self, shown: bool) {
        let to = if shown { 0.0 } else { STATUS_H - STATUS_PEEK };
        if (self.status_bottom.constant() - to).abs() < 0.5 {
            return;
        }
        objc2_app_kit::NSAnimationContext::beginGrouping();
        objc2_app_kit::NSAnimationContext::currentContext().setDuration(SLIDE);
        self.status_bottom.animator().setConstant(to);
        objc2_app_kit::NSAnimationContext::endGrouping();
    }

    /// Full screen was entered a moment ago: tuck the status bar away
    /// unless the pointer is on it.
    pub fn tuck_status(&self) {
        if self.full_chrome && !self.status_hovered {
            self.slide_status(false);
        }
    }

    /// Show the hidden library as a floating panel, as Arc does: it slides
    /// in from the window's edge.
    fn open_panel(&mut self) {
        if self.panel_open {
            return;
        }
        self.panel_open = true;
        self.panel_seq += 1;
        let body = &self.library_body;
        // Still in the panel when a close had not finished sliding out.
        // SAFETY: a plain getter, on the main thread.
        if !unsafe { body.superview() }.is_some_and(|v| *v == *self.panel_body) {
            body.removeFromSuperview();
            self.panel_body.addSubview(body);
            pin(body, &self.panel_body);
        }
        if self.panel.isHidden() {
            self.panel_leading.setConstant(-(SIDEBAR_W + PANEL_INSET));
            self.panel.setAlphaValue(0.0);
            self.panel.setHidden(false);
            // SAFETY: a plain getter, on the main thread.
            if let Some(root) = unsafe { self.panel.superview() } {
                root.layoutSubtreeIfNeeded();
            }
        }
        self.slide_panel(true);
    }

    /// Slide the floating library out; it goes back in the sidebar once out
    /// of sight (`library_panel_closed`).
    fn close_panel(&mut self) {
        if !self.panel_open {
            return;
        }
        self.panel_open = false;
        self.slide_panel(false);
        let seq = self.panel_seq;
        if let Ok(when) = dispatch2::DispatchTime::try_from(std::time::Duration::from_secs_f64(SLIDE)) {
            let _ = dispatch2::DispatchQueue::main().after(when, move || crate::app::library_panel_closed(seq));
        }
    }

    fn slide_panel(&self, shown: bool) {
        objc2_app_kit::NSAnimationContext::beginGrouping();
        objc2_app_kit::NSAnimationContext::currentContext().setDuration(SLIDE);
        self.panel_leading.animator().setConstant(if shown { PANEL_INSET } else { -(SIDEBAR_W + PANEL_INSET) });
        self.panel.animator().setAlphaValue(if shown { 1.0 } else { 0.0 });
        objc2_app_kit::NSAnimationContext::endGrouping();
    }

    /// The floating library finished sliding out: put the list back in the
    /// sidebar, unless it was opened again meanwhile.
    pub fn library_panel_closed(&mut self, seq: u64) {
        if self.panel_open || seq != self.panel_seq {
            return;
        }
        self.panel.setHidden(true);
        self.put_library_back();
    }

    /// The library's list back in the sidebar, at once.
    fn put_library_back(&self) {
        let body = &self.library_body;
        // SAFETY: a plain getter, on the main thread.
        if unsafe { body.superview() }.is_some_and(|v| *v == *self.sidebar_view) {
            return;
        }
        body.removeFromSuperview();
        self.sidebar_view.addSubview(body);
        place_on_card(body, &self.sidebar_view);
    }

    /// Full screen: the library only as the floating panel (from the left
    /// edge or the library button), and the status bar over the display,
    /// tucked away to a sliver after a moment and brought back from near the
    /// bottom edge. The display takes the whole screen, so neither moves it.
    ///
    /// The sidebar leaves the split meanwhile: AppKit slides a collapsed
    /// sidebar in from the screen's edge in full screen, narrowing the
    /// display's column. Leaving puts it back as it was, shown or hidden.
    fn set_full_chrome(&mut self, on: bool) {
        if on == self.full_chrome {
            return;
        }
        self.panel_open = false;
        self.panel_seq += 1;
        self.panel.setHidden(true);
        self.put_library_back();
        self.full_chrome = on;
        self.full_chrome_edges();
        // The edge zone and the button follow (`sync_library_chrome`).
        self.library_shown_as = None;
        if on {
            self._split_vc.removeSplitViewItem(&self.sidebar);
            self.split_to_bar.setActive(false);
            self.split_to_root.setActive(true);
            self.status_bottom.setConstant(0.0);
            self.edge_bottom.setHidden(false);
            // Shown first, then slid away, so it is clear where it went.
            if let Ok(when) = dispatch2::DispatchTime::try_from(std::time::Duration::from_millis(1500)) {
                let _ = dispatch2::DispatchQueue::main().after(when, crate::app::tuck_status);
            }
        } else {
            self._split_vc.insertSplitViewItem_atIndex(&self.sidebar, 0);
            self.split_to_root.setActive(false);
            self.split_to_bar.setActive(true);
            self.status_bottom.setConstant(0.0);
            self.edge_bottom.setHidden(true);
        }
        self.sync_library_chrome();
    }

    /// The display's leading and top edges for the chrome in force: the
    /// pane's own edges in full screen, its safe area otherwise.
    fn full_chrome_edges(&self) {
        let (off, on) = if self.full_chrome { (&self.display_safe, &self.display_edge) } else { (&self.display_edge, &self.display_safe) };
        for c in off {
            c.setActive(false);
        }
        for c in on {
            c.setActive(true);
        }
    }

    /// Size the window that shows the display so the picture fills it as it
    /// is drawn now: no black around it, nothing to scroll (at Actual Size a
    /// guest larger than the window brings the window up to its size, as far
    /// as the screen allows). The window keeps its top-left corner.
    pub fn fit_window(&self) {
        let w = self.detached.clone().unwrap_or_else(|| self.window.clone());
        if w.styleMask().contains(NSWindowStyleMask::FullScreen) || self.display_scroll.isHidden() {
            return;
        }
        let Some((pw, ph)) = self.display.picture_extent() else { return };
        // What the display has now, scroll bars included: they go once the
        // picture fits.
        let have = self.display_scroll.frame().size;
        let mut frame = w.frame();
        let (dw, dh) = (pw - have.width, ph - have.height);
        let top = frame.origin.y + frame.size.height;
        frame.size.width = (frame.size.width + dw).max(w.minSize().width);
        frame.size.height = (frame.size.height + dh).max(w.minSize().height);
        frame.origin.y = top - frame.size.height;
        if let Some(screen) = w.screen() {
            let vis = screen.visibleFrame();
            frame.size.width = frame.size.width.min(vis.size.width);
            frame.size.height = frame.size.height.min(vis.size.height);
            frame.origin.x = frame.origin.x.min(vis.origin.x + vis.size.width - frame.size.width).max(vis.origin.x);
            frame.origin.y = frame.origin.y.max(vis.origin.y).min(vis.origin.y + vis.size.height - frame.size.height);
        }
        w.setFrame_display_animate(frame, true, true);
    }

    /// The library button and menu item say what they will do.
    fn label_library_toggle(&self) {
        let text = if self.library_hidden() { t!("tb-show-library") } else { t!("tb-hide-library") };
        for item in self.toolbar.items().iter() {
            if item.itemIdentifier().to_string() == iris_gui_core::controller::ids::toolbar::SHOW_LIBRARY {
                item.setLabel(&ns(&text));
                item.setToolTip(Some(&ns(&text)));
            }
        }
        crate::menu::retitle(iris_gui_core::controller::ids::toolbar::SHOW_LIBRARY, &text);
    }

    /// Show a message in the status bar; it clears itself after a few
    /// seconds unless another one replaced it.
    pub fn toast(&self, text: &str) {
        self.toast.setStringValue(&ns(text));
        let n = TOAST_SEQ.with(|c| {
            c.set(c.get().wrapping_add(1));
            c.get()
        });
        if let Ok(when) = dispatch2::DispatchTime::try_from(std::time::Duration::from_secs(3)) {
            let _ = dispatch2::DispatchQueue::main().after(when, move || crate::app::toast_expired(n));
        }
    }

    pub fn clear_toast(&self, n: u64) {
        if TOAST_SEQ.with(|c| c.get()) == n {
            self.toast.setStringValue(&ns(""));
        }
    }

    /// Show the machine page's other tab on the next refresh.
    pub fn set_summary_tab(&mut self, tab: usize) {
        if self.summary_tab != tab {
            self.summary_tab = tab;
            self.last = None;
        }
    }

    /// Bring every view in line with the controller. Views are set only when
    /// their value changed.
    pub fn refresh(&mut self, mtm: MainThreadMarker, app: &AppController) {
        self.sync_display(mtm, app);
        self.sync_library_chrome();
        self.library.refresh(&app.library());
        // The window's title (for the Window menu and Mission Control) is
        // not drawn: the toolbar's title item shows the name and model right
        // of the library button.
        let title = app.selected().map(str::to_string).unwrap_or_else(|| t!("app-name"));
        if self.window.title().to_string() != title {
            self.window.setTitle(&ns(&title));
        }
        let model = app.overview().map(|o| o.spec).unwrap_or_default();
        // The toolbar follows the machine's state. A new layout builds a new
        // (empty) machine capsule, so fill the capsule after it.
        let layout = app.toolbar_layout();
        if layout != self.toolbar_layout {
            toolbar::set_layout(&self.toolbar, &layout);
            self.toolbar_layout = layout;
            self.label_library_toggle();
        }
        toolbar::set_title(&title, &model, app.overview().map(|o| o.icon));
        let o = app.overview();
        if self.last.as_ref() != Some(&o) {
            let (page, preview) = crate::summary::build(mtm, o.as_ref(), self.summary_tab);
            // A flipped document view holds the page at its top; the page
            // takes the pane's width and its own height.
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
            // The page is as wide as the window's content; it lays itself out.
            page.trailingAnchor().constraintEqualToAnchor(&doc.trailingAnchor()).setActive(true);
            doc.heightAnchor().constraintEqualToAnchor(&page.heightAnchor()).setActive(true);
            // The whole page fits the window: the preview shrinks (down to
            // its minimum) until nothing needs scrolling.
            if preview.is_some() {
                let fit = page.heightAnchor().constraintLessThanOrEqualToAnchor(&clip.heightAnchor());
                // Above the preview's growth, below the window's size.
                fit.setPriority(490.0);
                fit.setActive(true);
            }
            self.last = Some(o);
        }
        // The status line: the state when it says something (not a plain
        // "running"), what waits for a restart, the speed, and while captured
        // how to let go. No executor or clock rate: they are for developers.
        let bar = app.status_bar();
        let mut parts: Vec<String> = Vec::new();
        let plain_running = bar.run_state == iris_gui_core::controller::session::RunState::Running && bar.run_label == t!("status-running");
        if !plain_running {
            parts.push(bar.run_label.clone());
        }
        if let Some(p) = &bar.cpu_pending {
            parts.push(p.clone());
        }
        if bar.unsaved {
            parts.push("\u{2022}".into());
        }
        if let Some(m) = &bar.mips {
            parts.push(m.clone());
        }
        if app.display().captured() {
            let keys = iris_gui_core::controller::keys::CaptureChords::release_text(true);
            parts.push(t!("main-capture-release", keys = keys));
        }
        let s = parts.join("   ");
        if self.status.stringValue().to_string() != s {
            self.status.setStringValue(&ns(&s));
        }
        let mut converting = false;
        if let Some(job) = app.conversion() {
            // A disk conversion: its label, how far, and Cancel.
            let pct = (job.progress.fraction().unwrap_or(0.0) * 100.0).round();
            self.sync_bar.setHidden(false);
            self.sync_bar.setDoubleValue(pct as f64);
            self.sync_bar.setToolTip(None);
            self.sync_label.setStringValue(&ns(&format!("{}  {pct:.0}%", job.label)));
            self.sync_label.setHidden(false);
            self.sync_cancel.setHidden(!job.can_cancel || job.cancel_requested);
            converting = true;
        } else {
            self.sync_cancel.setHidden(true);
        }
        match app.session().syncing() {
            _ if converting => {}
            Some(p) => {
                let pct = (p.overall() * 100.0).round();
                self.sync_bar.setHidden(false);
                self.sync_bar.setDoubleValue(pct as f64);
                let mut text = format!("{pct:.0}%");
                if p.total > 1 {
                    text = format!("{text}  {}", t!("modal-sync-disk", disk = (p.disk + 1).min(p.total), total = p.total));
                }
                self.sync_label.setStringValue(&ns(&text));
                self.sync_label.setHidden(false);
                self.sync_bar.setToolTip(Some(&ns(&t!("modal-sync-dont-quit"))));
            }
            None => {
                self.sync_bar.setHidden(true);
                self.sync_label.setHidden(true);
            }
        }
        // The lights: one borderless button each, made when the set of
        // lights changes and restyled in place otherwise, so a click is not
        // lost to a refresh. A click runs the command the light stands for
        // (`AppController::light_command`), routed by its id.
        let same = self.light_buttons.len() == bar.lights.len()
            && self.last_lights.iter().zip(&bar.lights).all(|(a, b)| a.id == b.id);
        if !same {
            for v in self.lights.arrangedSubviews().iter() {
                self.lights.removeArrangedSubview(&v);
                v.removeFromSuperview();
            }
            self.light_buttons.clear();
            self.last_lights.clear();
            for l in &bar.lights {
                let b = crate::ui::button(mtm, "", &l.id);
                b.setBordered(false);
                self.lights.addArrangedSubview(&b);
                self.light_buttons.push(b);
            }
        }
        for (i, l) in bar.lights.iter().enumerate() {
            // A device's light opens its menu; the others run their command.
            let enabled = app.light_menu(&l.id).is_some()
                || app.light_command(&l.id).is_some_and(|c| app.command_enabled(&c) == iris_gui_core::controller::Enabled::Yes);
            if self.last_lights.get(i) == Some(l) && self.light_buttons[i].isEnabled() == enabled {
                continue;
            }
            style_light(&self.light_buttons[i], l, enabled);
        }
        self.last_lights = bar.lights.clone();
    }
}

impl MainWindow {
    /// The status-bar light with id `id`, to pop its menu up at.
    pub fn light_view(&self, id: &str) -> Option<Retained<NSView>> {
        let i = self.last_lights.iter().position(|l| l.id == id)?;
        self.light_buttons.get(i).map(|b| Retained::into_super(Retained::into_super(b.clone())))
    }
}

fn style_light(b: &NSButton, l: &DeviceLight, enabled: bool) {
    // A name VoiceOver can read (the icon alone says nothing).
    objc2_app_kit::NSAccessibility::setAccessibilityLabel(b, Some(&ns(&l.tooltip)));
    match crate::glyphs::icon(l.icon, &l.tooltip) {
        Some(img) => b.setImage(Some(&img)),
        None => b.setTitle(&ns("\u{2022}")),
    }
    if let Some(badge) = &l.badge {
        b.setTitle(&ns(badge));
        b.setImagePosition(objc2_app_kit::NSCellImagePosition::ImageLeading);
    } else if b.image().is_some() {
        b.setTitle(&ns(""));
        b.setImagePosition(objc2_app_kit::NSCellImagePosition::ImageOnly);
    }
    let color = match l.state {
        LightState::Off => NSColor::tertiaryLabelColor(),
        LightState::Idle => NSColor::secondaryLabelColor(),
        LightState::Reading | LightState::Active => NSColor::systemGreenColor(),
        LightState::Writing => NSColor::systemOrangeColor(),
        LightState::Fault => NSColor::systemRedColor(),
    };
    b.setContentTintColor(Some(&color));
    b.setEnabled(enabled);
    b.setToolTip(Some(&ns(&l.tooltip)));
}

/// The library's list in the sidebar: below the window buttons, clear of
/// the hairline on its edge.
fn place_on_card(v: &NSView, sidebar: &NSView) {
    v.leadingAnchor().constraintEqualToAnchor(&sidebar.leadingAnchor()).setActive(true);
    v.trailingAnchor().constraintEqualToAnchor_constant(&sidebar.trailingAnchor(), -1.0).setActive(true);
    v.topAnchor().constraintEqualToAnchor(&sidebar.safeAreaLayoutGuide().topAnchor()).setActive(true);
    v.bottomAnchor().constraintEqualToAnchor(&sidebar.bottomAnchor()).setActive(true);
}

/// The divider's autosave name (the sidebar's width).
const SPLIT_AUTOSAVE: &str = "IRISSidebarSplit";

/// Pin the display's scroll view in `to`: trailing and bottom at `to`'s
/// edges, and two sets for leading and top, the safe area's (active) and the
/// edges' (full screen; `MainWindow::full_chrome_edges`).
fn pin_display(v: &NSView, to: &NSView) -> (Vec<Retained<objc2_app_kit::NSLayoutConstraint>>, Vec<Retained<objc2_app_kit::NSLayoutConstraint>>) {
    let safe = to.safeAreaLayoutGuide();
    v.trailingAnchor().constraintEqualToAnchor(&safe.trailingAnchor()).setActive(true);
    v.bottomAnchor().constraintEqualToAnchor(&to.bottomAnchor()).setActive(true);
    let in_safe = vec![
        v.leadingAnchor().constraintEqualToAnchor(&safe.leadingAnchor()),
        v.topAnchor().constraintEqualToAnchor(&safe.topAnchor()),
    ];
    let at_edges = vec![v.leadingAnchor().constraintEqualToAnchor(&to.leadingAnchor()), v.topAnchor().constraintEqualToAnchor(&to.topAnchor())];
    for c in &in_safe {
        c.setActive(true);
    }
    (in_safe, at_edges)
}

/// Pin `v` inside `to`'s safe area, but to its bottom edge: below the title
/// bar and toolbar (the window's content runs under them) and clear of a
/// floating sidebar.
fn pin_safe(v: &NSView, to: &NSView) {
    let safe = to.safeAreaLayoutGuide();
    v.leadingAnchor().constraintEqualToAnchor(&safe.leadingAnchor()).setActive(true);
    v.trailingAnchor().constraintEqualToAnchor(&safe.trailingAnchor()).setActive(true);
    v.topAnchor().constraintEqualToAnchor(&safe.topAnchor()).setActive(true);
    v.bottomAnchor().constraintEqualToAnchor(&to.bottomAnchor()).setActive(true);
}

fn pin(v: &NSView, to: &NSView) {
    v.leadingAnchor().constraintEqualToAnchor(&to.leadingAnchor()).setActive(true);
    v.trailingAnchor().constraintEqualToAnchor(&to.trailingAnchor()).setActive(true);
    v.topAnchor().constraintEqualToAnchor(&to.topAnchor()).setActive(true);
    v.bottomAnchor().constraintEqualToAnchor(&to.bottomAnchor()).setActive(true);
}

fn new_display_window(mtm: MainThreadMarker, name: &str) -> Retained<NSWindow> {
    let style = NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Miniaturizable
        | NSWindowStyleMask::Resizable;
    // SAFETY: a standard window, kept by the main window while detached.
    let w = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            mtm.alloc(),
            NSRect::new(NSPoint::new(160.0, 160.0), NSSize::new(1282.0, 1024.0)),
            style,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    // SAFETY: the main window keeps it.
    unsafe { w.setReleasedWhenClosed(false) };
    w.setTitle(&ns(name));
    w.setFrameAutosaveName(&ns("IRISDisplayWindow"));
    w.setAcceptsMouseMovedEvents(true);
    w
}

thread_local! {
    /// Counts toasts, so a timer clears only the one it was set for.
    static TOAST_SEQ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// A window is in its full-screen animation.
    static FULL_SCREEN_MOVING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A window starts into or out of full screen (by the green button, the
/// menu or us): the controller hears where it is going at once.
fn full_screen_moving(on: bool) {
    FULL_SCREEN_MOVING.with(|m| m.set(true));
    crate::app::display(iris_gui_core::controller::display::DisplayAction::FullScreenChanged(on));
}

/// The animation ended (or failed) with the window `on` full screen.
fn full_screen_moved(on: bool) {
    FULL_SCREEN_MOVING.with(|m| m.set(false));
    crate::app::display(iris_gui_core::controller::display::DisplayAction::FullScreenChanged(on));
    // Whatever waited for the animation (a detached window closing) goes on.
    crate::app::later(Box::new(crate::app::refresh));
}

define_class!(
    /// The main window's delegate: focus for the capture rules, full screen
    /// the user left by other means.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISWindowDelegate"]
    pub struct WindowDelegate;

    unsafe impl NSObjectProtocol for WindowDelegate {}

    unsafe impl NSWindowDelegate for WindowDelegate {
        #[unsafe(method(windowDidResize:))]
        fn resized(&self, _n: &objc2_foundation::NSNotification) {
            // The display's size follows the window (Actual Size scrolls).
            crate::app::display_fit();
        }

        #[unsafe(method(windowDidChangeOcclusionState:))]
        fn occlusion(&self, _n: &objc2_foundation::NSNotification) {
            crate::app::display_redraw();
        }

        #[unsafe(method(windowDidBecomeKey:))]
        fn became_key(&self, _n: &objc2_foundation::NSNotification) {
            crate::app::display(iris_gui_core::controller::display::DisplayAction::Focus(true));
        }

        #[unsafe(method(windowDidResignKey:))]
        fn resigned_key(&self, _n: &objc2_foundation::NSNotification) {
            crate::app::display(iris_gui_core::controller::display::DisplayAction::Focus(false));
        }

        #[unsafe(method(window:willUseFullScreenPresentationOptions:))]
        fn full_screen_options(
            &self,
            _w: &NSWindow,
            proposed: objc2_app_kit::NSApplicationPresentationOptions,
        ) -> objc2_app_kit::NSApplicationPresentationOptions {
            use objc2_app_kit::NSApplicationPresentationOptions as O;
            proposed | O::FullScreen | O::AutoHideToolbar | O::AutoHideMenuBar
        }

        #[unsafe(method(windowWillEnterFullScreen:))]
        fn will_enter_full(&self, _n: &objc2_foundation::NSNotification) {
            full_screen_moving(true);
        }

        #[unsafe(method(windowWillExitFullScreen:))]
        fn will_exit_full(&self, _n: &objc2_foundation::NSNotification) {
            full_screen_moving(false);
        }

        #[unsafe(method(windowDidEnterFullScreen:))]
        fn entered_full(&self, _n: &objc2_foundation::NSNotification) {
            full_screen_moved(true);
        }

        #[unsafe(method(windowDidExitFullScreen:))]
        fn exited_full(&self, _n: &objc2_foundation::NSNotification) {
            full_screen_moved(false);
        }

        #[unsafe(method(windowDidFailToEnterFullScreen:))]
        fn failed_enter(&self, _w: &NSWindow) {
            full_screen_moved(false);
        }

        #[unsafe(method(windowDidFailToExitFullScreen:))]
        fn failed_exit(&self, _w: &NSWindow) {
            full_screen_moved(true);
        }
    }
);

impl WindowDelegate {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = mtm.alloc::<WindowDelegate>().set_ivars(());
        // SAFETY: NSObject's plain init.
        unsafe { objc2::msg_send![super(this), init] }
    }
}

define_class!(
    /// The detached display window's delegate: focus and full screen as the
    /// main window's, and closing it puts the display back.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISDetachedDelegate"]
    pub struct DetachedDelegate;

    unsafe impl NSObjectProtocol for DetachedDelegate {}

    unsafe impl NSWindowDelegate for DetachedDelegate {
        #[unsafe(method(windowShouldClose:))]
        fn should_close(&self, _w: &NSWindow) -> bool {
            // After this handler: putting the display back closes the window.
            crate::app::later(Box::new(|| {
                crate::app::display(iris_gui_core::controller::display::DisplayAction::ToggleDetached)
            }));
            false
        }

        #[unsafe(method(windowDidResize:))]
        fn resized(&self, _n: &objc2_foundation::NSNotification) {
            // The display's size follows the window (Actual Size scrolls).
            crate::app::display_fit();
        }

        #[unsafe(method(windowDidChangeOcclusionState:))]
        fn occlusion(&self, _n: &objc2_foundation::NSNotification) {
            crate::app::display_redraw();
        }

        #[unsafe(method(windowDidBecomeKey:))]
        fn became_key(&self, _n: &objc2_foundation::NSNotification) {
            crate::app::display(iris_gui_core::controller::display::DisplayAction::Focus(true));
        }

        #[unsafe(method(windowDidResignKey:))]
        fn resigned_key(&self, _n: &objc2_foundation::NSNotification) {
            crate::app::display(iris_gui_core::controller::display::DisplayAction::Focus(false));
        }

        #[unsafe(method(window:willUseFullScreenPresentationOptions:))]
        fn full_screen_options(
            &self,
            _w: &NSWindow,
            proposed: objc2_app_kit::NSApplicationPresentationOptions,
        ) -> objc2_app_kit::NSApplicationPresentationOptions {
            use objc2_app_kit::NSApplicationPresentationOptions as O;
            proposed | O::FullScreen | O::AutoHideToolbar | O::AutoHideMenuBar
        }

        #[unsafe(method(windowWillEnterFullScreen:))]
        fn will_enter_full(&self, _n: &objc2_foundation::NSNotification) {
            full_screen_moving(true);
        }

        #[unsafe(method(windowWillExitFullScreen:))]
        fn will_exit_full(&self, _n: &objc2_foundation::NSNotification) {
            full_screen_moving(false);
        }

        #[unsafe(method(windowDidEnterFullScreen:))]
        fn entered_full(&self, _n: &objc2_foundation::NSNotification) {
            full_screen_moved(true);
        }

        #[unsafe(method(windowDidExitFullScreen:))]
        fn exited_full(&self, _n: &objc2_foundation::NSNotification) {
            full_screen_moved(false);
        }

        #[unsafe(method(windowDidFailToEnterFullScreen:))]
        fn failed_enter(&self, _w: &NSWindow) {
            full_screen_moved(false);
        }

        #[unsafe(method(windowDidFailToExitFullScreen:))]
        fn failed_exit(&self, _w: &NSWindow) {
            full_screen_moved(true);
        }
    }
);

impl DetachedDelegate {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = mtm.alloc::<DetachedDelegate>().set_ivars(());
        // SAFETY: NSObject's plain init.
        unsafe { objc2::msg_send![super(this), init] }
    }
}

fn as_view(t: &Retained<NSTextField>) -> Retained<NSView> {
    let v: &NSView = t.as_ref();
    v.retain()
}
