//! The main window (D3): the library on the left, the selected machine on the
//! right, the status bar under it.
//!
//! `AdwOverlaySplitView` is the library's sidebar — collapsing it to an
//! overlay on a narrow window is the widget's own behaviour — inside an
//! `AdwToolbarView` that carries the header bar (and later the status bar),
//! with an `AdwToastOverlay` over the lot for `Effect::Toast`.

use std::rc::Rc;

use adw::prelude::*;
use gtk::glib;
use gtk4 as gtk;
use iris_gui_core::controller::app::AppController;
use iris_gui_core::controller::icons::Icon;
use iris_gui_core::controller::ids;
use libadwaita as adw;

use crate::icons;
use crate::library::Library;
use crate::status::StatusBar;
use crate::summary;
use crate::toolbar::Toolbar;

pub struct MainWindow {
    pub window: adw::ApplicationWindow,
    toasts: adw::ToastOverlay,
    split: adw::OverlaySplitView,
    library: Rc<Library>,
    toolbar: Toolbar,
    status: StatusBar,
    /// Holds the machine page (and later the guest display).
    pane: adw::ToolbarView,
    /// The overview the page was built from, so it is rebuilt only on a change.
    shown: Option<Option<iris_gui_core::controller::overview::Overview>>,
    /// Which tab the machine page shows.
    tab: String,
    /// The last toast shown, so a repeat does not stack up.
    last_toast: Option<adw::Toast>,
}

impl MainWindow {
    pub fn new(app: &adw::Application) -> Self {
        let header = adw::HeaderBar::builder().show_start_title_buttons(false).build();
        let toolbar = Toolbar::new(&header);

        let toggle = gtk::ToggleButton::builder()
            .child(&icons::image(Icon::ShowLibrary, 16, &t!("tb-show-library")))
            .tooltip_text(t!("tb-show-library"))
            .active(true)
            .build();
        toggle.set_widget_name(ids::toolbar::SHOW_LIBRARY);
        header.pack_start(&toggle);

        // GNOME has no menu bar: everything is in the header bar's primary
        // menu. `open-menu-symbolic` is the desktop's own glyph for it, not
        // one of the core's icon meanings, so it comes from the icon theme
        // libadwaita already requires.
        let primary = gtk::MenuButton::builder()
            .icon_name("open-menu-symbolic")
            .tooltip_text(t!("mb-menu"))
            .menu_model(&crate::menu::primary(app))
            .primary(true)
            .build();
        header.pack_end(&primary);

        let content = adw::ToolbarView::new();
        content.add_top_bar(&header);
        let status = StatusBar::new();
        content.add_bottom_bar(&status.root);

        let library = Library::new(app);
        let split = adw::OverlaySplitView::builder()
            .sidebar(&library.root)
            .content(&content)
            .min_sidebar_width(220.0)
            .max_sidebar_width(360.0)
            .sidebar_width_fraction(0.25)
            .build();

        // Narrow enough and the sidebar becomes an overlay over the machine,
        // as every GNOME app does it.
        let breakpoint = adw::Breakpoint::new(adw::BreakpointCondition::new_length(
            adw::BreakpointConditionLengthType::MaxWidth,
            640.0,
            adw::LengthUnit::Px,
        ));
        breakpoint.add_setter(&split, "collapsed", Some(&true.into()));

        let toasts = adw::ToastOverlay::new();
        toasts.set_child(Some(&split));

        let window = adw::ApplicationWindow::builder()
            .application(app)
            .default_width(1100)
            .default_height(700)
            .title(t!("app-name"))
            .content(&toasts)
            .build();
        window.add_breakpoint(breakpoint);
        // Closing the window leaves the application only when the controller
        // says so: it may have disks to synchronize first.
        window.connect_close_request(|_| crate::app::close_requested());

        toggle.connect_toggled(glib::clone!(
            #[weak]
            split,
            move |b| split.set_show_sidebar(b.is_active())
        ));
        split.connect_show_sidebar_notify(glib::clone!(
            #[weak]
            toggle,
            move |s| toggle.set_active(s.shows_sidebar())
        ));

        Self {
            window,
            toasts,
            split,
            library,
            toolbar,
            status,
            pane: content,
            shown: None,
            tab: summary::TAB_DETAILS.to_string(),
            last_toast: None,
        }
    }

    /// Draw what the controller holds now.
    pub fn refresh(&mut self, app: &AppController) {
        self.library.refresh(app.library());
        self.toolbar.refresh(app);
        self.status.refresh(app);
        let o = app.overview();
        if self.shown.as_ref() != Some(&o) {
            let gtk_app = crate::app::application();
            let page = summary::build(gtk_app.as_ref(), o.as_ref(), &self.tab, app.is_empty());
            self.pane.set_content(Some(&page));
            self.shown = Some(o);
        }
    }

    /// The machine page's tab, kept across refreshes.
    pub fn set_tab(&mut self, tab: &str) {
        if self.tab != tab {
            self.tab = tab.to_string();
            self.shown = None;
        }
    }

    pub fn toggle_library(&self) {
        self.split.set_show_sidebar(!self.split.shows_sidebar());
    }

    /// Draw the window into a PNG, for a test script's `shot:` step.
    ///
    /// The window paints itself through its own GSK renderer, so this needs
    /// no compositor permission, no screenshot tool and no unlocked screen —
    /// on Wayland there is no way to grab another process's window anyway.
    pub fn screenshot(&self, path: &str) -> Result<(), String> {
        let renderer = self.window.native().and_then(|n| n.renderer()).ok_or("the window has no renderer yet")?;
        let (w, h) = (self.window.width() as f64, self.window.height() as f64);
        let paintable = gtk::WidgetPaintable::new(Some(&self.window));
        let snapshot = gtk::Snapshot::new();
        paintable.snapshot(&snapshot, w, h);
        let node = snapshot.to_node().ok_or("the window drew nothing")?;
        renderer.render_texture(&node, None).save_to_png(path).map_err(|e| e.to_string())
    }

    pub fn toast(&mut self, text: &str) {
        if let Some(old) = self.last_toast.take() {
            old.dismiss();
        }
        let toast = adw::Toast::new(text);
        self.toasts.add_toast(toast.clone());
        self.last_toast = Some(toast);
    }
}
