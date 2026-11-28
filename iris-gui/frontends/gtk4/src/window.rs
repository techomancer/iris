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

use crate::display::Screen;
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
    /// The guest's screen, and the scroller that holds it at Actual Size.
    pub screen: Screen,
    /// The second display head (a dual Newport), beside the first.
    pub head1: Screen,
    head1_box: gtk::Box,
    head1_label: gtk::Label,
    screen_scroll: gtk::ScrolledWindow,
    /// Whether the pane is laid out for two heads.
    showing_head1: bool,
    /// Whether the pane is showing the display rather than the page.
    showing_display: bool,
    /// The display's own window, while it is detached.
    detached: Option<adw::Window>,
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
        let screen = Screen::new();
        let head1 = Screen::new();
        for v in [&screen.view, &head1.view] {
            v.set_hexpand(true);
            v.set_vexpand(true);
        }
        let head1_label = gtk::Label::new(None);
        head1_label.add_css_class("dim-label");
        head1_label.add_css_class("caption");
        let head1_box = gtk::Box::new(gtk::Orientation::Vertical, 2);
        head1_box.set_hexpand(true);
        head1_box.set_vexpand(true);
        head1_box.append(&head1_label);
        head1_box.append(&head1.view);
        head1_box.set_visible(false);

        // Head 0, and head 1 beside it when the machine has one. Each keeps
        // its own aspect, as egui's two viewports do; head 0 takes the input.
        let heads = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        heads.set_homogeneous(true);
        heads.append(&screen.view);
        heads.append(&head1_box);
        let screen_scroll = gtk::ScrolledWindow::builder().hexpand(true).vexpand(true).child(&heads).build();
        screen_scroll.add_css_class("iris-screen");

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
            screen,
            head1,
            head1_box,
            head1_label,
            screen_scroll,
            showing_head1: false,
            showing_display: false,
            detached: None,
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
        // A running machine shows its screen; a stopped one its page.
        let running = app.session().is_running();
        self.screen.view.set_look(
            iris_gui_core::controller::display::sizing(app.prefs()),
            app.prefs().display_filter,
        );
        let running = running && self.detached.is_none();
        if running != self.showing_display {
            self.showing_display = running;
            if running {
                self.pane.set_content(Some(&self.screen_scroll));
                self.screen.view.grab_focus();
            } else {
                self.shown = None;
            }
        }
        if running {
            // A dual Newport draws both heads side by side; head 0 keeps the
            // input. The label says when the second head has drawn nothing.
            let dual = app.session().dual_head();
            if dual != self.showing_head1 {
                self.showing_head1 = dual;
                self.head1_box.set_visible(dual);
                if !dual {
                    self.head1.clear();
                }
            }
            if dual {
                let live = app.session().head1_live();
                self.head1_label.set_label(&if live { t!("fb-head", head = 1) } else { t!("fb-head-waiting") });
            }
            return;
        }
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

    /// A pop-up menu hung on the header bar's button for `id`, or on the
    /// window when that button is not there.
    pub fn popup_at(&self, id: &str, menu: &gtk::gio::Menu) {
        let popover = gtk::PopoverMenu::from_model(Some(menu));
        let anchor: gtk::Widget = self.toolbar.item_widget(id).unwrap_or_else(|| self.window.clone().upcast());
        popover.set_parent(&anchor);
        popover.connect_closed(|p| p.unparent());
        popover.popup();
    }

    /// Full screen and the detached window follow the display controller.
    pub fn sync_display(&mut self, app: &AppController) {
        let want = app.display().fullscreen();
        if want != self.window.is_fullscreen() {
            if want {
                self.window.fullscreen();
            } else {
                self.window.unfullscreen();
            }
        }
        let detached = app.display().detached();
        if detached == self.detached.is_some() {
            return;
        }
        if detached {
            let window = adw::Window::builder()
                .title(app.selected().unwrap_or(&t!("app-name")))
                .default_width(1280)
                .default_height(1024)
                .build();
            let header = adw::HeaderBar::new();
            let view = adw::ToolbarView::new();
            view.add_top_bar(&header);
            self.pane.set_content(None::<&gtk::Widget>);
            view.set_content(Some(&self.screen_scroll));
            window.set_content(Some(&view));
            window.connect_close_request(|_| {
                crate::app::display(iris_gui_core::controller::display::DisplayAction::ToggleDetached);
                glib::Propagation::Proceed
            });
            window.present();
            self.detached = Some(window);
            self.showing_display = false;
            self.shown = None;
        } else if let Some(window) = self.detached.take() {
            window.set_content(None::<&gtk::Widget>);
            window.close();
            self.showing_display = false;
            self.shown = None;
        }
    }

    pub fn toggle_library(&self) {
        self.split.set_show_sidebar(!self.split.shows_sidebar());
    }

    /// Draw any window into a PNG, the way `shot:` does the main one.
    pub fn shoot(window: &impl IsA<gtk::Widget>, path: &str) -> Result<(), String> {
        let window = window.as_ref();
        let renderer = window.native().and_then(|n| n.renderer()).ok_or("the window has no renderer yet")?;
        let (w, h) = (window.width() as f64, window.height() as f64);
        if w < 1.0 || h < 1.0 {
            return Err("the window has no size yet".into());
        }
        let paintable = gtk::WidgetPaintable::new(Some(window));
        let snapshot = gtk::Snapshot::new();
        paintable.snapshot(&snapshot, w, h);
        let node = snapshot.to_node().ok_or("the window drew nothing")?;
        renderer.render_texture(&node, None).save_to_png(path).map_err(|e| e.to_string())
    }

    /// Draw the window into a PNG, for a test script's `shot:` step.
    ///
    /// The window paints itself through its own GSK renderer, so this needs
    /// no compositor permission, no screenshot tool and no unlocked screen —
    /// on Wayland there is no way to grab another process's window anyway.
    pub fn screenshot(&self, path: &str) -> Result<(), String> {
        Self::shoot(&self.window, path)
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
