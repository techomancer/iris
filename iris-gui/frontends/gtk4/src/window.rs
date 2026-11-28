//! The main window (D3): the library on the left, the selected machine on the
//! right, the status bar under it. Each of those arrives with its own slice;
//! this is the frame they hang in.
//!
//! `AdwToolbarView` carries the header bar (and later the status bar), with
//! an `AdwToastOverlay` over the lot for `Effect::Toast`.

use adw::prelude::*;
use gtk4 as gtk;
use iris_gui_core::controller::app::AppController;
use libadwaita as adw;

pub struct MainWindow {
    pub window: adw::ApplicationWindow,
    title: adw::WindowTitle,
    toasts: adw::ToastOverlay,
    placeholder: adw::StatusPage,
    /// The last toast shown, so a repeat does not stack up.
    last_toast: Option<adw::Toast>,
}

impl MainWindow {
    pub fn new(app: &adw::Application) -> Self {
        let title = adw::WindowTitle::new(&t!("app-name"), "");
        let header = adw::HeaderBar::new();
        header.set_title_widget(Some(&title));
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

        let placeholder = adw::StatusPage::builder().title(t!("toast-no-machine")).build();

        let view = adw::ToolbarView::new();
        view.add_top_bar(&header);
        view.set_content(Some(&placeholder));

        let toasts = adw::ToastOverlay::new();
        toasts.set_child(Some(&view));

        let window = adw::ApplicationWindow::builder()
            .application(app)
            .default_width(1000)
            .default_height(640)
            .title(t!("app-name"))
            .build();
        window.set_content(Some(&toasts));
        // Closing the window leaves the application only when the controller
        // says so: it may have disks to synchronize first.
        window.connect_close_request(|_| crate::app::close_requested());

        Self { window, title, toasts, placeholder, last_toast: None }
    }

    /// Draw what the controller holds now.
    pub fn refresh(&mut self, app: &AppController) {
        match app.selected() {
            Some(name) => {
                self.title.set_subtitle(name);
                self.placeholder.set_title(name);
            }
            None => {
                self.title.set_subtitle("");
                self.placeholder.set_title(&t!("toast-no-machine"));
            }
        }
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
