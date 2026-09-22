//! `egui::Window` stand-in that opens a real OS window.
//!
//! The dialogs in `main.rs` and `dialogs/` are written against `egui::Window`.
//! In the native build those files import `crate::macos_native::egui` instead of
//! `eframe::egui`, and that module swaps in this [`Window`]. So every dialog
//! becomes a separate macOS window without its code changing, and the classic
//! build keeps drawing them as in-window egui windows.
//!
//! Only the builder methods those call sites use are provided. If a new dialog
//! needs another method, add it here as well, or the native build won't compile.
//!
//! Each window is an immediate viewport. It is drawn inside the parent's pass,
//! so the body can borrow app state the way an `egui::Window` body does. Don't
//! nest them: showing one from inside another's body is not supported.

use eframe::egui::{
    self, Context, InnerResponse, Pos2, Ui, Vec2, ViewportBuilder, ViewportClass, ViewportId,
    WidgetText,
};

/// Size used for a non-resizable window before its content has been measured.
const UNMEASURED: Vec2 = Vec2::new(420.0, 180.0);
/// Size used for a resizable window that doesn't give a default.
const RESIZABLE_DEFAULT: Vec2 = Vec2::new(560.0, 440.0);
const MARGIN: i8 = 12;

pub struct Window<'open> {
    title: String,
    open: Option<&'open mut bool>,
    resizable: bool,
    default_size: Vec2,
}

/// Remembered between passes, keyed by the window's viewport id.
#[derive(Clone, Copy, Default)]
struct Placement {
    /// Measured content size of a non-resizable window.
    content: Option<Vec2>,
    /// Centre point picked when the window opened, from the main window's position.
    centre: Option<Pos2>,
    /// The parent pass this window was last shown in.
    last_pass: Option<u64>,
}

impl<'open> Window<'open> {
    pub fn new(title: impl Into<WidgetText>) -> Self {
        Self {
            title: title.into().text().to_owned(),
            open: None,
            // egui::Window's default.
            resizable: true,
            default_size: Vec2::splat(f32::NAN),
        }
    }

    /// Cleared when the user closes the window. A window without it has no
    /// close button and is dismissed only from its own buttons, just as the
    /// in-window version has no ×.
    pub fn open(mut self, open: &'open mut bool) -> Self {
        self.open = Some(open);
        self
    }

    /// A non-resizable window sizes itself to its content.
    pub fn resizable(mut self, resizable: bool) -> Self {
        self.resizable = resizable;
        self
    }

    pub fn default_width(mut self, width: f32) -> Self {
        self.default_size.x = width;
        self
    }

    pub fn default_height(mut self, height: f32) -> Self {
        self.default_size.y = height;
        self
    }

    /// OS windows can't collapse, so this is ignored.
    pub fn collapsible(self, _collapsible: bool) -> Self {
        self
    }

    /// Ignored: every window opens centred over the main window.
    pub fn anchor(self, _align: egui::Align2, _offset: impl Into<Vec2>) -> Self {
        self
    }

    pub fn show<R>(
        self,
        ctx: &Context,
        add_contents: impl FnOnce(&mut Ui) -> R,
    ) -> Option<InnerResponse<Option<R>>> {
        let Self { title, open, resizable, default_size } = self;
        if open.as_deref() == Some(&false) {
            return None;
        }

        let viewport_id = ViewportId::from_hash_of(("iris-native-window", &title));
        let state_id = egui::Id::new(viewport_id);
        let pass = ctx.cumulative_pass_nr();
        let mut place: Placement = ctx.data(|d| d.get_temp(state_id)).unwrap_or_default();

        // Not shown in the previous pass means the window was closed and is
        // opening again. Centre it over the main window's current position.
        let reopened = place.last_pass.is_none_or(|p| p + 1 < pass);
        if reopened {
            place.centre = ctx.input(|i| i.viewport().outer_rect).map(|r| r.center());
        }

        let size = if resizable {
            let or = |v: f32, d: f32| if v.is_nan() { d } else { v };
            Vec2::new(or(default_size.x, RESIZABLE_DEFAULT.x), or(default_size.y, RESIZABLE_DEFAULT.y))
        } else {
            place.content.unwrap_or(UNMEASURED)
        };

        // Resizable windows keep a fixed size and position in the builder, so
        // the user's resize or move sticks. For the others, egui sees the new
        // size when the content is measured and resizes the window, keeping it
        // centred as the in-window CENTER_CENTER anchor did.
        let mut builder = ViewportBuilder::default()
            .with_title(title.as_str())
            .with_inner_size(size)
            .with_resizable(resizable)
            .with_maximize_button(resizable)
            .with_minimize_button(false)
            .with_close_button(open.is_some());
        if let Some(centre) = place.centre {
            builder = builder.with_position(centre - size / 2.0);
        }

        let mut body = Some(add_contents);
        let mut close_requested = false;
        let mut measured = None;
        let shown = ctx.show_viewport_immediate(viewport_id, builder, |ui, class| {
            let mut run = |ui: &mut Ui| body.take().map(|body| body(ui));
            if class == ViewportClass::EmbeddedWindow {
                // The backend can't open another window, so egui has already
                // wrapped this viewport in an egui::Window.
                let inner = run(ui);
                return InnerResponse::new(inner, ui.response());
            }
            close_requested |= ui.input(|i| i.viewport().close_requested());
            let frame = egui::Frame::central_panel(ui.style()).inner_margin(MARGIN);
            if resizable {
                egui::CentralPanel::default().frame(frame).show(ui, run)
            } else {
                // The panel only paints the window background. The content goes
                // in an unconstrained area, so it takes its natural size (which
                // is then measured) rather than the window's current size.
                egui::CentralPanel::default().show(ui, |_| {});
                let area = egui::Area::new(state_id.with("content"))
                    .fixed_pos(Pos2::ZERO)
                    .constrain(false)
                    .show(ui.ctx(), |ui| frame.show(ui, run).inner);
                measured = Some(area.response.rect.size());
                area
            }
        });

        if let Some(m) = measured.filter(|m| m.x >= 1.0 && m.y >= 1.0) {
            place.content = Some(m.ceil());
        }
        place.last_pass = Some(pass);
        ctx.data_mut(|d| d.insert_temp(state_id, place));

        if close_requested {
            if let Some(open) = open {
                *open = false;
            }
        }
        Some(shown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without a multi-viewport backend (as in a headless test) the window
    /// falls back to an embedded egui window, and the body still runs.
    #[test]
    fn falls_back_to_embedded_without_viewports() {
        let ctx = Context::default();
        let mut drawn = false;
        let mut open = true;
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            Window::new("Test").open(&mut open).resizable(false).show(ui.ctx(), |ui| {
                drawn = true;
                ui.label("content");
            });
        });
        // Headless: no renderer consumes the font texture upload.
        output.textures_delta.clear();
        assert!(drawn);
        assert!(open);
    }

    #[test]
    fn closed_window_is_not_drawn() {
        let ctx = Context::default();
        let mut open = false;
        let mut drawn = false;
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let shown = Window::new("Test").open(&mut open).show(ui.ctx(), |_| drawn = true);
            assert!(shown.is_none());
        });
        // Headless: no renderer consumes the font texture upload.
        output.textures_delta.clear();
        assert!(!drawn);
    }
}
