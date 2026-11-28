//! egui helpers over the shared text catalogs (`iris_gui_core::model::i18n`).

pub use iris_gui_core::model::i18n::*;

/// A secondary caption under a control. Draws nothing when the catalog leaves
/// the string empty.
pub fn subtitle(ui: &mut egui::Ui, text: String) {
    if !text.is_empty() {
        ui.label(egui::RichText::new(text).weak().small());
    }
}

pub trait ResponseExt {
    /// `on_hover_text`, skipped when the catalog leaves the string empty.
    fn tip(self, text: String) -> Self;
}

impl ResponseExt for egui::Response {
    fn tip(self, text: String) -> Self {
        if text.is_empty() { self } else { self.on_hover_text(text) }
    }
}
