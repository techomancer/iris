//! Labels and tooltips. Every string a shell shows comes from a controller,
//! through the Fluent catalogs in `iris-gui/locales` (the `t!` macro, which
//! moves here with the model in Phase 1). A control's accessibility label is
//! its tooltip.

use super::icons::Icon;

/// What a shell needs to draw one icon control: the meaning, the tooltip (also
/// its accessibility label) and the shortcut to show beside it, if any.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tip {
    pub icon: Icon,
    pub tooltip: String,
    pub shortcut: String,
}

impl Tip {
    pub fn new(icon: Icon, tooltip: impl Into<String>) -> Self {
        Self { icon, tooltip: tooltip.into(), shortcut: String::new() }
    }

    /// The tooltip with the shortcut appended, as a hover text shows it.
    pub fn hover_text(&self) -> String {
        if self.shortcut.is_empty() {
            self.tooltip.clone()
        } else {
            format!("{} ({})", self.tooltip, self.shortcut)
        }
    }
}
