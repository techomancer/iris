//! The shell's own CSS: the few things the theme has no class for.
//!
//! Colours come from libadwaita's palette wherever one fits (D12, semantic
//! colours, no hard-coded hex), except the guest's screen, which the plan
//! keeps dark in both appearances.

use gtk::gdk;
use gtk4 as gtk;

const CSS: &str = "
/* The machine's screen: dark in light and dark appearance alike (D12). */
.iris-screen {
  background-color: #101010;
  border-radius: 12px;
}
/* The machine's picture on its own black screen: bright enough to read. */
.iris-screen-art {
  color: alpha(#ffffff, 0.45);
}
/* A saved state's thumbnail. */
.iris-thumb {
  background-color: alpha(currentColor, 0.08);
  border-radius: 6px;
}
";

pub fn load() {
    let Some(display) = gdk::Display::default() else { return };
    let provider = gtk::CssProvider::new();
    provider.load_from_string(CSS);
    gtk::style_context_add_provider_for_display(&display, &provider, gtk::STYLE_PROVIDER_PRIORITY_APPLICATION);
}
