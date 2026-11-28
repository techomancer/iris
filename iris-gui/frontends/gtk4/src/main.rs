//! IRIS, the native Linux front end: a pure view over
//! `iris_gui_core::controller` (clanker/plans/iris-newgui-plan.md, Phase 5).
//!
//! Layers and rules: the controller contract (Action in, Effect out,
//! exhaustive matches, widget ids as accessibility identifiers, shortcuts as
//! data). Nothing here names the emulator or the model;
//! `iris-gui/scripts/check-boundaries.sh` enforces that.

#[cfg(target_os = "linux")]
#[macro_use]
extern crate iris_gui_core;

#[cfg(target_os = "linux")]
mod app;
#[cfg(target_os = "linux")]
mod effects;
#[cfg(target_os = "linux")]
mod icons;
#[cfg(target_os = "linux")]
mod library;
#[cfg(target_os = "linux")]
mod menu;
#[cfg(target_os = "linux")]
mod style;
#[cfg(target_os = "linux")]
mod summary;
#[cfg(target_os = "linux")]
mod toolbar;
#[cfg(target_os = "linux")]
mod wake;
#[cfg(target_os = "linux")]
mod window;

#[cfg(target_os = "linux")]
fn main() -> gtk4::glib::ExitCode {
    app::run()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("iris-gtk runs on Linux; use iris-appkit on macOS and iris-gui elsewhere.");
}
