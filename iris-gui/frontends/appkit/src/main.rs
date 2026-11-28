//! IRIS, the native macOS front end: a pure view over `iris_gui_core::controller`
//! (clanker/plans/iris-newgui-plan.md, Phase 4).
//!
//! Layers and rules: the controller contract (Action in, Effect out,
//! exhaustive matches, widget ids as accessibility identifiers, shortcuts as
//! data). Nothing here names the emulator or the model;
//! `iris-gui/scripts/check-boundaries.sh` enforces that.

#[cfg(target_os = "macos")]
#[macro_use]
extern crate iris_gui_core;

#[cfg(target_os = "macos")]
mod app;
#[cfg(target_os = "macos")]
mod browser;
#[cfg(target_os = "macos")]
mod chrome;
#[cfg(target_os = "macos")]
mod display;
#[cfg(target_os = "macos")]
mod effects;
#[cfg(target_os = "macos")]
mod glyphs;
#[cfg(target_os = "macos")]
mod keys;
#[cfg(target_os = "macos")]
mod library;
#[cfg(target_os = "macos")]
mod menu;
#[cfg(target_os = "macos")]
mod new_machine;
#[cfg(target_os = "macos")]
mod panels;
#[cfg(target_os = "macos")]
mod settings;
#[cfg(target_os = "macos")]
mod summary;
#[cfg(target_os = "macos")]
mod toolbar;
#[cfg(target_os = "macos")]
mod ui;
#[cfg(target_os = "macos")]
mod wake;
#[cfg(target_os = "macos")]
mod window;

#[cfg(target_os = "macos")]
fn main() {
    app::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("iris-appkit runs on macOS only; use iris-gui elsewhere.");
}
