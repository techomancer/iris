//! The model: domain state and operations, with no UI state and no toolkit.
//!
//! A model module never imports a controller or a view.

// First, so `t!` is in scope for every module below.
#[macro_use]
pub mod i18n;

pub mod camera_test;
pub mod chd_convert;
pub mod capture_access;
pub mod disk_browser;
pub mod disk_probe;
pub mod framebuffer;
pub mod handle;
pub mod hardware;
pub mod job;
pub mod machines;
pub mod macos_sandbox;
pub mod netfix;
pub mod netplan;
pub mod pointer_follow;
pub mod ram;
pub mod safe_stop;
pub mod serial_console;
pub mod settings;
pub mod single_instance;
pub mod summary;
