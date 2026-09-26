//! Native macOS front-end, compiled in with `--features macos-gui`
//! (`cfg(native_mac)`; see build.rs).
//!
//! The default iris-gui layout puts a control column, the configuration editor
//! and a status footer in the same window as the emulated display. This backend
//! moves each of them out of that window:
//!
//! - the menus go in the system menu bar ([`menus`], [`menubar`]);
//! - the configuration editor and every dialog get OS windows of their own
//!   ([`window`]);
//! - the run state goes in the window title ([`App::window_title`]).
//!
//! That leaves the main window with nothing but the guest's display. The app
//! logic (emulator lifecycle, dialogs, framebuffer) is shared with the default
//! layout. `main.rs` calls in here from a few `cfg(native_mac)` hooks.

mod menubar;
mod menus;
mod window;

use crate::handle::NetState;
use crate::{input, App};
use eframe::egui::{self as e, RichText, ViewportCommand};
use std::time::{Duration, Instant};

/// `eframe::egui` with [`window::Window`] in place of `egui::Window`. The files
/// that draw dialogs import this as `egui` in the native build, so their
/// windows become OS windows with no change to the dialog code.
pub mod egui {
    pub use super::window::Window;
    pub use eframe::egui::*;
}

/// Backend state kept on the [`App`].
pub struct State {
    /// The menu model as last built, and when it was built. It is rebuilt a
    /// few times a second, not every frame: naming a SCSI slot stats its image
    /// file, and each change hands AppKit a whole new `NSMenu`.
    menus: Vec<menus::Menu>,
    menus_built: Instant,
    /// The last title pushed to the OS, and when.
    title: String,
    title_at: Instant,
    /// Whether Help → About IRIS is open.
    show_about: bool,
}

impl Default for State {
    fn default() -> Self {
        // Far enough in the past that the first frame builds both.
        let long_ago = Instant::now() - Duration::from_secs(60);
        Self {
            menus: Vec::new(),
            menus_built: long_ago,
            title: String::new(),
            title_at: long_ago,
            show_about: false,
        }
    }
}

pub use window::end_frame;

/// Runs in `main` before the event loop starts.
pub fn before_launch() {
    menubar::disable_automatic_window_tabbing();
}

impl App {
    /// Per-frame work, done where the default layout draws its side panels:
    /// track the fullscreen state, keep the menu bar current, apply what was
    /// picked from it, and update the title.
    pub(crate) fn native_frame(&mut self, ctx: &e::Context) {
        // The green button and Escape change fullscreen without going through
        // our own actions, so take the window's reported state as the truth.
        if let Some(fullscreen) = ctx.input(|i| i.viewport().fullscreen) {
            if self.fullscreen != fullscreen {
                self.fullscreen = fullscreen;
                self.invalidate_menus();
            }
        }
        self.refresh_menus(ctx);
        for action in menubar::take_actions() {
            self.apply_menu_action(action, ctx);
        }
        self.sync_window_title(ctx);
    }

    /// The windows only this backend has. Called after the central panel, at
    /// the top level of the frame: an OS window must not be opened from inside
    /// another one's body.
    pub(crate) fn native_windows(&mut self, ctx: &e::Context) {
        self.config_editor_window(ctx);
        self.about_window(ctx);
    }

    /// The main window's background. While a machine runs, the window holds
    /// only the emulated display, so paint the letterbox around it black, as
    /// a monitor would be, whatever the UI theme. The welcome panel shown while
    /// stopped keeps the theme's background.
    pub(crate) fn native_central_frame(&self, frame: e::Frame) -> e::Frame {
        if self.emu.is_running() {
            frame.fill(e::Color32::BLACK)
        } else {
            frame
        }
    }

    /// The welcome panel, inset from the window edge. The central panel has
    /// no margin (so the framebuffer can reach the edges), and in the default
    /// layout the sidebar is what keeps the panel off the window's left edge.
    pub(crate) fn native_welcome_panel(&mut self, ui: &mut e::Ui) {
        e::Frame::new()
            .inner_margin(e::Margin::symmetric(14, 10))
            .show(ui, |ui| self.welcome_panel(ui));
    }

    fn invalidate_menus(&mut self) {
        self.native.menus_built = Instant::now() - Duration::from_secs(1);
    }

    fn toggle_fullscreen(&mut self, ctx: &e::Context) {
        self.fullscreen = !ctx.input(|i| i.viewport().fullscreen).unwrap_or(self.fullscreen);
        ctx.send_viewport_cmd_to(e::ViewportId::ROOT, ViewportCommand::Fullscreen(self.fullscreen));
        self.invalidate_menus();
    }

    /// Rebuild the menu model if it may be stale, and give it to AppKit only if
    /// it changed. A fifth of a second is well under "the menu looks out of
    /// date" and well over "every frame".
    fn refresh_menus(&mut self, ctx: &e::Context) {
        menubar::install(ctx);
        if self.native.menus_built.elapsed() < Duration::from_millis(200) {
            return;
        }
        self.native.menus_built = Instant::now();
        let menus = self.build_menus();
        if menus != self.native.menus {
            self.native.menus = menus;
            menubar::rebuild(&self.native.menus);
        }
    }

    /// The tabbed configuration editor, in its own window so it can sit beside
    /// the display (or on another monitor) instead of sharing its window.
    fn config_editor_window(&mut self, ctx: &e::Context) {
        if !self.show_config_editor {
            return;
        }
        let machine = self.prefs.active_machine.as_deref().unwrap_or("default").to_string();
        let mut open = true;
        window::Window::new(format!("IRIS configuration \u{2014} {machine}"))
            .open(&mut open)
            .resizable(true)
            .default_width(620.0)
            .default_height(760.0)
            .show(ctx, |ui| {
                e::ScrollArea::vertical().show(ui, |ui| self.central_tabs(ui));
            });
        if !open {
            self.show_config_editor = false;
        }
    }

    /// Help → About IRIS: the version, and what this build has compiled in.
    /// (The default layout lists this at the bottom of its Help menu.)
    fn about_window(&mut self, ctx: &e::Context) {
        let mut open = self.native.show_about;
        window::Window::new("About IRIS").open(&mut open).resizable(false).show(ctx, |ui| {
            ui.set_max_width(380.0);
            ui.heading("IRIS");
            ui.label("SGI Indy (MIPS R4400) emulator");
            ui.label(format!("Version {}", env!("APP_VERSION")));
            ui.add_space(8.0);
            ui.label(RichText::new("Authors").strong());
            ui.label("Original: techomancer");
            ui.label("iris-gui fork: Dani Sarfati (danifunker)");
            ui.add_space(8.0);
            ui.label(RichText::new("Build features").strong());
            use iris::build_features as bf;
            let on = |b: bool| if b { "on" } else { "off" };
            ui.label(format!("  chd:       {}", on(bf::CHD)));
            ui.label(format!("  camera:    {}", on(bf::CAMERA)));
            // rex-jit is a compile-time feature, but the App Store build forces
            // the interpreter at runtime via IRIS_NO_JIT. Report what is
            // actually running.
            let jit_off = std::env::var_os("IRIS_NO_JIT").is_some();
            let jit = if !bf::REX_JIT { "off" } else if jit_off { "off (sandbox)" } else { "on" };
            ui.label(format!("  rex-jit:   {jit}"));
            ui.label(format!("  lightning: {}", if bf::LIGHTNING { "on (no debug)" } else { "off" }));
            ui.label(format!("  ultra64:   {}", on(bf::ULTRA64)));
        });
        self.native.show_about = open;
    }

    /// The window title, which carries the state the default layout shows in
    /// its status footer: machine, run state, speed, networking, on-screen
    /// scale, the capture hint and the latest notification.
    fn window_title(&self) -> String {
        use iris::config::NetMode;
        let mut parts: Vec<String> = Vec::new();
        let name = self.prefs.active_machine.as_deref().unwrap_or("(unsaved)");
        parts.push(format!("{name}{}", if self.cfg_dirty { " *" } else { "" }));

        let running = self.emu.is_running();
        let halted = running && self.emu.status.cpu_halted;
        parts.push(
            if running && self.emu.status.cpu_stopped {
                "powered off"
            } else if halted {
                "halted \u{2014} safe to stop"
            } else if self.emu.status.in_prom {
                "PROM"
            } else if running {
                "IRIX running"
            } else {
                "stopped"
            }
            .to_string(),
        );
        if running && !halted {
            // Instructions per wall-clock second on the host: real emulation
            // speed, not the PROM's inventory "MHz".
            parts.push(format!("{:.0} MIPS", self.emu.status.mips));
        }
        if running {
            // The backend is latched at Start; "idle" means the running guest
            // has produced no IP traffic yet.
            let backend = match self.launched_net.as_ref() {
                Some((NetMode::Pcap, iface)) => {
                    let iface = iface.as_deref().filter(|s| !s.is_empty()).unwrap_or("auto");
                    format!("PCAP {iface}")
                }
                _ => "NAT".to_string(),
            };
            let live = match self.emu.net_state() {
                NetState::Active => "",
                NetState::Idle => " idle",
                NetState::Off => " off",
            };
            parts.push(format!("net {backend}{live}"));
            if self.fb_scale > 0.0 {
                // "filtered": not a whole device-pixel multiple, so smoothed.
                let filtered = if self.fb_nearest { "" } else { " filtered" };
                parts.push(format!("{}{filtered}", menus::scale_label(self.fb_scale)));
            }
        }
        if self.input_state.captured {
            parts.push(format!("captured \u{2014} {} to release", input::RELEASE_HINT));
        }
        if let Some((msg, when)) = &self.toast {
            if when.elapsed().as_secs() < 5 {
                parts.push(msg.clone());
            }
        }
        format!("IRIS \u{2014} {}", parts.join("  \u{00b7}  "))
    }

    /// Push the title at about 4 Hz, and only when it changed. Most of its
    /// fields move constantly (MIPS especially), and retitling every frame is
    /// wasted work and visibly jittery.
    fn sync_window_title(&mut self, ctx: &e::Context) {
        // The default layout expires the toast in its status footer.
        if self.toast.as_ref().is_some_and(|(_, when)| when.elapsed().as_secs() >= 5) {
            self.toast = None;
        }
        if self.native.title_at.elapsed() < Duration::from_millis(250) {
            return;
        }
        self.native.title_at = Instant::now();
        let title = self.window_title();
        if title != self.native.title {
            self.native.title = title.clone();
            ctx.send_viewport_cmd_to(e::ViewportId::ROOT, ViewportCommand::Title(title));
        }
        // A stopped machine asks for no frames, so wake up to expire the toast.
        if self.toast.is_some() {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }
}
