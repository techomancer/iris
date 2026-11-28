//! The guest display: a widget of its own that draws the frame sink's
//! picture at the size the display controller's rule gives, and turns the
//! keyboard and pointer over it into display actions.
//!
//! The sink publishes damaged bands, so the shell keeps the whole picture
//! and copies each band into it, then hands GTK a texture of it.
//!
//! **Pointer.** GTK 4 offers no pointer lock or warp of its own, and Wayland
//! gives one only through protocols it does not expose, so this shell steers
//! the guest's pointer to the host's instead — the core's Follow mode, which
//! the plan names for exactly this case. The keyboard reaches the guest while
//! the display has the focus.

use std::cell::{Cell, RefCell};

use adw::prelude::*;
use gtk::subclass::prelude::*;
use gtk::{gdk, glib, graphene, gsk};
use gtk4 as gtk;
use iris_gui_core::controller::display::{DisplayAction, FrameSink, Sizing};
use libadwaita as adw;

mod imp {
    use super::*;

    pub struct GuestView {
        pub texture: RefCell<Option<gdk::Texture>>,
        /// The guest's own size in pixels.
        pub px: Cell<(f32, f32)>,
        pub sizing: Cell<Sizing>,
        /// Sample nearest rather than linear (the filter preference).
        pub nearest: Cell<bool>,
        /// Where the picture was last drawn, so a pointer position can be
        /// turned back into a guest pixel.
        pub rect: Cell<(f32, f32, f32, f32)>,
    }

    impl Default for GuestView {
        fn default() -> Self {
            Self {
                texture: RefCell::default(),
                px: Cell::new((0.0, 0.0)),
                sizing: Cell::new(Sizing::Fit { integer_only: false }),
                nearest: Cell::new(true),
                rect: Cell::new((0.0, 0.0, 0.0, 0.0)),
            }
        }
    }

    #[glib::object_subclass]
    impl ObjectSubclass for GuestView {
        const NAME: &'static str = "IrisGuestView";
        type Type = super::GuestView;
        type ParentType = gtk::Widget;
    }

    impl ObjectImpl for GuestView {
        fn constructed(&self) {
            self.parent_constructed();
            self.obj().set_focusable(true);
            self.obj().set_can_focus(true);
        }
    }

    impl WidgetImpl for GuestView {
        /// In Actual Size the view asks for the guest's own size, so the
        /// scrolled window gives it scroll bars; otherwise it takes what it
        /// is given.
        fn measure(&self, orientation: gtk::Orientation, _for_size: i32) -> (i32, i32, i32, i32) {
            let (w, h) = self.px.get();
            let want = match (self.sizing.get(), orientation) {
                (Sizing::Actual, gtk::Orientation::Horizontal) => w as i32,
                (Sizing::Actual, _) => h as i32,
                (_, gtk::Orientation::Horizontal) => 160,
                (_, _) => 120,
            };
            (want.max(1), want.max(1), -1, -1)
        }

        fn snapshot(&self, snapshot: &gtk::Snapshot) {
            let obj = self.obj();
            let (vw, vh) = (obj.width() as f32, obj.height() as f32);
            let Some(texture) = self.texture.borrow().clone() else {
                self.rect.set((0.0, 0.0, 0.0, 0.0));
                return;
            };
            let px = self.px.get();
            if px.0 < 1.0 || px.1 < 1.0 {
                return;
            }
            let ppp = obj.scale_factor().max(1) as f32;
            let (size, _) = iris_gui_core::controller::display::picture_size(self.sizing.get(), (vw, vh), px, ppp);
            // Centred, with the black of the screen around it.
            let (x, y) = (((vw - size.0) / 2.0).max(0.0), ((vh - size.1) / 2.0).max(0.0));
            self.rect.set((x, y, size.0, size.1));
            let bounds = graphene::Rect::new(x, y, size.0, size.1);
            let filter = if self.nearest.get() { gsk::ScalingFilter::Nearest } else { gsk::ScalingFilter::Linear };
            snapshot.append_scaled_texture(&texture, filter, &bounds);
        }
    }
}

glib::wrapper! {
    /// The guest's screen.
    pub struct GuestView(ObjectSubclass<imp::GuestView>)
        @extends gtk::Widget,
        @implements gtk::Accessible, gtk::Buildable, gtk::ConstraintTarget;
}

impl Default for GuestView {
    fn default() -> Self {
        Self::new()
    }
}

impl GuestView {
    pub fn new() -> Self {
        let view: Self = glib::Object::new();
        view.add_css_class("iris-screen");
        view.wire_input();
        view
    }

    /// Where the picture is drawn now, in widget points.
    fn rect(&self) -> (f32, f32, f32, f32) {
        self.imp().rect.get()
    }

    /// The guest pixel under widget point (x, y), when the pointer is over
    /// the picture at all.
    fn guest_pixel(&self, x: f64, y: f64) -> Option<(i32, i32)> {
        let (rx, ry, rw, rh) = self.rect();
        if rw <= 0.0 || rh <= 0.0 {
            return None;
        }
        let (px, py) = self.imp().px.get();
        let u = (x as f32 - rx) / rw;
        let v = (y as f32 - ry) / rh;
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        Some(((u * px).round() as i32, (v * py).round() as i32))
    }

    fn wire_input(&self) {
        // The keyboard, while the display has the focus.
        let keys = gtk::EventControllerKey::new();
        keys.connect_key_pressed(glib::clone!(
            #[weak(rename_to = view)]
            self,
            #[upgrade_or]
            glib::Propagation::Proceed,
            move |_, _keyval, code, _state| view.key(code, true)
        ));
        keys.connect_key_released(glib::clone!(
            #[weak(rename_to = view)]
            self,
            move |_, _keyval, code, _state| {
                view.key(code, false);
            }
        ));
        self.add_controller(keys);

        // Follow mode: wherever the host pointer is over the picture, the
        // guest's goes too.
        let motion = gtk::EventControllerMotion::new();
        motion.connect_motion(glib::clone!(
            #[weak(rename_to = view)]
            self,
            move |c, x, y| view.pointer(x, y, buttons_of(c.current_event_state()), 0.0)
        ));
        self.add_controller(motion);

        let click = gtk::GestureClick::new();
        click.set_button(0);
        click.connect_pressed(glib::clone!(
            #[weak(rename_to = view)]
            self,
            move |g, _, x, y| {
                view.grab_focus();
                view.pointer(x, y, buttons_of(g.current_event_state()) | button_bit(g.current_button()), 0.0);
            }
        ));
        click.connect_released(glib::clone!(
            #[weak(rename_to = view)]
            self,
            move |g, _, x, y| {
                let held = buttons_of(g.current_event_state()) & !button_bit(g.current_button());
                view.pointer(x, y, held, 0.0);
            }
        ));
        self.add_controller(click);

        let scroll = gtk::EventControllerScroll::new(gtk::EventControllerScrollFlags::BOTH_AXES);
        scroll.connect_scroll(glib::clone!(
            #[weak(rename_to = view)]
            self,
            #[upgrade_or]
            glib::Propagation::Proceed,
            move |c, _dx, dy| {
                // A wheel notch is a line; the guest's pointer stays where
                // it is, so the wheel goes to the middle of the picture.
                view.wheel(-dy as f32, buttons_of(c.current_event_state()));
                glib::Propagation::Stop
            }
        ));
        self.add_controller(scroll);

        let focus = gtk::EventControllerFocus::new();
        focus.connect_enter(|_| crate::app::display(DisplayAction::Focus(true)));
        focus.connect_leave(|_| crate::app::display(DisplayAction::Focus(false)));
        self.add_controller(focus);
    }

    fn key(&self, code: u32, down: bool) -> glib::Propagation {
        match crate::keys::guest_key(code) {
            Some(key) => {
                crate::app::display(DisplayAction::Key { key, down, repeat: false });
                glib::Propagation::Stop
            }
            None => glib::Propagation::Proceed,
        }
    }

    fn pointer(&self, x: f64, y: f64, buttons: u8, wheel: f32) {
        let Some((gx, gy)) = self.guest_pixel(x, y) else { return };
        crate::app::display(DisplayAction::PointerAt {
            x: gx,
            y: gy,
            buttons,
            wheel_lines: wheel,
            wheel_points: wheel * 16.0,
        });
    }

    /// The wheel turned: the pointer stays where the guest has it, so only
    /// the wheel moves.
    fn wheel(&self, lines: f32, buttons: u8) {
        let (px, py) = self.imp().px.get();
        let (gx, gy) = ((px / 2.0) as i32, (py / 2.0) as i32);
        crate::app::display(DisplayAction::PointerAt {
            x: gx,
            y: gy,
            buttons,
            wheel_lines: lines,
            wheel_points: lines * 16.0,
        });
    }

    /// How the picture is sized and sampled now.
    pub fn set_look(&self, sizing: Sizing, nearest: bool) {
        let imp = self.imp();
        if imp.sizing.get() != sizing {
            imp.sizing.set(sizing);
            self.queue_resize();
        }
        if imp.nearest.get() != nearest {
            imp.nearest.set(nearest);
            self.queue_draw();
        }
    }

    /// Forget the picture (a machine stopped or started).
    pub fn clear(&self) {
        *self.imp().texture.borrow_mut() = None;
        self.imp().px.set((0.0, 0.0));
        self.queue_draw();
    }

    pub fn has_picture(&self) -> bool {
        self.imp().texture.borrow().is_some()
    }
}

/// The whole picture, with the sink's bands copied into it.
pub struct Screen {
    pub view: GuestView,
    rgba: Vec<u8>,
    size: (usize, usize),
}

impl Default for Screen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen {
    pub fn new() -> Self {
        Self { view: GuestView::new(), rgba: Vec::new(), size: (0, 0) }
    }

    pub fn clear(&mut self) {
        self.rgba.clear();
        self.size = (0, 0);
        self.view.clear();
    }

    /// Take whatever the sink has published and show it. `true` when the
    /// picture changed.
    pub fn take(&mut self, sink: &FrameSink) -> bool {
        let Some(frame) = sink.take() else { return false };
        let (w, h) = (frame.width, frame.height);
        if w == 0 || h == 0 {
            sink.recycle(frame);
            return false;
        }
        let full = self.size != (w, h);
        if full {
            self.rgba = vec![0; w * h * 4];
            self.size = (w, h);
        }
        let (y, rows) = if full || frame.dirty_h == 0 || frame.dirty_h as usize >= h {
            (0usize, h)
        } else {
            (frame.dirty_y as usize, frame.dirty_h as usize)
        };
        let start = y * w * 4;
        let len = rows * w * 4;
        if let (Some(src), Some(dst)) = (frame.rgba.get(start..start + len), self.rgba.get_mut(start..start + len)) {
            dst.copy_from_slice(src);
        }
        sink.recycle(frame);

        // GTK takes a texture, not a buffer: one copy per frame. A partial
        // update needs GdkMemoryTextureBuilder, which is GTK 4.16 and above.
        let bytes = glib::Bytes::from(&self.rgba[..]);
        let texture =
            gdk::MemoryTexture::new(w as i32, h as i32, gdk::MemoryFormat::R8g8b8a8, &bytes, w * 4);
        let imp = self.view.imp();
        *imp.texture.borrow_mut() = Some(texture.upcast());
        let was = imp.px.replace((w as f32, h as f32));
        if was != (w as f32, h as f32) {
            self.view.queue_resize();
        }
        self.view.queue_draw();
        true
    }
}

/// Which mouse buttons a modifier state holds (bit 0 left, 1 right, 2 middle).
fn buttons_of(state: gdk::ModifierType) -> u8 {
    let mut b = 0;
    if state.contains(gdk::ModifierType::BUTTON1_MASK) {
        b |= 1;
    }
    if state.contains(gdk::ModifierType::BUTTON3_MASK) {
        b |= 2;
    }
    if state.contains(gdk::ModifierType::BUTTON2_MASK) {
        b |= 4;
    }
    b
}

fn button_bit(button: u32) -> u8 {
    match button {
        1 => 1,
        3 => 2,
        2 => 4,
        _ => 0,
    }
}
