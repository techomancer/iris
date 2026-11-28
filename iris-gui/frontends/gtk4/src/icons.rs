//! The shared icon set (D11, D25) as GTK paintables.
//!
//! The art is `iris-gui/assets/icons/*.svg`, black line work on transparent,
//! baked into `iris-gui-core` (`Icon::svg`, `MachineIcon::svg`) so every shell
//! draws the same glyphs. It is rendered with resvg — GTK loads SVG through
//! gdk-pixbuf, whose SVG loader is a separate package on several distributions,
//! and a missing one would mean no icons at all.
//!
//! What is rendered is a mask: [`SvgArt`] implements `GtkSymbolicPaintable`,
//! so GTK hands it the colour the widget is drawn in and the glyph follows the
//! theme, the light and dark appearance, and a dimmed or insensitive control,
//! with no redraw of ours.

use std::cell::{Cell, RefCell};

use gtk::prelude::*;
use gtk::subclass::prelude::*;
use gtk::{gdk, glib, graphene, gsk};
use gtk4 as gtk;
use iris_gui_core::controller::icons::Icon;
use iris_gui_core::controller::profile::MachineIcon;

mod imp {
    use super::*;

    #[derive(Default)]
    pub struct SvgArt {
        pub svg: RefCell<String>,
        /// The size the art is laid out at, in logical pixels.
        pub size: Cell<i32>,
        /// The last mask rendered, with the pixel size it was rendered for.
        pub mask: RefCell<Option<(u32, u32, gdk::Texture)>>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for SvgArt {
        const NAME: &'static str = "IrisSvgArt";
        type Type = super::SvgArt;
        type Interfaces = (gdk::Paintable, gtk::SymbolicPaintable);
    }

    impl ObjectImpl for SvgArt {}

    impl PaintableImpl for SvgArt {
        fn intrinsic_width(&self) -> i32 {
            self.size.get()
        }

        fn intrinsic_height(&self) -> i32 {
            self.size.get()
        }

        fn flags(&self) -> gdk::PaintableFlags {
            // The size never changes; the contents do, because the symbolic
            // colour comes from the widget drawing it.
            gdk::PaintableFlags::STATIC_SIZE
        }

        /// Drawn by something that has no symbolic colour to give (a plain
        /// `GtkPicture`): take the window's foreground.
        fn snapshot(&self, snapshot: &gdk::Snapshot, width: f64, height: f64) {
            self.obj().snapshot_symbolic(snapshot, width, height, &[gdk::RGBA::new(0.0, 0.0, 0.0, 1.0)]);
        }
    }

    impl SymbolicPaintableImpl for SvgArt {
        fn snapshot_symbolic(&self, snapshot: &gdk::Snapshot, width: f64, height: f64, colors: &[gdk::RGBA]) {
            if width <= 0.0 || height <= 0.0 {
                return;
            }
            let Some(snapshot) = snapshot.downcast_ref::<gtk::Snapshot>() else { return };
            let Some(mask) = self.mask(width, height) else { return };
            let bounds = graphene::Rect::new(0.0, 0.0, width as f32, height as f32);
            // The glyph's alpha masks a flat fill of the widget's own colour.
            snapshot.push_mask(gsk::MaskMode::Alpha);
            snapshot.append_texture(&mask, &bounds);
            snapshot.pop();
            snapshot.append_color(colors.first().unwrap_or(&gdk::RGBA::BLACK), &bounds);
            snapshot.pop();
        }
    }

    impl SvgArt {
        /// The art rendered at this size, from the cache when it fits.
        fn mask(&self, width: f64, height: f64) -> Option<gdk::Texture> {
            let (w, h) = (width.round().max(1.0) as u32, height.round().max(1.0) as u32);
            if let Some((cw, ch, tex)) = self.mask.borrow().as_ref() {
                if *cw == w && *ch == h {
                    return Some(tex.clone());
                }
            }
            let tex = super::render(&self.svg.borrow(), w, h)?;
            *self.mask.borrow_mut() = Some((w, h, tex.clone()));
            Some(tex)
        }
    }
}

glib::wrapper! {
    /// One piece of the shared line art, drawn in the widget's own colour.
    pub struct SvgArt(ObjectSubclass<imp::SvgArt>) @implements gdk::Paintable, gtk::SymbolicPaintable;
}

impl SvgArt {
    pub fn new(svg: &str, size: i32) -> Self {
        let art: Self = glib::Object::new();
        let imp = art.imp();
        *imp.svg.borrow_mut() = svg.to_string();
        imp.size.set(size.max(1));
        art
    }
}

/// Render `svg` into a `w` x `h` texture. Premultiplied RGBA, as tiny-skia
/// leaves it and as `MemoryFormat::R8g8b8a8Premultiplied` reads it.
fn render(svg: &str, w: u32, h: u32) -> Option<gdk::Texture> {
    use resvg::{tiny_skia, usvg};
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).ok()?;
    let mut pixmap = tiny_skia::Pixmap::new(w, h)?;
    let size = tree.size();
    let scale = (w as f32 / size.width()).min(h as f32 / size.height());
    // Centred, so a non-square box does not shift the glyph into a corner.
    let dx = (w as f32 - size.width() * scale) / 2.0;
    let dy = (h as f32 - size.height() * scale) / 2.0;
    resvg::render(&tree, tiny_skia::Transform::from_translate(dx, dy).pre_scale(scale, scale), &mut pixmap.as_mut());
    let bytes = glib::Bytes::from_owned(pixmap.take());
    Some(gdk::MemoryTexture::new(w as i32, h as i32, gdk::MemoryFormat::R8g8b8a8Premultiplied, &bytes, w as usize * 4).upcast())
}

/// A control's icon at `size` logical pixels, with its tooltip as the
/// accessible name (D11: the accessibility label is the tooltip).
// Used from the chrome and the library onwards.
#[allow(dead_code)]
pub fn image(icon: Icon, size: i32, description: &str) -> gtk::Image {
    let image = gtk::Image::from_paintable(Some(&SvgArt::new(icon.svg(), size)));
    image.set_pixel_size(size);
    if !description.is_empty() {
        image.update_property(&[gtk::accessible::Property::Label(description)]);
    }
    image
}

/// A machine's front view at `size` logical pixels.
#[allow(dead_code)]
pub fn machine(icon: MachineIcon, size: i32) -> gtk::Image {
    let image = gtk::Image::from_paintable(Some(&SvgArt::new(icon.svg(), size)));
    image.set_pixel_size(size);
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes every icon into one PNG for a look, since a glyph is only right
    /// if you can see it:
    ///   cargo test -p iris-gui-gtk --  --ignored --nocapture icon_sheet
    /// IRIS_ICON_SHEET names the file (default target/icon-sheet.png).
    #[test]
    #[ignore = "writes a picture for a human to look at"]
    fn icon_sheet() {
        use resvg::{tiny_skia, usvg};
        const CELL: u32 = 64;
        const PAD: u32 = 16;
        const COLS: u32 = 8;
        let icons = Icon::ALL;
        let rows = (icons.len() as u32).div_ceil(COLS);
        let (w, h) = (COLS * (CELL + PAD) + PAD, rows * (CELL + PAD) + PAD);
        let mut sheet = tiny_skia::Pixmap::new(w, h).expect("a sheet");
        sheet.fill(tiny_skia::Color::from_rgba8(0xff, 0xff, 0xff, 0xff));
        for (n, icon) in icons.iter().enumerate() {
            let tree = usvg::Tree::from_str(icon.svg(), &usvg::Options::default()).expect(icon.svg_name());
            let mut cell = tiny_skia::Pixmap::new(CELL, CELL).expect("a cell");
            let s = CELL as f32 / tree.size().width();
            resvg::render(&tree, tiny_skia::Transform::from_scale(s, s), &mut cell.as_mut());
            let x = PAD + (n as u32 % COLS) * (CELL + PAD);
            let y = PAD + (n as u32 / COLS) * (CELL + PAD);
            sheet.draw_pixmap(
                x as i32,
                y as i32,
                cell.as_ref(),
                &tiny_skia::PixmapPaint::default(),
                tiny_skia::Transform::identity(),
                None,
            );
        }
        let path = std::env::var("IRIS_ICON_SHEET").unwrap_or_else(|_| "target/icon-sheet.png".into());
        sheet.save_png(&path).expect("the sheet saves");
        println!("icon sheet: {path} ({} icons)", icons.len());
    }
}
