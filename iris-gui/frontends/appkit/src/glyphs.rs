//! The glyph for each icon meaning (D11): SF Symbols on macOS 11 and later.
//! Below 11 there are no symbols; controls fall back to their titles. The
//! machine pictures are the shared line art in `iris-gui/assets/icons/`,
//! drawn here with resvg, so they show on every version.

use std::cell::RefCell;
use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::AnyThread;
use objc2_app_kit::{NSBitmapImageRep, NSDeviceRGBColorSpace, NSImage};
use objc2_foundation::{NSCopying, NSSize};
use iris_gui_core::controller::icons::Icon;
use iris_gui_core::controller::profile::MachineIcon;

use crate::ui::{has_symbols, ns};

pub fn symbol_name(icon: Icon) -> &'static str {
    match icon {
        Icon::ShowLibrary => "sidebar.left",
        Icon::PowerOn => "play.fill",
        Icon::ShutDownGuest => "power",
        Icon::PowerOff => "stop.fill",
        Icon::Restart => "arrow.clockwise",
        Icon::Reset => "exclamationmark.arrow.circlepath",
        // A snapshot of the machine, as Time Machine draws one.
        Icon::SaveState => "externaldrive.badge.timemachine",
        Icon::SavedStates => "square.stack.3d.up",
        Icon::Screenshot => "camera",
        Icon::InsertDisc => "opticaldisc",
        Icon::BrowseFiles => "folder",
        Icon::DetachDisplay => "macwindow.on.rectangle",
        Icon::FullScreen => "arrow.up.left.and.arrow.down.right",
        Icon::SerialConsole => "terminal",
        Icon::MachineSettings => "slider.horizontal.3",
        Icon::NewMachine => "plus",
        Icon::NewFolder => "folder.badge.plus",
        Icon::Folder => "folder",
        Icon::Disk => "internaldrive",
        Icon::CdRom => "opticaldisc",
        Icon::Network => "network",
        Icon::Audio => "speaker.wave.2",
        Icon::Camera => "video",
        Icon::InputCapture => "keyboard",
        Icon::DisplayScale => "display",
        Icon::Warning => "exclamationmark.triangle",
        Icon::Error => "xmark.octagon",
        Icon::Info => "info.circle",
        Icon::Delete => "trash",
        Icon::Rename => "pencil",
        Icon::Restore => "clock.arrow.circlepath",
        Icon::Export => "square.and.arrow.up",
        Icon::Import => "square.and.arrow.down",
        Icon::Edit => "doc.text",
        Icon::Permissions => "lock",
        Icon::Check => "checkmark.seal",
        Icon::Up => "arrow.up",
        Icon::Refresh => "arrow.clockwise",
        Icon::Cancel => "xmark",
        Icon::Duplicate => "plus.square.on.square",
    }
}

thread_local! {
    static MACHINES: RefCell<HashMap<(&'static str, u32), Retained<NSImage>>> = RefCell::new(HashMap::new());
}

/// The machine's front view, `points` square: the shared line art
/// (`MachineIcon::svg`) drawn at 1x and 2x into a template image, so an image
/// view's tint colours it. Works on every macOS the app supports.
pub fn machine_image(icon: MachineIcon, points: f64, description: &str) -> Option<Retained<NSImage>> {
    let key = (icon.svg_name(), points.round() as u32);
    let img = match MACHINES.with(|m| m.borrow().get(&key).cloned()) {
        Some(img) => img,
        None => {
            let img = render_template(icon.svg(), points)?;
            MACHINES.with(|m| m.borrow_mut().insert(key, img.clone()));
            img
        }
    };
    // A copy, so each view carries its own description.
    let copy = img.copy();
    // SAFETY: `copy` of an NSImage is an NSImage.
    let copy: Retained<NSImage> = unsafe { Retained::cast_unchecked(copy) };
    copy.setAccessibilityDescription(Some(&ns(description)));
    Some(copy)
}

fn render_template(svg: &str, points: f64) -> Option<Retained<NSImage>> {
    use resvg::{tiny_skia, usvg};
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).ok()?;
    let size = NSSize::new(points, points);
    let image = NSImage::initWithSize(NSImage::alloc(), size);
    for scale in [1.0, 2.0] {
        let px = (points * scale).round().max(1.0) as u32;
        let mut pixmap = tiny_skia::Pixmap::new(px, px)?;
        let s = px as f32 / tree.size().width().max(tree.size().height());
        resvg::render(&tree, tiny_skia::Transform::from_scale(s, s), &mut pixmap.as_mut());
        // SAFETY: a fresh 8-bit RGBA rep with its own buffer, `px` rows of
        // `px * 4` bytes, which is exactly the pixmap's (premultiplied) layout.
        let rep = unsafe {
            let rep = NSBitmapImageRep::initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel(
                NSBitmapImageRep::alloc(),
                std::ptr::null_mut(),
                px as isize,
                px as isize,
                8,
                4,
                true,
                false,
                NSDeviceRGBColorSpace,
                (px * 4) as isize,
                32,
            )?;
            let dst = rep.bitmapData();
            if dst.is_null() {
                return None;
            }
            std::ptr::copy_nonoverlapping(pixmap.data().as_ptr(), dst, pixmap.data().len());
            rep
        };
        rep.setSize(size);
        image.addRepresentation(&rep);
    }
    image.setTemplate(true);
    Some(image)
}

/// The image for `name`, `None` below macOS 11.
pub fn symbol(name: &str, description: &str) -> Option<Retained<NSImage>> {
    if !has_symbols() {
        return None;
    }
    NSImage::imageWithSystemSymbolName_accessibilityDescription(&ns(name), Some(&ns(description)))
}

/// The symbol `name` drawn at `points`, `None` below macOS 11.
pub fn symbol_at(name: &str, points: f64) -> Option<Retained<NSImage>> {
    let img = symbol(name, "")?;
    let config = objc2_app_kit::NSImageSymbolConfiguration::configurationWithPointSize_weight(points, 0.0);
    img.imageWithSymbolConfiguration(&config)
}

pub fn icon(icon: Icon, description: &str) -> Option<Retained<NSImage>> {
    symbol(symbol_name(icon), description)
}
