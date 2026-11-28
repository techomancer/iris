//! The guest display (D4): a `CAMetalLayer`-backed view fed by the frame
//! sink's damaged bands (`replaceRegion` per band), drawn as one textured quad
//! with nearest or linear sampling at the size the display controller's
//! `draw_size` rule gives. The view also takes the keyboard and mouse while
//! the controller says input is captured.

use std::cell::{Cell, RefCell};
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::{define_class, msg_send, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSCursor, NSEvent, NSEventModifierFlags, NSView};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLDrawable, MTLLibrary, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLSize, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage, MTLViewport,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use iris_gui_core::controller::display::{DisplayAction, FrameSink};
use iris_gui_core::controller::keys::Mods;

use crate::keys;

extern "C" {
    /// CoreGraphics: decouple the cursor from the mouse while captured, so
    /// motion arrives as deltas and the (hidden) cursor stays put.
    fn CGAssociateMouseAndMouseCursorPosition(connected: u32) -> i32;
}

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct V { float4 pos [[position]]; float2 uv; };
vertex V vs(uint id [[vertex_id]]) {
    float2 p = float2((id & 1) ? 1.0 : -1.0, (id & 2) ? -1.0 : 1.0);
    V v; v.pos = float4(p, 0.0, 1.0); v.uv = float2((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5); return v;
}
fragment float4 fs(V v [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {
    return float4(t.sample(s, v.uv).rgb, 1.0);
}
"#;

type Device = ProtocolObject<dyn MTLDevice>;

/// The Metal objects shared by every display view.
struct Gpu {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    nearest: Retained<ProtocolObject<dyn MTLSamplerState>>,
    linear: Retained<ProtocolObject<dyn MTLSamplerState>>,
}

fn gpu() -> Option<Gpu> {
    let device = MTLCreateSystemDefaultDevice()?;
    let queue = device.newCommandQueue()?;
    let lib = device.newLibraryWithSource_options_error(&NSString::from_str(SHADER), None).ok()?;
    let vs = lib.newFunctionWithName(&NSString::from_str("vs"))?;
    let fs = lib.newFunctionWithName(&NSString::from_str("fs"))?;
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vs));
    desc.setFragmentFunction(Some(&fs));
    // SAFETY: index 0 always exists on a pipeline descriptor.
    unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) }.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    let pipeline = device.newRenderPipelineStateWithDescriptor_error(&desc).ok()?;
    let sampler = |f: MTLSamplerMinMagFilter| {
        let d = MTLSamplerDescriptor::new();
        d.setMinFilter(f);
        d.setMagFilter(f);
        device.newSamplerStateWithDescriptor(&d)
    };
    let nearest = sampler(MTLSamplerMinMagFilter::Nearest)?;
    let linear = sampler(MTLSamplerMinMagFilter::Linear)?;
    Some(Gpu { device, queue, pipeline, nearest, linear })
}

/// What a display view keeps between frames.
pub struct State {
    gpu: Option<Gpu>,
    layer: Retained<CAMetalLayer>,
    /// Each head's picture: head 0, and head 1 with a second Newport.
    textures: [RefCell<Option<(Retained<ProtocolObject<dyn MTLTexture>>, usize, usize)>>; 2],
    /// How many heads to draw: two while the session has a second Newport
    /// (`SessionController::dual_head`).
    heads: Cell<usize>,
    /// Over head 1 until it has drawn anything (`head1_live`).
    head1_label: RefCell<Option<Retained<objc2_app_kit::NSTextField>>>,
    /// The scale cap (points per guest pixel), integer-only and filter, from
    /// the app-wide display preferences.
    /// How the picture is sized, and a forced filter (`None`: sharp at
    /// whole multiples, smooth between).
    sizing: Cell<(iris_gui_core::controller::display::Sizing, Option<bool>)>,
    captured: Cell<bool>,
    /// Follow mode: the guest pointer follows the host's, with no capture.
    follow: Cell<bool>,
    buttons: Cell<u8>,
    /// Text shown when there is no picture (powered off, starting).
    overlay: RefCell<Option<Retained<objc2_app_kit::NSTextField>>>,
}

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "IRISDisplayView"]
    #[ivars = State]
    pub struct DisplayView;

    unsafe impl NSObjectProtocol for DisplayView {}

    impl DisplayView {
        #[unsafe(method(acceptsFirstResponder))]
        fn accepts_first_responder(&self) -> bool {
            true
        }

        #[unsafe(method(isOpaque))]
        fn is_opaque(&self) -> bool {
            true
        }

        #[unsafe(method(setFrameSize:))]
        fn set_frame_size(&self, size: NSSize) {
            // SAFETY: NSView's own setter.
            let _: () = unsafe { msg_send![super(self), setFrameSize: size] };
            self.resize_drawable();
            self.place_head1_label();
            self.draw();
        }

        #[unsafe(method(viewDidChangeBackingProperties))]
        fn backing_changed(&self) {
            self.resize_drawable();
            self.draw();
        }

        #[unsafe(method(mouseDown:))]
        fn mouse_down(&self, e: &NSEvent) {
            if self.ivars().follow.get() {
                // Follow mode: a click is a click, and the display takes the keyboard.
                if let Some(w) = self.window() {
                    w.makeFirstResponder(Some(self));
                }
                self.button(e, 0x01, true);
            } else if self.ivars().captured.get() {
                self.button(e, 0x01, true);
            } else {
                crate::app::display(DisplayAction::Capture);
            }
        }

        #[unsafe(method(mouseUp:))]
        fn mouse_up(&self, e: &NSEvent) {
            self.button(e, 0x01, false);
        }

        #[unsafe(method(rightMouseDown:))]
        fn right_down(&self, e: &NSEvent) {
            self.button(e, 0x02, true);
        }

        #[unsafe(method(rightMouseUp:))]
        fn right_up(&self, e: &NSEvent) {
            self.button(e, 0x02, false);
        }

        #[unsafe(method(otherMouseDown:))]
        fn other_down(&self, e: &NSEvent) {
            self.button(e, other_bit(e), true);
        }

        #[unsafe(method(otherMouseUp:))]
        fn other_up(&self, e: &NSEvent) {
            self.button(e, other_bit(e), false);
        }

        #[unsafe(method(mouseMoved:))]
        fn mouse_moved(&self, e: &NSEvent) {
            self.motion(e);
        }

        #[unsafe(method(mouseDragged:))]
        fn mouse_dragged(&self, e: &NSEvent) {
            self.motion(e);
        }

        #[unsafe(method(rightMouseDragged:))]
        fn right_dragged(&self, e: &NSEvent) {
            self.motion(e);
        }

        #[unsafe(method(otherMouseDragged:))]
        fn other_dragged(&self, e: &NSEvent) {
            self.motion(e);
        }

        #[unsafe(method(scrollWheel:))]
        fn scroll_wheel(&self, e: &NSEvent) {
            let dy = e.scrollingDeltaY() as f32;
            let (lines, points) = if e.hasPreciseScrollingDeltas() { (0.0, -dy) } else { (-dy, 0.0) };
            if self.ivars().follow.get() {
                self.follow_at(e, lines, points);
                return;
            }
            if !self.ivars().captured.get() {
                return;
            }
            crate::app::display(DisplayAction::Pointer {
                dx: 0.0,
                dy: 0.0,
                buttons: self.ivars().buttons.get(),
                wheel_lines: lines,
                wheel_points: points,
            });
        }

        #[unsafe(method(keyDown:))]
        fn key_down(&self, e: &NSEvent) {
            self.key(e, true);
        }

        #[unsafe(method(keyUp:))]
        fn key_up(&self, e: &NSEvent) {
            self.key(e, false);
        }

        #[unsafe(method(flagsChanged:))]
        fn flags_changed(&self, e: &NSEvent) {
            let follow = self.ivars().follow.get();
            if !self.ivars().captured.get() && !follow {
                return;
            }
            let code = e.keyCode();
            // Follow mode keeps Command for the Mac's own shortcuts.
            if follow && !self.ivars().captured.get() && matches!(code, 0x36 | 0x37) {
                return;
            }
            let Some(k) = keys::guest_key(code) else { return };
            // Down when the modifier's own flag is now set.
            let flags = e.modifierFlags();
            let down = match code {
                0x36 | 0x37 => flags.contains(NSEventModifierFlags::Command),
                0x38 | 0x3C => flags.contains(NSEventModifierFlags::Shift),
                0x3A | 0x3D => flags.contains(NSEventModifierFlags::Option),
                0x3B | 0x3E => flags.contains(NSEventModifierFlags::Control),
                0x39 => flags.contains(NSEventModifierFlags::CapsLock),
                _ => return,
            };
            // A Mac reports Caps Lock as one press: tap it.
            if code == 0x39 {
                crate::app::display(DisplayAction::Key { key: k, down: true, repeat: false });
                crate::app::display(DisplayAction::Key { key: k, down: false, repeat: false });
                return;
            }
            crate::app::display(DisplayAction::Key { key: k, down, repeat: false });
        }

        /// While captured, Cmd chords belong to the guest, not the menus.
        #[unsafe(method(performKeyEquivalent:))]
        fn perform_key_equivalent(&self, e: &NSEvent) -> objc2::runtime::Bool {
            if self.ivars().captured.get() {
                self.key(e, true);
                // AppKit does not deliver keyUp: for a key released while
                // Command is held, so a chord's key would stay down (and
                // repeat) in the guest: release it at once, as a tap. Without
                // Command the key's own keyUp: comes as usual.
                if e.modifierFlags().contains(NSEventModifierFlags::Command) {
                    self.key(e, false);
                }
                return objc2::runtime::Bool::YES;
            }
            objc2::runtime::Bool::NO
        }
    }
);

fn other_bit(e: &NSEvent) -> u8 {
    match e.buttonNumber() {
        2 => 0x04,
        3 => 0x08,
        4 => 0x10,
        _ => 0,
    }
}

/// The gap between two heads, in points.
const HEAD_GAP: f32 = 8.0;

/// Where each head is drawn in a view of `avail` points, as (origin, size)
/// from the top left: one head fills the view; two sit side by side in equal
/// columns, [`HEAD_GAP`] apart, as egui's `framebuffer_panel_dual` lays them
/// out. Each picture is then sized and centred in its area. Only head 0
/// takes the pointer.
fn head_areas(avail: (f32, f32), heads: usize) -> Vec<((f32, f32), (f32, f32))> {
    if heads < 2 {
        return vec![((0.0, 0.0), avail)];
    }
    let w = ((avail.0 - HEAD_GAP) / 2.0).max(0.0);
    vec![((0.0, 0.0), (w, avail.1)), ((w + HEAD_GAP, 0.0), (w, avail.1))]
}

/// The view size Actual Size wants for heads of these pixel sizes: side by
/// side, [`HEAD_GAP`] apart, as tall as the taller.
fn actual_size(heads: &[(f32, f32)]) -> (f32, f32) {
    let w: f32 = heads.iter().map(|h| h.0).sum::<f32>() + HEAD_GAP * heads.len().saturating_sub(1) as f32;
    (w, heads.iter().map(|h| h.1).fold(0.0, f32::max))
}

fn mods(e: &NSEvent) -> Mods {
    let f = e.modifierFlags();
    Mods {
        ctrl: f.contains(NSEventModifierFlags::Control),
        alt: f.contains(NSEventModifierFlags::Option),
        shift: f.contains(NSEventModifierFlags::Shift),
        cmd: f.contains(NSEventModifierFlags::Command),
    }
}

impl DisplayView {
    pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let layer = CAMetalLayer::new();
        let gpu = gpu();
        if let Some(g) = &gpu {
            layer.setDevice(Some(&g.device));
        }
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setFramebufferOnly(true);
        let state = State {
            gpu,
            layer: layer.clone(),
            textures: [RefCell::new(None), RefCell::new(None)],
            heads: Cell::new(1),
            head1_label: RefCell::new(None),
            sizing: Cell::new((iris_gui_core::controller::display::Sizing::Actual, None)),
            captured: Cell::new(false),
            follow: Cell::new(false),
            buttons: Cell::new(0),
            overlay: RefCell::new(None),
        };
        let this = mtm.alloc::<DisplayView>().set_ivars(state);
        // SAFETY: NSView's designated initialiser.
        let v: Retained<DisplayView> =
            unsafe { msg_send![super(this), initWithFrame: NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(640.0, 512.0))] };
        // A layer-hosting view: the layer is set before the view wants one.
        v.setLayer(Some(&layer));
        v.setWantsLayer(true);
        // Cursor moves over the view arrive without capture (Follow mode),
        // whichever view has the keyboard; the area follows the view's size.
        // SAFETY: a tracking area owned by the view it reports to.
        unsafe {
            use objc2_app_kit::{NSTrackingArea, NSTrackingAreaOptions as T};
            let area = NSTrackingArea::initWithRect_options_owner_userInfo(
                mtm.alloc(),
                NSRect::ZERO,
                T::MouseMoved | T::ActiveInKeyWindow | T::InVisibleRect,
                Some(&v),
                None,
            );
            v.addTrackingArea(&area);
        }
        crate::ui::set_id(&v, "display");
        v
    }

    fn resize_drawable(&self) {
        let scale = self.window().map(|w| w.backingScaleFactor()).unwrap_or(2.0);
        let b = self.bounds().size;
        self.ivars().layer.setContentsScale(scale);
        self.ivars().layer.setDrawableSize(NSSize::new((b.width * scale).max(1.0), (b.height * scale).max(1.0)));
    }

    /// How many heads to draw, side by side ([`head_areas`]).
    pub fn set_heads(&self, n: usize) {
        let n = n.clamp(1, 2);
        if self.ivars().heads.replace(n) != n {
            if n == 1 {
                *self.ivars().textures[1].borrow_mut() = None;
            }
            self.fit();
            self.place_head1_label();
            self.draw();
        }
    }

    /// Say over head 1 that it has drawn nothing yet, or take that away.
    pub fn set_head1_waiting(&self, mtm: MainThreadMarker, waiting: bool) {
        let mut slot = self.ivars().head1_label.borrow_mut();
        match (waiting, slot.as_ref()) {
            (true, None) => {
                let l = crate::ui::label(mtm, &t!("fb-head-waiting"));
                l.setTextColor(Some(&objc2_app_kit::NSColor::lightGrayColor()));
                l.setFont(Some(&objc2_app_kit::NSFont::systemFontOfSize(15.0)));
                crate::ui::set_id(&l, "display.head1-waiting");
                self.addSubview(&l);
                *slot = Some(l);
            }
            (false, Some(l)) => {
                l.removeFromSuperview();
                *slot = None;
            }
            _ => {}
        }
        drop(slot);
        self.place_head1_label();
    }

    /// Centre the head-1 label in head 1's area (the view's y grows up).
    fn place_head1_label(&self) {
        let st = self.ivars();
        let slot = st.head1_label.borrow();
        let Some(l) = slot.as_ref() else { return };
        let b = self.bounds().size;
        let areas = head_areas((b.width as f32, b.height as f32), st.heads.get());
        l.setHidden(areas.len() < 2);
        let Some(&((ax, ay), (aw, ah))) = areas.get(1) else { return };
        l.sizeToFit();
        let s = l.frame().size;
        let x = ax as f64 + (aw as f64 - s.width) / 2.0;
        let y = b.height - (ay as f64 + (ah as f64 + s.height) / 2.0);
        l.setFrameOrigin(NSPoint::new(x.round(), y.round()));
    }

    /// Take head 0's newest frame from `sink`; see [`DisplayView::take_head`].
    pub fn take(&self, sink: &FrameSink) -> bool {
        self.take_head(0, sink)
    }

    /// Take the newest frame of head `head` from `sink` and upload its
    /// damaged band. Returns whether there was one.
    pub fn take_head(&self, head: usize, sink: &FrameSink) -> bool {
        let Some(frame) = sink.take() else { return false };
        let st = self.ivars();
        let Some(g) = &st.gpu else {
            sink.recycle(frame);
            return false;
        };
        let (w, h) = (frame.width, frame.height);
        if w == 0 || h == 0 {
            sink.recycle(frame);
            return false;
        }
        let mut tex = st.textures[head.min(1)].borrow_mut();
        let full = match tex.as_ref() {
            Some((_, tw, th)) => *tw != w || *th != h,
            None => true,
        };
        if full {
            // SAFETY: plain texture descriptor values.
            let d = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA8Unorm, w, h, false)
            };
            d.setUsage(MTLTextureUsage::ShaderRead);
            match g.device.newTextureWithDescriptor(&d) {
                Some(t) => *tex = Some((t, w, h)),
                None => {
                    sink.recycle(frame);
                    return false;
                }
            }
        }
        let (y, rows) = if full || frame.dirty_h == 0 || frame.dirty_h as usize >= h {
            (0usize, h)
        } else {
            (frame.dirty_y as usize, frame.dirty_h as usize)
        };
        if let Some((t, _, _)) = tex.as_ref() {
            let region = MTLRegion { origin: MTLOrigin { x: 0, y, z: 0 }, size: MTLSize { width: w, height: rows, depth: 1 } };
            let start = y * w * 4;
            if let Some(bytes) = frame.rgba.get(start..start + rows * w * 4) {
                // SAFETY: `bytes` holds `rows` rows of `w` RGBA pixels, as the region says.
                unsafe {
                    t.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                        region,
                        0,
                        NonNull::new(bytes.as_ptr() as *mut _).unwrap(),
                        w * 4,
                    )
                };
            }
        }
        drop(tex);
        sink.recycle(frame);
        if full {
            // A new guest resolution: Actual Size changes the view's size.
            self.fit();
        }
        true
    }

    /// Forget the picture (a new run starts with the placeholder).
    pub fn clear(&self) {
        for t in &self.ivars().textures {
            *t.borrow_mut() = None;
        }
        self.draw();
    }

    /// What the view draws, for the `dump:display` script step: the head
    /// count, then each head's picture size and its rectangle in the view.
    pub fn describe(&self) -> String {
        let st = self.ivars();
        let b = self.bounds().size;
        let mut out = format!("display: {} head(s) in {:.0}x{:.0} pt", st.heads.get(), b.width, b.height);
        for head in 0..st.heads.get() {
            match st.textures[head].borrow().as_ref() {
                Some((_, w, h)) => {
                    let ((x, y), (dw, dh), _) = self.picture_rect(head, *w as f32, *h as f32);
                    out.push_str(&format!("\n  head {head}: {w}x{h} px at ({x:.0},{y:.0}) {dw:.0}x{dh:.0} pt"));
                }
                None => out.push_str(&format!("\n  head {head}: no picture yet")),
            }
        }
        if let Some(l) = st.head1_label.borrow().as_ref() {
            out.push_str(&format!("\n  head 1 label: {:?}", l.stringValue().to_string()));
        }
        out
    }

    /// The pixel size of each head that has a picture, in head order.
    fn head_sizes(&self) -> Vec<(f32, f32)> {
        let st = self.ivars();
        st.textures[..st.heads.get()]
            .iter()
            .filter_map(|t| t.borrow().as_ref().map(|(_, w, h)| (*w as f32, *h as f32)))
            .collect()
    }

    pub fn set_sizing(&self, sizing: iris_gui_core::controller::display::Sizing, nearest: Option<bool>) {
        if self.ivars().sizing.get() != (sizing, nearest) {
            self.ivars().sizing.set((sizing, nearest));
            self.fit();
            // Fit and Stretch share the view's size, so fit() does not
            // redraw; a still guest screen sends no new frame either.
            self.draw();
        }
    }

    /// Size the view in its scroll view: at Actual Size at least the guest's
    /// size (one pixel to a point; the scroll view scrolls the rest), else
    /// the visible area.
    pub fn fit(&self) {
        // SAFETY: a plain getter; the view is on the main thread.
        let Some(clip) = (unsafe { self.superview() }) else { return };
        let c = clip.bounds().size;
        let mut sizes = self.head_sizes();
        // A head with no picture yet is given the other's size, so the one
        // drawn already keeps its whole column.
        while !sizes.is_empty() && sizes.len() < self.ivars().heads.get() {
            sizes.push(sizes[0]);
        }
        let want = match self.ivars().sizing.get().0 {
            iris_gui_core::controller::display::Sizing::Actual if !sizes.is_empty() => {
                let (w, h) = actual_size(&sizes);
                NSSize::new(c.width.max(w as f64), c.height.max(h as f64))
            }
            _ => c,
        };
        let now = self.frame().size;
        if (now.width - want.width).abs() > 0.5 || (now.height - want.height).abs() > 0.5 {
            self.setFrameSize(want);
            // A picture larger than the window starts at its top-left, as a
            // page does (the view's y grows up, so the top is its far end).
            if want.height > c.height || want.width > c.width {
                self.scrollPoint(NSPoint::new(0.0, (want.height - c.height).max(0.0)));
            }
            self.draw();
        }
    }

    /// Head `head`'s picture rectangle in the view, in points from the top
    /// left: origin and size, centred in the head's area.
    fn picture_rect(&self, head: usize, w: f32, h: f32) -> ((f32, f32), (f32, f32), bool) {
        let b = self.bounds().size;
        let areas = head_areas((b.width as f32, b.height as f32), self.ivars().heads.get());
        let ((ax, ay), (aw, ah)) = areas.get(head).copied().unwrap_or(areas[0]);
        let scale = self.ivars().layer.contentsScale() as f32;
        let (sizing, _) = self.ivars().sizing.get();
        let ((dw, dh), integer) = iris_gui_core::controller::display::picture_size(sizing, (aw, ah), (w, h), scale);
        ((ax + (aw - dw) / 2.0, ay + (ah - dh) / 2.0), (dw, dh), integer)
    }

    /// Draw the texture into the layer's next drawable.
    pub fn draw(&self) {
        let st = self.ivars();
        let Some(g) = &st.gpu else { return };
        // Not on screen (closed, minimized, covered): no drawable may come
        // back, and nextDrawable would hold the main thread up to a second.
        // The window's occlusion change draws again (`app::display_redraw`).
        let shown = self.window().is_some_and(|w| {
            w.isVisible() && w.occlusionState().contains(objc2_app_kit::NSWindowOcclusionState::Visible)
        });
        if !shown {
            return;
        }
        let Some(drawable) = st.layer.nextDrawable() else { return };
        let pass = MTLRenderPassDescriptor::new();
        // SAFETY: index 0 always exists on a render pass descriptor.
        let ca = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        ca.setTexture(Some(&drawable.texture()));
        ca.setLoadAction(MTLLoadAction::Clear);
        ca.setStoreAction(MTLStoreAction::Store);
        ca.setClearColor(MTLClearColor { red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0 });
        let Some(cmd) = g.queue.commandBuffer() else { return };
        let Some(enc) = cmd.renderCommandEncoderWithDescriptor(&pass) else { return };
        for head in 0..st.heads.get() {
            let slot = st.textures[head].borrow();
            let Some((tex, w, h)) = slot.as_ref() else { continue };
            let scale = st.layer.contentsScale() as f32;
            let ((x, y), (dw, dh), integer) = self.picture_rect(head, *w as f32, *h as f32);
            let nearest = st.sizing.get().1.unwrap_or(integer);
            // Centre on the device-pixel grid.
            let ox = (x * scale).round() as f64;
            let oy = (y * scale).round() as f64;
            enc.setViewport(MTLViewport {
                originX: ox,
                originY: oy,
                width: (dw * scale).round() as f64,
                height: (dh * scale).round() as f64,
                znear: 0.0,
                zfar: 1.0,
            });
            enc.setRenderPipelineState(&g.pipeline);
            // SAFETY: slot 0 is what the fragment shader reads.
            unsafe {
                enc.setFragmentTexture_atIndex(Some(tex), 0);
                enc.setFragmentSamplerState_atIndex(Some(if nearest { &g.nearest } else { &g.linear }), 0);
                enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::TriangleStrip, 0, 4);
            }
        }
        enc.endEncoding();
        let d: &ProtocolObject<dyn MTLDrawable> = ProtocolObject::from_ref(&*drawable);
        cmd.presentDrawable(d);
        cmd.commit();
    }

    /// Follow the controller's capture state: hide and lock the cursor while
    /// captured, give it back otherwise.
    pub fn set_captured(&self, on: bool) {
        let st = self.ivars();
        if st.captured.get() == on {
            return;
        }
        st.captured.set(on);
        st.buttons.set(0);
        // SAFETY: plain CoreGraphics calls with no pointers.
        unsafe { CGAssociateMouseAndMouseCursorPosition(if on { 0 } else { 1 }) };
        if on {
            NSCursor::hide();
            if let Some(w) = self.window() {
                w.makeFirstResponder(Some(self));
            }
        } else {
            NSCursor::unhide();
        }
    }

    /// The message over a display with no picture.
    pub fn set_overlay(&self, mtm: MainThreadMarker, text: Option<&str>) {
        let st = self.ivars();
        let mut o = st.overlay.borrow_mut();
        match (text, o.as_ref()) {
            (None, Some(l)) => {
                l.removeFromSuperview();
                *o = None;
            }
            (Some(t), Some(l)) => l.setStringValue(&crate::ui::ns(t)),
            (Some(t), None) => {
                let l = crate::ui::label(mtm, t);
                l.setTextColor(Some(&objc2_app_kit::NSColor::lightGrayColor()));
                l.setFont(Some(&objc2_app_kit::NSFont::systemFontOfSize(18.0)));
                l.setTranslatesAutoresizingMaskIntoConstraints(false);
                self.addSubview(&l);
                l.centerXAnchor().constraintEqualToAnchor(&self.centerXAnchor()).setActive(true);
                l.centerYAnchor().constraintEqualToAnchor(&self.centerYAnchor()).setActive(true);
                *o = Some(l);
            }
            (None, None) => {}
        }
    }

    /// Follow the controller's pointer mode (Follow: no capture, steering).
    pub fn set_follow(&self, on: bool) {
        let st = self.ivars();
        if st.follow.get() != on {
            st.follow.set(on);
            st.buttons.set(0);
        }
    }

    /// The guest pixel under the event's location, clamped to the picture.
    fn guest_point(&self, e: &NSEvent) -> Option<(i32, i32)> {
        let st = self.ivars();
        // Only head 0 takes the pointer.
        let tex = st.textures[0].borrow();
        let (_, w, h) = tex.as_ref()?;
        let (w, h) = (*w as f32, *h as f32);
        let p = self.convertPoint_fromView(e.locationInWindow(), None);
        let b = self.bounds().size;
        drop(tex);
        let ((ox, oy), (dw, dh), _) = self.picture_rect(0, w, h);
        if dw <= 0.0 || dh <= 0.0 {
            return None;
        }
        // The view's y grows up; the guest's grows down.
        let gx = ((p.x as f32 - ox) * w / dw).floor().clamp(0.0, w - 1.0);
        let gy = ((b.height as f32 - p.y as f32 - oy) * h / dh).floor().clamp(0.0, h - 1.0);
        Some((gx as i32, gy as i32))
    }

    /// Follow mode: tell the controller where the host pointer is.
    fn follow_at(&self, e: &NSEvent, wheel_lines: f32, wheel_points: f32) {
        let Some((x, y)) = self.guest_point(e) else { return };
        crate::app::display(DisplayAction::PointerAt { x, y, buttons: self.ivars().buttons.get(), wheel_lines, wheel_points });
    }

    fn button(&self, e: &NSEvent, bit: u8, down: bool) {
        let st = self.ivars();
        if st.follow.get() {
            let b = if down { st.buttons.get() | bit } else { st.buttons.get() & !bit };
            st.buttons.set(b);
            self.follow_at(e, 0.0, 0.0);
            return;
        }
        if !st.captured.get() {
            return;
        }
        let b = if down { st.buttons.get() | bit } else { st.buttons.get() & !bit };
        st.buttons.set(b);
        let _ = e;
        crate::app::display(DisplayAction::Pointer { dx: 0.0, dy: 0.0, buttons: b, wheel_lines: 0.0, wheel_points: 0.0 });
    }

    fn motion(&self, e: &NSEvent) {
        let st = self.ivars();
        if st.follow.get() {
            self.follow_at(e, 0.0, 0.0);
            return;
        }
        if !st.captured.get() {
            return;
        }
        crate::app::display(DisplayAction::Pointer {
            dx: e.deltaX() as f32,
            dy: e.deltaY() as f32,
            buttons: st.buttons.get(),
            wheel_lines: 0.0,
            wheel_points: 0.0,
        });
    }

    fn key(&self, e: &NSEvent, down: bool) {
        let code = e.keyCode();
        // Follow mode: keys go to the guest while the display has the
        // keyboard, except Command chords, which stay the Mac's.
        let command = e.modifierFlags().contains(NSEventModifierFlags::Command);
        if self.ivars().follow.get() && !self.ivars().captured.get() && !command {
            if keys::is_modifier(code) {
                return;
            }
            if let Some(k) = keys::guest_key(code) {
                crate::app::display(DisplayAction::Key { key: k, down, repeat: down && e.isARepeat() });
            }
            return;
        }
        if !self.ivars().captured.get() {
            // Not captured: a shortcut the key map knows (F11); anything else
            // goes on to the window as usual (Tab moves the focus on, an
            // unused key beeps).
            let chars = e.charactersIgnoringModifiers().map(|s| s.to_string()).unwrap_or_default();
            match keys::shortcut_key(code, &chars) {
                Some(k) if down => crate::app::shortcut(k, mods(e)),
                Some(_) => {}
                // SAFETY: NSResponder's own handlers, with the event they got.
                None if down => unsafe { objc2::msg_send![super(self), keyDown: e] },
                None => unsafe { objc2::msg_send![super(self), keyUp: e] },
            }
            return;
        }
        if keys::is_modifier(code) {
            return;
        }
        if let Some(k) = keys::guest_key(code) {
            crate::app::display(DisplayAction::Key { key: k, down, repeat: down && e.isARepeat() });
        }
    }
}
