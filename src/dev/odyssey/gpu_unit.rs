//! The GPU raster unit as the board sees it (`wgpu/`, behind the `wgpu`
//! feature): the unit, a draw's slot on it, and the refusal a plane the
//! kernels cannot address gets. Without the feature the types are
//! uninhabited, so the board's dispatch compiles to the CPU path alone.

/// Which view a blit stores into: one of its slot's, or one of its own
/// (a DMA put's surface).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "wgpu"), allow(dead_code, reason = "own-view blits are made only with the unit"))]
pub(crate) enum BlitView {
    Slot(u32),
    Own([u32; 4]),
}

/// How a run of blit pixels stores: the view, the bits stored (`mask`,
/// all ones for a plain store), the window's tests, and a logical
/// operation (`op`, 0 none, else 0x100 | the GL code) whose result is
/// masked and or-ed (`and`, `or`) before the store, as Board::paint_integer
/// does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlitKey {
    pub view: BlitView,
    pub mask: u32,
    pub window: bool,
    pub op: u32,
    pub and: u32,
    pub or: u32,
}

impl BlitKey {
    /// A plain store into slot view `view`.
    pub fn store(view: u32, mask: u32) -> BlitKey {
        BlitKey { view: BlitView::Slot(view), mask, window: false, op: 0, and: u32::MAX, or: 0 }
    }
}

#[cfg(feature = "wgpu")]
pub(super) use super::wgpu::scanout::VproScanout as GpuScanout;
#[cfg(feature = "wgpu")]
pub(super) use super::wgpu::{GpuRaster, Slot, TexSlot};

#[cfg(not(feature = "wgpu"))]
mod stub {
    use super::super::pipe::backend::{ClearDraw, DestMem};
    use super::super::pipe::ring::BackQuad;
    use super::super::pipe::setup::Tri;
    use super::super::pipe::state::{BackendWords, RasterWords};

    pub(crate) enum GpuRaster {}

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Slot {}

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum TexSlot {}

    impl TexSlot {
        pub fn new(_r: &super::super::pipe::texture::TexRegs, _l: &super::super::pipe::texture::Latch) -> TexSlot {
            unreachable!("built without the wgpu feature")
        }
    }

    /// A plane the kernels cannot address (here: no kernels at all).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct Unsupported(pub String);

    impl GpuRaster {
        pub fn read_back(&mut self) -> Result<(), String> {
            match *self {}
        }

        pub fn read_back_now(&mut self) -> Result<(), String> {
            match *self {}
        }

        pub fn wait(&mut self) -> Result<(), String> {
            match *self {}
        }

        #[allow(clippy::too_many_arguments, reason = "the record's sources")]
        pub fn push_tri(&mut self, _t: &Tri, _ox: i32, _oy: i32, _bounds: (i32, i32, i32, i32), _steps: i64, _depth_range: [f64; 2], _depth_slack: f64, _mode: u32, _alpha: u32, _slot: &Slot, _tex: Option<&TexSlot>, _specular: bool) {
            match *self {}
        }

        pub fn push_frag(&mut self, _q: &BackQuad, _slot: &Slot) {
            match *self {}
        }

        pub fn push_rect(&mut self, _x0: i32, _y0: i32, _x1: i32, _y1: i32, _c: &ClearDraw, _slot: &Slot) {
            match *self {}
        }

        pub fn clear_all(&mut self) {
            match *self {}
        }

        pub fn blit_pixel(&mut self, _s: &Slot, _key: &super::BlitKey, _x: i32, _y: i32, _word: u32) {
            match *self {}
        }

        pub fn note_region(&mut self, _v: &super::super::view::View, _x: i32, _y: i32) {
            match *self {}
        }

        pub fn tick(&mut self, _idle: bool, _retired: u64) {
            match *self {}
        }


        pub fn trace_waits(&mut self, _on: bool) {
            match *self {}
        }

        pub fn report(&self) -> String {
            match *self {}
        }

        pub fn read_back_view(&mut self, _v: &super::super::view::View, _x0: i32, _y0: i32, _x1: i32, _y1: i32) -> Result<(), String> {
            match *self {}
        }

        pub fn upload_region(&mut self, _v: &super::super::view::View, _x0: i32, _y0: i32, _x1: i32, _y1: i32) {
            match *self {}
        }
    }

    #[allow(dead_code, reason = "the own-view blit paths are compiled with the wgpu feature only")]
    impl Slot {
        pub fn new(_w: &BackendWords, _planes: &DestMem, _index: bool, _to_aux: bool, _raster: &RasterWords, _stipple: Option<&[u32; 32]>, _background: Option<[f64; 4]>) -> Result<Slot, Unsupported> {
            Err(Unsupported("built without the wgpu feature".into()))
        }

        pub fn clear(_planes: &DestMem, _raster: &RasterWords) -> Result<Slot, Unsupported> {
            Err(Unsupported("built without the wgpu feature".into()))
        }

        pub fn none() -> Slot {
            unreachable!("built without the wgpu feature")
        }

        pub fn own_view(_rows: &super::super::view::Rows) -> Result<[u32; 4], Unsupported> {
            Err(Unsupported("built without the wgpu feature".into()))
        }
    }
}

#[cfg(not(feature = "wgpu"))]
mod stub_scanout {
    /// The display thread's scan-out on the GPU.
    pub(crate) enum GpuScanout {}

    impl GpuScanout {
        pub fn run(&mut self, _frame: &super::super::frame::Frame, _regs: &super::super::reader::DisplayRegs, _width: u32, _height: u32, _out: &mut [u32]) -> Result<(), String> {
            match *self {}
        }
    }
}

#[cfg(not(feature = "wgpu"))]
pub(super) use stub::{GpuRaster, Slot, TexSlot};
#[cfg(not(feature = "wgpu"))]
pub(super) use stub_scanout::GpuScanout;
