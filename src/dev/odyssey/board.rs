//! The Odyssey (VPro) board's constants and state. The interrupt packet path
//! is the `irq` field of the host interface (`HostInterface`, interface.rs).
//!
//! Ported from include/hw/misc/sgi-odyssey.h in jamesbraid/qemu-sgi-origami.

use super::consts::{
    clip, config, flctl, gl_cmd, intr, nic, tex, write, ODSY_CONTROL_WIDGET_MASK, ODSY_DEPTH24_MAX,
    ODSY_DEPTH_FUNC_RESET, ODSY_DRAW_MASK_RESET, ODSY_F32_ONE, ODSY_F32_POINT_EIGHT,
    ODSY_F32_POINT_TWO, ODSY_XT_REQ_TIMEOUT_RESET,
};
use super::fifo::CommandFifo;
use super::geometry::LightScratch;
use super::gl_context::{op, vtx, GL_CLIP_PLANES, GL_MODELVIEW, GL_SCISSOR_TEST, GlContext, GlDrawable};
use super::interface::HostInterface;
use super::memory::{BoardMemory, ClipRect, Packing, Plane, PlaneFormat};
use super::pixel_dma::GuestMemory;
use super::raster_unit::RasterQueue;
use super::reader::{DisplayShared, PlaneView};
use super::scanout::Scanout;
#[cfg(test)]
use super::reader::ScanoutReader;
/// Distinct unimplemented offsets reported before the device falls silent
/// about them. Diagnostic bookkeeping, not guest state.
pub(super) const ODSY_UNIMP_REPORTS: u32 = 64;

/// The vertices an integer GL primitive holds while it streams: a QUADS
/// rectangle's four, or a strip's first and previous vertex.
pub(super) const ODSY_GL_HELD_VERTICES: u32 = 4;

/// The operands a register-select word's gather keeps: the arity field's
/// largest count, a 16-word burst.
pub(super) const ODSY_CMD_OPERANDS_MAX: u32 = 16;

/// The window-clip register's four rectangles, two words each.
pub(super) const ODSY_CLIP_RECTS: u32 = 4;

/// Geometry engine program state: constant-memory entries an 11-bit index
/// reaches.
pub(super) const ODSY_GL_CONSTANTS: u32 = 2048;
/// Off-screen board-memory objects the model keeps (texture levels). A
/// model budget, not a board limit: TyrQuake holds some 1,500 levels at once
/// and samples its oldest every frame, which a table of 1024 forgot.
pub(super) const ODSY_OBJECTS: u32 = 16384;
/// The slices of one volume texture a draw samples (Board::texture_setup()).
pub(super) const ODSY_VOLUME_SLICES: u32 = 1024;
pub(super) const ODSY_OBJ_POOL_BYTES: usize = 128 * 1024 * 1024;
pub(super) const ODSY_OBJ_POOL_WORDS: usize = ODSY_OBJ_POOL_BYTES / 4;
pub(super) const ODSY_OBJ_VALID_BYTES: usize = ODSY_OBJ_POOL_WORDS / 8;
/// The texture level table's entries (0x0e80 .. 0x0ebf).
pub(super) const ODSY_TEX_LEVELS: u32 = 64;
pub(super) const ODSY_TEX_COLOUR_TABLE_ENTRIES: u32 = 256;
pub(super) const ODSY_TEX_DETAIL_FUNCTION_ENTRIES: u32 = 256;
pub(super) const ODSY_GL_ENTRIES: u32 = 4;
pub(super) const ODSY_GL_LIGHTS: u32 = 8;
pub(super) const ODSY_GL_LIGHT_COLOURS: u32 = 25;
pub(super) const ODSY_GL_PROGRAM_REPORTS: u32 = 16;
/// Program memory: 2048 instructions of six halfwords, of which the window
/// at 0x738 is the last 0xc7 (prog::WINDOW_BASE).
pub(super) const ODSY_GL_PROGRAM_ENTRIES: u32 = 2048;
pub(super) const ODSY_GL_WINDOW_ENTRIES: u32 = 0xc7;

/// The one modeled mode: the desktop target size.
pub(super) const ODSY_WIDTH: u32 = 1280;
pub(super) const ODSY_HEIGHT: u32 = 1024;

#[derive(Clone, Copy, Default)]
pub(super) struct OpenVertex {
    pub word: u32,
    pub operands: [u32; 4],
    pub attribute: [[u32; 5]; 19],
    pub slot: u32,
}

/// Diagnostic bookkeeping, not guest state: what has been reported once.
pub(super) struct Reports {
    pub memory_control_reported: bool,
    pub setup_7f0_reported: bool,
    pub stats_reported: bool,
    pub overlay_mode_reported: bool,
    pub cursor_reported: u32,
    pub draw_mask_stereo_reported: bool,
    pub draw_mask_high_reported: bool,
    pub drawable_word_reported: bool,
    pub obj_full_reported: bool,
    pub obj_forget_reported: bool,
    pub board_oob_reported: bool,
    pub tex_rect_reported: bool,
    pub gl_aux_reported: bool,
    pub aux_mask_reported: bool,
    /// A rectangle copy with exactly one base at the id-plane base was
    /// refused and noted once.
    pub copy_mixed_base_reported: bool,
    pub aux_image_reported: bool,
    pub gl_rect_depth_logged: bool,
    pub gl_rect_stencil_logged: bool,
    pub gl_rect_pattern_logged: bool,
    pub gl_index_mask_logged: bool,
    pub gl_program_logged: [u32; (ODSY_GL_PROGRAM_REPORTS * 3) as usize],
    pub gl_program_logged_count: u32,
    pub gl_program_mem_reported: bool,
    pub gl_two_sided_logged: bool,
    pub gl_texgen_logged: bool,
    pub gl_clip_logged: bool,
    pub gl_texture_stage_logged: bool,
    pub gl_block_logged: bool,
    pub ctx_open_prim_logged: bool,
    pub ctx_shadow_fail_reported: bool,
    pub gl_dither_reported: bool,
    pub colour_mode_reported: bool,
    pub gl_index_lit_reported: bool,
    pub image_code_reported: bool,
    pub reported_addr: [std::sync::atomic::AtomicU32; ODSY_UNIMP_REPORTS as usize],
    pub reported_addr_count: std::sync::atomic::AtomicU32,
    pub reported_cmd: [u32; ODSY_UNIMP_REPORTS as usize],
    pub reported_cmd_count: u32,
    pub line_stipple_reported: bool,
    pub dma_bpp_reported: bool,
    pub dma_readback_reported: bool,
    pub dma_code_reported: bool,
    /// One bit per dma::REPORT_* reason.
    pub dma_read_reported: u32,
}

/// The board-memory objects (ODSY_OBJ_POOL_BYTES): off-screen images,
/// textures and saved state.
pub(super) struct Objects {
    // The board-memory objects (ODSY_OBJ_POOL_BYTES).
    pub mem: BoardMemory,
    pub count: u32,
    pub used: u32,
    pub base: Vec<u32>,
    /// obj_base inverted, base to index + 1 (Board::object_find()).
    pub index: std::collections::HashMap<u32, u32>,
    pub width: Vec<u32>,
    pub offset: Vec<u32>,
    pub rows: Vec<u32>,
    /// When each object was last selected or grown (obj_clock's count).
    pub stamp: Vec<u64>,
    pub clock: u64,
    /// ODSY_OBJ_VALID_BYTES bytes.
    pub valid: Vec<u8>,
}

/// The texture state words as last pushed.
pub(super) struct TexState {
    // The texture state words as last pushed.
    pub mode: u32,
    pub sampler: [u32; 7],
    pub colour_table: [[u32; ODSY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2],
    pub colour_table_valid: [u32; 2],
    pub detail_function: [u32; ODSY_TEX_DETAIL_FUNCTION_ENTRIES as usize],
    pub detail_function_valid: u32,
    pub env: [u32; 5],
    pub levels: [u32; (ODSY_TEX_LEVELS * 3) as usize],
    pub slice_offset: [u32; (ODSY_VOLUME_SLICES + 2) as usize],
    pub slice_width: [u32; (ODSY_VOLUME_SLICES + 2) as usize],
    pub level0: u32,
    pub lod: [u32; 4],
    /// The 0x203b post-filter scale and bias words, R, G, B, A.
    pub scale_bias: [u32; 8],
    /// The 0x0c3e texture offset words.
    pub offset: [u32; 3],
    pub coord: [u32; 2],
    pub coord_r: u32,
    pub coord_q: u32,
    pub linear_plane: [[u32; 4]; 2],
    pub linear_plane_r: [u32; 4],
    pub linear_plane_q: [u32; 4],
    pub linear_valid: u32,
}

/// The current-query and raster-position records the kernel reads back.
pub(super) struct Query {
    // The command-FIFO interpreter (0x110000).
    pub current: [[u32; 5]; 5],
    pub captured: [[u32; 5]; 5],
    /// The index, colour and texture-coordinate records the raster position
    /// latched (indexed like `query_current`; the others are unused).
    pub latched: [[u32; 5]; 5],
    pub current_valid: u32,
    pub latched_valid: u32,
    pub captured_valid: u32,
    pub stage: u32,
    /// The open query's kind (word 0x4c0 + kind).
    pub kind: u32,
    pub count: u32,
    pub sync_index: u32,
    pub epoch: u64,
    pub captured_epoch: u64,
    pub base: u64,
    pub sync_base: u64,
}

/// The BEGIN-0xa image raster path.
pub(super) struct ImageRaster {
    // The BEGIN-0xa image raster path.
    pub active: bool,
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
    pub cur_x: u32,
    pub cur_y: u32,
    pub fmt: u32,
    pub group_len: u32,
    pub group_painted: u32,
    pub code: u32,
    /// The header's whole write code, format and type (write::CODE_MASK).
    pub write_code: u32,
    /// The bytes of the pixel a write-table image is part way through.
    pub pixel: [u8; 16],
    pub bytes: u32,
    pub nbytes: u32,
}

/// The command-FIFO interpreter (0x110000): the word being parsed and its
/// operands.
pub(super) struct CommandParse {
    pub state: u32,
    pub tag: u32,
    pub operand: [u32; ODSY_CMD_OPERANDS_MAX as usize],
    pub operand_count: u32,
    pub operand_index: u32,
    pub from_kernel: bool,
}

/// Where the current context's window is, and the window as the kernel last
/// pushed it on its own port.
pub(super) struct Window {
    // Where the current context's window is.
    pub origin: [i32; 2],
    pub read_origin: [i32; 2],
    pub clip_words: [u32; (2 * ODSY_CLIP_RECTS) as usize],
    /// The second operand of the last 0x840 (enable, mask and value).
    pub id_compare: u32,
    pub clip_kept: u32,
    pub clip_rect: [ClipRect; ODSY_CLIP_RECTS as usize],
    // The window as the kernel last pushed it on its own port.
    pub committed_origin: [i32; 2],
    pub committed_clip: [u32; (2 * ODSY_CLIP_RECTS) as usize],
    pub committed_id_compare: u32,
    pub prior_origin: [i32; 2],
    pub origin_pending: bool,
    pub after_tail: bool,
}

/// The widget's configuration and the host-side registers that live on the
/// board: the properties, the modelled registers and the back end's indexed
/// file.
pub(super) struct Widget {
    // Properties.
    pub revision: u32,
    /// megabytes
    pub memory_size: u32,
    pub memory_banks: u32,
    pub cas_latency: u32,
    /// bits 2:0 of 0x23380c
    pub backend_version: u32,
    pub product: String,
    pub nic_serial: u64,
    pub nic_record: String,
    // Composed in Board::configure from the properties above.
    /// The product's record, if nic-record unset.
    pub product_record: String,
    pub memory_config: u32,
    // Modeled registers.
    pub flctl_config: u32,
    /// The memory controller's configuration word (+0x1004), as written.
    pub memory_control: u32,
    /// The FIFO set-up registers 0x6b0-0x6b3 and 0x7b5, in that order, as
    /// last written. Values only: their roles are unknown.
    pub fifo_setup: [u32; 5],
    /// Register 0x7f0 as last written. Its unit and meaning are unknown; the
    /// model records the value and runs no timer. Whether a nonzero value has
    /// been reported.
    pub setup_7f0: u32,
    // The DMA-sync command's host-write base (48-bit guest physical address).
    pub host_write_addr_upper: u32,
    pub host_write_addr_lower: u32,
    // The statistics buffer address (0x401c, 0x4024) and control (0x402c).
    pub stats_addr_upper: u32,
    pub stats_addr_lower: u32,
    pub stats_control: u32,
    // The kernel driver's context switch save address (0x4004/0x400c).
    pub ctx_shadow_addr_upper: u32,
    pub ctx_shadow_addr_lower: u32,
    pub buzz: [u32; 6],
    pub nic: u32,
    pub i2c_main_cmd: u32,
    pub i2c_opt_cmd: u32,
    /// Back-end entries 0x2000-0x2fff as last written (0x1000 entries). The
    /// colour lookup RAM, the display modes, the overlay modes and the
    /// cursor, which the display reads, are in `scanout.shared.regs`.
    pub backend_ctl: Vec<u32>,
    pub backend_index: u32,
    pub backend_count: u32,
    // Command-FIFO self-test.
    pub selftest_progress: u32,
    pub cfifo_selftest_ready: bool,
}

/// The geometry engine's program state: lighting, materials, the transforms
/// it keeps and the program it runs.
pub(super) struct Geometry {
    // Lighting: the geometry engine's program state.
    pub constants: [u32; ODSY_GL_CONSTANTS as usize],
    pub constant_index: u32,
    pub entry: [u32; ODSY_GL_ENTRIES as usize],
    pub light_colour: [u32; (ODSY_GL_LIGHT_COLOURS * 4) as usize],
    pub light_vec: [u32; (ODSY_GL_LIGHTS * 16) as usize],
    pub light_colour_sel: u32,
    pub light_vec_sel: u32,
    pub material: [u32; 16],
    pub shininess_sqrt: u32,
    /// Raw back material retained for idle context save, not lighting.
    pub back_material: [u32; 16],
    pub back_shininess_sqrt: u32,
    /// The class 0x35 colour-material word's eight bits.
    pub colour_material: u32,
    pub normal: [u32; 3],
    pub normal_matrix: [u32; 16],
    pub colour: [u32; 4],
    /// The model-view (slot 0x24), row-major float bits as sent.
    pub modelview: [u32; 16],
    /// The user clip planes in eye coordinates, four floats' bits a plane.
    pub clip_eye: [u32; (GL_CLIP_PLANES * 4) as usize],
    pub lit: bool,
    pub local_viewer: bool,
    /// The geometry program renormalises transformed normals.
    pub normalize: bool,
    pub colour_overridden: bool,
    /// Program memory as uploaded: ODSY_GL_PROGRAM_ENTRIES * 6
    /// halfwords.
    pub program_mem: Vec<u16>,
    pub program_addr: u32,
    /// The classes 0x20 to 0x35 as written; 0x27 also sets gl_program_addr.
    pub regs_20_35: [u32; 22],
    pub program_half: u32,
    /// Derived from the retained program list.
    pub linear_texgen: u32,
}

/// The board's planes: colour buffers, depth, stencil, drawable id and
/// overlay. The raster unit owns them while it has jobs (raster_unit.rs).
pub(super) struct Planes {
    /// Colour buffer 0's RGBA8 plane over the shared board memory.
    pub colour: BoardMemory,
    /// The id BoardMemory::add_plane returned.
    pub colour_plane: u32,
    /// The second colour buffer (buffer 1 in the displayed-buffer sense).
    pub buffer1: BoardMemory,
    pub buffer1_plane: u32,
    /// The per-pixel drawable-id plane, one byte per framebuffer pixel.
    pub did: Vec<u8>,
    /// The overlay pixel byte beside each id byte; same lifetime as did.
    pub overlay: Vec<u8>,
    pub depth: BoardMemory,
    pub depth_plane: u32,
    pub stencil: BoardMemory,
    pub stencil_plane: u32,
}

/// The 2D stream's GL immediate-mode subset, the X server's and greeter's
/// drawing: points, lines, polygons and the one-bit text pattern.
pub(super) struct Stream2d {
    // The 2D stream's GL immediate-mode subset.
    pub primitive: u32,
    pub color: u32,
    pub vertex_x: [u32; ODSY_GL_HELD_VERTICES as usize],
    pub vertex_y: [u32; ODSY_GL_HELD_VERTICES as usize],
    pub vertex_count: u32,
    pub line_stipple: [u32; 3],
    /// One byte per framebuffer pixel; crossing marks of an open POLYGON.
    pub poly_marks: Vec<u8>,
    pub poly_row_lo: u32,
    pub poly_row_hi: u32,
    pub poly_col_lo: u32,
    // The one-bit pattern the greeter draws its text with.
    pub pattern_buffer: [u32; 32],
    pub pattern_shape: u32,
    pub pattern_mode: u32,
    pub pattern_load_offset: u32,
}

/// The GL primitive in flight between glBegin and glEnd, and what a context
/// switch needs to restart it.
pub(super) struct OpenPrimitive {
    pub in_prim: bool,
    pub begin_open: bool,
    pub integer_prim: bool,
    pub open_r: u32,
    pub open_c: u32,
    pub open_m3: u32,
    pub open_m4: u32,
    pub open_held: u32,
    pub open_vertex: [OpenVertex; 3],
    pub restart_type: u32,
    pub restart_select: u32,
    pub restart_absorb: u32,
    pub restart_total: u32,
    pub restart_c: u32,
    pub restart_m3: u32,
    pub restart_m4: u32,
    pub restart_armed: bool,
    pub restart_vertex: [[f32; vtx::FLOATS]; 3],
}

/// The front end's state, which the board's thread owns while it runs
/// stores (`BoardCell`), in parts by concern, each part with a single
/// owner (MGRAS instead splits its front, engine and raster state by
/// thread). A processor reaches it only through `BoardLocks::board()`, with
/// the FIFO drained or for the whole-word reads `BoardCell` names. The other threads' state is not here: the raster
/// unit's is `RasterCtx`, the display's `DisplayShared`, and the registers a
/// processor touches without waiting are the host interface's (`front`).
pub(super) struct Board {
    /// The crosstalk widget number.
    pub slot: u32,
    /// The host interface, under its own lock (interface.rs): the registers a
    /// processor may touch while the board works, the interrupt routing
    /// and the command-FIFO queue.
    pub front: std::sync::Arc<parking_lot::Mutex<HostInterface>>,
    /// The command FIFO, shared with the host interface.
    pub fifo: std::sync::Arc<CommandFifo>,
    /// The queue to BUZZ's raster unit (raster_unit.rs), which owns the
    /// planes below while it has jobs.
    pub raster: std::sync::Arc<RasterQueue>,
    /// Guest memory as the board's DMA reaches it (HEART's RAM on the
    /// Octane).
    pub guest: GuestMemory,
    pub nic_bus: Option<crate::dev::w1::Bus>,

    /// The widget's configuration and the host-side registers that live
    /// on the board: the properties, the modelled registers and the back
    /// end's indexed file.
    pub widget: Widget,

    /// Diagnostic bookkeeping, not guest state: what has been reported once.
    pub diag: Reports,

    /// The board's planes: colour buffers, depth, stencil, drawable id
    /// and overlay. The raster unit owns them while it has jobs
    /// (raster_unit.rs).
    pub planes: Planes,

    // Buffer selection.
    pub draw_mask: u32,
    pub draw_word: u32,
    pub read_word: u32,

    /// The board-memory objects (ODSY_OBJ_POOL_BYTES): off-screen images,
    /// textures and saved state.
    pub objects: Objects,
    pub dest_base: u32,

    /// The texture state words as last pushed.
    pub tex: TexState,
    pub dma_seq: u64,
    /// Monotonic for this device instance.
    pub trace_sequence: std::sync::atomic::AtomicU64,

    /// Tests: the next slice on the board's thread first sleeps this long.
    #[cfg(test)]
    pub test_slice_sleep_ms: u32,

    /// The current-query and raster-position records the kernel reads back.
    pub query: Query,

    /// The command-FIFO interpreter (0x110000): the word being parsed and
    /// its operands.
    pub cmd: CommandParse,

    /// Where the current context's window is, and the window as the
    /// kernel last pushed it on its own port.
    pub win: Window,

    /// The 2D stream's GL immediate-mode subset, the X server's and
    /// greeter's drawing: points, lines, polygons and the one-bit text
    /// pattern.
    pub stream: Stream2d,

    // The true-colour OpenGL path.
    /// The GL context. During Board::gl_command() it runs outside the board
    /// and this holds a stand-in (`gl_spare`), which the draws that command
    /// emits must not reach.
    pub gl_ctx: Box<GlContext>,
    /// The stand-in for `gl_ctx` while a command runs; None during one.
    pub gl_spare: Option<Box<GlContext>>,
    /// The drawable Board::gl_bind() last built for the shared context.
    pub gl_bind_drawable: GlDrawable,
    /// The lighting state Board::gl_light() keeps between vertices; None
    /// until the first lit vertex.
    pub gl_light_scratch: Option<Box<LightScratch>>,
    /// The GL primitive in flight between glBegin and glEnd, and what a
    /// context switch needs to restart it.
    pub prim: OpenPrimitive,
    pub gl_matrix_slot: u32,
    pub gl_matrix: [u32; 16],
    pub gl_texture_matrix: [u32; 16],
    pub gl_rect: [u32; 10],

    // Depth.
    pub gl_depth_func: u32,
    pub gl_write_mask: u32,
    pub gl_combined: [u32; 16],
    pub gl_block16: [u32; 4],

    /// The scissor and window clip box, 0x081d, as last written.
    pub gl_clip_box: [u32; 2],
    pub gl_fog: [u32; 6],

    /// The geometry engine's program state: lighting, materials, the
    /// transforms it keeps and the program it runs.
    pub geom: Geometry,

    // The current fill's colour source.
    pub gl_color_index: u32,
    pub gl_color_is_index: bool,

    /// The colour-mode words as last sent, class 0x22 then 0x23.
    pub colour_mode: [u32; 2],

    pub dest_buffer_word: u32,
    pub plane_write_mask: u32,

    /// The colour buffer's base: the first operand of the last colour 0x1017
    /// block (0x2000 for the screen).
    pub dma_colour_base: u32,
    /// The first colour buffer the X server selected: the screen's. A pixel
    /// buffer's select can look the same, and moves dma_colour_base.
    pub screen_base: u32,

    // The screen-to-screen copy command's fixed literal template's three runtime slots.
    pub copy_src: u32,
    pub copy_size: u32,
    pub copy_dst: u32,

    /// 0x0c21 operands 1 and 2, the constant colour.
    pub const_colour: [u32; 2],
    /// 0x0c21 operand 0, the factors and equation, as last written.
    pub blend_word: u32,

    /// The BEGIN-0xa image raster path.
    pub image: ImageRaster,

    /// The scanout's mode and the marks the display takes. The presented
    /// frame and the retrace are the display's own (`ScanoutReader`).
    pub scanout: Scanout,
    /// Tests: the reader that stands in for the display thread
    /// (Board::scanout_update()).
    #[cfg(test)]
    pub reader: ScanoutReader,
}

impl Default for Board {
    fn default() -> Self {
        let front: std::sync::Arc<parking_lot::Mutex<HostInterface>> = Default::default();
        let fifo = front.lock().fifo.clone();
        let raster: std::sync::Arc<RasterQueue> = Default::default();
        let shared = std::sync::Arc::new(DisplayShared::new(Default::default(), front.clone(), raster.clone()));
        let scanout = Scanout::new(shared);
        Board {
            slot: 0,
            front,
            fifo,
            raster,
            guest: None,
            nic_bus: None,
            widget: Widget {
                revision: 0,
                memory_size: 0,
                memory_banks: 0,
                cas_latency: 0,
                backend_version: 0,
                product: String::new(),
                nic_serial: 0,
                nic_record: String::new(),
                product_record: String::new(),
                memory_config: 0,
                flctl_config: 0,
                memory_control: 0,
                fifo_setup: [0; 5],
                setup_7f0: 0,
                host_write_addr_upper: 0,
                host_write_addr_lower: 0,
                stats_addr_upper: 0,
                stats_addr_lower: 0,
                stats_control: 0,
                ctx_shadow_addr_upper: 0,
                ctx_shadow_addr_lower: 0,
                buzz: [0; 6],
                nic: 0,
                i2c_main_cmd: 0,
                i2c_opt_cmd: 0,
                backend_ctl: vec![0; 0x1000],
                backend_index: 0,
                backend_count: 0,
                selftest_progress: 0,
                cfifo_selftest_ready: false,
            },
            diag: Reports {
                memory_control_reported: false,
                setup_7f0_reported: false,
                stats_reported: false,
                overlay_mode_reported: false,
                cursor_reported: 0,
                draw_mask_stereo_reported: false,
                draw_mask_high_reported: false,
                drawable_word_reported: false,
                obj_full_reported: false,
                obj_forget_reported: false,
                board_oob_reported: false,
                tex_rect_reported: false,
                gl_aux_reported: false,
                aux_mask_reported: false,
                copy_mixed_base_reported: false,
                aux_image_reported: false,
                gl_rect_depth_logged: false,
                gl_rect_stencil_logged: false,
                gl_rect_pattern_logged: false,
                gl_index_mask_logged: false,
                gl_program_logged: [0; (ODSY_GL_PROGRAM_REPORTS * 3) as usize],
                gl_program_logged_count: 0,
                gl_program_mem_reported: false,
                gl_two_sided_logged: false,
                gl_texgen_logged: false,
                gl_clip_logged: false,
                gl_texture_stage_logged: false,
                gl_block_logged: false,
                ctx_open_prim_logged: false,
                ctx_shadow_fail_reported: false,
                gl_dither_reported: false,
                colour_mode_reported: false,
                gl_index_lit_reported: false,
                image_code_reported: false,
                reported_addr: std::array::from_fn(|_| std::sync::atomic::AtomicU32::new(0)),
                reported_addr_count: std::sync::atomic::AtomicU32::new(0),
                reported_cmd: [0; ODSY_UNIMP_REPORTS as usize],
                reported_cmd_count: 0,
                line_stipple_reported: false,
                dma_bpp_reported: false,
                dma_readback_reported: false,
                dma_code_reported: false,
                dma_read_reported: 0,
            },
            planes: Planes {
                colour: Default::default(),
                colour_plane: 0,
                buffer1: Default::default(),
                buffer1_plane: 0,
                did: Vec::new(),
                overlay: Vec::new(),
                depth: Default::default(),
                depth_plane: 0,
                stencil: Default::default(),
                stencil_plane: 0,
            },
            draw_mask: 0,
            draw_word: 0,
            read_word: 0,
            objects: Objects {
                mem: Default::default(),
                count: 0,
                used: 0,
                base: vec![0; ODSY_OBJECTS as usize],
                index: Default::default(),
                width: vec![0; ODSY_OBJECTS as usize],
                offset: vec![0; ODSY_OBJECTS as usize],
                rows: vec![0; ODSY_OBJECTS as usize],
                stamp: vec![0; ODSY_OBJECTS as usize],
                clock: 0,
                valid: vec![0; ODSY_OBJ_VALID_BYTES],
            },
            dest_base: 0,
            tex: TexState {
                mode: 0,
                sampler: [0; 7],
                colour_table: [[0; ODSY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2],
                colour_table_valid: [0; 2],
                detail_function: [0; ODSY_TEX_DETAIL_FUNCTION_ENTRIES as usize],
                detail_function_valid: 0,
                env: [0; 5],
                levels: [0; (ODSY_TEX_LEVELS * 3) as usize],
                slice_offset: [0; (ODSY_VOLUME_SLICES + 2) as usize],
                slice_width: [0; (ODSY_VOLUME_SLICES + 2) as usize],
                level0: 0,
                lod: [0; 4],
                scale_bias: [0; 8],
                offset: [0; 3],
                coord: [0; 2],
                coord_r: 0,
                coord_q: 0,
                linear_plane: [[0; 4]; 2],
                linear_plane_r: [0; 4],
                linear_plane_q: [0; 4],
                linear_valid: 0,
            },
            dma_seq: 0,
            trace_sequence: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            test_slice_sleep_ms: 0,
            query: Query {
                current: [[0; 5]; 5],
                captured: [[0; 5]; 5],
                latched: [[0; 5]; 5],
                current_valid: 0,
                latched_valid: 0,
                captured_valid: 0,
                stage: 0,
                kind: 0,
                count: 0,
                sync_index: 0,
                epoch: 0,
                captured_epoch: 0,
                base: 0,
                sync_base: 0,
            },
            cmd: CommandParse {
                state: 0,
                tag: 0,
                operand: [0; ODSY_CMD_OPERANDS_MAX as usize],
                operand_count: 0,
                operand_index: 0,
                from_kernel: false,
            },
            win: Window {
                origin: [0; 2],
                read_origin: [0; 2],
                clip_words: [0; (2 * ODSY_CLIP_RECTS) as usize],
                id_compare: 0,
                clip_kept: 0,
                clip_rect: std::array::from_fn(|_| Default::default()),
                committed_origin: [0; 2],
                committed_clip: [0; (2 * ODSY_CLIP_RECTS) as usize],
                committed_id_compare: 0,
                prior_origin: [0; 2],
                origin_pending: false,
                after_tail: false,
            },
            stream: Stream2d {
                primitive: 0,
                color: 0,
                vertex_x: [0; ODSY_GL_HELD_VERTICES as usize],
                vertex_y: [0; ODSY_GL_HELD_VERTICES as usize],
                vertex_count: 0,
                line_stipple: [0; 3],
                poly_marks: Vec::new(),
                poly_row_lo: 0,
                poly_row_hi: 0,
                poly_col_lo: 0,
                pattern_buffer: [0; 32],
                pattern_shape: 0,
                pattern_mode: 0,
                pattern_load_offset: 0,
            },
            gl_ctx: Box::default(),
            gl_spare: Some(Box::default()),
            gl_bind_drawable: Default::default(),
            gl_light_scratch: None,
            prim: OpenPrimitive {
                in_prim: false,
                begin_open: false,
                integer_prim: false,
                open_r: 0,
                open_c: 0,
                open_m3: 0,
                open_m4: 0,
                open_held: 0,
                open_vertex: std::array::from_fn(|_| Default::default()),
                restart_type: 0,
                restart_select: 0,
                restart_absorb: 0,
                restart_total: 0,
                restart_c: 0,
                restart_m3: 0,
                restart_m4: 0,
                restart_armed: false,
                restart_vertex: [[0.0; vtx::FLOATS]; 3],
            },
            gl_matrix_slot: 0,
            gl_matrix: [0; 16],
            gl_texture_matrix: [0; 16],
            gl_rect: [0; 10],
            gl_depth_func: 0,
            gl_write_mask: 0,
            gl_combined: [0; 16],
            gl_block16: [0; 4],
            gl_clip_box: [0; 2],
            gl_fog: [0; 6],
            geom: Geometry {
                constants: [0; ODSY_GL_CONSTANTS as usize],
                constant_index: 0,
                entry: [0; ODSY_GL_ENTRIES as usize],
                light_colour: [0; (ODSY_GL_LIGHT_COLOURS * 4) as usize],
                light_vec: [0; (ODSY_GL_LIGHTS * 16) as usize],
                light_colour_sel: 0,
                light_vec_sel: 0,
                material: [0; 16],
                shininess_sqrt: 0,
                back_material: [0; 16],
                back_shininess_sqrt: 0,
                colour_material: 0,
                normal: [0; 3],
                normal_matrix: [0; 16],
                colour: [0; 4],
                modelview: [0; 16],
                clip_eye: [0; (GL_CLIP_PLANES * 4) as usize],
                lit: false,
                local_viewer: false,
                normalize: false,
                colour_overridden: false,
                program_mem: vec![0; (ODSY_GL_PROGRAM_ENTRIES * 6) as usize],
                program_addr: 0,
                regs_20_35: [0; 22],
                program_half: 0,
                linear_texgen: 0,
            },
            gl_color_index: 0,
            gl_color_is_index: false,
            colour_mode: [0; 2],
            dest_buffer_word: 0,
            plane_write_mask: 0,
            dma_colour_base: 0,
            screen_base: 0,
            copy_src: 0,
            copy_size: 0,
            copy_dst: 0,
            const_colour: [0; 2],
            blend_word: 0,
            image: ImageRaster {
                active: false,
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 0,
                cur_x: 0,
                cur_y: 0,
                fmt: 0,
                group_len: 0,
                group_painted: 0,
                code: 0,
                write_code: 0,
                pixel: [0; 16],
                bytes: 0,
                nbytes: 0,
            },
            scanout,
            #[cfg(test)]
            reader: Default::default(),
        }
    }
}

impl Board {
    /// The board-identity record. Its field spelling is the one captured off
    /// real firmware, and the prologue (a format byte, four spaces and a
    /// semicolon, so a scanning reader meets ";Name:" and a positional reader
    /// finds the value at byte 11) follows the Bridge model's convention.
    ///
    /// One field by default, because the stock PROM was observed reading
    /// exactly one 32-byte page and stopping; a longer record is truncated
    /// where the master stops clocking. Without a nic-record, a selected
    /// product supplies its board's name, and otherwise the default is the
    /// public owner record of the board the Odyssey contract describes.
    pub fn build_nic_rom(&mut self, rom: &mut [u8; nic::ROM_SIZE]) {
        let fields: &str = if !self.widget.nic_record.is_empty() {
            &self.widget.nic_record
        } else if !self.widget.product_record.is_empty() {
            &self.widget.product_record
        } else {
            nic::RECORD_DEFAULT
        };
        let record = format!("\u{1}    ;{};", fields);
        let record = record.as_bytes();
        let length = record.len().min(nic::ROM_SIZE);

        rom.fill(0xff);
        rom[..length].copy_from_slice(&record[..length]);
        if length < nic::PAGE as usize {
            rom[length..nic::PAGE as usize].fill(b' ');
        }
    }

    pub fn attach_nic(&mut self) -> Result<(), String> {
        let mut rom = [0u8; nic::ROM_SIZE];

        if self.widget.nic_serial & !nic::SERIAL_MASK != 0 {
            return Err(format!("the sgi-odyssey one-wire serial {:#x} does not fit in 48 bits", self.widget.nic_serial));
        }
        self.build_nic_rom(&mut rom);
        let mut bus = crate::dev::w1::Bus::new();
        bus.add(Box::new(crate::dev::w1::Ds2505::new(nic::FAMILY as u8, self.widget.nic_serial, &rom)));
        self.nic_bus = Some(bus);
        Ok(())
    }

    pub fn compose_memory_config(&mut self) -> Result<(), String> {
        let code: u32 = match self.widget.memory_size {
            32 => 0,
            64 => 1,
            128 => 2,
            _ => {
                return Err(format!("sgi-odyssey memory-size must be 32, 64 or 128 megabytes, not {}", self.widget.memory_size));
            }
        };
        if self.widget.memory_banks != 2 && self.widget.memory_banks != 4 {
            return Err(format!("sgi-odyssey memory-banks must be 2 or 4, not {}", self.widget.memory_banks));
        }
        if self.widget.cas_latency != 2 && self.widget.cas_latency != 3 {
            return Err(format!("sgi-odyssey cas-latency must be 2 or 3, not {}", self.widget.cas_latency));
        }
        // The outstanding-operation count and the data-FIFO level, which the
        // driver waits to reach zero, stay zero: neither is modeled.
        self.widget.memory_config = code << config::SIZE_SHIFT
            | (if self.widget.memory_banks == 4 { config::FOUR_BANKS } else { 0 })
            | (if self.widget.cas_latency == 2 { config::LATENCY_2 } else { 0 })
            | config::COPY_FIELD;
        Ok(())
    }

    pub fn reset_registers(&mut self) {
        // The driver reads its own widget number out of the low nibble before
        // anything has written the register. The rest of the reset value is not
        // established, so it is zero.
        let identity = self.identity();
        let mut f = self.front.lock();
        f.identity = identity;
        f.widget_control = self.slot & ODSY_CONTROL_WIDGET_MASK;
        f.xt_req_timeout = ODSY_XT_REQ_TIMEOUT_RESET;
        f.int_dest_upper = 0;
        f.int_dest_lower = 0;
        f.error_command = 0;
        // The command FIFO is empty out of reset, so its low-water condition
        // holds and the low-water interrupt is pending. The kernel driver's
        // board reset arms a watchdog and then tears the board down unless the
        // driver sees this interrupt once, which is what lets it create the
        // direct rendering node the X server opens. Latch it at reset, because
        // the guest cannot reach the later FIFO-drain edge (a command
        // submission) until that node exists. Masked here until the driver
        // enables the source; enabling it then asserts.
        f.intr_flags = intr::FIFO_LOW_WATER;
        f.intr_enable = 0;
        // The level falls with the enables. The routed destination is reset
        // below, so no clearing packet has anywhere to go.
        f.interrupt_asserted = false;
        f.flag_held = false;
        f.sw_token = 0;
        f.flctl_target = 0;
        f.flctl_target_reported = false;
        f.route_dest_upper = 0;
        f.route_dest_lower = 0;
        f.route_vectors = 0;
        f.swap_arm = 0;
        f.swap_pending = 0;
        f.swap_control = 0;
        f.swap_control_companion = 0;
        f.swap_set = 0;
        f.swap_clear = 0;
        f.displayed_buffer = 0;
        f.displayed_written = false;
        drop(f);
        self.widget.flctl_config = flctl::CONFIG_RESET;
        self.widget.memory_control = 0;
        self.widget.fifo_setup = [0; 5];
        self.widget.setup_7f0 = 0;
        self.widget.host_write_addr_upper = 0;
        self.widget.host_write_addr_lower = 0;
        self.widget.stats_addr_upper = 0;
        self.widget.stats_addr_lower = 0;
        self.widget.stats_control = 0;
        self.widget.ctx_shadow_addr_upper = 0;
        self.widget.ctx_shadow_addr_lower = 0;
        self.widget.buzz = [0; 6];
        self.draw_mask = ODSY_DRAW_MASK_RESET;
        self.draw_word = 0;
        self.read_word = 0;
        self.diag.draw_mask_stereo_reported = false;
        self.diag.draw_mask_high_reported = false;
        self.diag.drawable_word_reported = false;
        self.widget.nic = 0;
        self.widget.i2c_main_cmd = 0;
        self.widget.i2c_opt_cmd = 0;
        self.widget.backend_ctl.fill(0);
        self.regs().reset();
        self.diag.overlay_mode_reported = false;
        self.diag.cursor_reported = 0;
        self.diag.gl_aux_reported = false;
        self.diag.aux_mask_reported = false;
        self.diag.copy_mixed_base_reported = false;
        self.diag.aux_image_reported = false;
        self.widget.backend_index = 0;
        self.widget.backend_count = 0;
        self.diag.reported_addr_count.store(0, std::sync::atomic::Ordering::Relaxed);
        self.diag.reported_cmd_count = 0;
        self.diag.line_stipple_reported = false;
        self.diag.dma_bpp_reported = false;
        self.diag.dma_readback_reported = false;
        self.diag.dma_code_reported = false;
        self.diag.dma_read_reported = 0;
        self.widget.selftest_progress = 0;
        self.widget.cfifo_selftest_ready = false;

        self.query_cancel();
        self.query_records_reset();
        self.win.origin = [0; 2];
        self.win.read_origin = [0; 2];
        for i in 0..ODSY_CLIP_RECTS as usize {
            self.win.clip_words[2 * i] = if i != 0 { clip::RESET_EMPTY } else { 0 };
            self.win.clip_words[2 * i + 1] = if i != 0 { 0 } else { clip::RESET_FULL };
        }
        self.clip_update();
        self.win.id_compare = 0;
        self.win.committed_origin = self.win.origin;
        self.win.committed_clip = self.win.clip_words;
        self.win.committed_id_compare = 0;
        self.win.prior_origin = [0; 2];
        self.win.origin_pending = false;
        self.win.after_tail = false;
        self.cmd.from_kernel = false;
        self.stream.primitive = 0;
        self.stream.color = 0;
        // GL's initial index is 1 (OpenGL 1.2.1 section 2.7), the value the
        // index record's default saves and a replay restores, so a context that
        // never sets an index draws GL vertices in index 1 before and after its
        // first switch. The board's reset index is unknown; this is a model
        // default. Limitation: the integer rectangle fill takes the index only
        // after an index word, so with none since reset it draws the packed
        // colour 0, and index 1 after a replay. Only a register test reaches
        // that: the X server sends an index or a colour before every fill.
        self.gl_color_index = 1;
        self.gl_color_is_index = false;
        // Out of reset the mode is unknown; RGBA until a context says otherwise.
        self.colour_mode = [0; 2];
        self.diag.colour_mode_reported = false;
        self.diag.gl_index_lit_reported = false;
        self.dest_buffer_word = 0;
        self.plane_write_mask = 0;
        self.dma_colour_base = 0;
        self.screen_base = 0;
        self.dest_base = 0;
        self.objects.count = 0;
        self.objects.used = 0;
        self.objects.base.fill(0);
        self.objects.index.clear();
        self.objects.width.fill(0);
        self.objects.offset.fill(0);
        self.objects.rows.fill(0);
        self.objects.stamp.fill(0);
        self.objects.clock = 0;
        self.objects.valid.fill(0);
        self.diag.obj_full_reported = false;
        self.diag.obj_forget_reported = false;
        self.diag.board_oob_reported = false;
        self.tex.mode = 0;
        self.tex.sampler = [0; 7];
        for t in self.tex.colour_table.iter_mut() {
            t.fill(0);
        }
        self.tex.colour_table_valid = [0; 2];
        self.tex.detail_function.fill(0);
        self.tex.detail_function_valid = 0;
        self.tex.env = [0; 5];
        self.tex.levels.fill(0);
        self.tex.level0 = 0;
        self.tex.lod[0] = 0;
        self.tex.lod[1] = 0;
        self.tex.lod[2] = 0x3100;
        self.tex.lod[3] = 0x0f00;
        for c in 0..4usize {
            self.tex.scale_bias[c * 2] = tex::SCALE_BIAS_ONE;
            self.tex.scale_bias[c * 2 + 1] = 0;
        }
        self.tex.offset = [0; 3];
        self.tex.coord = [0; 2];
        self.tex.coord_r = 0;
        self.tex.coord_q = ODSY_F32_ONE;
        self.diag.tex_rect_reported = false;
        self.stream.vertex_x.fill(0);
        self.stream.vertex_y.fill(0);
        self.stream.vertex_count = 0;
        self.stream.line_stipple[0] = 0xffff;
        self.stream.line_stipple[1] = 0xffff;
        self.stream.line_stipple[2] = 0;
        // No polygon is open; Board::reset clears the marks themselves.
        self.stream.poly_row_lo = ODSY_HEIGHT;
        self.stream.poly_row_hi = 0;
        self.stream.poly_col_lo = ODSY_WIDTH;
        // The shared GL context to OpenGL's initial state (identity matrices,
        // white current colour, depth test off), unbound until the first GL
        // command binds the drawable. A guest reset can land mid-primitive, with
        // an open polygon's vertex buffer still allocated; GlContext::reset() frees
        // it before reinitializing. It is safe here even before Board::allocate
        // allocates the colour plane. Its viewport starts empty: nothing is
        // drawn until the client's combined matrix arrives.
        self.gl().reset(gl_cmd::CONTEXT_HW_ID);
        {
            // Opt into the shared context's per-face polygon modes (0x0415) and
            // line stipple: unset, it silently keeps every face at legacy FILL
            // no matter what 0x0415 decodes to. GlContext::reset() above has
            // already cleared in_begin, the one precondition, even when reset
            // lands mid-primitive.
            assert!(self.gl().enable_polygon_modes());
            assert!(self.gl().enable_line_stipple());
        }
        self.prim.in_prim = false;
        self.prim.begin_open = false;
        self.prim.integer_prim = false;
        self.prim.open_r = 0;
        self.prim.open_c = 0;
        self.prim.open_m3 = 0;
        self.prim.open_m4 = 0;
        self.prim.open_held = 0;
        self.prim.restart_type = 0;
        self.prim.restart_select = 0;
        self.prim.restart_absorb = 0;
        self.prim.restart_total = 0;
        self.prim.restart_c = 0;
        self.prim.restart_m3 = 0;
        self.prim.restart_m4 = 0;
        self.prim.restart_armed = false;
        // The clip box and its scissor test: GlContext::init() leaves the shared
        // context's scissor box and test both at zero, which would clip away
        // every draw if the test were simply switched on. There is no reset
        // pattern-mode value that already matches a usable scissor default, so
        // the full window is set here explicitly before the test is enabled,
        // once and for good.
        self.gl_clip_box[0] = clip::BOX_FIELD_MASK & (ODSY_WIDTH - 1);
        self.gl_clip_box[1] = clip::BOX_FIELD_MASK & (ODSY_HEIGHT - 1);
        self.gl_fog = [0; 6];
        self.gl_apply_clip_box();
        {
            let scissor_test = GL_SCISSOR_TEST;

            self.gl_feed(op::ENABLE as u16, &[scissor_test]);
        }
        self.gl_matrix_slot = GL_MODELVIEW;
        self.geom.modelview = [0; 16];
        for i in 0..4usize {
            self.geom.modelview[i * 5] = ODSY_F32_ONE;
        }
        self.geom.clip_eye.fill(0);
        self.gl_matrix = [0; 16];
        self.gl_texture_matrix = [0; 16];
        for i in 0..4usize {
            self.gl_texture_matrix[i * 5] = ODSY_F32_ONE;
        }
        self.gl_rect = [0; 10];
        self.gl_depth_func = ODSY_DEPTH_FUNC_RESET;
        self.gl_write_mask = write::MASK_RESET;
        self.gl_combined = [0; 16];
        // The identity projection's (-0, -0, -0, 1): the depth row unflipped.
        self.gl_block16[0] = 0x80000000;
        self.gl_block16[1] = 0x80000000;
        self.gl_block16[2] = 0x80000000;
        self.gl_block16[3] = 0x3f800000;
        self.diag.gl_rect_depth_logged = false;
        self.diag.gl_rect_stencil_logged = false;
        self.diag.gl_rect_pattern_logged = false;
        self.diag.gl_index_mask_logged = false;
        // Lighting out of reset. The hardware's values are unknown; the model
        // starts at GL's initial material, light-model ambient, normal and
        // colour, zero light colours and vectors (the library loads every light
        // before a lit draw), an identity normal matrix and no program.
        self.geom.constants.fill(0);
        // Program memory's reset content is unknown; the kernel loads it.
        self.geom.program_mem.fill(0);
        self.geom.program_addr = 0;
        self.geom.regs_20_35 = [0; 22];
        self.geom.program_half = 0;
        self.diag.gl_program_mem_reported = false;
        self.geom.constant_index = 0;
        self.geom.entry = [0; ODSY_GL_ENTRIES as usize];
        // GL's own initial state, colour material off, as the library sends.
        self.geom.colour_material = 0;
        self.geom.light_colour.fill(0);
        for i in 0..3usize {
            self.geom.light_colour[gl_cmd::LIGHT_MODEL_AMBIENT as usize * 4 + i] = ODSY_F32_POINT_TWO;
            self.geom.material[i] = ODSY_F32_POINT_TWO; // ambient
            self.geom.material[4 + i] = ODSY_F32_POINT_EIGHT; // diffuse
            self.geom.material[8 + i] = 0; // specular
            self.geom.material[12 + i] = 0; // emission
        }
        for i in 0..4usize {
            self.geom.material[i * 4 + 3] = ODSY_F32_ONE;
            self.geom.colour[i] = ODSY_F32_ONE;
        }
        self.geom.light_colour[gl_cmd::LIGHT_MODEL_AMBIENT as usize * 4 + 3] = ODSY_F32_ONE;
        self.geom.light_vec.fill(0);
        self.geom.light_colour_sel = 0;
        self.geom.light_vec_sel = 0;
        self.geom.shininess_sqrt = 0;
        self.back_material_reset();
        self.geom.normal[0] = 0;
        self.geom.normal[1] = 0;
        self.geom.normal[2] = ODSY_F32_ONE;
        self.geom.normal_matrix = [0; 16];
        for i in 0..4usize {
            self.geom.normal_matrix[i * 5] = ODSY_F32_ONE;
        }
        self.geom.lit = false;
        self.geom.local_viewer = false;
        self.geom.normalize = false;
        self.geom.colour_overridden = false;
        self.diag.gl_program_logged_count = 0;
        self.diag.gl_two_sided_logged = false;
        self.diag.gl_texgen_logged = false;
        self.geom.linear_texgen = 0;
        self.tex.linear_plane = [[0; 4]; 2];
        self.tex.linear_plane_r = [0; 4];
        self.tex.linear_plane_q = [0; 4];
        self.tex.linear_valid = 0;
        self.diag.gl_clip_logged = false;
        self.diag.gl_texture_stage_logged = false;
        self.diag.gl_block_logged = false;
        self.diag.ctx_open_prim_logged = false;
        self.diag.ctx_shadow_fail_reported = false;
        self.diag.gl_dither_reported = false;
        self.scanout.shared.group25.store(false, std::sync::atomic::Ordering::Relaxed);
        self.copy_src = 0;
        self.copy_size = 0;
        self.copy_dst = 0;
        self.stream.pattern_buffer = [0; 32];
        self.stream.pattern_shape = 0;
        self.const_colour = [0; 2];
        self.blend_word = 0;
        self.stream.pattern_mode = 0;
        self.stream.pattern_load_offset = 0;
        self.image.active = false;
        self.image.x1 = 0;
        self.image.y1 = 0;
        self.image.x2 = 0;
        self.image.y2 = 0;
        self.image.cur_x = 0;
        self.image.cur_y = 0;
        self.image.fmt = 0;
        self.image.group_len = 0;
        self.image.group_painted = 0;
        self.image.code = 0;
        self.image.write_code = 0;
        self.image.bytes = 0;
        self.image.nbytes = 0;
        self.diag.image_code_reported = false;
        self.dma_seq = 0;
    }
}

/// One marketed board: name, revision, memory size, banks, back-end
/// version, identity record.
pub(super) struct Product {
    pub name: &'static str,
    pub revision: u32,
    pub memory_size: u32,
    pub memory_banks: u32,
    pub backend_version: u32,
    pub record: &'static str,
}

/// The four marketed boards. A driver turns widget revision 1-3 into one
/// part generation and 9-15 into the next, and the board-info tool prints
/// the minor number as the revision's offset less one: revision 2 is "A.1",
/// 10 is "B.1". The inventory names the board from those two facts and the
/// size: A at 32 or 128 MB is V6 or V8, B is V10 or V12. Public captures
/// give A.1 for a 32 MB V6, B.1 for a 32 MB V10 and a 128 MB V12, and four
/// banks for the V12. The V8's A.1 and four banks are inferred from its
/// generation and size; the V10's revision 10 is inferred from working
/// Fuel and Tezro V10s (B.2 boards, revision 11, also exist); the 32 MB
/// boards' two banks are inferred, since the tool prints a fixed "Banks: 2"
/// line for any 32 MB board. Every product is CAS latency 3 with back-end
/// (PB&J) version 1. The identity record is the Octane board's name as the
/// inventory prints it, with its part number beside each entry. Fuel and
/// Tezro boards are other assemblies; a profile answers the Octane name.
pub(super) static PRODUCTS: [Product; 4] = [
    Product { name: "v12", revision: 0xa, memory_size: 128, memory_banks: 4, backend_version: 1, record: "Name:ODY128VERSIONB" }, // 030-1611
    Product { name: "v10", revision: 0xa, memory_size: 32, memory_banks: 2, backend_version: 1, record: "Name:ODY32VERSIONB" }, // 030-1655
    Product { name: "v8", revision: 0x2, memory_size: 128, memory_banks: 4, backend_version: 1, record: "Name:ODY128" }, // 030-1404
    Product { name: "v6", revision: 0x2, memory_size: 32, memory_banks: 2, backend_version: 1, record: "Name:ODY32" }, // 030-1403
];

impl Board {
    pub fn resolve_product(&mut self) -> Result<(), String> {
        if self.widget.product.is_empty() {
            return Ok(());
        }
        for p in PRODUCTS.iter() {
            if self.widget.product == p.name {
                self.widget.revision = p.revision;
                self.widget.memory_size = p.memory_size;
                self.widget.memory_banks = p.memory_banks;
                self.widget.backend_version = p.backend_version;
                self.widget.product_record = p.record.to_string();
                return Ok(());
            }
        }
        Err(format!("sgi-odyssey product must be v12, v10, v8 or v6, not {}", self.widget.product))
    }

    pub fn configure(&mut self) -> Result<(), String> {
        self.resolve_product()?;
        if self.widget.revision > 0xf {
            return Err(format!("sgi-odyssey revision must fit in four bits, not {}", self.widget.revision));
        }
        self.compose_memory_config()?;
        self.reset_registers();
        Ok(())
    }

    pub fn allocate(&mut self) -> Result<(), String> {
        self.attach_nic()?;

        // The colour framebuffer is two RGBA8 planes, one per colour buffer,
        // each over device-backed board memory of its own: a model-owned byte
        // array. It is not page-backed onto guest RAM. add_plane can only fail
        // on a programming error in these fixed sizes.
        let pixels = ODSY_WIDTH as u64 * ODSY_HEIGHT as u64;
        self.planes.colour.init(pixels * 4);
        {
            let plane = Plane {
                format: PlaneFormat::Rgba8,
                packing: Packing::BeRedHigh,
                width: ODSY_WIDTH,
                height: ODSY_HEIGHT,
                stride: ODSY_WIDTH * 4,
                base: 0,
                len: pixels * 4,
                bottom_origin: false,
                ..Default::default()
            };

            self.planes.colour_plane = self.planes.colour.add_plane(&plane).map_err(|e| format!("sgi-odyssey: colour plane: {e}"))?;
            // Colour buffer 1, the same shape in its own memory. Two model
            // planes preserve the visible contract; they do not claim the
            // board's two-buffer layout.
            self.planes.buffer1.init(plane.len);
            self.planes.buffer1_plane = self.planes.buffer1.add_plane(&plane).map_err(|e| format!("sgi-odyssey: colour buffer 1 plane: {e}"))?;
        }

        // The board-memory object store: device-backed words the objects are
        // laid in by Board::object_rows(), addressed directly.
        self.objects.mem.init(ODSY_OBJ_POOL_BYTES as u64);

        self.planes.did = vec![0u8; pixels as usize];
        self.planes.overlay = vec![0u8; pixels as usize];
        self.stream.poly_marks = vec![0u8; pixels as usize];

        // The depth plane: 24-bit depth in a 32-bit word per frame pixel, in
        // device-backed memory of its own. Where the board keeps the depth
        // buffer, and its width and layout, are unknown, so no aperture reaches
        // this plane. 24 bits matches the Z24 names in the GL library and the depth
        // word (depth_word.rs).
        self.planes.depth.init(pixels * 4);
        {
            let depth = Plane {
                format: PlaneFormat::Z24,
                packing: Packing::BeRedHigh,
                width: ODSY_WIDTH,
                height: ODSY_HEIGHT,
                stride: ODSY_WIDTH * 4,
                base: 0,
                len: pixels * 4,
                bottom_origin: false,
                ..Default::default()
            };

            self.planes.depth_plane = self.planes.depth.add_plane(&depth).map_err(|e| format!("sgi-odyssey: depth plane: {e}"))?;
        }

        // The stencil plane: 8 bits a pixel, in device-backed memory of its own;
        // like the depth plane, no aperture reaches it.
        self.planes.stencil.init(pixels);
        {
            let stencil = Plane {
                format: PlaneFormat::S8,
                packing: Packing::BeRedHigh,
                width: ODSY_WIDTH,
                height: ODSY_HEIGHT,
                stride: ODSY_WIDTH,
                base: 0,
                len: pixels,
                bottom_origin: false,
                ..Default::default()
            };

            self.planes.stencil_plane = self.planes.stencil.add_plane(&stencil).map_err(|e| format!("sgi-odyssey: stencil plane: {e}"))?;
        }

        // The display reads the planes through a view that lives as long as
        // they do: they are never resized or moved after this.
        if let Some(view) = PlaneView::of(self) {
            let _ = self.scanout.shared.planes.set(view);
        }

        // Setting the mode marks the scanout's frame all dirty and
        // restarts the retrace at the mode's rate.
        self.set_scanout_size(ODSY_WIDTH, ODSY_HEIGHT);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.work_reset();
        // The raster unit finishes what it was given before the planes clear.
        self.raster_wait();
        self.reset_registers();
        self.front.lock().update_irq();
        // Board::configure() also calls Board::reset_registers(), before
        // the board memory exists; this runs after Board::allocate(), so the
        // memory exists here. Zero the colour planes' bytes directly; the
        // planes themselves are fixed at allocation and stay. Redraw because
        // reset clears them.
        self.planes.colour.bytes.fill(0);
        self.planes.buffer1.bytes.fill(0);
        self.objects.mem.bytes.fill(0);
        self.planes.stencil.bytes.fill(0);
        self.planes.did.fill(0);
        self.planes.overlay.fill(0);
        self.stream.poly_marks.fill(0);
        // The depth plane starts at all ones (a model choice). The far plane,
        // 2S, is word 0xfffffe, and every drawn depth is at most that, so all
        // ones, the largest word, compares as farther than any fragment.
        let depth = self.planes.depth.plane(self.planes.depth_plane).expect("the depth plane");
        for y in 0..ODSY_HEIGHT as i32 {
            for x in 0..ODSY_WIDTH as i32 {
                self.planes.depth.plane_store(&depth, x, y, ODSY_DEPTH24_MAX);
            }
        }
        // Back to the boot size, then repaint the cleared plane and restart
        // the phase.
        if self.scanout.mode.width != ODSY_WIDTH || self.scanout.mode.height != ODSY_HEIGHT {
            self.set_scanout_size(ODSY_WIDTH, ODSY_HEIGHT);
        }
        self.scanout.reset();
    }
}

/// The synthetic one-wire serial for a component letter and index.
pub(super) fn synthetic_onewire_serial(letter: char, index: u32) -> u64 {
    0x51454d55u64 << 16 | (letter as u64) << 8 | index as u64
}

impl Board {
    /// A board of `product` (v12, v10, v8 or v6) in crosstalk widget
    /// `slot`: board revision A with a synthetic serial number in its
    /// one-wire NIC, and the memory the product carries. An error names a
    /// configuration the board cannot have.
    pub fn new(product: &str, slot: u32) -> Result<Box<Board>, String> {
        let mut s = Box::new(Board::default());

        s.objects.index = Default::default();
        // The fixed part of the drawable Board::gl_bind() hands over.
        s.gl_bind_drawable.bound = true;
        s.gl_bind_drawable.front = 0;
        s.gl_bind_drawable.back = 1;
        s.widget.revision = 0xa;
        s.widget.memory_size = 128;
        s.widget.memory_banks = 4;
        s.widget.cas_latency = 3;
        s.widget.backend_version = 1;
        s.widget.nic_serial = synthetic_onewire_serial('G', 0);
        s.widget.product = product.to_string();
        s.slot = slot;
        s.configure()?;
        s.allocate()?;
        Ok(s)
    }
}
