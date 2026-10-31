//! The Odyssey (VPro) board's constants and state. The interrupt packet path
//! is the `irq` field of the host interface (`HostInterface`, interface.rs).
//!
//! Ported from include/hw/misc/sgi-odyssey.h in jamesbraid/qemu-sgi-origami.
use super::*;

/// Distinct unimplemented offsets reported before the device falls silent
/// about them. Diagnostic bookkeeping, not guest state.
pub const SGI_ODYSSEY_UNIMP_REPORTS: u32 = 64;

/// The vertices an integer GL primitive holds while it streams: a QUADS
/// rectangle's four, or a strip's first and previous vertex.
pub const SGI_ODYSSEY_GL_HELD_VERTICES: u32 = 4;

/// The operands a register-select word's gather keeps: the arity field's
/// largest count, a 16-word burst.
pub const SGI_ODYSSEY_CMD_OPERANDS: u32 = 16;

/// The window-clip register's four rectangles, two words each.
pub const SGI_ODYSSEY_CLIP_RECTS: u32 = 4;

/// Geometry engine program state: constant-memory entries an 11-bit index
/// reaches.
pub const SGI_ODYSSEY_GL_CONSTANTS: u32 = 2048;
/// Off-screen board-memory objects the model keeps (texture levels).
pub const SGI_ODYSSEY_OBJECTS: u32 = 1024;
pub const SGI_ODYSSEY_OBJ_POOL_BYTES: u32 = 128 * 1024 * 1024;
pub const SGI_ODYSSEY_OBJ_POOL_WORDS: u32 = SGI_ODYSSEY_OBJ_POOL_BYTES / 4;
pub const SGI_ODYSSEY_OBJ_VALID_BYTES: u32 = SGI_ODYSSEY_OBJ_POOL_WORDS / 8;
/// The texture level table's entries (0x0e80 .. 0x0ebf).
pub const SGI_ODYSSEY_TEX_LEVELS: u32 = 64;
pub const SGI_ODYSSEY_TEX_COLOUR_TABLE_ENTRIES: u32 = 256;
pub const SGI_ODYSSEY_TEX_DETAIL_FUNCTION_ENTRIES: u32 = 256;
pub const SGI_ODYSSEY_GL_ENTRIES: u32 = 4;
pub const SGI_ODYSSEY_GL_LIGHTS: u32 = 8;
pub const SGI_ODYSSEY_GL_LIGHT_COLOURS: u32 = 25;
pub const SGI_ODYSSEY_GL_PROGRAM_REPORTS: u32 = 16;
/// Program memory: 2048 instructions of six halfwords, of which the window
/// at 0x738 is the last 0xc7 (ODSY_PROG_WINDOW_BASE).
pub const SGI_ODYSSEY_GL_PROGRAM_ENTRIES: u32 = 2048;
pub const SGI_ODYSSEY_GL_WINDOW_ENTRIES: u32 = 0xc7;

/// The one modeled mode: the desktop target size.
pub const SGI_ODYSSEY_WIDTH: u32 = 1280;
pub const SGI_ODYSSEY_HEIGHT: u32 = 1024;

#[derive(Clone, Copy, Default)]
pub struct OpenVertex {
    pub word: u32,
    pub operands: [u32; 4],
    pub attribute: [[u32; 5]; 19],
    pub slot: u32,
}

pub struct Board {
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
    pub memory_control_reported: bool,
    /// The FIFO set-up registers 0x6b0-0x6b3 and 0x7b5, as last written.
    pub fifo_setup: [u32; 5],
    /// The hardware-sync timeout (0x7f0) as last written, and whether a
    /// nonzero one has been reported as unable to expire.
    pub sync_timeout: u32,
    pub sync_timeout_reported: bool,

    // The DMA-sync command's host-write base (48-bit guest physical address).
    pub host_write_addr_upper: u32,
    pub host_write_addr_lower: u32,
    // The statistics buffer address (0x401c, 0x4024) and control (0x402c).
    pub stats_addr_upper: u32,
    pub stats_addr_lower: u32,
    pub stats_control: u32,
    pub stats_reported: bool,

    // The kernel driver's context switch save address (0x4004/0x400c).
    pub ctx_shadow_addr_upper: u32,
    pub ctx_shadow_addr_lower: u32,

    pub buzz: [u32; 6],
    pub nic: u32,
    pub i2c_main_cmd: u32,
    pub i2c_opt_cmd: u32,

    /// The display back end's colour lookup RAM (0x2000 entries), one 30-bit
    /// colour per entry.
    pub lut: Vec<u32>,
    /// Back-end entries 0x2000-0x2fff as last written (0x1000 entries).
    pub backend_ctl: Vec<u32>,
    pub dispmode: [u32; 32],
    /// The overlay ids' display modes, backend entries 0x2920 + id.
    pub overlay_mode: [u32; 8],
    pub overlay_mode_reported: bool,
    /// The hardware cursor's backend entries 0x2500-0x25ff.
    pub cursor: [u32; 0x100],
    pub cursor_reported: u32,
    pub backend_index: u32,
    pub backend_count: u32,

    // Command-FIFO self-test.
    pub selftest_progress: u32,
    pub cfifo_selftest_ready: bool,

    /// Colour buffer 0's RGBA8 plane over the shared board memory.
    pub mem: BoardMemory,
    /// The id BoardMemory::add_plane returned.
    pub colour_plane: u32,

    /// The second colour buffer (buffer 1 in the displayed-buffer sense).
    pub buffer1_mem: BoardMemory,
    pub buffer1_plane: u32,

    // Buffer selection.
    pub draw_mask: u32,
    pub draw_word: u32,
    pub read_word: u32,
    pub draw_mask_stereo_reported: bool,
    pub draw_mask_high_reported: bool,
    pub drawable_word_reported: bool,

    // The board-memory objects (ODSY_OBJ_POOL_BYTES).
    pub obj_mem: BoardMemory,
    pub dest_base: u32,
    pub obj_count: u32,
    pub obj_used: u32,
    pub obj_base: [u32; SGI_ODYSSEY_OBJECTS as usize],
    /// obj_base inverted, base to index + 1 (Board::object_find()).
    pub obj_index: std::collections::HashMap<u32, u32>,
    pub obj_width: [u32; SGI_ODYSSEY_OBJECTS as usize],
    pub obj_offset: [u32; SGI_ODYSSEY_OBJECTS as usize],
    pub obj_rows: [u32; SGI_ODYSSEY_OBJECTS as usize],
    /// When each object was last selected or grown (obj_clock's count).
    pub obj_stamp: [u64; SGI_ODYSSEY_OBJECTS as usize],
    pub obj_clock: u64,
    /// SGI_ODYSSEY_OBJ_VALID_BYTES bytes.
    pub obj_valid: Vec<u8>,
    pub obj_full_reported: bool,
    pub obj_forget_reported: bool,
    pub board_oob_reported: bool,

    // The texture state words as last pushed.
    pub tex_mode: u32,
    pub tex_sampler: [u32; 7],
    pub tex_colour_table: [[u32; SGI_ODYSSEY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2],
    pub tex_colour_table_valid: [u32; 2],
    pub tex_detail_function: [u32; SGI_ODYSSEY_TEX_DETAIL_FUNCTION_ENTRIES as usize],
    pub tex_detail_function_valid: u32,
    pub tex_env: [u32; 5],
    pub tex_levels: [u32; (SGI_ODYSSEY_TEX_LEVELS * 3) as usize],
    pub tex_slice_offset: [u32; (SGI_ODYSSEY_OBJECTS + 2) as usize],
    pub tex_slice_width: [u32; (SGI_ODYSSEY_OBJECTS + 2) as usize],
    pub tex_level0: u32,
    pub tex_lod: [u32; 4],
    /// The 0x203b post-filter scale and bias words, R, G, B, A.
    pub tex_scale_bias: [u32; 8],
    /// The 0x0c3e texture offset words.
    pub tex_offset: [u32; 3],
    pub tex_coord: [u32; 2],
    pub tex_coord_r: u32,
    pub tex_coord_q: u32,
    pub tex_linear_plane: [[u32; 4]; 2],
    pub tex_linear_plane_r: [u32; 4],
    pub tex_linear_plane_q: [u32; 4],
    pub tex_linear_valid: u32,
    pub tex_rect_reported: bool,
    pub dma_seq: u64,
    /// Monotonic for this device instance.
    pub trace_sequence: u64,

    /// Tests: the next slice on the board's thread first sleeps this long.
    #[cfg(test)]
    pub test_slice_sleep_ms: u32,

    /// The per-pixel drawable-id plane, one byte per framebuffer pixel.
    pub did: Vec<u8>,
    /// The overlay pixel byte beside each id byte; same lifetime as did.
    pub overlay: Vec<u8>,
    pub gl_aux_reported: bool,
    pub aux_mask_reported: bool,
    pub aux_image_reported: bool,

    // The command-FIFO interpreter (0x110000).
    pub query_current: [[u32; 5]; 5],
    pub query_captured: [[u32; 5]; 5],
    /// The index, colour and texture-coordinate records the raster position
    /// latched (indexed like `query_current`; the others are unused).
    pub query_latched: [[u32; 5]; 5],
    pub query_current_valid: u32,
    pub query_latched_valid: u32,
    pub query_captured_valid: u32,
    pub query_stage: u32,
    /// The open query's kind (word 0x4c0 + kind).
    pub query_kind: u32,
    pub query_count: u32,
    pub query_sync_index: u32,
    pub query_epoch: u64,
    pub query_captured_epoch: u64,
    pub query_base: u64,
    pub query_sync_base: u64,

    pub cmd_state: u32,
    pub cmd_tag: u32,
    pub cmd_operand: [u32; SGI_ODYSSEY_CMD_OPERANDS as usize],
    pub cmd_operand_count: u32,
    pub cmd_operand_index: u32,

    // Where the current context's window is.
    pub win_origin: [i32; 2],
    pub read_origin: [i32; 2],
    pub clip_words: [u32; (2 * SGI_ODYSSEY_CLIP_RECTS) as usize],
    /// The second operand of the last 0x840 (enable, mask and value).
    pub id_compare: u32,
    pub clip_kept: u32,
    pub clip_rect: [ClipRect; SGI_ODYSSEY_CLIP_RECTS as usize],
    // The window as the kernel last pushed it on its own port.
    pub win_committed_origin: [i32; 2],
    pub win_committed_clip: [u32; (2 * SGI_ODYSSEY_CLIP_RECTS) as usize],
    pub win_committed_id_compare: u32,
    pub win_prior_origin: [i32; 2],
    pub win_origin_pending: bool,
    pub win_after_tail: bool,
    pub cmd_from_kernel: bool,

    // The 2D stream's GL immediate-mode subset.
    pub gl_primitive: u32,
    pub gl_color: u32,
    pub gl_vertex_x: [u32; SGI_ODYSSEY_GL_HELD_VERTICES as usize],
    pub gl_vertex_y: [u32; SGI_ODYSSEY_GL_HELD_VERTICES as usize],
    pub gl_vertex_count: u32,
    pub line_stipple: [u32; 3],
    /// One byte per framebuffer pixel; crossing marks of an open POLYGON.
    pub poly_marks: Vec<u8>,
    pub poly_row_lo: u32,
    pub poly_row_hi: u32,
    pub poly_col_lo: u32,

    // The true-colour OpenGL path.
    /// None only during Board::gl_command().
    pub gl_ctx: Option<Box<GlContext>>,
    /// The drawable Board::gl_bind() last built for the shared context.
    pub gl_bind_drawable: GlDrawable,
    /// The lighting state Board::gl_light() keeps between vertices; None
    /// until the first lit vertex.
    pub gl_light_scratch: Option<Box<LightScratch>>,
    pub gl_in_prim: bool,
    pub gl_begin_open: bool,
    pub gl_integer_prim: bool,
    pub gl_open_r: u32,
    pub gl_open_c: u32,
    pub gl_open_m3: u32,
    pub gl_open_m4: u32,
    pub gl_open_held: u32,
    pub gl_open_vertex: [OpenVertex; 3],
    pub gl_restart_type: u32,
    pub gl_restart_select: u32,
    pub gl_restart_absorb: u32,
    pub gl_restart_total: u32,
    pub gl_restart_c: u32,
    pub gl_restart_m3: u32,
    pub gl_restart_m4: u32,
    pub gl_restart_armed: bool,
    pub gl_restart_vertex: [[f32; SGI_GL_VERTEX_FLOATS]; 3],
    pub gl_matrix_slot: u32,
    pub gl_matrix: [u32; 16],
    pub gl_texture_matrix: [u32; 16],
    pub gl_rect: [u32; 10],

    // Depth.
    pub gl_depth_func: u32,
    pub gl_write_mask: u32,
    pub gl_combined: [u32; 16],
    pub gl_block16: [u32; 4],
    pub gl_rect_depth_logged: bool,
    pub gl_rect_stencil_logged: bool,
    pub gl_rect_pattern_logged: bool,
    pub gl_index_mask_logged: bool,
    pub depth_mem: BoardMemory,
    pub depth_plane: u32,
    pub stencil_mem: BoardMemory,
    pub stencil_plane: u32,

    /// The scissor and window clip box, 0x081d, as last written.
    pub gl_clip_box: [u32; 2],
    pub gl_fog: [u32; 6],

    // Lighting: the geometry engine's program state.
    pub gl_const: [u32; SGI_ODYSSEY_GL_CONSTANTS as usize],
    pub gl_const_index: u32,
    pub gl_entry: [u32; SGI_ODYSSEY_GL_ENTRIES as usize],
    pub gl_light_colour: [u32; (SGI_ODYSSEY_GL_LIGHT_COLOURS * 4) as usize],
    pub gl_light_vec: [u32; (SGI_ODYSSEY_GL_LIGHTS * 16) as usize],
    pub gl_light_colour_sel: u32,
    pub gl_light_vec_sel: u32,
    pub gl_material: [u32; 16],
    pub gl_shininess_sqrt: u32,
    /// Raw back material retained for idle context save, not lighting.
    pub gl_back_material: [u32; 16],
    pub gl_back_shininess_sqrt: u32,
    /// The class 0x35 colour-material word's eight bits.
    pub gl_colour_material: u32,
    pub gl_normal: [u32; 3],
    pub gl_normal_matrix: [u32; 16],
    pub gl_colour: [u32; 4],
    /// The model-view (slot 0x24), row-major float bits as sent.
    pub gl_modelview: [u32; 16],
    /// The user clip planes in eye coordinates, four floats' bits a plane.
    pub gl_clip_eye: [u32; (SGI_GL_CLIP_PLANES * 4) as usize],
    pub gl_lit: bool,
    pub gl_local_viewer: bool,
    /// The geometry program renormalises transformed normals.
    pub gl_normalize: bool,
    pub gl_colour_overridden: bool,
    pub gl_program_logged: [u32; (SGI_ODYSSEY_GL_PROGRAM_REPORTS * 3) as usize],
    pub gl_program_logged_count: u32,
    /// Program memory as uploaded: SGI_ODYSSEY_GL_PROGRAM_ENTRIES * 6
    /// halfwords.
    pub gl_program_mem: Vec<u16>,
    pub gl_program_addr: u32,
    /// The program-sequencer registers, classes 0x20 to 0x35, as written.
    pub gl_seq: [u32; 22],
    pub gl_program_half: u32,
    pub gl_program_mem_reported: bool,
    pub gl_two_sided_logged: bool,
    pub gl_texgen_logged: bool,
    /// Derived from the retained program list.
    pub gl_linear_texgen: u32,
    pub gl_clip_logged: bool,
    pub gl_texture_stage_logged: bool,
    pub gl_block_logged: bool,
    pub ctx_open_prim_logged: bool,
    pub ctx_shadow_fail_reported: bool,
    /// GL_DITHER, 0x0404 bit 18, as last pushed.
    pub gl_dither: bool,
    pub gl_dither_reported: bool,
    pub scanout_group25_reported: bool,

    // The current fill's colour source.
    pub gl_color_index: u32,
    pub gl_color_is_index: bool,

    /// The colour-mode words as last sent, class 0x22 then 0x23.
    pub colour_mode: [u32; 2],
    pub colour_mode_reported: bool,
    pub gl_index_lit_reported: bool,

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

    // The one-bit pattern the greeter draws its text with.
    pub pattern_buffer: [u32; 32],
    pub pattern_shape: u32,
    pub pattern_mode: u32,
    pub pattern_load_offset: u32,
    /// 0x0c21 operands 1 and 2, the constant colour.
    pub const_colour: [u32; 2],
    /// 0x0c21 operand 0, the factors and equation, as last written.
    pub blend_word: u32,

    // The BEGIN-0xa image raster path.
    pub image_active: bool,
    pub image_x1: u32,
    pub image_y1: u32,
    pub image_x2: u32,
    pub image_y2: u32,
    pub image_cur_x: u32,
    pub image_cur_y: u32,
    pub image_fmt: u32,
    pub image_group_len: u32,
    pub image_group_painted: u32,
    pub image_code: u32,
    /// The header's whole write code, format and type (ODSY_WRITE_CODE_MASK).
    pub image_write_code: u32,
    /// The bytes of the pixel a write-table image is part way through.
    pub image_pixel: [u8; 16],
    pub image_bytes: u32,
    pub image_nbytes: u32,
    pub image_code_reported: bool,

    /// The presented frame and retrace; Board::scanout_update() resolves
    /// the colour planes into it.
    pub scanout: Scanout,
    /// The lines of the update in progress between scanout_stage() and
    /// scanout_resolve().
    pub scanout_stage: ScanoutStage,
    /// The resolved lines' buffer, kept between frames for its allocation.
    pub scanout_rows: Vec<u32>,

    // Diagnostic bookkeeping, not guest state.
    pub reported_addr: [u32; SGI_ODYSSEY_UNIMP_REPORTS as usize],
    pub reported_addr_count: u32,
    pub reported_cmd: [u32; SGI_ODYSSEY_UNIMP_REPORTS as usize],
    pub reported_cmd_count: u32,
    pub line_stipple_reported: bool,
    pub dma_bpp_reported: bool,
    pub dma_readback_reported: bool,
    pub dma_code_reported: bool,
    /// One bit per ODSY_DMA_REPORT_* reason.
    pub dma_read_reported: u32,
}

impl Default for Board {
    fn default() -> Self {
        let front: std::sync::Arc<parking_lot::Mutex<HostInterface>> = Default::default();
        let fifo = front.lock().fifo.clone();
        Board {
            slot: 0,
            front,
            fifo,
            raster: Default::default(),
            guest: None,
            nic_bus: None,
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
            memory_control_reported: false,
            fifo_setup: [0; 5],
            sync_timeout: 0,
            sync_timeout_reported: false,
            host_write_addr_upper: 0,
            host_write_addr_lower: 0,
            stats_addr_upper: 0,
            stats_addr_lower: 0,
            stats_control: 0,
            stats_reported: false,
            ctx_shadow_addr_upper: 0,
            ctx_shadow_addr_lower: 0,
            buzz: [0; 6],
            nic: 0,
            i2c_main_cmd: 0,
            i2c_opt_cmd: 0,
            lut: vec![0; 0x2000],
            backend_ctl: vec![0; 0x1000],
            dispmode: [0; 32],
            overlay_mode: [0; 8],
            overlay_mode_reported: false,
            cursor: [0; 0x100],
            cursor_reported: 0,
            backend_index: 0,
            backend_count: 0,
            selftest_progress: 0,
            cfifo_selftest_ready: false,
            mem: Default::default(),
            colour_plane: 0,
            buffer1_mem: Default::default(),
            buffer1_plane: 0,
            draw_mask: 0,
            draw_word: 0,
            read_word: 0,
            draw_mask_stereo_reported: false,
            draw_mask_high_reported: false,
            drawable_word_reported: false,
            obj_mem: Default::default(),
            dest_base: 0,
            obj_count: 0,
            obj_used: 0,
            obj_base: [0; SGI_ODYSSEY_OBJECTS as usize],
            obj_index: Default::default(),
            obj_width: [0; SGI_ODYSSEY_OBJECTS as usize],
            obj_offset: [0; SGI_ODYSSEY_OBJECTS as usize],
            obj_rows: [0; SGI_ODYSSEY_OBJECTS as usize],
            obj_stamp: [0; SGI_ODYSSEY_OBJECTS as usize],
            obj_clock: 0,
            obj_valid: vec![0; SGI_ODYSSEY_OBJ_VALID_BYTES as usize],
            obj_full_reported: false,
            obj_forget_reported: false,
            board_oob_reported: false,
            tex_mode: 0,
            tex_sampler: [0; 7],
            tex_colour_table: [[0; SGI_ODYSSEY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2],
            tex_colour_table_valid: [0; 2],
            tex_detail_function: [0; SGI_ODYSSEY_TEX_DETAIL_FUNCTION_ENTRIES as usize],
            tex_detail_function_valid: 0,
            tex_env: [0; 5],
            tex_levels: [0; (SGI_ODYSSEY_TEX_LEVELS * 3) as usize],
            tex_slice_offset: [0; (SGI_ODYSSEY_OBJECTS + 2) as usize],
            tex_slice_width: [0; (SGI_ODYSSEY_OBJECTS + 2) as usize],
            tex_level0: 0,
            tex_lod: [0; 4],
            tex_scale_bias: [0; 8],
            tex_offset: [0; 3],
            tex_coord: [0; 2],
            tex_coord_r: 0,
            tex_coord_q: 0,
            tex_linear_plane: [[0; 4]; 2],
            tex_linear_plane_r: [0; 4],
            tex_linear_plane_q: [0; 4],
            tex_linear_valid: 0,
            tex_rect_reported: false,
            dma_seq: 0,
            trace_sequence: 0,
            #[cfg(test)]
            test_slice_sleep_ms: 0,
            did: Vec::new(),
            overlay: Vec::new(),
            gl_aux_reported: false,
            aux_mask_reported: false,
            aux_image_reported: false,
            query_current: [[0; 5]; 5],
            query_captured: [[0; 5]; 5],
            query_latched: [[0; 5]; 5],
            query_current_valid: 0,
            query_latched_valid: 0,
            query_captured_valid: 0,
            query_stage: 0,
            query_kind: 0,
            query_count: 0,
            query_sync_index: 0,
            query_epoch: 0,
            query_captured_epoch: 0,
            query_base: 0,
            query_sync_base: 0,
            cmd_state: 0,
            cmd_tag: 0,
            cmd_operand: [0; SGI_ODYSSEY_CMD_OPERANDS as usize],
            cmd_operand_count: 0,
            cmd_operand_index: 0,
            win_origin: [0; 2],
            read_origin: [0; 2],
            clip_words: [0; (2 * SGI_ODYSSEY_CLIP_RECTS) as usize],
            id_compare: 0,
            clip_kept: 0,
            clip_rect: std::array::from_fn(|_| Default::default()),
            win_committed_origin: [0; 2],
            win_committed_clip: [0; (2 * SGI_ODYSSEY_CLIP_RECTS) as usize],
            win_committed_id_compare: 0,
            win_prior_origin: [0; 2],
            win_origin_pending: false,
            win_after_tail: false,
            cmd_from_kernel: false,
            gl_primitive: 0,
            gl_color: 0,
            gl_vertex_x: [0; SGI_ODYSSEY_GL_HELD_VERTICES as usize],
            gl_vertex_y: [0; SGI_ODYSSEY_GL_HELD_VERTICES as usize],
            gl_vertex_count: 0,
            line_stipple: [0; 3],
            poly_marks: Vec::new(),
            poly_row_lo: 0,
            poly_row_hi: 0,
            poly_col_lo: 0,
            gl_ctx: Some(Box::default()),
            gl_bind_drawable: Default::default(),
            gl_light_scratch: None,
            gl_in_prim: false,
            gl_begin_open: false,
            gl_integer_prim: false,
            gl_open_r: 0,
            gl_open_c: 0,
            gl_open_m3: 0,
            gl_open_m4: 0,
            gl_open_held: 0,
            gl_open_vertex: std::array::from_fn(|_| Default::default()),
            gl_restart_type: 0,
            gl_restart_select: 0,
            gl_restart_absorb: 0,
            gl_restart_total: 0,
            gl_restart_c: 0,
            gl_restart_m3: 0,
            gl_restart_m4: 0,
            gl_restart_armed: false,
            gl_restart_vertex: [[0.0; SGI_GL_VERTEX_FLOATS]; 3],
            gl_matrix_slot: 0,
            gl_matrix: [0; 16],
            gl_texture_matrix: [0; 16],
            gl_rect: [0; 10],
            gl_depth_func: 0,
            gl_write_mask: 0,
            gl_combined: [0; 16],
            gl_block16: [0; 4],
            gl_rect_depth_logged: false,
            gl_rect_stencil_logged: false,
            gl_rect_pattern_logged: false,
            gl_index_mask_logged: false,
            depth_mem: Default::default(),
            depth_plane: 0,
            stencil_mem: Default::default(),
            stencil_plane: 0,
            gl_clip_box: [0; 2],
            gl_fog: [0; 6],
            gl_const: [0; SGI_ODYSSEY_GL_CONSTANTS as usize],
            gl_const_index: 0,
            gl_entry: [0; SGI_ODYSSEY_GL_ENTRIES as usize],
            gl_light_colour: [0; (SGI_ODYSSEY_GL_LIGHT_COLOURS * 4) as usize],
            gl_light_vec: [0; (SGI_ODYSSEY_GL_LIGHTS * 16) as usize],
            gl_light_colour_sel: 0,
            gl_light_vec_sel: 0,
            gl_material: [0; 16],
            gl_shininess_sqrt: 0,
            gl_back_material: [0; 16],
            gl_back_shininess_sqrt: 0,
            gl_colour_material: 0,
            gl_normal: [0; 3],
            gl_normal_matrix: [0; 16],
            gl_colour: [0; 4],
            gl_modelview: [0; 16],
            gl_clip_eye: [0; (SGI_GL_CLIP_PLANES * 4) as usize],
            gl_lit: false,
            gl_local_viewer: false,
            gl_normalize: false,
            gl_colour_overridden: false,
            gl_program_logged: [0; (SGI_ODYSSEY_GL_PROGRAM_REPORTS * 3) as usize],
            gl_program_logged_count: 0,
            gl_program_mem: vec![0; (SGI_ODYSSEY_GL_PROGRAM_ENTRIES * 6) as usize],
            gl_program_addr: 0,
            gl_seq: [0; 22],
            gl_program_half: 0,
            gl_program_mem_reported: false,
            gl_two_sided_logged: false,
            gl_texgen_logged: false,
            gl_linear_texgen: 0,
            gl_clip_logged: false,
            gl_texture_stage_logged: false,
            gl_block_logged: false,
            ctx_open_prim_logged: false,
            ctx_shadow_fail_reported: false,
            gl_dither: false,
            gl_dither_reported: false,
            scanout_group25_reported: false,
            gl_color_index: 0,
            gl_color_is_index: false,
            colour_mode: [0; 2],
            colour_mode_reported: false,
            gl_index_lit_reported: false,
            dest_buffer_word: 0,
            plane_write_mask: 0,
            dma_colour_base: 0,
            screen_base: 0,
            copy_src: 0,
            copy_size: 0,
            copy_dst: 0,
            pattern_buffer: [0; 32],
            pattern_shape: 0,
            pattern_mode: 0,
            pattern_load_offset: 0,
            const_colour: [0; 2],
            blend_word: 0,
            image_active: false,
            image_x1: 0,
            image_y1: 0,
            image_x2: 0,
            image_y2: 0,
            image_cur_x: 0,
            image_cur_y: 0,
            image_fmt: 0,
            image_group_len: 0,
            image_group_painted: 0,
            image_code: 0,
            image_write_code: 0,
            image_pixel: [0; 16],
            image_bytes: 0,
            image_nbytes: 0,
            image_code_reported: false,
            scanout: Default::default(),
            scanout_stage: Default::default(),
            scanout_rows: Vec::new(),
            reported_addr: [0; SGI_ODYSSEY_UNIMP_REPORTS as usize],
            reported_addr_count: 0,
            reported_cmd: [0; SGI_ODYSSEY_UNIMP_REPORTS as usize],
            reported_cmd_count: 0,
            line_stipple_reported: false,
            dma_bpp_reported: false,
            dma_readback_reported: false,
            dma_code_reported: false,
            dma_read_reported: 0,
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
    pub fn build_nic_rom(&mut self, rom: &mut [u8; ODSY_NIC_ROM_SIZE]) {
        let fields: &str = if !self.nic_record.is_empty() {
            &self.nic_record
        } else if !self.product_record.is_empty() {
            &self.product_record
        } else {
            ODSY_NIC_RECORD_DEFAULT
        };
        let record = format!("\u{1}    ;{};", fields);
        let record = record.as_bytes();
        let length = record.len().min(ODSY_NIC_ROM_SIZE);

        rom.fill(0xff);
        rom[..length].copy_from_slice(&record[..length]);
        if length < ODSY_NIC_PAGE as usize {
            rom[length..ODSY_NIC_PAGE as usize].fill(b' ');
        }
    }

    pub fn attach_nic(&mut self) -> Result<(), String> {
        let mut rom = [0u8; ODSY_NIC_ROM_SIZE];

        if self.nic_serial & !ODSY_NIC_SERIAL_MASK != 0 {
            return Err(format!("the sgi-odyssey one-wire serial {:#x} does not fit in 48 bits", self.nic_serial));
        }
        self.build_nic_rom(&mut rom);
        let mut bus = crate::dev::w1::Bus::new();
        bus.add(Box::new(crate::dev::w1::Ds2505::new(ODSY_NIC_FAMILY as u8, self.nic_serial, &rom)));
        self.nic_bus = Some(bus);
        Ok(())
    }

    pub fn compose_memory_config(&mut self) -> Result<(), String> {
        let code: u32 = match self.memory_size {
            32 => 0,
            64 => 1,
            128 => 2,
            _ => {
                return Err(format!("sgi-odyssey memory-size must be 32, 64 or 128 megabytes, not {}", self.memory_size));
            }
        };
        if self.memory_banks != 2 && self.memory_banks != 4 {
            return Err(format!("sgi-odyssey memory-banks must be 2 or 4, not {}", self.memory_banks));
        }
        if self.cas_latency != 2 && self.cas_latency != 3 {
            return Err(format!("sgi-odyssey cas-latency must be 2 or 3, not {}", self.cas_latency));
        }
        // The outstanding-operation count and the data-FIFO level, which the
        // driver waits to reach zero, stay zero: neither is modeled.
        self.memory_config = code << ODSY_CONFIG_SIZE_SHIFT
            | (if self.memory_banks == 4 { ODSY_CONFIG_FOUR_BANKS } else { 0 })
            | (if self.cas_latency == 2 { ODSY_CONFIG_LATENCY_2 } else { 0 })
            | ODSY_CONFIG_COPY_FIELD;
        Ok(())
    }

    pub fn reset_registers(&mut self) {
        // The driver reads its own widget number out of the low nibble before
        // anything has written the register. The rest of the reset value is not
        // established, so it is zero.
        let identity = self.identity();
        let mut f = self.front.lock();
        f.slot = self.slot;
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
        f.intr_flags = ODSY_INTR_FIFO_LOW_WATER;
        f.intr_enable = 0;
        // The level falls with the enables. The routed destination is reset
        // below, so no clearing packet has anywhere to go.
        f.interrupt_asserted = false;
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
        self.flctl_config = ODSY_FLCTL_CONFIG_RESET;
        self.memory_control = 0;
        self.fifo_setup = [0; 5];
        self.sync_timeout = 0;
        self.host_write_addr_upper = 0;
        self.host_write_addr_lower = 0;
        self.stats_addr_upper = 0;
        self.stats_addr_lower = 0;
        self.stats_control = 0;
        self.ctx_shadow_addr_upper = 0;
        self.ctx_shadow_addr_lower = 0;
        self.buzz = [0; 6];
        self.draw_mask = ODSY_DRAW_MASK_RESET;
        self.draw_word = 0;
        self.read_word = 0;
        self.draw_mask_stereo_reported = false;
        self.draw_mask_high_reported = false;
        self.drawable_word_reported = false;
        self.nic = 0;
        self.i2c_main_cmd = 0;
        self.i2c_opt_cmd = 0;
        self.lut.fill(0);
        self.backend_ctl.fill(0);
        self.dispmode = [0; 32];
        self.overlay_mode = [0; 8];
        self.overlay_mode_reported = false;
        self.cursor = [0; 0x100];
        self.cursor_reported = 0;
        self.gl_aux_reported = false;
        self.aux_mask_reported = false;
        self.aux_image_reported = false;
        self.backend_index = 0;
        self.backend_count = 0;
        self.reported_addr_count = 0;
        self.reported_cmd_count = 0;
        self.line_stipple_reported = false;
        self.dma_bpp_reported = false;
        self.dma_readback_reported = false;
        self.dma_code_reported = false;
        self.dma_read_reported = 0;
        self.selftest_progress = 0;
        self.cfifo_selftest_ready = false;

        self.query_cancel();
        self.query_records_reset();
        self.win_origin = [0; 2];
        self.read_origin = [0; 2];
        for i in 0..SGI_ODYSSEY_CLIP_RECTS as usize {
            self.clip_words[2 * i] = if i != 0 { ODSY_CLIP_RESET_EMPTY } else { 0 };
            self.clip_words[2 * i + 1] = if i != 0 { 0 } else { ODSY_CLIP_RESET_FULL };
        }
        self.clip_update();
        self.id_compare = 0;
        self.win_committed_origin = self.win_origin;
        self.win_committed_clip = self.clip_words;
        self.win_committed_id_compare = 0;
        self.win_prior_origin = [0; 2];
        self.win_origin_pending = false;
        self.win_after_tail = false;
        self.cmd_from_kernel = false;
        self.gl_primitive = 0;
        self.gl_color = 0;
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
        self.colour_mode_reported = false;
        self.gl_index_lit_reported = false;
        self.dest_buffer_word = 0;
        self.plane_write_mask = 0;
        self.dma_colour_base = 0;
        self.screen_base = 0;
        self.dest_base = 0;
        self.obj_count = 0;
        self.obj_used = 0;
        self.obj_base.fill(0);
        self.obj_index.clear();
        self.obj_width.fill(0);
        self.obj_offset.fill(0);
        self.obj_rows.fill(0);
        self.obj_stamp.fill(0);
        self.obj_clock = 0;
        self.obj_valid.fill(0);
        self.obj_full_reported = false;
        self.obj_forget_reported = false;
        self.board_oob_reported = false;
        self.tex_mode = 0;
        self.tex_sampler = [0; 7];
        for t in self.tex_colour_table.iter_mut() {
            t.fill(0);
        }
        self.tex_colour_table_valid = [0; 2];
        self.tex_detail_function.fill(0);
        self.tex_detail_function_valid = 0;
        self.tex_env = [0; 5];
        self.tex_levels.fill(0);
        self.tex_level0 = 0;
        self.tex_lod[0] = 0;
        self.tex_lod[1] = 0;
        self.tex_lod[2] = 0x3100;
        self.tex_lod[3] = 0x0f00;
        for c in 0..4usize {
            self.tex_scale_bias[c * 2] = ODSY_TEX_SCALE_BIAS_ONE;
            self.tex_scale_bias[c * 2 + 1] = 0;
        }
        self.tex_offset = [0; 3];
        self.tex_coord = [0; 2];
        self.tex_coord_r = 0;
        self.tex_coord_q = ODSY_F32_ONE;
        self.tex_rect_reported = false;
        self.gl_vertex_x.fill(0);
        self.gl_vertex_y.fill(0);
        self.gl_vertex_count = 0;
        self.line_stipple[0] = 0xffff;
        self.line_stipple[1] = 0xffff;
        self.line_stipple[2] = 0;
        // No polygon is open; Board::reset clears the marks themselves.
        self.poly_row_lo = SGI_ODYSSEY_HEIGHT;
        self.poly_row_hi = 0;
        self.poly_col_lo = SGI_ODYSSEY_WIDTH;
        // The shared GL context to OpenGL's initial state (identity matrices,
        // white current colour, depth test off), unbound until the first GL
        // command binds the drawable. A guest reset can land mid-primitive, with
        // an open polygon's vertex buffer still allocated; GlContext::reset() frees
        // it before reinitializing. It is safe here even before Board::allocate
        // allocates the colour plane. Its viewport starts empty: nothing is
        // drawn until the client's combined matrix arrives.
        self.gl().reset(ODSY_GL_CONTEXT_HW_ID);
        {
            // Opt into the shared context's per-face polygon modes (0x0415) and
            // line stipple: unset, it silently keeps every face at legacy FILL
            // no matter what 0x0415 decodes to. GlContext::reset() above has
            // already cleared in_begin, the one precondition, even when reset
            // lands mid-primitive.
            assert!(self.gl().enable_polygon_modes());
            assert!(self.gl().enable_line_stipple());
        }
        self.gl_in_prim = false;
        self.gl_begin_open = false;
        self.gl_integer_prim = false;
        self.gl_open_r = 0;
        self.gl_open_c = 0;
        self.gl_open_m3 = 0;
        self.gl_open_m4 = 0;
        self.gl_open_held = 0;
        self.gl_restart_type = 0;
        self.gl_restart_select = 0;
        self.gl_restart_absorb = 0;
        self.gl_restart_total = 0;
        self.gl_restart_c = 0;
        self.gl_restart_m3 = 0;
        self.gl_restart_m4 = 0;
        self.gl_restart_armed = false;
        // The clip box and its scissor test: GlContext::init() leaves the shared
        // context's scissor box and test both at zero, which would clip away
        // every draw if the test were simply switched on. There is no reset
        // pattern-mode value that already matches a usable scissor default, so
        // the full window is set here explicitly before the test is enabled,
        // once and for good.
        self.gl_clip_box[0] = ODSY_CLIP_BOX_FIELD_MASK & (SGI_ODYSSEY_WIDTH - 1);
        self.gl_clip_box[1] = ODSY_CLIP_BOX_FIELD_MASK & (SGI_ODYSSEY_HEIGHT - 1);
        self.gl_fog = [0; 6];
        self.gl_apply_clip_box();
        {
            let scissor_test = SGI_GL_SCISSOR_TEST;

            self.gl_feed(SGI_GL_OP_ENABLE as u16, &[scissor_test]);
        }
        self.gl_matrix_slot = SGI_GL_MODELVIEW;
        self.gl_modelview = [0; 16];
        for i in 0..4usize {
            self.gl_modelview[i * 5] = ODSY_F32_ONE;
        }
        self.gl_clip_eye.fill(0);
        self.gl_matrix = [0; 16];
        self.gl_texture_matrix = [0; 16];
        for i in 0..4usize {
            self.gl_texture_matrix[i * 5] = ODSY_F32_ONE;
        }
        self.gl_rect = [0; 10];
        self.gl_depth_func = ODSY_DEPTH_FUNC_RESET;
        self.gl_write_mask = ODSY_WRITE_MASK_RESET;
        self.gl_combined = [0; 16];
        // The identity projection's (-0, -0, -0, 1): the depth row unflipped.
        self.gl_block16[0] = 0x80000000;
        self.gl_block16[1] = 0x80000000;
        self.gl_block16[2] = 0x80000000;
        self.gl_block16[3] = 0x3f800000;
        self.gl_rect_depth_logged = false;
        self.gl_rect_stencil_logged = false;
        self.gl_rect_pattern_logged = false;
        self.gl_index_mask_logged = false;
        // Lighting out of reset. The hardware's values are unknown; the model
        // starts at GL's initial material, light-model ambient, normal and
        // colour, zero light colours and vectors (the library loads every light
        // before a lit draw), an identity normal matrix and no program.
        self.gl_const.fill(0);
        // Program memory's reset content is unknown; the kernel loads it.
        self.gl_program_mem.fill(0);
        self.gl_program_addr = 0;
        self.gl_seq = [0; 22];
        self.gl_program_half = 0;
        self.gl_program_mem_reported = false;
        self.gl_const_index = 0;
        self.gl_entry = [0; SGI_ODYSSEY_GL_ENTRIES as usize];
        // GL's own initial state, colour material off, as the library sends.
        self.gl_colour_material = 0;
        self.gl_light_colour.fill(0);
        for i in 0..3usize {
            self.gl_light_colour[ODSY_GL_LIGHT_MODEL_AMBIENT as usize * 4 + i] = ODSY_F32_POINT_TWO;
            self.gl_material[i] = ODSY_F32_POINT_TWO; // ambient
            self.gl_material[4 + i] = ODSY_F32_POINT_EIGHT; // diffuse
            self.gl_material[8 + i] = 0; // specular
            self.gl_material[12 + i] = 0; // emission
        }
        for i in 0..4usize {
            self.gl_material[i * 4 + 3] = ODSY_F32_ONE;
            self.gl_colour[i] = ODSY_F32_ONE;
        }
        self.gl_light_colour[ODSY_GL_LIGHT_MODEL_AMBIENT as usize * 4 + 3] = ODSY_F32_ONE;
        self.gl_light_vec.fill(0);
        self.gl_light_colour_sel = 0;
        self.gl_light_vec_sel = 0;
        self.gl_shininess_sqrt = 0;
        self.back_material_reset();
        self.gl_normal[0] = 0;
        self.gl_normal[1] = 0;
        self.gl_normal[2] = ODSY_F32_ONE;
        self.gl_normal_matrix = [0; 16];
        for i in 0..4usize {
            self.gl_normal_matrix[i * 5] = ODSY_F32_ONE;
        }
        self.gl_lit = false;
        self.gl_local_viewer = false;
        self.gl_normalize = false;
        self.gl_colour_overridden = false;
        self.gl_program_logged_count = 0;
        self.gl_two_sided_logged = false;
        self.gl_texgen_logged = false;
        self.gl_linear_texgen = 0;
        self.tex_linear_plane = [[0; 4]; 2];
        self.tex_linear_plane_r = [0; 4];
        self.tex_linear_plane_q = [0; 4];
        self.tex_linear_valid = 0;
        self.gl_clip_logged = false;
        self.gl_texture_stage_logged = false;
        self.gl_block_logged = false;
        self.ctx_open_prim_logged = false;
        self.ctx_shadow_fail_reported = false;
        self.gl_dither = false;
        self.gl_dither_reported = false;
        self.scanout_group25_reported = false;
        self.copy_src = 0;
        self.copy_size = 0;
        self.copy_dst = 0;
        self.pattern_buffer = [0; 32];
        self.pattern_shape = 0;
        self.const_colour = [0; 2];
        self.blend_word = 0;
        self.pattern_mode = 0;
        self.pattern_load_offset = 0;
        self.image_active = false;
        self.image_x1 = 0;
        self.image_y1 = 0;
        self.image_x2 = 0;
        self.image_y2 = 0;
        self.image_cur_x = 0;
        self.image_cur_y = 0;
        self.image_fmt = 0;
        self.image_group_len = 0;
        self.image_group_painted = 0;
        self.image_code = 0;
        self.image_write_code = 0;
        self.image_bytes = 0;
        self.image_nbytes = 0;
        self.image_code_reported = false;
        self.dma_seq = 0;
    }
}

/// One marketed board: name, revision, memory size, banks, back-end
/// version, identity record.
pub struct Product {
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
pub static PRODUCTS: [Product; 4] = [
    Product { name: "v12", revision: 0xa, memory_size: 128, memory_banks: 4, backend_version: 1, record: "Name:ODY128VERSIONB" }, // 030-1611
    Product { name: "v10", revision: 0xa, memory_size: 32, memory_banks: 2, backend_version: 1, record: "Name:ODY32VERSIONB" }, // 030-1655
    Product { name: "v8", revision: 0x2, memory_size: 128, memory_banks: 4, backend_version: 1, record: "Name:ODY128" }, // 030-1404
    Product { name: "v6", revision: 0x2, memory_size: 32, memory_banks: 2, backend_version: 1, record: "Name:ODY32" }, // 030-1403
];

impl Board {
    pub fn resolve_product(&mut self) -> Result<(), String> {
        if self.product.is_empty() {
            return Ok(());
        }
        for p in PRODUCTS.iter() {
            if self.product == p.name {
                self.revision = p.revision;
                self.memory_size = p.memory_size;
                self.memory_banks = p.memory_banks;
                self.backend_version = p.backend_version;
                self.product_record = p.record.to_string();
                return Ok(());
            }
        }
        Err(format!("sgi-odyssey product must be v12, v10, v8 or v6, not {}", self.product))
    }

    pub fn configure(&mut self) -> Result<(), String> {
        self.resolve_product()?;
        if self.revision > 0xf {
            return Err(format!("sgi-odyssey revision must fit in four bits, not {}", self.revision));
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
        let pixels = SGI_ODYSSEY_WIDTH as u64 * SGI_ODYSSEY_HEIGHT as u64;
        self.mem.init(pixels * 4);
        {
            let plane = Plane {
                format: PlaneFormat::Rgba8,
                packing: Packing::BeRedHigh,
                width: SGI_ODYSSEY_WIDTH,
                height: SGI_ODYSSEY_HEIGHT,
                stride: SGI_ODYSSEY_WIDTH * 4,
                base: 0,
                len: pixels * 4,
                bottom_origin: false,
                ..Default::default()
            };

            self.colour_plane = self.mem.add_plane(&plane).map_err(|e| format!("sgi-odyssey: colour plane: {e}"))?;
            // Colour buffer 1, the same shape in its own memory. Two model
            // planes preserve the visible contract; they do not claim the
            // board's two-buffer layout.
            self.buffer1_mem.init(plane.len);
            self.buffer1_plane = self.buffer1_mem.add_plane(&plane).map_err(|e| format!("sgi-odyssey: colour buffer 1 plane: {e}"))?;
        }

        // The board-memory object store: device-backed words the objects are
        // laid in by Board::object_rows(), addressed directly.
        self.obj_mem.init(ODSY_OBJ_POOL_BYTES as u64);

        self.did = vec![0u8; pixels as usize];
        self.overlay = vec![0u8; pixels as usize];
        self.poly_marks = vec![0u8; pixels as usize];

        // The depth plane: 24-bit depth in a 32-bit word per frame pixel, in
        // device-backed memory of its own. Where the board keeps the depth
        // buffer, and its width and layout, are unknown, so no aperture reaches
        // this plane. 24 bits matches the Z24 names in the GL library and the depth
        // word (depth_word.rs).
        self.depth_mem.init(pixels * 4);
        {
            let depth = Plane {
                format: PlaneFormat::Z24,
                packing: Packing::BeRedHigh,
                width: SGI_ODYSSEY_WIDTH,
                height: SGI_ODYSSEY_HEIGHT,
                stride: SGI_ODYSSEY_WIDTH * 4,
                base: 0,
                len: pixels * 4,
                bottom_origin: false,
                ..Default::default()
            };

            self.depth_plane = self.depth_mem.add_plane(&depth).map_err(|e| format!("sgi-odyssey: depth plane: {e}"))?;
        }

        // The stencil plane: 8 bits a pixel, in device-backed memory of its own;
        // like the depth plane, no aperture reaches it.
        self.stencil_mem.init(pixels);
        {
            let stencil = Plane {
                format: PlaneFormat::S8,
                packing: Packing::BeRedHigh,
                width: SGI_ODYSSEY_WIDTH,
                height: SGI_ODYSSEY_HEIGHT,
                stride: SGI_ODYSSEY_WIDTH,
                base: 0,
                len: pixels,
                bottom_origin: false,
                ..Default::default()
            };

            self.stencil_plane = self.stencil_mem.add_plane(&stencil).map_err(|e| format!("sgi-odyssey: stencil plane: {e}"))?;
        }

        // Setting the mode allocates the scanout's frame, marks it all dirty
        // and arms the retrace at the mode's rate.
        self.set_scanout_size(SGI_ODYSSEY_WIDTH, SGI_ODYSSEY_HEIGHT);
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
        self.mem.bytes.fill(0);
        self.buffer1_mem.bytes.fill(0);
        self.obj_mem.bytes.fill(0);
        self.stencil_mem.bytes.fill(0);
        self.did.fill(0);
        self.overlay.fill(0);
        self.poly_marks.fill(0);
        // The depth plane starts at all ones (a model choice). The far plane,
        // 2S, is word 0xfffffe, and every drawn depth is at most that, so all
        // ones, the largest word, compares as farther than any fragment.
        let depth = self.depth_mem.plane(self.depth_plane).expect("the depth plane");
        for y in 0..SGI_ODYSSEY_HEIGHT as i32 {
            for x in 0..SGI_ODYSSEY_WIDTH as i32 {
                self.depth_mem.plane_store(&depth, x, y, ODSY_DEPTH24_MAX);
            }
        }
        // Back to the boot size, then repaint the cleared plane and restart
        // the phase.
        if self.scanout.mode.width != SGI_ODYSSEY_WIDTH || self.scanout.mode.height != SGI_ODYSSEY_HEIGHT {
            self.set_scanout_size(SGI_ODYSSEY_WIDTH, SGI_ODYSSEY_HEIGHT);
        }
        self.scanout.reset();
    }
}

/// The synthetic one-wire serial for a component letter and index.
pub fn synthetic_onewire_serial(letter: char, index: u32) -> u64 {
    0x51454d55u64 << 16 | (letter as u64) << 8 | index as u64
}

impl Board {
    /// A board of `product` (v12, v10, v8 or v6) in crosstalk widget
    /// `slot`: board revision A with a synthetic serial number in its
    /// one-wire NIC, and the memory the product carries. An error names a
    /// configuration the board cannot have.
    pub fn new(product: &str, slot: u32) -> Result<Box<Board>, String> {
        let mut s = Box::new(Board::default());

        s.obj_index = Default::default();
        // The fixed part of the drawable Board::gl_bind() hands over.
        s.gl_bind_drawable.bound = 1;
        s.gl_bind_drawable.front = 0;
        s.gl_bind_drawable.back = 1;
        s.revision = 0xa;
        s.memory_size = 128;
        s.memory_banks = 4;
        s.cas_latency = 3;
        s.backend_version = 1;
        s.nic_serial = synthetic_onewire_serial('G', 0);
        s.product = product.to_string();
        s.slot = slot;
        s.configure()?;
        s.allocate()?;
        Ok(s)
    }
}
