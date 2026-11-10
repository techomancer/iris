//! The selector tables: for each command word a port can open, how many
//! argument words follow. Counts are those of the senders read.
//!
//! Ported from hw/misc/sgi-kona-ports.c in jamesbraid/qemu-sgi-origami.

use super::defs::*;
use super::frame::Selector;

/// The kernel FIFO push port: the DMA packet, the list-write record and the pool
/// registration, then the display-mode and select-feed selectors.
pub(super) const DMA_FIFO: &[Selector] = &[
    Selector::new(START_INPUT_DMA, 6),
    Selector::new(COPY_LIST_BY_INPUT_DMA_THEN_LIST_DMA, 2),
    Selector::new(DESCRIBE_DMA_PAGE_POOL, 3),
    Selector::new(DISPLAY_MODE_BYTE, 3),
    Selector::new(START_SELECT_FEEDBACK_RETURN_DMA, 0),
    Selector::new(FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE, 0),
];

/// The textport's commands, which follow a sync word.
pub(super) const TEXTPORT: &[Selector] = &[
    Selector::new(TEXTPORT_UNMODELLED_1, 1),
    Selector::new(TEXTPORT_SET_COLOUR, 1),
    Selector::new(TEXTPORT_LOAD_COLOUR_MAP, 4),
    Selector::new(TEXTPORT_SET_RASTER_POSITION, 2),
    Selector::new(TEXTPORT_FILL_BOX, 4),
    Selector::new(TEXTPORT_PLOT_POINT, 2),
    Selector::new(TEXTPORT_UNMODELLED_7, 2),
    Selector::new(TEXTPORT_UNMODELLED_8, 2),
    Selector::new(TEXTPORT_SYNC_PROBE, 1),
    Selector::new(TEXTPORT_GRAPHICS_INFO, 1),
];

/// The command page's tokens: kernel, graphics-process and X server commands.
pub(super) const KERNEL: &[Selector] = &[
    Selector::new(PIPE_SET_UP, 3),
    Selector::new(SET_UP_OUTPUT_DMA_FOR_HOST_READ, 3),
    Selector::new(CONTEXT_INITIALISE, 3),
    Selector::new(CONTEXT_CLEAN_UP_WITH_EVENT_CODE, 1),
    Selector::new(RAISE_HOST_INTERRUPT_WITH_EVENT_CODE, 1),
    Selector::new(QUERY_BOARD_CONFIGURATION_BLOCK, 1),
    Selector::new(EEPROM_OPERATION, 1),
    Selector::new(DISPLAY_MODE_BYTE, 3),
    Selector::new(CONTEXT_DEACTIVATE, 1),
    Selector::new(CONTEXT_IMAGE_LOAD, 4),
    Selector::new(WINDOW_CLIP_SETUP, 11),
    Selector::new(WINDOW_READ_CLIP_SET_UP, 3),
    Selector::new(DISPLAY_BANK_SET_UP, 2),
    Selector::new(DESCRIBE_DMA_PAGE_POOL, 3),
    Selector::new(START_INPUT_DMA, 6),
    Selector::new(COPY_LIST_BY_INPUT_DMA_THEN_LIST_DMA, 2),
    Selector::new(START_SELECT_FEEDBACK_RETURN_DMA, 6),
    Selector::new(FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE, 2),
    Selector::new(EXECUTE_LIST_BY_LIST_DMA, 2),
    Selector::new(EXECUTE_LIST_BY_INPUT_DMA, 2),
    Selector::new(EXECUTE_LIST_BY_LIST_DMA_REPEATED, 1),
    Selector::new(EXECUTE_LIST_BY_INPUT_DMA_REPEATED, 1),
    Selector::new(COPY_USER_LIST_BY_LIST_DMA, 4),
    Selector::new(MOVE_LIST_BY_LIST_DMA, 3),
    Selector::new(PUBLISH_RETRACE_SEQUENCE_NUMBER, 1),
    Selector::new(PIPE_MEMORY_RESERVE, 4),
    Selector::new(PIPE_MEMORY_COPY_DOWN, 4),
    Selector::new(PIPE_MEMORY_SWAP_OUT, 8),
    Selector::new(PIPE_MEMORY_SWAP_IN, 6),
    Selector::new(PIPE_MEMORY_SUSPEND_WITH_EVENT_CODE, 1),
    Selector::new(PIPE_MEMORY_RESUME, 0),
    Selector::new(DISTRIBUTOR_NO_OP_PADDING_WORD, 1),
    Selector::new(WINDOW_SYSTEM_CONTEXT_INITIALISE, 3),
    Selector::new(WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, 13),
    Selector::new(WINDOW_SYSTEM_DRAWABLE_ORIGIN_AND_SIZE_LOAD, 4),
    Selector::new(WINDOW_SYSTEM_FOUR_WORD_DATA_PRELOAD_BYTE_FORM, 4),
    Selector::new(WINDOW_SYSTEM_CLIP_LOAD, 5),
    Selector::new(WINDOW_SYSTEM_SCISSOR_LOAD, 4),
    Selector::new(LOAD_COLOUR_MAP_ENTRIES, 2),
    Selector::new(LOAD_GAMMA_TABLE_ENTRIES, 2),
    Selector::new(LOAD_CURSOR_IMAGE_BLOCK, 1),
    Selector::new(LOAD_CURSOR_COLOUR_WORDS, 1),
    Selector::new(LOAD_ONE_ENTRY_OF_DISPLAY_ID_TABLE, 2),
    Selector::new(WINDOW_CREATED_OR_DESTROYED_NOTICE_TO_COLOUR_MAP_CHIP, 1),
    Selector::new(WINDOW_SYSTEM_PIXEL_PATH_SET_UP, 7),
    Selector::new(WINDOW_SYSTEM_PIXEL_READ, 2),
    Selector::new(WINDOW_SYSTEM_PIXEL_WRITE, 2),
    Selector::new(WINDOW_SYSTEM_PIXEL_PATH_FINISH, 1),
    Selector::new(WINDOW_SYSTEM_PIXEL_COPY, 4),
    Selector::new(LOAD_WINDOW_SYSTEM_STIPPLE, 1),
    Selector::new(STORE_WINDOW_SYSTEM_STIPPLE, 3),
    Selector::new(ROTATE_WINDOW_SYSTEM_STIPPLE, 3),
    Selector::new(DRAW_ABSOLUTE_POINTS, 1),
    Selector::new(DRAW_ABSOLUTE_POINTS_BURST_OF_6, 6),
    Selector::new(WINDOW_SYSTEM_1_WORD_DATA_PRELOAD, 1),
    Selector::new(WINDOW_SYSTEM_3_WORD_DATA_PRELOAD, 3),
    Selector::new(WINDOW_SYSTEM_5_WORD_DATA_PRELOAD, 5),
    Selector::new(DRAW_ABSOLUTE_POINTS_BURST_OF_24, 24),
    Selector::new(DRAW_POINTS_IN_BACKGROUND_COLOUR, 1),
    Selector::new(DRAW_POINTS_IN_BACKGROUND_COLOUR_BURST_OF_6, 6),
    Selector::new(DRAW_POINTS_IN_BACKGROUND_COLOUR_BURST_OF_24, 24),
    Selector::new(DRAW_RELATIVE_POINTS, 1),
    Selector::new(DRAW_RELATIVE_POINTS_BURST_OF_6, 6),
    Selector::new(DRAW_RELATIVE_POINTS_BURST_OF_24, 24),
    Selector::new(DRAW_LINE_SEGMENTS, 2),
    Selector::new(DRAW_LINE_SEGMENTS_BURST_OF_6, 12),
    // The polyline tokens' word counts are the X server's sender's (established):
    // 1, 7, 2 and 7. A longer polyline arrives as n - 1 words of data preload and
    // the one-word token for the last point (the pipe's consumption is inferred).
    Selector::new(DRAW_CONNECTED_POLYLINE, 1),
    Selector::new(DRAW_CONNECTED_POLYLINE_BURST_OF_6, 7),
    Selector::new(DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS, 2),
    Selector::new(DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS_BURST_OF_6, 7),
    Selector::new(WINDOW_SYSTEM_2_WORD_DATA_PRELOAD, 2),
    Selector::new(WINDOW_SYSTEM_4_WORD_DATA_PRELOAD, 4),
    Selector::new(WINDOW_SYSTEM_6_WORD_DATA_PRELOAD, 6),
    Selector::new(WINDOW_SYSTEM_8_WORD_DATA_PRELOAD, 8),
    Selector::new(WINDOW_SYSTEM_10_WORD_DATA_PRELOAD, 10),
    Selector::new(FILL_BOXES_TWO_CORNERS, 2),
    Selector::new(FILL_BOXES_TWO_CORNERS_BURST_OF_6, 12),
    Selector::new(FILL_RECTANGLES_ORIGIN_WIDTH_HEIGHT, 2),
    Selector::new(FILL_RECTANGLES_ORIGIN_WIDTH_HEIGHT_BURST_OF_6, 12),
    Selector::new(FILL_SPANS, 3),
    Selector::new(FILL_SPANS_BURST_OF_2, 5),
    Selector::new(FILL_SPANS_BURST_OF_6, 13),
    Selector::new(WRITE_GATHER_FLUSH_LAST_SLOT, 1),
    Selector::new(DISPLAY_GENERATOR_REQUEST_CARRIER, 1),
];

/// The GL client's commands, keyed by the low 16 bits of the opening word.
pub(super) const GL: &[Selector] = &[
    Selector::new(CONFIGURE_FRAME_BUFFER_LAYOUT, 10),
    Selector::new(FLUSH_AND_REBUILD_HARDWARE_TOKEN_MAP_FOR_PACKED_VERTEX, 1),
    Selector::new(CONFIGURE_READ_SIDE_FRAME_BUFFER_LAYOUT, 10),
    Selector::new(SELECT_VIDEO_SOURCE, 1),
    Selector::new(SELECT_ACTIVE_TEXTURE_UNIT, 1),
    Selector::new(CHANGE_POST_TEXTURE_SCALE_AND_BIAS, 4),
    Selector::new(LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE, 1),
    Selector::new(RESTART_SELECT_OR_FEEDBACK, 1),
    Selector::new(QUERY_LINE_MODE_LIMIT_TABLE, 1),
    Selector::new(SET_LAZY_VALIDATION_MODE, 1),
    Selector::new(SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES, 1),
    Selector::new(CLEAR_SELECTION_NAME_STACK, 1),
    Selector::new(REPLACE_TOP_SELECTION_NAME, 1),
    Selector::new(POP_SELECTION_NAME, 1),
    Selector::new(PUSH_SELECTION_NAME, 1),
    Selector::new(SELECT_FEEDBACK_RECORD_TYPE, 1),
    Selector::new(INSERT_FEEDBACK_PASS_THROUGH_MARKER, 1),
    Selector::new(SET_VIEWPORT, 4),
    Selector::new(SET_SCISSOR_BOX, 4),
    Selector::new(RECORD_ERROR_CODE, 1),
    Selector::new(SET_COLOUR_CLEAR_VALUE, 4),
    Selector::new(CLEAR_SELECTED_BUFFERS, 1),
    Selector::new(SET_ACCUMULATION_BUFFER_CLEAR_VALUE, 4),
    Selector::new(ACCUMULATION_BUFFER_OPERATION, 2),
    Selector::new(SET_COLOUR_INDEX_CLEAR_VALUE, 1),
    Selector::new(SET_STENCIL_CLEAR_VALUE, 1),
    Selector::new(SET_DEPTH_CLEAR_VALUE, 2),
    Selector::new(BEGIN_PRIMITIVE_POINTS, 1), // gl_begin_first + 0
    Selector::new(BEGIN_PRIMITIVE_INDEPENDENT_LINES, 1), // gl_begin_first + 1
    Selector::new(BEGIN_PRIMITIVE_LINE_LOOP, 1), // gl_begin_first + 2
    Selector::new(BEGIN_PRIMITIVE_LINE_STRIP, 1), // gl_begin_first + 3
    Selector::new(BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, 1), // gl_begin_first + 4
    Selector::new(BEGIN_PRIMITIVE_TRIANGLE_STRIP, 1), // gl_begin_first + 5
    Selector::new(BEGIN_PRIMITIVE_TRIANGLE_FAN, 1), // gl_begin_first + 6
    Selector::new(BEGIN_PRIMITIVE_INDEPENDENT_QUADS, 1), // gl_begin_first + 7
    Selector::new(BEGIN_PRIMITIVE_QUAD_STRIP, 1), // gl_begin_first + 8
    Selector::new(BEGIN_PRIMITIVE_POLYGON, 1),
    Selector::new(END_OPEN_PRIMITIVE, 1),
    Selector::new(BEGIN_PRIMITIVE_RECTANGLE, 1),
    Selector::new(FLUSH_VERTEX_PIPELINE, 1),
    Selector::new(FINISH_DRAIN_PIPE_AND_SIGNAL_COMPLETION, 1),
    Selector::new(FLUSH_RASTER_BOARDS, 1),
    Selector::new(QUERY_PENDING_ERROR, 1),
    Selector::new(QUERY_CURRENT_VERTEX_ATTRIBUTES, 1),
    Selector::new(QUERY_RASTER_POSITION_STATE, 1),
    Selector::new(LOAD_ENGINE_TRANSFORM_ATTRIBUTES, 1),
    Selector::new(POP_TEXTURE_GENERATION_STATE, 1),
    Selector::new(SET_DEPTH_RANGE, 4),
    Selector::new(SELECT_CURRENT_MATRIX_STACK, 1),
    Selector::new(MULTIPLY_CURRENT_MATRIX_BY_FRUSTUM, 12),
    Selector::new(LOAD_IDENTITY_MATRIX, 1),
    Selector::new(LOAD_MATRIX_16_FLOATS, 16),
    Selector::new(LOAD_MATRIX_16_DOUBLES, 32),
    Selector::new(MULTIPLY_BY_MATRIX_16_FLOATS, 16),
    Selector::new(MULTIPLY_BY_MATRIX_16_DOUBLES, 32),
    Selector::new(MULTIPLY_CURRENT_MATRIX_BY_ORTHOGRAPHIC_PROJECTION, 12),
    Selector::new(POP_MATRIX_STACK, 1),
    Selector::new(PUSH_MATRIX_STACK, 1),
    Selector::new(ROTATE_DOUBLES, 8),
    Selector::new(ROTATE_FLOATS, 4),
    Selector::new(SCALE_DOUBLES, 6),
    Selector::new(SCALE_FLOATS, 3),
    Selector::new(TRANSLATE_DOUBLES, 6),
    Selector::new(TRANSLATE_FLOATS, 3),
    Selector::new(SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_BYTE, 1),
    Selector::new(SET_CURRENT_COLOUR_3_COMPONENTS_DOUBLE, 6),
    Selector::new(SET_CURRENT_COLOUR_3_COMPONENTS_SINGLE_FLOAT, 3),
    Selector::new(SET_CURRENT_COLOUR_3_COMPONENTS_SIGNED_32_BIT_INTEGER, 3),
    Selector::new(SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_BYTE, 1),
    Selector::new(SET_CURRENT_COLOUR_3_COMPONENTS_UNSIGNED_32_BIT_INTEGER, 3),
    Selector::new(SET_CURRENT_COLOUR_4_COMPONENTS_DOUBLE, 8),
    Selector::new(SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, 4),
    Selector::new(SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_32_BIT_INTEGER, 4),
    Selector::new(SET_CURRENT_COLOUR_4_COMPONENTS_SIGNED_SHORT, 2),
    Selector::new(SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_BYTE, 1),
    Selector::new(SET_CURRENT_COLOUR_4_COMPONENTS_UNSIGNED_32_BIT_INTEGER, 4),
    Selector::new(SET_CURRENT_COLOUR_INDEX_DOUBLE, 2),
    Selector::new(SET_CURRENT_COLOUR_INDEX_SINGLE_FLOAT, 1),
    Selector::new(SET_CURRENT_COLOUR_INDEX_SIGNED_32_BIT_INTEGER, 1),
    Selector::new(SET_CURRENT_COLOUR_INDEX_SIGNED_SHORT, 1),
    Selector::new(SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_BYTE, 1),
    Selector::new(SET_CURRENT_NORMAL_3_COMPONENTS_DOUBLE, 6),
    Selector::new(SET_CURRENT_NORMAL_3_COMPONENTS_SINGLE_FLOAT, 3),
    Selector::new(SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_32_BIT_INTEGER, 3),
    Selector::new(SET_CURRENT_NORMAL_3_COMPONENTS_SIGNED_SHORT, 1),
    Selector::new(SET_CURRENT_TEXTURE_COORDINATE_1_COMPONENTS_SINGLE_FLOAT, 1),
    Selector::new(SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT, 2),
    Selector::new(SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SIGNED_SHORT, 1),
    Selector::new(SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT, 3),
    Selector::new(SET_CURRENT_TEXTURE_COORDINATE_4_COMPONENTS_SINGLE_FLOAT, 4),
    Selector::new(SUBMIT_VERTEX_2_COMPONENTS_DOUBLE, 4),
    Selector::new(SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT, 2),
    Selector::new(SUBMIT_VERTEX_2_COMPONENTS_SIGNED_32_BIT_INTEGER, 2),
    Selector::new(SUBMIT_VERTEX_2_COMPONENTS_SIGNED_SHORT, 1),
    Selector::new(SUBMIT_VERTEX_3_COMPONENTS_DOUBLE, 6),
    Selector::new(SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, 3),
    Selector::new(SUBMIT_VERTEX_3_COMPONENTS_SIGNED_32_BIT_INTEGER, 3),
    Selector::new(SUBMIT_VERTEX_3_COMPONENTS_SIGNED_SHORT, 1),
    Selector::new(SUBMIT_VERTEX_4_COMPONENTS_DOUBLE, 8),
    Selector::new(SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT, 4),
    Selector::new(SUBMIT_VERTEX_4_COMPONENTS_SIGNED_32_BIT_INTEGER, 4),
    Selector::new(SUBMIT_VERTEX_4_COMPONENTS_SIGNED_SHORT, 2),
    Selector::new(SET_CURRENT_EDGE_FLAG, 1),
    Selector::new(SET_RASTER_POSITION_2_COMPONENTS_FLOAT, 2),
    Selector::new(SET_RASTER_POSITION_3_COMPONENTS_FLOAT, 3),
    Selector::new(SET_RASTER_POSITION_4_COMPONENTS_FLOAT, 4),
    Selector::new(SET_CURRENT_RASTER_POSITION, 15),
    Selector::new(SET_UP_1D_EVALUATOR_GRID, 4),
    Selector::new(SET_UP_2D_EVALUATOR_GRID, 8),
    Selector::new(SKIP_2_WORDS, 1),
    Selector::new(SKIP_4_WORDS, 3),
    Selector::new(SKIP_5_WORDS, 4),
    Selector::new(SET_POINT_PARAMETER, 2),
    Selector::new(SET_POINT_PARAMETERS, 4),
    Selector::new(SET_SPRITE_PARAMETER, 2),
    Selector::new(SET_SPRITE_PARAMETERS, 4),
    Selector::new(SET_MULTISAMPLE_COVERAGE_MASK, 2),
    Selector::new(SET_MULTISAMPLE_PATTERN, 1),
    Selector::new(LOAD_1D_CONVOLUTION_FILTER, 7),
    Selector::new(LOAD_2D_CONVOLUTION_FILTER, 8),
    Selector::new(COPY_1D_CONVOLUTION_FILTER_FROM_FRAME_BUFFER, 4),
    Selector::new(COPY_2D_CONVOLUTION_FILTER_FROM_FRAME_BUFFER, 5),
    Selector::new(QUERY_CONVOLUTION_FILTER, 3),
    Selector::new(LOAD_SEPARABLE_2D_FILTER, 9),
    Selector::new(QUERY_SEPARABLE_FILTER, 3),
    Selector::new(SET_PIXEL_TRANSFORM_PARAMETERS_INTEGER, 3),
    Selector::new(SET_PIXEL_TRANSFORM_PARAMETERS_FLOAT, 6),
    Selector::new(RESOLVE_WHETHER_PIXEL_FORMAT_IS_SINGLE_COMPONENT, 4),
    Selector::new(LOAD_COLOUR_TABLE, 9),
    Selector::new(QUERY_COLOUR_TABLE, 4),
    Selector::new(SET_COLOUR_TABLE_PARAMETERS, 6),
    Selector::new(QUERY_COLOUR_TABLE_PARAMETERS, 2),
    Selector::new(SET_UP_HISTOGRAM, 5),
    Selector::new(RESET_HISTOGRAM, 1),
    Selector::new(QUERY_HISTOGRAM, 1),
    Selector::new(QUERY_HISTOGRAM_PARAMETERS, 2),
    Selector::new(SET_UP_MIN_MAX_TRACKING, 2),
    Selector::new(RESET_MIN_MAX_TRACKING, 1),
    Selector::new(QUERY_MIN_MAX_RESULTS, 1),
    Selector::new(QUERY_MIN_MAX_PARAMETERS, 1),
    Selector::new(SET_LIGHT_PARAMETER_FLOAT_FORM, 6),
    Selector::new(SET_LIGHT_PARAMETER_INTEGER_FORM, 6),
    Selector::new(SET_LIGHTING_MODEL_PARAMETER_FLOAT_FORM, 5),
    Selector::new(SET_LIGHTING_MODEL_PARAMETER_INTEGER_FORM, 5),
    Selector::new(SET_MATERIAL_PARAMETER_FLOAT_FORM, 6),
    Selector::new(SET_COLOUR_MATERIAL_TRACKING, 2),
    Selector::new(ENABLE_CAPABILITY, 1),
    Selector::new(DISABLE_CAPABILITY, 1),
    Selector::new(SET_FLAT_OR_SMOOTH_SHADING, 1),
    Selector::new(SET_DEPTH_TEST, 1),
    Selector::new(SET_DEPTH_WRITE_MASK, 1),
    Selector::new(SET_BLEND_FACTORS, 2),
    Selector::new(SET_CONSTANT_BLEND_COLOUR, 4),
    Selector::new(SET_BLEND_EQUATION, 1),
    Selector::new(SET_ALPHA_TEST, 2),
    Selector::new(SET_STENCIL_TEST, 3),
    Selector::new(SET_STENCIL_OPERATIONS, 3),
    Selector::new(SET_STENCIL_WRITE_MASK, 1),
    Selector::new(SET_COLOUR_WRITE_MASK, 4),
    Selector::new(SET_COLOUR_INDEX_WRITE_MASK, 1),
    Selector::new(SET_LOGIC_OPERATION, 1),
    Selector::new(SET_FACE_CULLING, 1),
    Selector::new(SET_FRONT_FACE_WINDING, 1),
    Selector::new(SET_POLYGON_FILL_MODE, 2),
    Selector::new(SET_POLYGON_OFFSET, 2),
    Selector::new(SET_POLYGON_OFFSET_EXTENSION_FORM, 2),
    Selector::new(LOAD_POLYGON_STIPPLE_PATTERN, 3),
    Selector::new(SET_LINE_WIDTH, 2),
    Selector::new(SET_POINT_SIZE, 2),
    Selector::new(SET_LINE_STIPPLE, 2),
    Selector::new(SET_FOG_PARAMETER_FLOAT_FORM, 5),
    Selector::new(SET_FOG_PARAMETER_INTEGER_FORM, 5),
    Selector::new(SET_USER_CLIP_PLANE, 9),
    Selector::new(SET_QUALITY_HINT, 2),
    Selector::new(SELECT_DRAW_BUFFER, 1),
    Selector::new(SELECT_READ_BUFFER, 1),
    Selector::new(SET_PIXEL_ZOOM_FACTORS, 2),
    Selector::new(SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM, 2),
    Selector::new(SET_PIXEL_STORE_PARAMETER_INTEGER_FORM, 2),
    Selector::new(LOAD_PIXEL_MAP_FLOAT_VALUES, 4),
    Selector::new(LOAD_PIXEL_MAP_UNSIGNED_32_BIT_VALUES, 4),
    Selector::new(LOAD_PIXEL_MAP_UNSIGNED_16_BIT_VALUES, 4),
    Selector::new(SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, 5),
    Selector::new(SET_TEXTURE_ENVIRONMENT_PARAMETER_INTEGER_FORM, 5),
    Selector::new(DRAW_BITMAP_FROM_DISPLAY_LIST, 8),
    Selector::new(DRAW_BITMAP, 9),
    Selector::new(DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, 8),
    Selector::new(READ_PIXEL_RECTANGLE_BACK_TO_HOST, 7),
    Selector::new(COPY_PIXEL_RECTANGLE_WITHIN_FRAME_BUFFER, 7),
    Selector::new(ASK_WHETHER_COPY_SOURCE_OVERLAPS_ITS_DESTINATION, 4),
    Selector::new(QUERY_TEXTURE_IMAGE, 18),
    Selector::new(SUSPEND_PIXEL_MODE_STATE_AROUND_TRANSFER, 2),
    Selector::new(ASK_WHETHER_CONVOLUTION_COPY_OVERLAPS, 4),
    Selector::new(QUERY_STATE_GENERIC, 1),
    Selector::new(PUSH_ATTRIBUTE_STACK, 1),
    Selector::new(POP_ATTRIBUTE_STACK, 1),
    Selector::new(POP_SAVED_LIGHT_STATE, 1),
    Selector::new(BIND_ARM_TEXTURE_CACHE_ENTRY, 1),
    Selector::new(FREE_ARM_TEXTURE_CACHE_ENTRY, 1),
    Selector::new(DRAW_TEXELS_INTO_TEXTURE_MEMORY, 20),
    Selector::new(SET_TEXEL_STORE_PARAMETER, 2),
    Selector::new(TEXTURE_LOAD_BY_IDENTIFIER, 1),
    Selector::new(BIND_TEXTURE_BY_IDENTIFIER, 1),
    Selector::new(BIND_2D_MIP_MAPPED_TEXTURE, 22),
    Selector::new(BIND_3D_MIP_MAPPED_TEXTURE, 14),
    Selector::new(BIND_3D_TEXTURE, 4),
    Selector::new(BIND_DETAIL_TEXTURE, 6),
    Selector::new(BIND_BICUBIC_TEXTURE, 5),
    Selector::new(BIND_2D_CLIP_MAP_TEXTURE, 8),
    Selector::new(SET_SHADOW_TEXTURE_STATE, 3),
    Selector::new(TEXTURE_LOAD_SET_UP, 11),
    Selector::new(BIND_TEXTURE_CURRENT_FORM, 5),
    // Five words, as the table's count and text agree; the recorded GL demos frame correctly with five and misframe with seven.
    Selector::new(BIND_LINEAR_TEXTURE, 5),
    Selector::new(LOAD_POST_TEXTURE_SCALE_AND_BIAS, 4),
    Selector::new(INVALIDATE_ARM_TEXTURE_CACHE_INDEX, 2),
    Selector::new(SWITCH_LEGACY_API_COMPATIBILITY_MODE, 1),
    Selector::new(SET_COLOUR_MATERIAL_VALUE, 1),
    Selector::new(SWAP_TRIANGLE_MESH_VERTEX, 1),
    Selector::new(SET_COLOUR_NORMAL_MODE_VALUE, 1),
    Selector::new(SET_TRIANGLE_MESH_MODE, 1),
    Selector::new(TEXTURE_COORDINATE_GENERATION_FLOAT_FORM, 9),
    Selector::new(SET_RASTER_POSITION_2_COMPONENTS_DOUBLE, 4),
    Selector::new(SET_RASTER_POSITION_2_COMPONENTS_32_BIT_INTEGER, 2),
    Selector::new(SET_RASTER_POSITION_2_COMPONENTS_SHORT, 2),
    Selector::new(SET_RASTER_POSITION_3_COMPONENTS_DOUBLE, 6),
    Selector::new(SET_RASTER_POSITION_3_COMPONENTS_32_BIT_INTEGER, 3),
    Selector::new(SET_RASTER_POSITION_3_COMPONENTS_SHORT, 3),
    Selector::new(SET_RASTER_POSITION_4_COMPONENTS_DOUBLE, 8),
    Selector::new(SET_RASTER_POSITION_4_COMPONENTS_32_BIT_INTEGER, 4),
    Selector::new(SET_RASTER_POSITION_4_COMPONENTS_SHORT, 4),
    Selector::new(SET_VERTEX_ARRAY_COMMAND_TABLE, 1),
    Selector::new(VERTEX_ARRAY_SET_UP, 1),
    Selector::new(BEGIN_VERTEX_ARRAY_LOOP, 3),
    Selector::new(VERTEX_ARRAY_TEAR_DOWN, 1),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_1, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_2, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_3, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_4, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_5, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_6, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_7, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_8, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_9, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_10, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_11, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_12, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_13, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_14, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_15, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_16, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_17, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_18, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_19, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_20, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_21, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_22, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_23, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_24, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_25, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_26, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_27, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_28, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_29, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_30, 0),
    Selector::new(VERTEX_ARRAY_LOOP_ENTRY_31, 0),
];

/// The video command table: the data words each command carries between its
/// number and its checksum (established for all 93 requests).
pub(super) const VIDEO: &[Option<u16>] = &[
    Some(1), // 0x00 Test
    Some(2), // 0x01 VCWrite
    Some(1), // 0x02 VCRead
    Some(10), // 0x03 DiagXmapColor
    Some(1), // 0x04 DisableDiagXmapColor
    Some(4), // 0x05 VCBusTrace
    Some(4107), // 0x06 DiagCursorFromImage
    Some(1), // 0x07 DebugMask
    Some(3), // 0x08 SetupFrameBuffer
    Some(1), // 0x09 InputSync
    Some(1), // 0x0a GetInputSync
    Some(2), // 0x0b HPhase
    Some(1), // 0x0c GetHPhase
    Some(1), // 0x0d VPhase
    Some(1), // 0x0e GetVPhase
    Some(2), // 0x0f ChannelGammaMap
    Some(1), // 0x10 GetChannelGammaMap
    Some(1), // 0x11 Fetal
    Some(1), // 0x12 GetBoardConfig
    Some(1), // 0x13 GetChannelConfig
    Some(8), // 0x14 Resize
    Some(1), // 0x15 GetResize
    Some(383), // 0x16 LoadFormat
    Some(3), // 0x17 EnableChannel
    Some(1), // 0x18 DisableChannel
    Some(16), // 0x19 LoadGenlockParams
    Some(1), // 0x1a EnableGenlock
    Some(2), // 0x1b CursorPosition
    Some(1), // 0x1c CursorChannel
    Some(1), // 0x1d CursorType
    Some(2), // 0x1e CursorPriority
    Some(2), // 0x1f ChannelBlanking
    Some(1), // 0x20 GetChannelBlanking
    Some(2), // 0x21 OutputGain
    Some(1), // 0x22 GetOutputGain
    Some(4), // 0x23 SyncSelects
    Some(1), // 0x24 GetSyncSelects
    Some(2), // 0x25 OutputPedestal
    Some(1), // 0x26 GetOutputPedestal
    Some(1), // 0x27 InitTextport
    Some(1), // 0x28 LoadTextportGlyph
    Some(4), // 0x29 LoadTextportColorIndex
    Some(1), // 0x2a EnableVocReadback
    Some(1), // 0x2b MasterClockSelect
    Some(1), // 0x2c GetMasterClockSelect
    Some(1), // 0x2d AutoInit
    Some(1), // 0x2e LoadEncoder
    Some(1), // 0x2f EncoderOutputGain
    Some(1), // 0x30 GetEncoderOutputGain
    Some(1), // 0x31 SCHPhase
    Some(1), // 0x32 GetSCHPhase
    Some(1), // 0x33 EncoderTestPattern
    Some(2), // 0x34 BindChannelToXWid
    Some(1328), // 0x35 LoadPabXilinx
    Some(4), // 0x36 LoadPabSizeRegs
    Some(2), // 0x37 LoadPabCtrlRegs
    Some(263), // 0x38 LoadDacCursor
    Some(7), // 0x39 VampireAndFieldDominance
    Some(1), // 0x3a GetVampire
    Some(1), // 0x3b FastlistChunksize
    Some(1), // 0x3c TempBlankAllChannels
    Some(1), // 0x3d GetDGInterruptState
    Some(1), // 0x3e XMapClockSelect
    Some(1), // 0x3f GetXMapClockSelect
    Some(3), // 0x40 ChannelResetRatios
    Some(1), // 0x41 ResetRMBoards
    Some(1), // 0x42 GetChannelEnables
    Some(9), // 0x43 PlacementRes
    Some(1), // 0x44 GenlockSwitching
    Some(1), // 0x45 GetGenlockSwitching
    Some(2), // 0x46 ChannelFieldDominance
    Some(1), // 0x47 GetChannelFieldDominance
    Some(1), // 0x48 BoardResetRatio
    Some(1), // 0x49 GetBoardResetRatio
    Some(1), // 0x4a GetChannelFormatInfo
    Some(5), // 0x4b ChannelGammaMapState
    Some(1), // 0x4c GetChannelGammaMapState
    Some(1), // 0x4d EnableEncoder
    Some(2), // 0x4e ResizeSyncType
    Some(7), // 0x4f Vampire
    Some(1), // 0x50 QueryMonitorName
    Some(34), // 0x51 MonitorCommand
    Some(6), // 0x52 LoadXilinx
    Some(1), // 0x53 XServerVideoInit
    Some(1), // 0x54 InternalGenlock
    Some(1), // 0x55 GetInternalGenlock
    Some(3), // 0x56 BufferParams
    Some(1), // 0x57 GetBufferParams
    Some(3), // 0x58 DataFormat
    Some(1), // 0x59 GetDataFormat
    Some(1), // 0x5a DVPMult2Freq
    Some(42), // 0x5b VBOBCommand
    Some(1), // 0x5c GetVBOBReply
];

/// The reply each video command waits for, by command number: the reply id and the bytes of data
/// after it, or `None` for a command nobody waits on. Established for all 93 requests (32 of them
/// reply; the monitor command, which carries a payload, does not).
pub(super) const VIDEO_REPLY: &[Option<(u8, u8)>] = &[
    Some((0, 4)), // 0x00 Test
    None, // 0x01 VCWrite
    Some((1, 4)), // 0x02 VCRead
    None, // 0x03 DiagXmapColor
    None, // 0x04 DisableDiagXmapColor
    None, // 0x05 VCBusTrace
    None, // 0x06 DiagCursorFromImage
    None, // 0x07 DebugMask
    None, // 0x08 SetupFrameBuffer
    None, // 0x09 InputSync
    Some((2, 4)), // 0x0a GetInputSync
    None, // 0x0b HPhase
    Some((3, 4)), // 0x0c GetHPhase
    None, // 0x0d VPhase
    Some((4, 4)), // 0x0e GetVPhase
    None, // 0x0f ChannelGammaMap
    Some((5, 4)), // 0x10 GetChannelGammaMap
    None, // 0x11 Fetal
    Some((6, 20)), // 0x12 GetBoardConfig
    Some((7, 76)), // 0x13 GetChannelConfig
    None, // 0x14 Resize
    Some((8, 72)), // 0x15 GetResize
    None, // 0x16 LoadFormat
    None, // 0x17 EnableChannel
    None, // 0x18 DisableChannel
    None, // 0x19 LoadGenlockParams
    None, // 0x1a EnableGenlock
    None, // 0x1b CursorPosition
    None, // 0x1c CursorChannel
    None, // 0x1d CursorType
    None, // 0x1e CursorPriority
    None, // 0x1f ChannelBlanking
    Some((9, 4)), // 0x20 GetChannelBlanking
    None, // 0x21 OutputGain
    Some((10, 4)), // 0x22 GetOutputGain
    None, // 0x23 SyncSelects
    Some((11, 24)), // 0x24 GetSyncSelects
    None, // 0x25 OutputPedestal
    Some((12, 4)), // 0x26 GetOutputPedestal
    None, // 0x27 InitTextport
    None, // 0x28 LoadTextportGlyph
    None, // 0x29 LoadTextportColorIndex
    None, // 0x2a EnableVocReadback
    None, // 0x2b MasterClockSelect
    Some((13, 4)), // 0x2c GetMasterClockSelect
    None, // 0x2d AutoInit
    None, // 0x2e LoadEncoder
    None, // 0x2f EncoderOutputGain
    Some((14, 4)), // 0x30 GetEncoderOutputGain
    None, // 0x31 SCHPhase
    Some((15, 4)), // 0x32 GetSCHPhase
    None, // 0x33 EncoderTestPattern
    None, // 0x34 BindChannelToXWid
    None, // 0x35 LoadPabXilinx
    None, // 0x36 LoadPabSizeRegs
    None, // 0x37 LoadPabCtrlRegs
    None, // 0x38 LoadDacCursor
    None, // 0x39 VampireAndFieldDominance
    Some((16, 24)), // 0x3a GetVampire
    None, // 0x3b FastlistChunksize
    None, // 0x3c TempBlankAllChannels
    Some((17, 12)), // 0x3d GetDGInterruptState
    None, // 0x3e XMapClockSelect
    Some((18, 4)), // 0x3f GetXMapClockSelect
    None, // 0x40 ChannelResetRatios
    None, // 0x41 ResetRMBoards
    Some((19, 8)), // 0x42 GetChannelEnables
    None, // 0x43 PlacementRes
    None, // 0x44 GenlockSwitching
    Some((20, 4)), // 0x45 GetGenlockSwitching
    None, // 0x46 ChannelFieldDominance
    Some((21, 4)), // 0x47 GetChannelFieldDominance
    None, // 0x48 BoardResetRatio
    Some((22, 4)), // 0x49 GetBoardResetRatio
    Some((23, 20)), // 0x4a GetChannelFormatInfo
    None, // 0x4b ChannelGammaMapState
    Some((24, 16)), // 0x4c GetChannelGammaMapState
    None, // 0x4d EnableEncoder
    None, // 0x4e ResizeSyncType
    None, // 0x4f Vampire
    Some((25, 68)), // 0x50 QueryMonitorName
    None, // 0x51 MonitorCommand
    Some((27, 4)), // 0x52 LoadXilinx
    None, // 0x53 XServerVideoInit
    None, // 0x54 InternalGenlock
    Some((28, 4)), // 0x55 GetInternalGenlock
    None, // 0x56 BufferParams
    Some((29, 8)), // 0x57 GetBufferParams
    None, // 0x58 DataFormat
    Some((30, 8)), // 0x59 GetDataFormat
    None, // 0x5a DVPMult2Freq
    None, // 0x5b VBOBCommand
    Some((31, 164)), // 0x5c GetVBOBReply
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every selector of the kernel and GL tables is a host token the definitions
    /// name, once, and the table's own entries do not repeat a token.
    #[test]
    fn the_selector_tables_and_the_definitions_agree() {
        for (what, table) in [("kernel", KERNEL), ("GL", GL), ("dma fifo", DMA_FIFO)] {
            let mut seen = std::collections::BTreeSet::new();
            for s in table {
                assert!(!token_name(s.selector).is_empty(), "{what} table: {:#x} is not a token", s.selector);
                assert!(seen.insert(s.selector), "{what} table: {:#x} ({}) twice", s.selector, token_name(s.selector));
            }
        }
        // The textport grammar is its own: its numbers are not the tokens'.
        assert!(TEXTPORT.iter().all(|s| s.selector < 0x10));
    }

    /// The four polyline tokens carry 1, 7, 2 and 7 words (the X server's
    /// sender counts). A one-word token that framed as two would pull the next
    /// command's token word in as a point and desync everything after it.
    #[test]
    fn the_polyline_tokens_frame_one_seven_two_and_seven_words() {
        let argc = |sel| KERNEL.iter().find(|s| s.selector == sel).map(|s| s.argc);
        assert_eq!(argc(DRAW_CONNECTED_POLYLINE), Some(1));
        assert_eq!(argc(DRAW_CONNECTED_POLYLINE_BURST_OF_6), Some(7));
        assert_eq!(argc(DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS), Some(2));
        assert_eq!(argc(DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS_BURST_OF_6), Some(7));
    }
}
