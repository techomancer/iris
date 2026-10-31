//! Register offsets, command words, enums and the command arity helper.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.

use super::*;

// The whole aperture a driver maps.
pub const ODSY_APERTURE: u64 = 0x410000;
pub const ODSY_PART: u32 = 0xc013;
pub const ODSY_MANUFACTURER: u32 = 0x023;

// The one modeled scanout mode's retrace period, from the compiled 60 Hz 1280x1024 mode-description file
pub const ODSY_FRAME_CLOCKS: u32 = 1688 * 1066;
pub const ODSY_PIXEL_CLOCK_HZ: u32 = 107964476;

// The display channel's active size, backend entries 0x2802 (pixels) and 0x2803 (lines).
pub const ODSY_CHANNEL_WIDTH: u32 = 0x2802;
pub const ODSY_CHANNEL_HEIGHT: u32 = 0x2803;

// The crosstalk configuration block
pub const ODSY_BLOCK_END: u64 = 0x60;
pub const ODSY_IDENTITY: u64 = 0x00;
pub const ODSY_STATUS: u64 = 0x08;
pub const ODSY_ERROR_ADDR_UPPER: u64 = 0x10;
pub const ODSY_ERROR_ADDR_LOWER: u64 = 0x18;
pub const ODSY_CONTROL: u64 = 0x20;
pub const ODSY_XT_REQ_TIMEOUT: u64 = 0x28;
pub const ODSY_INT_DEST_UPPER: u64 = 0x30;
pub const ODSY_INT_DEST_LOWER: u64 = 0x38;
pub const ODSY_ERROR_COMMAND: u64 = 0x40;

// int_dest_upper/int_dest_lower
pub const ODSY_INT_DEST_WIDGET_BITS: u32 = 4;

// +0x1004, the memory controller configuration.
pub const ODSY_MEMORY_CONTROL: u64 = 0x1004;
pub const ODSY_MEMORY_CONTROL_LATENCY_2: u32 = 1 << 1;
pub const ODSY_MEMORY_CONTROL_FOUR_BANKS: u32 = 1 << 2;
pub const ODSY_MEMORY_CONTROL_SIZE_SHIFT: u32 = 3;
pub const ODSY_MEMORY_CONTROL_SIZE_MASK: u32 = 0x3;

// +0x100c, the low-level reset
pub const ODSY_RESET_CONTROL: u64 = 0x100c;

// The flag interrupt's registers, as the shipped guest OS kernel uses them
pub const ODSY_INTR_FLAGS: u64 = 0x1014;
pub const ODSY_INTR_CLEAR: u64 = 0x101c;
pub const ODSY_INTR_ENABLE_SET: u64 = 0x1034;
pub const ODSY_INTR_ENABLE_CLEAR: u64 = 0x103c;

// The routed-interrupt destination and per-source vector table
pub const ODSY_ROUTE_DEST_UPPER: u64 = 0x1044;
pub const ODSY_ROUTE_DEST_LOWER: u64 = 0x104c;
pub const ODSY_ROUTE_VECTORS: u64 = 0x1054;
pub const ODSY_STATUS0: u64 = 0x1064;
pub const ODSY_MEMORY_CONFIG: u64 = 0x106c;
pub const ODSY_FLCTL_CONFIG: u64 = 0x1070;
pub const ODSY_FLCTL_TARGET: u64 = 0x1074;
pub const ODSY_SW_TOKEN: u64 = 0x107c;
pub const ODSY_NIC_PROM: u64 = 0x10d8;
pub const ODSY_NIC_KERNEL: u64 = 0x10dc;

// The DMA-sync command's (ODSY_REG_HOST_WRITE, below) host-write base
pub const ODSY_HOST_WRITE_ADDR_UPPER: u64 = 0x4034;
pub const ODSY_HOST_WRITE_ADDR_LOWER: u64 = 0x403c;

// The statistics region.
pub const ODSY_STATS_ADDR_UPPER: u64 = 0x401c;
pub const ODSY_STATS_ADDR_LOWER: u64 = 0x4024;
pub const ODSY_STATS_CONTROL: u64 = 0x402c;

// The context-shadow address
pub const ODSY_CTX_SHADOW_ADDR_UPPER: u64 = 0x4004;
pub const ODSY_CTX_SHADOW_ADDR_LOWER: u64 = 0x400c;

// bits 28:25 of ROUTE_DEST_UPPER, and two of the four source bytes within ROUTE_VECTORS.
pub const ODSY_ROUTE_DEST_WIDGET_SHIFT: u32 = 25;
pub const ODSY_ROUTE_VECTOR_SOFTSYNC_SHIFT: u32 = 24;
pub const ODSY_ROUTE_VECTOR_RETRACE_SHIFT: u32 = 16;
pub const ODSY_ROUTE_VECTOR_DMA_SHIFT: u32 = 8;
pub const ODSY_ROUTE_VECTOR_FLAG_SHIFT: u32 = 0;

// The "Buzz registers" the stock IP35 BaseIO PROM's graphics diagnostic tests, adjoining but distinct from the kernel driver's own register scan.
pub const ODSY_BUZZ_1: u64 = 0x1078;
pub const ODSY_BUZZ_2: u64 = 0x1088;
pub const ODSY_BUZZ_3: u64 = 0x1098;
pub const ODSY_BUZZ_4: u64 = 0x10a8;
pub const ODSY_BUZZ_5: u64 = 0x10b8;
pub const ODSY_BUZZ_6: u64 = 0x10c0;

// The swap-control register bank, as the guest driver's buffer swap and vertical-retrace handler use it, interleaved with the Buzz registers above but ...
pub const ODSY_SWAP_ARM: u64 = 0x10b4;
pub const ODSY_SWAP_PENDING: u64 = 0x10bc;
pub const ODSY_SWAP_CONTROL: u64 = 0x10c4;
pub const ODSY_SWAP_CONTROL_COMPANION: u64 = 0x10cc;
pub const ODSY_DISPLAYED_BUFFER: u64 = 0x10d4;

// Front and back storage.
pub const ODSY_REG_DRAW_MASK: u32 = 0x0424;
pub const ODSY_BUFFERS: u32 = 2;
pub const ODSY_DRAW_MASK_BITS: u32 = 0xf;
pub const ODSY_DRAW_MASK_BUFFER0: u32 = 0x3; // left and right eye of buffer 0
pub const ODSY_DRAW_MASK_BUFFER1: u32 = 0xc; // left and right eye of buffer 1
pub const ODSY_DRAW_MASK_RIGHT: u32 = 0xa; // the right-eye bits
pub const ODSY_DRAW_MASK_RESET: u32 = 0x1; // buffer 0, left: an approximation

// Reset values a real Tezro V12's firmware diagnostics printed for a healthy board
pub const ODSY_XT_REQ_TIMEOUT_RESET: u32 = 0x000fffff;
pub const ODSY_FLCTL_CONFIG_RESET: u32 = 0x0001c000;

// Status register 0x1064 (odsy status0).
pub const ODSY_STATUS0_FIFO_ENABLED: u32 = 0x10000000;
pub const ODSY_STATUS0_FIFO_LOW_WATER: u32 = 0x00020000;
pub const ODSY_STATUS0_FIFO_EMPTY: u32 = 0x00010000;
pub const ODSY_STATUS0_FIFO_HIGH_WATER: u32 = 0x00008000;

// The command FIFO's low-water mark, in words
pub const ODSY_CFIFO_LOW_WATER: u32 = 16;

// 0x1074, flow-control attach.
pub const ODSY_FLCTL_TARGET_SHIFT: u32 = 12;
pub const ODSY_FLCTL_TARGET_MASK: u32 = 0x1f;
pub const ODSY_FLCTL_INTR_TARGET_SHIFT: u32 = 17;

// FIFO set-up registers.
pub const ODSY_REG_FIFO_SETUP_FIRST: u32 = 0x6b0;
pub const ODSY_REG_FIFO_SETUP_LAST: u32 = 0x6b3;
pub const ODSY_REG_FIFO_HIGH_WATER: u32 = 0x6b2;
pub const ODSY_REG_FIFO_SETUP_7B5: u32 = 0x7b5;
pub const ODSY_FIFO_SETUP_7B5_INDEX: usize = 4;
// The hardware-sync timeout in board clocks. The kernels' FIFO set-up writes
// 10^9 after the program load, and when a hardware sync waits longer the
// board raises flags bit 1. The model has no hardware-sync wait (which
// command words are syncs is not established), so the timeout is stored and
// never expires.
pub const ODSY_REG_SYNC_TIMEOUT: u32 = 0x7f0;

// ODSY_INTR_FLAGS' low-water bit, the command-FIFO low-water interrupt (the kernel's flags decode names it as the low-water flag bit).
pub const ODSY_INTR_FIFO_LOW_WATER: u32 = 0x00000200;
// Flags bit 25, raised as the command FIFO reaches its high-water mark (inferred from its name).
pub const ODSY_INTR_FIFO_HIGH_WATER: u32 = 0x02000000;

// SGI's MicroLAN control register, the layout the Bridge, the XBridge and the Hub all carry
pub const ODSY_NIC_LEVEL: u32 = 1 << 0;
pub const ODSY_NIC_DONE: u32 = 1 << 1;
pub const ODSY_NIC_SAMPLE_SHIFT: u32 = 2;
pub const ODSY_NIC_SAMPLE_MASK: u32 = 0xff << ODSY_NIC_SAMPLE_SHIFT;
pub const ODSY_NIC_PULSE_SHIFT: u32 = 10;
pub const ODSY_NIC_PULSE_MASK: u32 = 0x3ff << ODSY_NIC_PULSE_SHIFT;

// Board-identity part family 0x0b, one 32-byte page, 48-bit serial.
pub const ODSY_NIC_FAMILY: u32 = 0x0b;
pub const ODSY_NIC_PAGE: u32 = 32;
pub const ODSY_NIC_ROM_SIZE: usize = 160;
pub const ODSY_NIC_SERIAL_MASK: u64 = (1u64 << 48) - 1;
pub const ODSY_NIC_RECORD_DEFAULT: &str = "Name:ODY128VERSIONB";

// The display back end's indexed register file is a 32-bit word FIFO, not a selector-plus-data register.
pub const ODSY_BACKEND_SELECT: u64 = 0x400000;
pub const ODSY_BACKEND_READ_BASE: u64 = 0x220000;
pub const ODSY_BACKEND_SELECTOR_INDEX_SHIFT: u32 = 14;

// The header's index (bits 27:14) and count (bits 13:0) fields.
pub const ODSY_BACKEND_SELECTOR_INDEX_MASK: u32 = 0x3fff;
pub const ODSY_BACKEND_COUNT_MASK: u32 = 0x3fff;

// A header word's tag, bits 29:28, always 0b11 in every observed record.
pub const ODSY_BACKEND_TAG_SHIFT: u32 = 28;
pub const ODSY_BACKEND_TAG_MASK: u32 = 0x3;
pub const ODSY_BACKEND_TAG_HEADER: u32 = 0x3;

// The colour lookup RAM
pub const ODSY_LUT_SIZE: u32 = 0x2000;
pub const ODSY_LUT_ENTRY_MASK: u32 = 0x3fffffff;
pub const ODSY_LUT_RED_SHIFT: u32 = 22;
pub const ODSY_LUT_GREEN_SHIFT: u32 = 12;
pub const ODSY_LUT_BLUE_SHIFT: u32 = 2;

// The per-drawable display-mode registers at ODSY_DISPMODE_BASE + DID (drawable id 0..31).
pub const ODSY_DISPMODE_BASE: u32 = 0x2900;
pub const ODSY_DISPMODE_COUNT: u32 = 32;
pub const ODSY_DISPMODE_GROUP_SHIFT: u32 = 4;
pub const ODSY_DISPMODE_GROUP_MASK: u32 = 0x1f;

// Bits 11:9 of an RGB display mode follow the visual's channel width.
pub const ODSY_DISPMODE_WIDTH_SHIFT: u32 = 9;
pub const ODSY_DISPMODE_WIDTH_MASK: u32 = 0x7;
pub const ODSY_DISPMODE_WIDTH_4: u32 = 4;
pub const ODSY_DISPMODE_WIDTH_5: u32 = 3;

// Colour resolution at scanout.
pub const ODSY_GAMMA_GROUP: u32 = 26; // base 0x1a00: groups >= 26 are RGB
pub const ODSY_GAMMA_BASE: u32 = ODSY_GAMMA_GROUP * 256;
pub const ODSY_CMAP_UNVERIFIED_GROUP: u32 = 25; // base 0x1900, no software gamma
pub const ODSY_DID_MAIN_SHIFT: u32 = 3;
pub const ODSY_DID_OVERLAY_MASK: u32 = 0x7;

// The overlay.
pub const ODSY_OVERLAY_MODE_BASE: u32 = 0x2920;
pub const ODSY_OVERLAY_IDS: u32 = 8;
pub const ODSY_OVERLAY_MODE_ENABLE: u32 = 0x1;
pub const ODSY_OVERLAY_MODE_GROUP_SHIFT: u32 = 3;
pub const ODSY_OVERLAY_MODE_GROUP_MASK: u32 = 0x1f;
pub const ODSY_OVERLAY_MODE_KNOWN: u32 = 0xf9;

// The id-plane destination's drawing-mask buffers
pub const ODSY_AUX_ID_BUFFER: u32 = 0;
pub const ODSY_AUX_OVERLAY_BUFFER: u32 = 1;

// 0x608's bit 31, the software buffer state, beside the drawable's base.
pub const ODSY_DRAW_WORD_STATE: u32 = 0x80000000;

// The colour plane word of an index pixel
pub const ODSY_INDEX_MASK: u32 = 0xfff;
pub const ODSY_I2C_MAIN_CMD_INDEX: u32 = 0x2780;
pub const ODSY_I2C_OPT_CMD_INDEX: u32 = 0x2740;
pub const ODSY_I2C_MAIN_CMD_READ: u64 = 0x233c00;
pub const ODSY_I2C_MAIN_STATUS_READ: u64 = 0x233c08;
pub const ODSY_I2C_OPT_CMD_READ: u64 = 0x233a00;
pub const ODSY_I2C_OPT_STATUS_READ: u64 = 0x233a08;

// The back-end (PB&J) version, bits 2:0.
pub const ODSY_BACKEND_VERSION: u64 = 0x23380c;

// Status bits this model computes at read time
pub const ODSY_I2C_STATUS_DONE: u32 = 1 << 9;
pub const ODSY_I2C_STATUS_READY: u32 = 1 << 22;

// The command FIFO
pub const ODSY_CMD_FIFO: u64 = 0x110000;

// The per-context USER command FIFO.
pub const ODSY_USER_CMD_FIFO: u64 = 0x100000;

// The per-context command port
pub const ODSY_CTX_SWITCH_PORT: u64 = 0x118000;
pub const ODSY_CTX_SWITCH_OPCODE_SHIFT: u32 = 40;
pub const ODSY_CTX_SWITCH_OPCODE: u32 = 0x7;
pub const ODSY_CTX_SWITCH_VALID: u64 = 1 << 0;
pub const ODSY_CTX_SWITCH_DONE_OFFSET: u64 = 0x8fc;

// The command-FIFO self-test read-back.
pub const ODSY_CFIFO_SELFTEST: u64 = 0x201594;
pub const ODSY_CFIFO_SELFTEST_OK: u32 = 0x003d0046;

// A "register-select" word, 0x0001nnnn with the register in the low 16 bits.
pub const ODSY_CMD_REGISTER_SHAPE: u32 = 0xffff0000;
pub const ODSY_CMD_REGISTER_TAG: u32 = 0x00010000;

// The operand-count field every command word carries
pub const ODSY_CMD_ARITY_SHIFT: u32 = 10;
pub const ODSY_CMD_ARITY_MASK: u32 = 0xf;
pub const ODSY_CMD_ARITY_BURST: u32 = 16;

// A GL-class word's operands-follow bit (Board::cmd_classify()).
pub const ODSY_CMD_GL_OPERANDS: u32 = 0x00800000;
pub const ODSY_REG_QUICK_FLUSH_ARG: u32 = 0x0443; // 1 operand: 0xfa or 0x96 seen
pub const ODSY_REG_STROBE: u32 = 0x0046; // 0 operands, repeated in flush
pub const ODSY_REG_FULL_FLUSH_EXTRA: u32 = 0x0019; // 0 operands, full flush only
pub const ODSY_REG_GL_BEGIN: u32 = 0x4400; // 1 operand: the GL primitive
pub const ODSY_REG_GL_END: u32 = 0x4001; // 0 operands

// 0x0405, glyph pixel setup, is also the texturing mode
pub const ODSY_REG_PIXEL_MODE_EXT: u32 = 0x1453; // 5 operands: glyph pixel setup

// Block 0x258 with one operand, 0x00120000
pub const ODSY_REG_BLIT_PREFIX: u32 = 0x0658;

// The host-write/sync command the kernel driver's DMA completion wait uses
pub const ODSY_REG_HOST_WRITE: u32 = 0x0a5a;

// The page-table pixel/texture DMA descriptor.
pub const ODSY_DMA_DESC_UPLOAD: u32 = 0x00000e01; // opcode 0xe01: either direction
pub const ODSY_DMA_DESC_READBACK: u32 = 0x00000e03; // opcode 0xe03: region path only
pub const ODSY_DMA_DESC_OPERANDS: u32 = 3; // the three words after the opcode
pub const ODSY_DMA_DESC_ADDR_LOW_MASK: u32 = 0x1ffff; // bits 16:0 of a packed half
pub const ODSY_DMA_DESC_ADDR_HIGH_SHIFT: u32 = 23; // bits 29:23: half bits 23:17
pub const ODSY_DMA_DESC_ADDR_HIGH_MASK: u32 = 0x7f;

// The two buffer-select doublewords the kernel driver's buffer-select push sends ahead of a swap, 0x608 and 0x609 each with one operand word taken from ...
pub const ODSY_SWAP_SYNC_A: u32 = 0x00000608;
pub const ODSY_SWAP_SYNC_B: u32 = 0x00000609;

// The bound reading drawable's packed y:x origin, distinct from 0x841.
pub const ODSY_REG_READ_ORIGIN: u32 = 0x0000060b;
pub const ODSY_SWAP_SYNC_OPERANDS: u32 = 1;

// The rest of that block
pub const ODSY_REG_DRAWABLE_FIRST: u32 = 0x00000608;
pub const ODSY_REG_DRAWABLE_LAST: u32 = 0x0000060f;

// The fence word the 0x0a5a host-write command's operands form (see ODSY_REG_HOST_WRITE).
pub const ODSY_DMA_FENCE_SLOT_MASK: u32 = 0x7fff;
pub const ODSY_DMA_FENCE_INTERRUPT: u32 = 1 << 15;
pub const ODSY_DMA_FENCE_TOKEN_SHIFT: u32 = 16;

// The completion cookie a head-ops publish() carries
pub const ODSY_FENCE_VALID: u64 = 1 << 32;
pub const ODSY_FENCE_WORD_MASK: u32 = 0xffffffff;

// The translation table
pub const ODSY_DMA_PAGE_SIZE: u64 = 0x4000;
pub const ODSY_DMA_TABLE_HEADER: u64 = 0x20;
pub const ODSY_DMA_HDR_WORDS: usize = 8;
pub const ODSY_DMA_HDR_DEST: usize = 0;
pub const ODSY_DMA_HDR_ORIGIN: usize = 1;
pub const ODSY_DMA_HDR_WIDTH: usize = 2;
pub const ODSY_DMA_HDR_SIZE: usize = 3;
pub const ODSY_DMA_HDR_SKIP: usize = 5;
pub const ODSY_DMA_HDR_ROWBYTES: usize = 6;
pub const ODSY_DMA_HDR_CONTROL: usize = 7;
pub const ODSY_DMA_ROWBYTES_MASK: u32 = 0xfffff;
pub const ODSY_DMA_BPP_SHIFT: u32 = 16;
pub const ODSY_DMA_BPP_MASK: u32 = 0x7;
pub const ODSY_DMA_IRQ_REQUEST: u32 = 1 << 31;
pub const ODSY_DMA_PTE_LEN_MASK: u32 = 0xffff;
pub const ODSY_DMA_PTE_ADDR_SHIFT: u32 = 16;

// One kick names one 128-byte work-list entry
pub const ODSY_DMA_ENTRY_PAGES: usize = 12;
pub const ODSY_DMA_MAX_PAGES: usize = ODSY_DMA_ENTRY_PAGES;

// Board-to-host transfers.
pub const ODSY_DMA_HDR_FORMAT: usize = 4;
pub const ODSY_DMA_TO_HOST: u32 = 1 << 24; // word 7
pub const ODSY_DMA_BOTTOM_UP: u32 = 1 << 30; // word 7
pub const ODSY_DMA_SENTINEL: u32 = 1 << 29; // word 0
pub const ODSY_DMA_SENTINEL_READ: u32 = 0x30000019;
pub const ODSY_DMA_SENTINEL_DRAW: u32 = 0x30000010;
pub const ODSY_DMA_FMT_FRONT: u32 = 1 << 31; // word 4
pub const ODSY_DMA_FMT_SELECTOR_SHIFT: u32 = 28; // word 4 bits 30:28
pub const ODSY_DMA_FMT_SELECTOR_MASK: u32 = 0x7;
pub const ODSY_DMA_FMT_SELECTOR_READ: u32 = 1;
pub const ODSY_DMA_FMT_SELECTOR_DRAW: u32 = 0;
pub const ODSY_DMA_FMT_SIZE_SHIFT: u32 = 13; // word 4 bits 15:13
pub const ODSY_DMA_FMT_SIZE_MASK: u32 = 0x7;
pub const ODSY_DMA_FMT_SIZE_STENCIL: u32 = 0; // a write's size field: stencil
pub const ODSY_DMA_FMT_SIZE_DEPTH: u32 = 1; // and depth
pub const ODSY_DMA_FMT_CLASS_MASK: u32 = 0x7; // word 4 bits 2:0
pub const ODSY_DMA_FMT_CODE_SHIFT: u32 = 3; // word 4 bits 7:3
pub const ODSY_DMA_FMT_CODE_MASK: u32 = 0x1f;

// The GL library's source classes for 8- and 12-bit RGBA visuals.
pub const ODSY_DMA_GL_CLASS_RGBA8: u32 = 2;
pub const ODSY_DMA_GL_CLASS_RGBA12: u32 = 3;

// The kernel's board-memory manager pages an object out to its backing store in host memory and back in with raw copies of the object's texels.
pub const ODSY_DMA_FMT_RAW16: u32 = 0x1001;
pub const ODSY_DMA_FMT_RAW32: u32 = 0x13002;
pub const ODSY_DMA_FMT_RAW64: u32 = 0x25004;

// Off-screen board-memory objects.
pub const ODSY_OBJ_POOL_BYTES: usize = SGI_ODYSSEY_OBJ_POOL_BYTES as usize;
pub const ODSY_OBJ_POOL_WORDS: usize = SGI_ODYSSEY_OBJ_POOL_WORDS as usize;
pub const ODSY_DEST_WIDTH_INDEX: usize = 1; // 0x1017's row length, 16-pixel units
pub const ODSY_DEST_WIDTH_UNIT: u32 = 16;

// The select's format word
pub const ODSY_DEST_FORMAT_SHIFT: u32 = 28;
pub const ODSY_DEST_FORMAT_SCREEN: u32 = 0x2;

// The 2D stream's GL primitives.
pub const ODSY_GL_POINTS: u32 = 0x00000000;
pub const ODSY_GL_LINES: u32 = 0x00000001;
pub const ODSY_GL_LINE_LOOP: u32 = 0x00000002;
pub const ODSY_GL_LINE_STRIP: u32 = 0x00000003;
pub const ODSY_GL_TRIANGLES: u32 = 0x00000004;
pub const ODSY_GL_TRIANGLE_STRIP: u32 = 0x00000005;
pub const ODSY_GL_TRIANGLE_FAN: u32 = 0x00000006;
pub const ODSY_GL_QUADS: u32 = 0x00000007;
pub const ODSY_GL_QUAD_STRIP: u32 = 0x00000008;
pub const ODSY_GL_POLYGON: u32 = 0x00000009;

// Two GL vertex-stream words carry their own tag instead of the register-select shape above (same primary source, section 4).
pub const ODSY_GL_COLOR_3UB: u32 = 0xc580cc08; // 3 operands: R, G, B, one byte each
pub const ODSY_GL_VERTEX_2I: u32 = 0x8080c800; // 2 operands: X, Y

// The greeter draws in PseudoColor
pub const ODSY_GL_COLOR_INDEX: u32 = 0xc580c406; // 1 operand: PseudoColor index
pub const ODSY_GL_COLOR_INDEX_F: u32 = 0xc580c407; // 1 operand: the index as a float
pub const ODSY_GL_REPLAY_INDEX_F: u32 = 0xce80c407; // 1 operand: the index as a float
pub const ODSY_GL_INDEX_MODE_COLOR_3F: u32 = 0xce80cc0e; // 3 float operands
pub const ODSY_GL_INDEX_MODE_COLOR_4F: u32 = 0xce80d015; // 4 float operands
pub const ODSY_MODE_INDEX_ZEROS: u32 = 0xce80cc08; // 3 operands: three zero words

// The other current-colour and normal words of the guest OS GL library (software expectations, not measured hardware).
pub const ODSY_GL_COLOR_3I: u32 = 0xc580cc0d; // 3 signed integers
pub const ODSY_GL_INDEX_MODE_COLOR_3I: u32 = 0xce80cc0d;
pub const ODSY_GL_COLOR_4F: u32 = 0xc580d015; // 4 floats: R, G, B, A

// The integer colour words, one per glColor variant, in the same shape
pub const ODSY_GL_COLOR_3US: u32 = 0xc580cc09;
pub const ODSY_GL_COLOR_3UI: u32 = 0xc580cc0a;
pub const ODSY_GL_COLOR_3B: u32 = 0xc580cc0b;
pub const ODSY_GL_COLOR_3S: u32 = 0xc580cc0c;
pub const ODSY_GL_COLOR_4UB: u32 = 0xc580d00f;
pub const ODSY_GL_COLOR_4US: u32 = 0xc580d010;
pub const ODSY_GL_COLOR_4UI: u32 = 0xc580d011;
pub const ODSY_GL_COLOR_4B: u32 = 0xc580d012;
pub const ODSY_GL_COLOR_4S: u32 = 0xc580d013;
pub const ODSY_GL_COLOR_4I: u32 = 0xc580d014;
pub const ODSY_GL_COLOR_TYPE_MASK: u32 = 0x3f;
pub const ODSY_GL_COLOR_TYPE_3: u32 = 8; // glColor3ub's type code
pub const ODSY_GL_COLOR_TYPE_4: u32 = 15; // glColor4ub's type code
pub const ODSY_GL_NORMAL_3B: u32 = 0xc180cc16; // 3 signed bytes, one per word
pub const ODSY_GL_NORMAL_3S: u32 = 0xc180cc17; // 3 signed shorts, one per word
pub const ODSY_GL_NORMAL_3I: u32 = 0xc180cc18; // 3 signed integers

// The true-colour immediate-mode GL words a real OpenGL client draws with, the float siblings of the integer 2D words above.
pub const ODSY_GL_COLOR_3F: u32 = 0xc580cc0e; // 3 operands: R, G, B as IEEE floats
pub const ODSY_GL_VERTEX_2F: u32 = 0x8080c801; // 2 operands: X, Y as IEEE floats

// The GL library's glVertex3f, object coordinates (ESTABLISHED).
pub const ODSY_GL_VERTEX_3F: u32 = 0x8080cc03; // 3 operands: X, Y, Z as IEEE floats

// The GL library's glVertex3i, glVertex3s and their vector forms send this word and the three coordinates as raw signed 32-bit integers, shorts ...
pub const ODSY_GL_VERTEX_3I: u32 = 0x8080cc02; // 3 operands: X, Y, Z as int32

// glVertex4f
pub const ODSY_GL_VERTEX_4F: u32 = 0x8080d005;

// Depth state (ESTABLISHED from the GL library's entry points and the captures)
pub const ODSY_MODE_DEPTH_TEST: u32 = 1 << 16;
pub const ODSY_REG_DEPTH_FUNC: u32 = 0x0420; // 1 operand
pub const ODSY_DEPTH_FUNC_MASK: u32 = 0x7;
pub const ODSY_REG_WRITE_MASK: u32 = 0x0452; // 1 operand
pub const ODSY_WRITE_DEPTH: u32 = 1 << 24;
pub const ODSY_WRITE_COLOUR_SHIFT: u32 = 20;
pub const ODSY_WRITE_COLOUR_MASK: u32 = 0xf;

// Out of reset
pub const ODSY_DEPTH_FUNC_RESET: u32 = 0x1;
pub const ODSY_WRITE_MASK_RESET: u32 = 0x01f00000;

// Lighting data (section 1, ESTABLISHED from the entry points and two captures with distinct values).
pub const ODSY_GL_NORMAL_3F: u32 = 0xc180cc19; // 3 operands
pub const ODSY_GL_MATERIAL_AMBIENT: u32 = 0xc680d02f; // 4 operands, as the next 7
pub const ODSY_GL_MATERIAL_DIFFUSE: u32 = 0xc780d030;
pub const ODSY_GL_MATERIAL_SPECULAR: u32 = 0xc880d031;
pub const ODSY_GL_MATERIAL_EMISSION: u32 = 0xc980d032;
pub const ODSY_GL_BACK_AMBIENT: u32 = 0xca80d035;
pub const ODSY_GL_BACK_DIFFUSE: u32 = 0xcb80d036;
pub const ODSY_GL_BACK_SPECULAR: u32 = 0xcc80d037;
pub const ODSY_GL_BACK_EMISSION: u32 = 0xcd80d038;
pub const ODSY_GL_SHININESS_SQRT: u32 = 0xd080c433; // 1 operand
pub const ODSY_GL_BACK_SHININESS_SQRT: u32 = 0xd080c439; // 1 operand
pub const ODSY_GL_SHININESS_DERIVED: u32 = 0x908091ea; // 4 operands
pub const ODSY_GL_LIGHT_COLOUR_LOAD: u32 = 0x8e809015; // 4 operands
pub const ODSY_GL_LIGHT_VEC_LOAD: u32 = 0x9080bda2; // 16 operands
pub const ODSY_GL_CLIP_PLANE_LOAD: u32 = 0x908091bb; // 4 operands
pub const ODSY_GL_LIGHT_MODEL_AMBIENT: u32 = 24;
pub const ODSY_GL_LIGHT_VEC_FLOATS: u32 = 16;
pub const ODSY_GL_MATRIX_SELECT_NORMAL: u32 = 0x34;

// Light block floats
pub const ODSY_LIGHT_POS: usize = 0;
pub const ODSY_LIGHT_HALF: usize = 3;
pub const ODSY_LIGHT_ATTEN: usize = 7;
pub const ODSY_LIGHT_SPOT_DIR: usize = 10;
pub const ODSY_LIGHT_SPOT_COS: usize = 13;
pub const ODSY_LIGHT_SPOT_EXP: usize = 14;
pub const ODSY_PROG_SHAPE: u32 = 0x007ffc00;
pub const ODSY_PROG_TAG: u32 = 0x00008000;
pub const ODSY_PROG_LOW_MASK: u32 = 0x3ff;
pub const ODSY_PROG_HIGH_SHIFT: u32 = 23;
pub const ODSY_PROG_CLASS_SHIFT: u32 = 24;

// The colour mode, in the same word shape.
pub const ODSY_PROG_COLOUR_MODE: u32 = 0x22;
pub const ODSY_PROG_COLOUR_MODE_B: u32 = 0x23;
pub const ODSY_COLOUR_MODE_INDEX: u32 = 0x1;
pub const ODSY_PROG_SET_UCODE: u32 = 0x25;
pub const ODSY_PROG_PROGRAM_ADDR: u32 = 0x27;
pub const ODSY_PROG_CONST_SELECT: u32 = 0x28;
pub const ODSY_PROG_ENTRY_29: u32 = 0x29;
pub const ODSY_PROG_ENTRY_2A: u32 = 0x2a;
pub const ODSY_PROG_ENTRY_2C: u32 = 0x2c;
pub const ODSY_PROG_LIGHT_VEC_SELECT: u32 = 0x31;
pub const ODSY_PROG_LIGHT_COLOUR_SELECT: u32 = 0x32;
pub const ODSY_PROG_LIST_INDEX: u32 = 0x33;

// The colour-material word, class 0x35
pub const ODSY_PROG_COLOUR_MATERIAL: u32 = 0x35;
pub const ODSY_COLOUR_MATERIAL_MASK: u32 = 0xff;
pub const ODSY_COLOUR_MATERIAL_AMBIENT: u32 = 0x1;
pub const ODSY_COLOUR_MATERIAL_DIFFUSE: u32 = 0x2;
pub const ODSY_COLOUR_MATERIAL_SPECULAR: u32 = 0x4;
pub const ODSY_COLOUR_MATERIAL_EMISSION: u32 = 0x8;

// Classes 0x20 to 0x35 are program-sequencer registers the host never reads back
pub const ODSY_PROG_SEQ_FIRST: u32 = 0x20;
pub const ODSY_PROG_SEQ_LAST: u32 = 0x35;
pub const ODSY_PROG_PROGRAM_HALFWORD: u32 = 0x40;
pub const ODSY_PROG_PROGRAM_HALFWORD_LAST: u32 = 0x5f;
pub const ODSY_PROG_CONST_STORE: u32 = 0x60;
pub const ODSY_PROG_HALFWORD_HIGH_MASK: u32 = 0x3f;
pub const ODSY_PROG_ENTRY_HALFWORDS: u32 = 6;
pub const ODSY_PROG_RUN_ONCE: u32 = 0x90808800; // 2 operands

// How far a program list is read before it counts as unrecognised.
pub const ODSY_PROG_LIST_MAX: u32 = 64;
// The older GL libraries' program image lacks the 99-record spot routine at
// records 792 to 890 and every record from 2365 on, so they number each
// routine from 891 on 99 lower than the current libraries do.
pub const ODSY_UCODE_SPOT_FIRST: u32 = 792;
pub const ODSY_UCODE_SPOT_RECORDS: u32 = 99;
// The older image's length: records 0-791 and 891-2364 of the current one.
pub const ODSY_UCODE_OLDER_RECORDS: u32 = 2266;
pub const ODSY_UCODE_END: u32 = 0x4c5;
pub const ODSY_UCODE_UNLIT: u32 = 0x0d6;
pub const ODSY_UCODE_UNLIT_2A: u32 = 0x0e0;
pub const ODSY_UCODE_UNLIT_2C: u32 = 0x0ea;
pub const ODSY_UCODE_UNLIT_CLIP: u32 = 0x0f3;
pub const ODSY_UCODE_UNLIT_CLIP_2A: u32 = 0x100;
pub const ODSY_UCODE_UNLIT_CLIP_2C: u32 = 0x10c;
pub const ODSY_PROG_WINDOW_BASE: u32 = 0x738;
pub const ODSY_PROG_WINDOW_ENTRIES: u32 = SGI_ODYSSEY_GL_WINDOW_ENTRIES;

// The assembled unlit list
pub const ODSY_UCODE_HEAD_UNLIT: u32 = 0x4ae;
pub const ODSY_UCODE_UNLIT_TEXTURE: u32 = 0x4bb;
pub const ODSY_UCODE_UNLIT_FINISH: u32 = 0x4c0;

// Assembled lit list heads
pub const ODSY_UCODE_HEAD_INF_DIR: u32 = 0x290;
pub const ODSY_UCODE_HEAD_INF_POS: u32 = 0x272;
pub const ODSY_UCODE_HEAD_LOCAL_DIR: u32 = 0x281;
pub const ODSY_UCODE_HEAD_LOCAL_POS: u32 = 0x263;

// The normal transform without and with GL_NORMALIZE.
pub const ODSY_UCODE_NORMAL: u32 = 0x2b6;
pub const ODSY_UCODE_NORMALIZE: u32 = 0x29c;

// Per-light fragments
pub const ODSY_UCODE_LIGHT_DIR: u32 = 0x300;
pub const ODSY_UCODE_LIGHT_POS: u32 = 0x398;
pub const ODSY_UCODE_LIGHT_PLAIN: u32 = 0x2cf;
pub const ODSY_UCODE_LIGHT_ATTEN: u32 = 0x2de;

// The lighting finish
pub const ODSY_UCODE_FINISH_A: u32 = 0x3cf;
pub const ODSY_UCODE_FINISH_A_TWO_SIDED: u32 = 0x3e5;
pub const ODSY_UCODE_FINISH_A_TEXTURE: u32 = 0x3da;
pub const ODSY_UCODE_FINISH_ONE_SIDED: u32 = 0x3f5;
pub const ODSY_UCODE_FINISH_TWO_SIDED: u32 = 0x415;
pub const ODSY_UCODE_FINISH_TEXTURE: u32 = 0x40b;
pub const ODSY_UCODE_END_ONE_SIDED: u32 = 0x49a;
pub const ODSY_UCODE_END_TWO_SIDED: u32 = 0x49f;

// The list's tail
pub const ODSY_UCODE_TAIL_ONE_SHOT: u32 = 0x203;
pub const ODSY_UCODE_TAIL: u32 = 0x25e;

// A hardware clip-plane stage is a 0 word followed by one of these.
pub const ODSY_UCODE_CLIP: u32 = 0x00c;
pub const ODSY_UCODE_CLIP_TEXTURED: u32 = 0x01a;
pub const ODSY_UCODE_CLIP_TEXGEN: u32 = 0x028;

// Texture state (sections 1 and 2
pub const ODSY_REG_TEX_MODE: u32 = 0x0405; // 1 operand
pub const ODSY_TEX_MODE_1D: u32 = 1 << 3;
pub const ODSY_TEX_MODE_2D: u32 = 1 << 4;
pub const ODSY_TEX_MODE_3D: u32 = 1 << 5;
pub const ODSY_REG_TEX_SAMPLER: u32 = 0x1c16; // 7 operands
pub const ODSY_REG_TEX_ENV: u32 = 0x141b; // 5 operands
pub const ODSY_REG_TEX_LEVEL: u32 = 0x0e80; // 3 operands, 0x0e80 .. 0x0ebf
pub const ODSY_REG_TEX_LEVEL0: u32 = 0x0406; // 1 operand
pub const ODSY_REG_TEX_LOD: u32 = 0x1013; // 4 operands
pub const ODSY_REG_TEX_SCALE_BIAS: u32 = 0x203b; // 8 operands

// The GL library's other two post-filter layouts
pub const ODSY_REG_POST_FILTER_DEFAULT: u32 = 0x083d;
pub const ODSY_REG_TEX_COLOUR_TABLE0: u32 = 0x3e00;
pub const ODSY_REG_TEX_COLOUR_TABLE1: u32 = 0x3e10;
pub const ODSY_TEX_COLOUR_TABLE_BLOCKS: u32 = 16;
pub const ODSY_TEX_COLOUR_TABLE_BLOCK_WORDS: u32 = 16;
pub const ODSY_REG_TEX_DETAIL_FUNCTION: u32 = 0x3e40;

// The texture offset block, three words the GL library copies from the texture object at activation.
pub const ODSY_REG_TEX_OFFSET: u32 = 0x0c3e; // 3 operands

// A fourth 256-entry table, sixteen 16-word blocks.
pub const ODSY_REG_TEX_TABLE3: u32 = 0x3e30;
pub const ODSY_TEX_TABLE3_BLOCKS: u32 = 16;
pub const ODSY_TEX_DETAIL_FUNCTION_BLOCKS: u32 = 16;
pub const ODSY_TEX_DETAIL_FUNCTION_BLOCK_WORDS: u32 = 16;
pub const ODSY_GL_TEXCOORD_1F: u32 = 0xc380c41f; // 1 operand: s
pub const ODSY_GL_TEXCOORD_2F: u32 = 0xc380c821; // 2 operands: s, t
pub const ODSY_GL_TEXCOORD_3F: u32 = 0xc380cc23; // 3 operands: s, t, r
pub const ODSY_GL_TEXCOORD_4F: u32 = 0xc380d025; // 4 operands: s, t, r, q
pub const ODSY_GL_TEXCOORD_SHAPE: u32 = 0xffffff00;
pub const ODSY_GL_TEXCOORD_TAG: u32 = 0xc380c800;

// glTexCoord1i to 4i
pub const ODSY_GL_TEXCOORD_1I: u32 = 0xc380c41e;
pub const ODSY_GL_TEXCOORD_2I: u32 = 0xc380c820;
pub const ODSY_GL_TEXCOORD_3I: u32 = 0xc380cc22;
pub const ODSY_GL_TEXCOORD_4I: u32 = 0xc380d024;
pub const ODSY_GL_TEX_RECT: u32 = 0x8f80b968; // 14 operands

// Six retained words
pub const ODSY_REG_FOG: u32 = 0x181a;
pub const ODSY_MODE_FOG: u32 = 1 << 22;

// The board-side geometry transform.
pub const ODSY_GL_MATRIX_SELECT_SHAPE: u32 = 0xffffff00;
pub const ODSY_GL_MATRIX_SELECT_TAG: u32 = 0x34008000;
pub const ODSY_GL_MATRIX_SELECT_COMBINED: u32 = 0x00; // the select word's low byte
pub const ODSY_GL_MATRIX_SELECT_TEXTURE: u32 = 0x14;
pub const ODSY_GL_MATRIX_SELECT_MODELVIEW: u32 = 0x24;
pub const ODSY_GL_MATRIX_LOAD: u32 = 0x9080bdc8;
pub const ODSY_GL_MATRIX_FLOATS: u32 = 16;

// The GL library's combined-matrix validation follows each slot-0x00 load with a select of index 16 and this word carrying four floats (ESTABLISHED)
pub const ODSY_GL_MATRIX_SELECT_BLOCK16: u32 = 0x10;
pub const ODSY_GL_LOAD_4F: u32 = 0x908091df;
pub const ODSY_GL_BLOCK16_FLOATS: u32 = 4;
pub const ODSY_GL_BLOCK16_SIGN: u32 = 3;

// The select's own slot, so the four-float load knows its target.
pub const ODSY_GL_SLOT_BLOCK16: u32 = 0x10002;

// The normal transform's slot (ODSY_GL_MATRIX_SELECT_NORMAL).
pub const ODSY_GL_SLOT_NORMAL_MATRIX: u32 = 0x10003;

// The board's depth unit
pub const ODSY_GL_DEPTH_S: f64 = 4294950912.0;
pub const ODSY_F32_ONE: u32 = 0x3f800000;
pub const ODSY_F32_POINT_TWO: u32 = 0x3e4ccccd;
pub const ODSY_F32_POINT_EIGHT: u32 = 0x3f4ccccd;
pub const ODSY_DEPTH24_MAX: u32 = 0xffffff;

// A GL matrix mode the shared context does not keep (neither modelview nor projection), so GlContext::set_matrix() logs and refuses a load aimed at it.
pub const ODSY_GL_SLOT_NORMAL: u32 = 0x1702;

// This model's own slots for the combined and texture matrices.
pub const ODSY_GL_SLOT_COMBINED: u32 = 0x10000;
pub const ODSY_GL_SLOT_TEXTURE: u32 = 0x10001;

// Half the extent of the fixed viewport the shared context draws through (Board::gl_load_combined())
pub const ODSY_GL_WINDOW_REACH: i32 = 8192;

// The screen-aligned rectangle.
pub const ODSY_GL_RECT: u32 = 0x8f80a944;
pub const ODSY_GL_RECT_FLOATS: u32 = 10;

// The kernel driver's buffer-clear request, made by a GL client with its parameters, sends the same ten-float record under this word inside BEGIN 0xa
pub const ODSY_GL_CLEAR_BOX: u32 = 0x8f80a92f;

// The identity the shared GL context keys draws by
pub const ODSY_GL_CONTEXT_HW_ID: u32 = 1;

// The screen-to-screen copy command is a fixed 8-word literal template following a bare trigger, not decomposed through the generic register-select ...
pub const ODSY_CMD_COPY_RECT_TRIGGER: u32 = 0x00002031;
pub const ODSY_CMD_COPY_RECT_WORDS: u32 = 8;
pub const ODSY_CMD_COPY_RECT_SRC_INDEX: u32 = 1;
pub const ODSY_CMD_COPY_RECT_SIZE_INDEX: u32 = 3;
pub const ODSY_CMD_COPY_RECT_DST_INDEX: u32 = 6;
pub const ODSY_COPY_RECT_SOURCE_SHIFT: u32 = 12;
pub const ODSY_COPY_RECT_DEST_SHIFT: u32 = 20;
pub const ODSY_COPY_RECT_FORMAT_SOURCE_SHIFT: u32 = 24;

// The one-bit pattern-quad text path.
pub const ODSY_REG_PATTERN_LOW: u32 = 0x3c4e; // pattern buffer words 0-15
pub const ODSY_REG_PATTERN_HIGH: u32 = 0x3c4f; // pattern buffer words 16-31
pub const ODSY_REG_PATTERN_SHAPE: u32 = 0x043f; // 1 operand: 0x4/0x5/0x6, 0x0 = off
pub const ODSY_REG_PATTERN_MODE: u32 = 0x0404; // 1 operand: pattern, GL mode bits
pub const ODSY_PATTERN_WORDS: u32 = 16;

// The line stipple.
pub const ODSY_REG_LINE_STIPPLE: u32 = 0x0c48; // 3 operands
pub const ODSY_MODE_GL_LINE_STIPPLE: u32 = 1 << 3;
pub const ODSY_MODE_POINT_SMOOTH: u32 = 1 << 0;
pub const ODSY_MODE_LINE_SMOOTH: u32 = 1 << 1;
pub const ODSY_LINE_STIPPLE_PATTERN: usize = 1; // the 32-bit rotated row
pub const ODSY_LINE_STIPPLE_CONTROL: usize = 2;
pub const ODSY_LINE_STIPPLE_CONTROL_32: u32 = 0x30f;

// The drawable-id plane write.
pub const ODSY_REG_DEST_SELECT: u32 = 0x1017; // 4 operands: destination-buffer select
pub const ODSY_REG_PLANE_MASK: u32 = 0x1023; // 4 operands: plane write mask << 12
pub const ODSY_DEST_ID_PLANES: u32 = 0x20080080; // 0x1017 third word: write id planes

// Two bits of the 0x1017 third word that the X server changes from one draw to the next for the same buffer, so they do not name the buffer.
pub const ODSY_DEST_NO_READ: u32 = 1 << 29;
pub const ODSY_DEST_BIT23: u32 = 1 << 23;
pub const ODSY_DEST_FLAGS: u32 = ODSY_DEST_NO_READ | ODSY_DEST_BIT23;
pub const ODSY_DEST_SELECT_VALUE_INDEX: usize = 2; // the third operand
pub const ODSY_DEST_SELECT_BASE_INDEX: usize = 0; // the buffer's base, 0x2000
pub const ODSY_PLANE_MASK_SHIFT: u32 = 12; // the 0x1023 operand holds mask << 12

// The 0x043f shape operand.
pub const ODSY_PATTERN_SHAPE_64X16: u32 = 0x5;
pub const ODSY_PATTERN_SHAPE_32X32: u32 = 0x4;
pub const ODSY_PATTERN_SHAPE_16X64: u32 = 0x6;

// Bit 4 of the 0x043f operand makes a pattern opaque
pub const ODSY_PATTERN_SHAPE_OPAQUE: u32 = 0x10;

// The 0x0404 mode operand's pattern-enable bits
pub const ODSY_PATTERN_MODE_ENABLE: u32 = 0x18;

// Proposed shape-4 screen-aligned rectangle policy.
pub const ODSY_GL_RECT_PATTERN_ENABLE: u32 = 1 << 4;

// The GL library keeps the shade model in the same register
pub const ODSY_MODE_FLAT_SHADE: u32 = 1 << 12;

// Bit 18 of the same register is GL_DITHER (ESTABLISHED)
pub const ODSY_MODE_DITHER: u32 = 1 << 18;

// Bit 14 of the same register is GL_ALPHA_TEST
pub const ODSY_MODE_ALPHA_TEST: u32 = 1 << 14;
pub const ODSY_REG_ALPHA_FUNC: u32 = 0x041e; // 1 operand
pub const ODSY_ALPHA_FUNC_SHIFT: u32 = 12;
pub const ODSY_ALPHA_FUNC_MASK: u32 = 0x7;
pub const ODSY_ALPHA_REF_MASK: u32 = 0xfff;
pub const ODSY_ALPHA_REF_SCALE: f32 = 4096.0;

// Bit 17 of the same register is a computed enable, not a plain mirror of glEnable(GL_BLEND)
pub const ODSY_MODE_BLEND: u32 = 1 << 17;
pub const ODSY_REG_BLEND: u32 = 0x0c21; // 3 operands
pub const ODSY_BLEND_DST_MASK: u32 = 0x3f;
pub const ODSY_BLEND_SRC_SHIFT: u32 = 6;
pub const ODSY_BLEND_SRC_MASK: u32 = 0x7f;
pub const ODSY_BLEND_EQ_SHIFT: u32 = 13;
pub const ODSY_BLEND_EQ_MASK: u32 = 0x7;
pub const ODSY_BLEND_COLOUR_MASK: u32 = 0xfff;
pub const ODSY_BLEND_COLOUR_SHIFT: u32 = 12;
pub const ODSY_BLEND_COLOUR_SCALE: f32 = 4096.0;

// The X server's opaque-span 0x0c21 operand 0
pub const ODSY_BLEND_X_OPAQUE: u32 = 0x200;

// Bits 19 and 20 of the same register are GL_INDEX_LOGIC_OP and GL_COLOR_LOGIC_OP.
pub const ODSY_MODE_INDEX_LOGIC_OP: u32 = 1 << 19;
pub const ODSY_MODE_COLOR_LOGIC_OP: u32 = 1 << 20;
pub const ODSY_REG_LOGIC_OP: u32 = 0x0422; // 1 operand
pub const ODSY_LOGIC_OP_MASK: u32 = 0xf;

// Bit 15 of the same register is GL_STENCIL_TEST, which the library sets only for a drawable with stencil planes.
pub const ODSY_MODE_STENCIL_TEST: u32 = 1 << 15;
pub const ODSY_REG_STENCIL_FUNC: u32 = 0x041f; // 1 operand
pub const ODSY_STENCIL_FUNC_SHIFT: u32 = 25;
pub const ODSY_STENCIL_FUNC_MASK: u32 = 0x7;
pub const ODSY_STENCIL_REF_MASK: u32 = 0xff;
pub const ODSY_STENCIL_VALUE_MASK_SHIFT: u32 = 8;
pub const ODSY_STENCIL_VALUE_MASK_MASK: u32 = 0xff;
pub const ODSY_STENCIL_OP_FAIL_SHIFT: u32 = 22;
pub const ODSY_STENCIL_OP_ZFAIL_SHIFT: u32 = 19;
pub const ODSY_STENCIL_OP_ZPASS_SHIFT: u32 = 16;
pub const ODSY_STENCIL_OP_MASK: u32 = 0x7;
pub const ODSY_WRITE_STENCIL_SHIFT: u32 = 12;
pub const ODSY_WRITE_STENCIL_MASK: u32 = 0xff;
/// Where a depth or stencil burst's word holds the stencil index.
pub const ODSY_BURST_STENCIL_SHIFT: u32 = 16;
pub const ODSY_STENCIL_MAX: u32 = 0xff;

// 0x452 bits 11:0 are glIndexMask ANDed with the drawable's index-plane mask
pub const ODSY_WRITE_INDEX_MASK: u32 = 0xfff;

// Bit 8 of the 0x0404 mode register is GL_CULL_FACE, bits 10:9 the face it culls -- 01 GL_BACK, 10 GL_FRONT, 11 GL_FRONT_AND_BACK, with 00 never sent ...
pub const ODSY_MODE_CULL_FACE: u32 = 1 << 8;
pub const ODSY_MODE_CULL_MODE_SHIFT: u32 = 9;
pub const ODSY_MODE_CULL_MODE_MASK: u32 = 0x3;
pub const ODSY_MODE_FRONT_FACE_CW: u32 = 1 << 23;

// Bits 5, 6 and 7 of the same register are GL_POLYGON_OFFSET_POINT, _LINE and _FILL
pub const ODSY_MODE_OFFSET_POINT: u32 = 1 << 5;
pub const ODSY_MODE_OFFSET_LINE: u32 = 1 << 6;
pub const ODSY_MODE_OFFSET_FILL: u32 = 1 << 7;
pub const ODSY_REG_POLYGON_OFFSET: u32 = 0x0458; // 1 operand
pub const ODSY_POLYGON_OFFSET_BITS: u32 = 25;
// The field is a signed count of steps of the encoded 24-bit depth word
// (depth_word.rs); the shared context carries it as that count over 2^24, which
// RasterRules::split_offset() turns back into steps.
pub const ODSY_POLYGON_OFFSET_SCALE: f32 = 1.0f32 / (1u32 << 24) as f32;

// 0x0415 packs glPolygonMode's two faces in one word
pub const ODSY_REG_POLYGON_MODE: u32 = 0x0415; // 1 operand
pub const ODSY_POLYGON_MODE_FRONT_SHIFT: u32 = 2;
pub const ODSY_POLYGON_MODE_BACK_SHIFT: u32 = 0;
pub const ODSY_POLYGON_MODE_MASK: u32 = 0x3;

// 0x084a packs the line width and point size together
pub const ODSY_REG_LINE_POINT_SIZE: u32 = 0x084a; // 2 operands: line width, point size
pub const ODSY_LINE_POINT_SIZE_MASK: u32 = 0xffffff;
pub const ODSY_LINE_POINT_SIZE_SCALE: f32 = 128.0;

// The scissor and window clip box, 0x081d, two operands
pub const ODSY_REG_CLIP_BOX: u32 = 0x081d; // 2 operands
pub const ODSY_CLIP_BOX_MIN_SHIFT: u32 = 12;
pub const ODSY_CLIP_BOX_FIELD_MASK: u32 = 0xfff;

// User clip planes, class 0x31 index 0x120 + 4j, reusing the same index-select field as the light-vector loads (ODSY_PROG_LIGHT_VEC_SELECT, ...
pub const ODSY_GL_CLIP_PLANE_BASE: u32 = 0x120;
pub const ODSY_GL_CLIP_PLANE_STRIDE: u32 = 4;
pub const ODSY_TEX_MODE_CLIP_COUNT_MASK: u32 = 0x7;

// The BEGIN-0xa image raster path
pub const ODSY_CMD_STATE_WORD: u32 = 0x29008000;
pub const ODSY_CMD_STATE_WORD_MASK: u32 = 0xfffffc00;
pub const ODSY_CMD_IMAGE_BOX: u32 = 0xcf80a92f; // 10 operands: raster-box floats
pub const ODSY_CMD_IMAGE_HEADER: u32 = 0x00004570; // 1 operand: transfer header
pub const ODSY_CMD_IMAGE_HEADER_TAIL: u32 = 0x00000071; // 0 operands

// BEGIN's operand for an image (as opposed to 0x7, QUADS).
pub const ODSY_GL_PRIMITIVE_IMAGE: u32 = 0xa;

// The floats 4, 5, 7, 8 of the 0xcf80a92f box record are the destination corners
pub const ODSY_IMAGE_BOX_X1_INDEX: u32 = 4;
pub const ODSY_IMAGE_BOX_Y1_INDEX: u32 = 5;
pub const ODSY_IMAGE_BOX_X2_INDEX: u32 = 7;
pub const ODSY_IMAGE_BOX_Y2_INDEX: u32 = 8;

// A pixel burst is 0x00014000 | (words - words/16) << 10 | fmt
pub const ODSY_IMAGE_BURST_BASE: u32 = 0x4000;
pub const ODSY_IMAGE_BURST_FMT_MASK: u32 = 0x1f;
pub const ODSY_IMAGE_BURST_RESERVED_MASK: u32 = 0x3e0;
pub const ODSY_IMAGE_FMT_CI: u32 = 0x10;
pub const ODSY_IMAGE_FMT_RGB: u32 = 0x11;
pub const ODSY_IMAGE_FMT_RGBA16: u32 = 0x12;

// The source pixel format
pub const ODSY_IMAGE_CODE_MASK: u32 = 0xff;
/// A GL write's format (bits 4:0) and type (8:5) together.
pub const ODSY_WRITE_CODE_MASK: u32 = 0x1ff;
/// Format 0 (depth or stencil) with type 4 (four bytes a pixel).
pub const ODSY_IMAGE_CODE_DEPTH_STENCIL: u32 = 0x80;
pub const ODSY_IMAGE_CODE_CI: u32 = 0x41;
pub const ODSY_IMAGE_CODE_CI16: u32 = 0x61;
pub const ODSY_IMAGE_CODE_LUM: u32 = 0x42;
pub const ODSY_IMAGE_CODE_LUM_ALPHA: u32 = 0x48;
pub const ODSY_IMAGE_CODE_RGB: u32 = 0x49;
pub const ODSY_IMAGE_CODE_RGBA: u32 = 0x4b;
pub const ODSY_IMAGE_CODE_ABGR: u32 = 0x4c;

// Not a board code
pub const ODSY_IMAGE_CODE_FALLBACK_XRGB: u32 = 0x100;
pub const ODSY_IMAGE_CODE_RGBA12: u32 = 0x6b;
pub const ODSY_IMAGE_CODE_LUM12: u32 = 0x62;
pub const ODSY_IMAGE_CODE_LUM_ALPHA12: u32 = 0x68;
pub const ODSY_IMAGE_CODE_RGB12: u32 = 0x69;
pub const ODSY_IMAGE_CODE_ABGR12: u32 = 0x6c;

// The transfer header's first-burst count, bits 23:20, 15 standing for 16.
pub const ODSY_IMAGE_HEADER_GROUP_SHIFT: u32 = 20;
pub const ODSY_IMAGE_HEADER_GROUP_MASK: u32 = 0xf;
pub const ODSY_IMAGE_HEADER_GROUP_FULL: u32 = 16;

// The window restore.
pub const ODSY_CMD_WINDOW_RESTORE: u32 = 0x90008116; // 0 operands
pub const ODSY_REG_WINDOW_ORIGIN: u32 = 0x0841; // 2 operands: x, y
pub const ODSY_REG_ID_COMPARE: u32 = 0x0840; // 2 operands: id-plane base, test
pub const ODSY_ID_COMPARE_ENABLE: u32 = 1 << 16;
pub const ODSY_ID_COMPARE_MASK_SHIFT: u32 = 8;
pub const ODSY_REG_WINDOW_CLIP: u32 = 0x204b; // 8 operands: four rectangles
pub const ODSY_ORIGIN_BITS: u32 = 13;
pub const ODSY_CLIP_FIELD_MASK: u32 = 0xfff;
pub const ODSY_CLIP_X_SHIFT: u32 = 12;

// The off-screen 0x204b
pub const ODSY_CLIP_OFFSCREEN_HI: u32 = 0x00ffffff;
pub const ODSY_CLIP_WORD_MASK: u32 = 0x00ffffff;

// The clip rectangles out of reset
pub const ODSY_CLIP_RESET_FULL: u32 = 0x00ffffff;
pub const ODSY_CLIP_RESET_EMPTY: u32 = 0x00001001;
pub const ODSY_CONTROL_WIDGET_MASK: u32 = 0xf;

// Memory configuration
pub const ODSY_CONFIG_SIZE_SHIFT: u32 = 30;
pub const ODSY_CONFIG_FOUR_BANKS: u32 = 1 << 29;
pub const ODSY_CONFIG_LATENCY_2: u32 = 1 << 28;

// Bits 27:24 the live V12 capture read back as 0x1
pub const ODSY_CONFIG_COPY_FIELD: u32 = 1 << 24;

// Reads and writes are reported apart
pub const ODSY_UNIMP_WRITE: u32 = 1 << 31;

// The display back end's 32 by 32 hardware cursor, drawn over the resolved picture
pub const ODSY_CURSOR_BASE: u32 = 0x2500;
pub const ODSY_CURSOR_ENTRIES: u32 = 0x100;
pub const ODSY_CURSOR_GLYPH: u32 = 0x00;
pub const ODSY_CURSOR_POSITION: u32 = 0x80;
pub const ODSY_CURSOR_CONTROL: u32 = 0x81;
pub const ODSY_CURSOR_COLOURS: u32 = 0x90;
pub const ODSY_CURSOR_COLOUR_COUNT: u32 = 15;
pub const ODSY_CURSOR_PLANE: u32 = 0xc0;
pub const ODSY_CURSOR_SIZE: u32 = 32;

// Control bits.
pub const ODSY_CURSOR_CTL_ENABLE: u32 = 1 << 0;
pub const ODSY_CURSOR_CTL_CROSSHAIR: u32 = 1 << 2;
pub const ODSY_CURSOR_CTL_CHANNEL1: u32 = 1 << 5;
pub const ODSY_CURSOR_CTL_KNOWN: u32 = 0xc000003f;

// Once-only reports, bits of cursor_reported.
pub const ODSY_CURSOR_REPORT_GLYPH15: u32 = 1 << 0;
pub const ODSY_CURSOR_REPORT_PLANE: u32 = 1 << 1;
pub const ODSY_CURSOR_REPORT_CROSSHAIR: u32 = 1 << 2;
pub const ODSY_CURSOR_REPORT_CHANNEL1: u32 = 1 << 3;
pub const ODSY_CURSOR_REPORT_CONTROL: u32 = 1 << 4;
pub const ODSY_CURSOR_REPORT_ENTRY: u32 = 1 << 5;

// Back-end entries 0x2000-0x2fff
pub const ODSY_BACKEND_CTL_BASE: u32 = 0x2000;
pub const ODSY_BACKEND_CTL_ENTRIES: u32 = 0x1000;

// The hardware shadow page
pub const ODSY_HW_PIPE_STATE: u32 = 0x80;
pub const ODSY_HW_PIPE_IDLE: u32 = 0xf;

// Pipe-state bit 11 of an open triangle strip
pub const ODSY_HW_PIPE_STRIP_ODD: u32 = 1 << 11;
pub const ODSY_HW_CURRENT_ATTRIBUTES: u32 = 0x84;
pub const ODSY_HW_DESCRIPTOR_WORDS: u32 = 10; // +0x84 to +0xab
pub const ODSY_HW_INDEX_WORD: u32 = 0x240;
pub const ODSY_HW_COLOUR_WORD: u32 = 0x250;
pub const ODSY_HW_INDEX_OPERANDS: u32 = 0x800;
pub const ODSY_HW_COLOUR_OPERANDS: u32 = 0x840;
pub const ODSY_HW_RING_WORD: u32 = 0x100;
pub const ODSY_HW_RING_WORD_STRIDE: u32 = 16;
pub const ODSY_HW_RING_OPERANDS: u32 = 0x300;
pub const ODSY_HW_RING_OPERAND_STRIDE: u32 = 64;
pub const ODSY_HW_ATTR_INDEX: usize = 1;
pub const ODSY_HW_ATTR_COLOUR: usize = 2;
pub const ODSY_HW_ATTR_NORMAL: usize = 3;
pub const ODSY_HW_ATTR_TEXTURE: usize = 4;
pub const ODSY_HW_ATTR_EDGE: usize = 7;
pub const ODSY_HW_ATTR_MATERIAL: usize = 8;
pub const ODSY_HW_ATTR_SHININESS: usize = 12;
pub const ODSY_HW_ATTR_BACK_MATERIAL: usize = 14;
pub const ODSY_HW_ATTR_BACK_SHININESS: usize = 18;

// The index word the kernel's template carries
pub const ODSY_GL_REPLAY_INDEX: u32 = 0xce80c406; // 1 operand

// How far past [0, 1] a fragment's depth may round and still be drawn, clamped
pub const ODSY_GL_DEPTH_SLACK: f64 = 1.0f64 / (1u32 << 20) as f64;

// A volume level's slices sit 64 << word 2 address units apart.
pub const ODSY_VOLUME_SLICE_SHIFT_MAX: u32 = 20;

// The stock driver's two-point function (-4, 1/4), (0, 1/2) fills entry 0 with 1/2, falls by 1/512 per entry to 1/4 at entry 128 and holds 1/4 through ...
pub const ODSY_TEX_DETAIL_FUNCTION_LAST_LOD: f64 = -((SGI_ODYSSEY_TEX_DETAIL_FUNCTION_ENTRIES as f64) - 1.0) / 32.0;

// Post-filter scale and bias (0x203b, where texture validation sends a texture's words)
pub const ODSY_TEX_SCALE_BIAS_WORDS: u32 = 8;
pub const ODSY_TEX_SCALE_BIAS_BITS: u32 = 16;
pub const ODSY_TEX_SCALE_BIAS_ONE: u32 = 4096;

// Sampler word 4 bits 7:5 are the minification code (0 nearest, 1 linear, 2 to 5 the four mipmap modes in GL order) and bits 4:1 the magnification code ...
pub const ODSY_TEX_MIN_MASK: u32 = 0xe0;
pub const ODSY_TEX_MIN_NEAREST_MIPMAP_NEAREST: u32 = 0x40;
pub const ODSY_TEX_MIN_LINEAR_MIPMAP_LINEAR: u32 = 0xa0;

// Word 4 bit 17
pub const ODSY_TEX_SAMPLER_BIT17: u32 = 1 << 17;

// The four 0x1013 words are 14-bit two's complement with 8 fraction bits
pub const ODSY_TEX_LOD_MASK: u32 = 0x3fff;
pub const ODSY_TEX_LOD_BITS: u32 = 14;
pub const ODSY_TEX_LOD_ONE: u32 = 256;
pub const ODSY_TEX_LOD_LIMIT: u32 = 15 * ODSY_TEX_LOD_ONE;
pub const ODSY_CRC64_ECMA_POLY: u64 = 0x42f0e1eba9ea3693;

// Source format code 0x6c
pub const ODSY_DMA_ABGR_CONTROL: u32 = 0x4001016c;
pub const ODSY_DMA_ABGR_CONTROL_IGNORED: u32 = 0x3e000000;

// dmplay's movie frames
pub const ODSY_DMA_MOVIE_FORMAT: u32 = 0x82233000;
pub const ODSY_DMA_MOVIE_CONTROL: u32 = 0x4001004c;

// A read's host bytes for one pixel
pub const ODSY_DMA_BYTE_ZERO: u32 = 4;
pub const ODSY_DMA_BYTE_INDEX: u32 = 5;
/// The index word's bits 15:8.
pub const ODSY_DMA_BYTE_INDEX_HIGH: u32 = 6;
/// The index when the pixel is shown through a colour map, else the red byte.
pub const ODSY_DMA_BYTE_INDEX_OR_RED: u32 = 7;

// The 16 KiB pages an entry's longest piece can touch.
pub const ODSY_DMA_PIECE_PAGES: usize = (ODSY_DMA_PTE_LEN_MASK as u64 + ODSY_DMA_PAGE_SIZE).div_ceil(ODSY_DMA_PAGE_SIZE) as usize;

// The raster-position query.
pub const ODSY_QUERY_KIND_RASTER: u32 = 3;
/// The last of the four zero-operand blocks (0x3cf8 to 0x3cfb) that end a
/// glRasterPos stream. INFERRED as the point where the board latches the
/// raster index, colour and texture coordinate (odyssey datasheet,
/// "Raster-position query and glyph colour").
pub const ODSY_RASTER_POS_LATCH_WORD: u32 = 0x3cfb;
pub const ODSY_QUERY_RASTER_INDEX: u32 = 0x928;
pub const ODSY_QUERY_RASTER_COLOUR: u32 = 0x93c;

// The GL library's conversion of a signed-integer colour or normal operand to a float, in single precision as its query decoder does.
pub const ODSY_F32_SCALE_8: u32 = 0x3b808081;
pub const ODSY_F32_SCALE_16: u32 = 0x37800080;
pub const ODSY_F32_SCALE_32: u32 = 0x2f800005;

// The host ring under the command FIFO, in entries (fifo.rs). A store is one
// entry whatever its width, so it holds at least twice the board's FIFO in
// words; a store is refused only when it is full.
pub const ODSY_WORK_QUEUE_MAX: u32 = 65536;

// The board's command FIFO, in 32-bit words: sixteen 8 KB tiles of board
// memory, as the kernel's set-up (0x6b1) gives it.
pub const ODSY_CFIFO_WORDS: u32 = 32768;

// The waiting stores whose credits have been returned (an approximation,
// see fifo.rs), in stores.
pub const ODSY_CFIFO_DEPTH: u32 = 64;

// Board-to-host and sentinel reports, each made once (dma_read_reported).
pub const ODSY_DMA_REPORT_DRAW_SENTINEL: u32 = 1 << 0;
pub const ODSY_DMA_REPORT_SENTINEL: u32 = 1 << 1;
pub const ODSY_DMA_REPORT_LAYOUT: u32 = 1 << 2;
pub const ODSY_DMA_REPORT_SOURCE: u32 = 1 << 3;
pub const ODSY_DMA_REPORT_BOUNDS: u32 = 1 << 4;
pub const ODSY_DMA_REPORT_INDEX: u32 = 1 << 5;
pub const ODSY_DMA_REPORT_SHAPE: u32 = 1 << 6;
pub const ODSY_DMA_REPORT_WRITE: u32 = 1 << 7;
pub const ODSY_DMA_REPORT_UPLOAD_SENTINEL: u32 = 1 << 8;
pub const ODSY_DMA_REPORT_INDEX_OBJECT: u32 = 1 << 9;
pub const ODSY_DMA_REPORT_COLOUR_OVER_INDEX: u32 = 1 << 10;
pub const ODSY_DMA_REPORT_RAW64: u32 = 1 << 11;
pub const ODSY_DMA_REPORT_SHARED: u32 = 1 << 12;
pub const ODSY_DMA_REPORT_STENCIL_OR_INDEX: u32 = 1 << 13;
pub const ODSY_DMA_REPORT_UNLISTED_FIELDS: u32 = 1 << 14;

// gl_entry[] slots.
pub const ODSY_ENTRY_29: usize = 0;
pub const ODSY_ENTRY_2A: usize = 1;
pub const ODSY_ENTRY_2C: usize = 2;
pub const ODSY_ENTRY_33: usize = 3;

// Every command word carries its own operand count in bits 13:10, the field
// value 15 encoding a 16-word burst. A register-select's low 16 bits carry
// the same field, so this reads correctly given the whole word or just the
// register.
pub fn cmd_arity(word: u32) -> u32 {
    let count = (word >> ODSY_CMD_ARITY_SHIFT) & ODSY_CMD_ARITY_MASK;

    if count == ODSY_CMD_ARITY_MASK { ODSY_CMD_ARITY_BURST } else { count }
}
