/// VINO — Video-In, No Out
///
/// SGI VINO ASIC (GIO64 slot 0, base 0x00080000).
/// Two independent video capture channels (A and B), each with:
///   - Philips SAA7191 / SGI CDMC input source selection
///   - Clipping, decimation, colour-space conversion, dithering
///   - 1 KB FIFO (128 × 64-bit words)
///   - Descriptor-based DMA with 4-entry cache
/// Master I2C bus for programming SAA7191 (DMSD) and CDMC camera controller.
///
/// References:
///   docs/vino/vino.md         — SGI VINO Design Spec 099-8937-001 v2.0
///   docs/vino/vino.{h,cpp}   — MAME reference implementation (Ryan Holtz)
///   irix/stand/arcs/ide/IP22/video/VINO/vinohw.h — IRIX diagnostic headers

use std::sync::Arc;
use parking_lot::{Mutex, Condvar};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device};
use crate::saa7191::Saa7191;
use crate::devlog::{LogModule, devlog_is_active};

/// Interrupt callback — implemented by the machine glue to assert/deassert
/// the VINO interrupt line on the IOC.  Keeps vino.rs free of IOC details.
pub trait VinoIrq: Send + Sync {
    fn set_interrupt(&self, active: bool);
}

// ─── GIO64 mapping ──────────────────────────────────────────────────────────

/// Physical base address of the VINO register block.
/// Diags use PHYS_TO_K1(0x00080000) — physical address is 0x00080000, not in GIO slot space.
pub const VINO_BASE: u32 = 0x00080000;
/// Total register aperture size.
pub const VINO_SIZE: u32 = 0x00001000; // 4 KB covers all registers (0x000–0x138)

// ─── Register byte offsets (relative to VINO_BASE) ──────────────────────────
//
// VINO registers are 64-bit on the GIO bus: the meaningful 32-bit value lives
// in the *low word* (+4 relative to the 8-byte-aligned slot).
// uses offset>>2 word addressing and masks ~1 to collapse hi/lo accesses.
//
//   Global regs:    0x0000–0x0028
//   Channel A regs: 0x0028–0x00B0
//   Channel B regs: 0x00B0–0x0138

pub mod reg {
    // ── Global ────────────────────────────────────────────────────────────
    pub const REV_ID:       u32 = 0x0000; // r   Revision/ID (chip_id[7:4], rev[3:0])
    pub const CONTROL:      u32 = 0x0008; // rw  Global control (see ctrl module)
    pub const INTR_STATUS:  u32 = 0x0010; // rw  Interrupt status (write 0 to clear bits)
    pub const I2C_CONTROL:  u32 = 0x0018; // rw  I2C control/status
    pub const I2C_DATA:     u32 = 0x0020; // rw  I2C data register

    // ── Per-channel base offsets ──────────────────────────────────────────
    pub const CHA_BASE: u32 = 0x0028;
    pub const CHB_BASE: u32 = 0x00B0;

    // ── Per-channel register offsets (relative to CHA_BASE / CHB_BASE) ───
    pub const CH_ALPHA:          u32 = 0x0000; // rw  8-bit alpha blend factor
    pub const CH_CLIP_START:     u32 = 0x0008; // rw  Clip start (x[9:0], y_odd[18:10], y_even[27:19])
    pub const CH_CLIP_END:       u32 = 0x0010; // rw  Clip end (same encoding)
    pub const CH_FRAME_RATE:     u32 = 0x0018; // rw  Frame-rate mask + NTSC/PAL bit
    pub const CH_FIELD_COUNTER:  u32 = 0x0020; // r   Field counter (16-bit, read-only)
    pub const CH_LINE_SIZE:      u32 = 0x0028; // rw  Line stride in bytes (bits [11:3])
    pub const CH_LINE_COUNT:     u32 = 0x0030; // rw  Current line counter
    pub const CH_PAGE_INDEX:     u32 = 0x0038; // rw  Byte offset within current 4K page
    pub const CH_NEXT_4_DESC:    u32 = 0x0040; // rw  Pointer to next four descriptors
    pub const CH_DESC_TABLE_PTR: u32 = 0x0048; // rw  Pointer to start of descriptor table
    pub const CH_DESC_0:         u32 = 0x0050; // rw  Descriptor cache entry 0
    pub const CH_DESC_1:         u32 = 0x0058; // rw  Descriptor cache entry 1
    pub const CH_DESC_2:         u32 = 0x0060; // rw  Descriptor cache entry 2
    pub const CH_DESC_3:         u32 = 0x0068; // rw  Descriptor cache entry 3
    pub const CH_FIFO_THRESHOLD: u32 = 0x0070; // rw  FIFO DMA threshold (bits [9:3])
    pub const CH_FIFO_READ:      u32 = 0x0078; // r   FIFO GIO (read) pointer
    pub const CH_FIFO_WRITE:     u32 = 0x0080; // r   FIFO video (write) pointer
}

// ─── Revision/ID register ────────────────────────────────────────────────────

pub mod rev_id {
    /// Expected chip ID in bits [7:4].
    pub const CHIP_ID: u32 = 0xB;
    /// Reset value: chip_id=0xB, rev=0 → 0xB0.
    pub const RESET_VAL: u32 = 0xB0;
}

// ─── Control register (offset 0x0008) ────────────────────────────────────────

pub mod ctrl {
    // Bit 0
    pub const ENDIAN_LITTLE: u32        = 1 << 0;  // 0=big-endian (default), 1=little-endian

    // Channel A interrupt enables (bits 1–3)
    pub const CHA_FIELD_INT_EN: u32     = 1 << 1;  // end-of-field interrupt enable
    pub const CHA_FIFO_INT_EN: u32      = 1 << 2;  // FIFO overflow interrupt enable
    pub const CHA_DESC_INT_EN: u32      = 1 << 3;  // end-of-descriptor interrupt enable

    // Channel B interrupt enables (bits 4–6)
    pub const CHB_FIELD_INT_EN: u32     = 1 << 4;
    pub const CHB_FIFO_INT_EN: u32      = 1 << 5;
    pub const CHB_DESC_INT_EN: u32      = 1 << 6;

    // Channel A control (bits 7–18)
    pub const CHA_DMA_EN: u32           = 1 << 7;  // enable channel A DMA capture
    pub const CHA_INTERLEAVE_EN: u32    = 1 << 8;  // interleave odd+even fields into one frame
    pub const CHA_SYNC_EN: u32          = 1 << 9;  // sync channels A and B
    pub const CHA_SELECT_D1: u32        = 1 << 10; // 0=Philips SAA7191, 1=D1/camera (CDMC)
    pub const CHA_COLOR_SPACE_RGB: u32  = 1 << 11; // 0=YUV, 1=RGB
    pub const CHA_LUMA_ONLY: u32        = 1 << 12; // output Y-only (8 bpp greyscale)
    pub const CHA_DECIMATE_EN: u32      = 1 << 13; // enable spatial decimation
    pub const CHA_DECIMATION_SHIFT: u32 = 14;      // decimation factor field [16:14]
    pub const CHA_DECIMATION_MASK: u32  = 0x7;     // factor = (field + 1): 1,2,3,4
    pub const CHA_DECIMATE_HORIZ: u32   = 1 << 17; // decimate horizontally only (not vertically)
    pub const CHA_DITHER_EN: u32        = 1 << 18; // dither RGB24→RGB8

    // Channel B control (bits 19–30) — same layout as channel A
    pub const CHB_DMA_EN: u32           = 1 << 19;
    pub const CHB_INTERLEAVE_EN: u32    = 1 << 20;
    pub const CHB_SYNC_EN: u32          = 1 << 21;
    pub const CHB_SELECT_D1: u32        = 1 << 22;
    pub const CHB_COLOR_SPACE_RGB: u32  = 1 << 23;
    pub const CHB_LUMA_ONLY: u32        = 1 << 24;
    pub const CHB_DECIMATE_EN: u32      = 1 << 25;
    pub const CHB_DECIMATION_SHIFT: u32 = 26;
    pub const CHB_DECIMATION_MASK: u32  = 0x7;
    pub const CHB_DECIMATE_HORIZ: u32   = 1 << 29;
    pub const CHB_DITHER_EN: u32        = 1 << 30;

    /// Writable bits mask (bit 31 reserved).
    pub const MASK: u32 = 0x7FFF_FFFF;
}

// ─── Interrupt status register (offset 0x0010) ───────────────────────────────
//
// Bits are set by hardware; software clears by writing 0 to individual bits.
// `interrupts_w()` masks status with enabled bits from CONTROL before asserting IRQ.

pub mod isr {
    pub const CHA_EOF:  u32 = 1 << 0; // channel A end-of-field
    pub const CHA_FIFO: u32 = 1 << 1; // channel A FIFO overflow
    pub const CHA_DESC: u32 = 1 << 2; // channel A end-of-descriptor (STOP bit)
    pub const CHB_EOF:  u32 = 1 << 3; // channel B end-of-field
    pub const CHB_FIFO: u32 = 1 << 4; // channel B FIFO overflow
    pub const CHB_DESC: u32 = 1 << 5; // channel B end-of-descriptor (STOP bit)
    pub const MASK: u32 = 0x3F;
}

// ─── I2C control/status register (offset 0x0018) ─────────────────────────────

pub mod i2c_ctrl {
    pub const NOT_IDLE: u32    = 1 << 0; // 0=idle (force idle when written), 1=bus active
    pub const READ: u32        = 1 << 1; // 0=write direction, 1=read direction
    pub const HOLD_BUS: u32    = 1 << 2; // 0=release after xfer, 1=hold (repeated start)
    // bit 3 reserved
    pub const XFER_BUSY: u32   = 1 << 4; // r: 1=transfer in progress
    pub const NACK: u32        = 1 << 5; // r: 1=no acknowledge received
    // bit 6 reserved
    pub const BUS_ERR: u32     = 1 << 7; // r: 1=bus error (arbitration lost)
    pub const MASK: u32        = 0xB7;   // writable bits
}

// ─── I2C data register (offset 0x0020) ───────────────────────────────────────

pub mod i2c_data {
    pub const MASK: u32 = 0xFF;
}

// ─── I2C slave addresses ──────────────────────────────────────────────────────

pub mod i2c_addr {
    /// Philips SAA7191 DMSD (composite / S-Video decoder).
    pub const DMSD: u8 = 0x8A;
    /// SGI CDMC camera controller.
    pub const CDMC: u8 = 0xAE;
}

// ─── Clip register encoding ───────────────────────────────────────────────────

pub mod clip {
    pub const X_SHIFT: u32      = 0;
    pub const X_MASK: u32       = 0x03FF; // 10 bits
    pub const YODD_SHIFT: u32   = 10;
    pub const YODD_MASK: u32    = 0x01FF; // 9 bits
    pub const YEVEN_SHIFT: u32  = 19;
    pub const YEVEN_MASK: u32   = 0x01FF; // 9 bits
    pub const REG_MASK: u32     = (X_MASK << X_SHIFT)
                                | (YODD_MASK  << YODD_SHIFT)
                                | (YEVEN_MASK << YEVEN_SHIFT);
}

// ─── Frame-rate register encoding ────────────────────────────────────────────

pub mod frame_rate {
    /// Bit 0: 0 = NTSC (30 fps / 60 fields), 1 = PAL (25 fps / 50 fields).
    pub const PAL: u32         = 1 << 0;
    /// Bits [12:1]: 12-bit frame-skip mask (1 bit per field in a 12/10-field window).
    pub const MASK_SHIFT: u32  = 1;
    pub const MASK_BITS: u32   = 0x0FFF;
    /// Full register mask.
    pub const REG_MASK: u32    = 0x1FFF;
}

// ─── DMA descriptor encoding ──────────────────────────────────────────────────
//
// Each descriptor is a 32-bit word stored in memory.  The emulator caches four
// descriptors per channel as u64 with validity/control flags in the upper half.

pub mod desc {
    /// Physical page address mask (bits [31:4], 16-byte aligned).
    pub const PTR_MASK: u32     = 0xFFFF_FFF0;
    /// Control bit: STOP — terminate DMA after this descriptor; raise DESC interrupt.
    pub const STOP_BIT: u64     = 1 << 31;
    /// Control bit: JUMP — bits [29:0] are a pointer to the next descriptor block.
    pub const JUMP_BIT: u64     = 1 << 30;
    /// Internal: valid flag (set by emulator to track cache state).
    pub const VALID_BIT: u64    = 1u64 << 32;
    /// Mask for the data/address portion of a cached descriptor.
    pub const DATA_MASK: u64    = 0x0000_0000_FFFF_FFFF;
}

// ─── FIFO threshold mask ──────────────────────────────────────────────────────
// The hardware FIFO (1KB, 128×64-bit) is not emulated as a buffer; assembled
// dwords are written directly to memory.  We only keep the threshold register.
pub mod fifo {
    pub const THRESHOLD_MASK: u32 = 0x03F8; // bits [9:3]
}

// ─── Pixel formats ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 32-bit RGBA, 2 pixels per 64-bit FIFO word.
    Rgba32,
    /// 16-bit YUV 4:2:2 (UYVY), 4 pixels per 64-bit FIFO word.
    Yuv422,
    /// 8-bit dithered RGB (2:3:3 BGR), 8 pixels per 64-bit FIFO word.
    Rgba8,
    /// 8-bit luma only, 8 pixels per 64-bit FIFO word.
    Y8,
}

// ─── Channel index ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    A = 0,
    B = 1,
}

// ─── Per-channel state ────────────────────────────────────────────────────────

struct ChannelState {
    // ── Visible registers ──
    alpha:          u32, // [7:0] blend factor
    clip_start:     u32,
    clip_end:       u32,
    frame_rate:     u32,
    field_counter:  u32, // incremented each field; even bit = even/odd field
    line_size:      u32, // stride in bytes (bits [11:3])
    line_counter:   u32,
    page_index:     u32, // byte offset within current 4 K page (bits [11:3])
    next_desc_ptr:  u32, // pointer to next group-of-four descriptors in memory
    start_desc_ptr: u32, // pointer to start of descriptor table (for interleaved rewind)
    descriptors:    [u64; 4], // cached descriptor entries (VALID_BIT in bit 32)
    fifo_threshold: u32,
    fifo_gio_ptr:   u32, // GIO (DMA-read) FIFO pointer
    fifo_video_ptr: u32, // video (capture-write) FIFO pointer

    // ── Internal / derived ──
    // No FIFO buffer — assembled dwords are written directly to memory.
    decimation:      u32, // effective decimation factor (1–8)
    next_dword:      u64, // dword being assembled from incoming pixels
    word_pixel_cnt:  u32, // pixels packed into next_dword so far
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            alpha:          0,
            clip_start:     0,
            clip_end:       0,
            frame_rate:     0,
            field_counter:  0,
            line_size:      0,
            line_counter:   0,
            page_index:     0,
            next_desc_ptr:  0,
            start_desc_ptr: 0,
            descriptors:    [0u64; 4],
            fifo_threshold: 0,
            fifo_gio_ptr:   0,
            fifo_video_ptr: 0,
            decimation:     1,
            next_dword:     0,
            word_pixel_cnt: 0,
        }
    }
}

// ─── Top-level VINO state (lives inside Mutex) ────────────────────────────────

struct VinoState {
    rev_id:     u32,
    control:    u32,
    int_status: u32,
    i2c_ctrl:   u32,
    i2c_data:   u32,
    channels:   [ChannelState; 2],
    dmsd:       Saa7191,  // Philips SAA7191B on the VINO I2C bus
}

impl Default for VinoState {
    fn default() -> Self {
        Self {
            rev_id:     rev_id::RESET_VAL,
            control:    0,
            int_status: 0,
            i2c_ctrl:   0,
            i2c_data:   0,
            channels:   [ChannelState::default(), ChannelState::default()],
            dmsd:       Saa7191::new(),
        }
    }
}

// ─── DMA wake signal ──────────────────────────────────────────────────────────

struct DmaWake {
    cond:  Condvar,
    mutex: Mutex<()>,
}

impl DmaWake {
    fn new() -> Arc<Self> {
        Arc::new(Self { cond: Condvar::new(), mutex: Mutex::new(()) })
    }
    fn notify(&self) { self.cond.notify_all(); }
    fn wait(&self)   { self.cond.wait(&mut self.mutex.lock()); }
}

// ─── Public device handle ─────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Vino {
    state:   Arc<Mutex<VinoState>>,
    irq:     Arc<Mutex<Option<Arc<dyn VinoIrq>>>>,
    sys_mem: Arc<Mutex<Option<Arc<dyn BusDevice>>>>,
    wake:    Arc<DmaWake>,
    running: Arc<AtomicBool>,
    thread:  Arc<Mutex<Option<thread::JoinHandle<()>>>>,
}

impl Vino {
    pub fn new() -> Self {
        Self {
            state:   Arc::new(Mutex::new(VinoState::default())),
            irq:     Arc::new(Mutex::new(None)),
            sys_mem: Arc::new(Mutex::new(None)),
            wake:    DmaWake::new(),
            running: Arc::new(AtomicBool::new(false)),
            thread:  Arc::new(Mutex::new(None)),
        }
    }

    /// Set the interrupt callback (called from machine setup after IOC is ready).
    pub fn set_irq(&self, irq: Arc<dyn VinoIrq>) {
        *self.irq.lock() = Some(irq);
    }

    /// Connect the physical bus so DMA writes can reach system memory.
    pub fn set_phys(&self, mem: Arc<dyn BusDevice>) {
        *self.sys_mem.lock() = Some(mem);
    }

    // ── Power-on reset ────────────────────────────────────────────────────

    pub fn power_on(&self) {
        let mut st = self.state.lock();
        st.dmsd.power_on(); // reset before overwriting the field
        *st = VinoState::default();
    }

    // ── Interrupt assertion ───────────────────────────────────────────────

    fn raise_interrupt(st: &mut VinoState, irq: &Option<Arc<dyn VinoIrq>>, new_status: u32) {
        // Only bits enabled in CONTROL (shifted right by 1) appear in int_status.
        // ctrl bits [6:1] are the six enable bits; isr bits [5:0] are the six status bits.
        let enable_mask = (st.control >> 1) & isr::MASK;
        let old = st.int_status;
        st.int_status = (new_status & enable_mask) & isr::MASK;

        let newly_raised = !old & st.int_status;
        if newly_raised != 0 {
            if let Some(irq) = irq {
                irq.set_interrupt(true);
            }
        } else if st.int_status == 0 && old != 0 {
            if let Some(irq) = irq {
                irq.set_interrupt(false);
            }
        }
    }

    // ── Control register write ────────────────────────────────────────────

    /// Returns true if a DMA channel was just enabled (caller should notify wake).
    fn control_w(st: &mut VinoState, irq: &Option<Arc<dyn VinoIrq>>, val: u32) -> bool {
        let old = st.control;
        st.control = val & ctrl::MASK;

        // Update derived decimation factors for each channel.
        for (ch_idx, (dec_en, dec_shift, dec_mask)) in [
            (ctrl::CHA_DECIMATE_EN, ctrl::CHA_DECIMATION_SHIFT, ctrl::CHA_DECIMATION_MASK),
            (ctrl::CHB_DECIMATE_EN, ctrl::CHB_DECIMATION_SHIFT, ctrl::CHB_DECIMATION_MASK),
        ].iter().enumerate() {
            st.channels[ch_idx].decimation = if st.control & dec_en != 0 {
                ((st.control >> dec_shift) & dec_mask) + 1
            } else {
                1
            };
        }

        if old == st.control { return false; }

        // Re-evaluate masked interrupts after enable-bit change.
        let cur = st.int_status;
        Self::raise_interrupt(st, irq, cur);

        // DMA enable/disable for each channel.
        let mut any_started = false;
        for (ch_idx, dma_en_bit) in [ctrl::CHA_DMA_EN, ctrl::CHB_DMA_EN].iter().enumerate() {
            let changed = (old ^ st.control) & dma_en_bit;
            if changed == 0 { continue; }
            if st.control & dma_en_bit != 0 {
                Self::start_channel(st, ch_idx);
                any_started = true;
            } else {
                Self::stop_channel(st, ch_idx);
            }
        }
        any_started
    }

    fn start_channel(st: &mut VinoState, ch: usize) {
        let chan = &mut st.channels[ch];
        chan.field_counter  = 0;
        chan.fifo_gio_ptr   = 0;
        chan.fifo_video_ptr = 0;
        eprintln!("VINO: channel {} DMA enabled", if ch == 0 { 'A' } else { 'B' });
        // DMA thread is notified by control_w() after returning from here.
    }

    fn stop_channel(_st: &mut VinoState, ch: usize) {
        eprintln!("VINO: channel {} DMA disabled", if ch == 0 { 'A' } else { 'B' });
    }

    // ── Interrupt status write (write 0 to individual bits to clear) ──────

    fn intr_status_w(st: &mut VinoState, irq: &Option<Arc<dyn VinoIrq>>, val: u32) {
        // A 0-bit in the written value clears the corresponding status bit.
        for bit in 0..6u32 {
            if val & (1 << bit) == 0 {
                st.int_status &= !(1 << bit);
            }
        }
        let cur = st.int_status;
        Self::raise_interrupt(st, irq, cur);
    }

    // ── Descriptor operations ─────────────────────────────────────────────

    fn invalidate_descriptors(chan: &mut ChannelState) {
        for d in &mut chan.descriptors {
            *d &= !desc::VALID_BIT;
        }
    }

    /// Fetch four 32-bit descriptors from `addr` in system memory into the
    /// channel's descriptor cache.  Each entry gets VALID_BIT set.
    /// If descriptor[0] has STOP_BIT the DMA thread will handle the interrupt.
    fn descriptor_fetch(chan: &mut ChannelState, addr: u32, mem: &Arc<dyn BusDevice>) {
        for i in 0..4usize {
            let word_addr = addr.wrapping_add((i as u32) * 4);
            let word = { let _r = mem.read32(word_addr); if _r.is_ok() { _r.data } else {
                    eprintln!("VINO: descriptor_fetch read error at {:#010x}", word_addr);
                    0
                }
            };
            chan.descriptors[i] = (word as u64 & desc::DATA_MASK) | desc::VALID_BIT;
        }
    }

    /// Write next_desc_ptr, invalidate the descriptor cache, and fetch four
    /// new descriptors from memory.
    fn next_desc_w_with_mem(chan: &mut ChannelState, ptr: u32, mem: &Arc<dyn BusDevice>) {
        chan.next_desc_ptr = ptr;
        Self::invalidate_descriptors(chan);
        Self::descriptor_fetch(chan, ptr, mem);
    }

    // Same as above but without memory access (used when sys_mem not yet set).
    fn next_desc_w(chan: &mut ChannelState, ptr: u32) {
        chan.next_desc_ptr = ptr;
        Self::invalidate_descriptors(chan);
    }

    // ── Page-index write with 4 K roll-over and descriptor shift ─────────

    fn page_index_w(chan: &mut ChannelState, val: u32) -> bool {
        let old = chan.page_index;
        chan.page_index = val;
        while chan.page_index >= 0x1000 {
            chan.page_index -= 0x1000;
        }
        if chan.page_index < old {
            Self::shift_descriptors_nomem(chan);
            return true;
        }
        false
    }

    fn shift_descriptors_nomem(chan: &mut ChannelState) {
        for i in 0..3 {
            chan.descriptors[i] = chan.descriptors[i + 1];
        }
        // Descriptor[3] is now stale; caller (DMA thread) will refetch when needed.
        chan.descriptors[3] &= !desc::VALID_BIT;
    }

    fn shift_descriptors(chan: &mut ChannelState, mem: &Arc<dyn BusDevice>) {
        for i in 0..3 {
            chan.descriptors[i] = chan.descriptors[i + 1];
        }
        chan.descriptors[3] &= !desc::VALID_BIT;

        if chan.descriptors[0] & desc::VALID_BIT == 0 {
            // Head is invalid — fetch a new group from next_desc_ptr.
            let ptr = chan.next_desc_ptr;
            Self::descriptor_fetch(chan, ptr, mem);
            chan.next_desc_ptr = chan.next_desc_ptr.wrapping_add(16);
        } else if chan.descriptors[0] & desc::JUMP_BIT != 0 {
            let target = (chan.descriptors[0] as u32) & 0x3FFF_FFFF;
            Self::descriptor_fetch(chan, target, mem);
        }
    }

    // ── DMA thread ────────────────────────────────────────────────────────

    /// Process one 64-bit dword of DMA output for channel `ch`.
    /// Called with the state lock dropped; re-acquires and releases per call.
    /// Returns false when DMA for this channel is complete or stopped.
    fn process_channel_dword(&self, ch: usize, mem: &Arc<dyn BusDevice>) -> bool {
        let mut st = self.state.lock();

        let dma_en = [ctrl::CHA_DMA_EN, ctrl::CHB_DMA_EN][ch];
        if st.control & dma_en == 0 {
            return false; // channel disabled
        }

        let chan = &mut st.channels[ch];

        // Check for STOP bit on the head descriptor.
        if chan.descriptors[0] & desc::VALID_BIT != 0
            && chan.descriptors[0] & desc::STOP_BIT != 0
        {
            // Raise DESC interrupt, disable DMA for this channel.
            let isr_desc = [isr::CHA_DESC, isr::CHB_DESC][ch];
            let new_status = st.int_status | isr_desc;
            let irq = self.irq.lock().clone();
            Self::raise_interrupt(&mut st, &irq, new_status);
            let dma_bit = [ctrl::CHA_DMA_EN, ctrl::CHB_DMA_EN][ch];
            st.control &= !dma_bit;
            return false;
        }

        // Compute physical write address: descriptor base | page_index offset.
        let desc_base = (chan.descriptors[0] as u32) & desc::PTR_MASK as u32;
        let write_addr = desc_base | (chan.page_index & 0x0FF8);

        // In a real capture we'd assemble pixel data; for now write zero
        // (the pixel pipeline is a separate future stub).
        let dword: u64 = chan.next_dword;
        drop(st); // release lock before memory write

        mem.write64(write_addr, dword);

        // Re-acquire to advance page_index (and line_counter in interleaved mode).
        let mut st = self.state.lock();
        let mem = mem.clone();

        let interleave = st.control & [ctrl::CHA_INTERLEAVE_EN, ctrl::CHB_INTERLEAVE_EN][ch] != 0;
        let chan = &mut st.channels[ch];

        let old_page = chan.page_index;
        chan.page_index = (chan.page_index + 8) & 0x0FFF;

        if interleave {
            // In interleaved mode the field is split into two interlaced lines
            // within the same 4K page.  line_counter tracks our offset within
            // the current scan-line; when it reaches line_size we have finished
            // one line and must skip to the next (which starts line_size+8 bytes
            // into the page, leaving room for the other field's line).
            // MAME line_count_w: if line_counter == line_size → reset to 0,
            // advance page_index by line_size+8 (jumping over the interleaved
            // partner line), then let the normal 4K roll-over handle desc shift.
            chan.line_counter += 8;
            if chan.line_counter >= chan.line_size {
                chan.line_counter = 0;
                // Jump page_index forward past the partner scan-line.
                let skip = chan.line_size.wrapping_add(8);
                let new_page = chan.page_index.wrapping_add(skip);
                chan.page_index = new_page & 0x0FFF;
                // If either the +8 step or the line skip crossed 4K, shift descs.
                if chan.page_index < old_page || new_page >= 0x1000 {
                    Self::shift_descriptors(chan, &mem);
                }
                return true;
            }
        }

        if chan.page_index < old_page {
            // Rolled over 4K boundary — shift descriptor cache.
            Self::shift_descriptors(chan, &mem);
        }

        true
    }

    fn process_dma(&self) {
        loop {
            // Sleep until a DMA enable bit is written.
            self.wake.wait();
            if !self.running.load(Ordering::Relaxed) { break; }

            // Drain both channels one dword at a time, dropping and reacquiring
            // the state lock on every iteration to allow CPU register updates.
            loop {
                if !self.running.load(Ordering::Relaxed) { return; }

                let mem_opt = self.sys_mem.lock().clone();
                let mem = match mem_opt {
                    Some(m) => m,
                    None => break,
                };

                let st = self.state.lock();
                let a_en = st.control & ctrl::CHA_DMA_EN != 0;
                let b_en = st.control & ctrl::CHB_DMA_EN != 0;
                drop(st);

                if !a_en && !b_en { break; }

                if a_en { self.process_channel_dword(0, &mem); }
                if b_en { self.process_channel_dword(1, &mem); }
            }
        }
    }

    // ── Channel register decode ───────────────────────────────────────────

    /// Map a bus offset to (channel_index, per-channel register offset).
    /// Returns None for global registers (< CHA_BASE) or unknown offsets.
    fn decode_channel(offset: u32) -> Option<(usize, u32)> {
        if offset >= reg::CHB_BASE {
            Some((1, offset - reg::CHB_BASE))
        } else if offset >= reg::CHA_BASE {
            Some((0, offset - reg::CHA_BASE))
        } else {
            None
        }
    }

    // ── Register read ─────────────────────────────────────────────────────

    fn read_reg(&self, offset: u32) -> u32 {
        let st = self.state.lock();
        // Each VINO register occupies 8 bytes (64-bit GIO slot); the meaningful
        // 32-bit value is in the low word (+4).  Both words alias to the same reg,
        // so mask off bit 2 to collapse the pair — same pattern as mc.rs `& !4`.
        let off = offset & !4u32;

        let val = if let Some((ch, ch_off)) = Self::decode_channel(off) {
            Self::read_channel_reg(&st.channels[ch], ch_off)
        } else {
            match off {
                reg::REV_ID      => st.rev_id,
                reg::CONTROL     => st.control,
                reg::INTR_STATUS => st.int_status,
                reg::I2C_CONTROL => st.i2c_ctrl,
                reg::I2C_DATA    => st.i2c_data,
                _ => {
                    eprintln!("VINO: unknown read at offset {:#06x}", offset);
                    0
                }
            }
        };

        dlog_dev!(LogModule::Vino, "VINO Read  [{:#06x}] ({}) -> {:#010x}",
            off, vino_reg_name(off), val);
        val
    }

    fn read_channel_reg(chan: &ChannelState, off: u32) -> u32 {
        match off {
            reg::CH_ALPHA          => chan.alpha,
            reg::CH_CLIP_START     => chan.clip_start,
            reg::CH_CLIP_END       => chan.clip_end,
            reg::CH_FRAME_RATE     => chan.frame_rate,
            reg::CH_FIELD_COUNTER  => chan.field_counter & 0xFFFF,
            reg::CH_LINE_SIZE      => chan.line_size,
            reg::CH_LINE_COUNT     => chan.line_counter,
            reg::CH_PAGE_INDEX     => chan.page_index,
            reg::CH_NEXT_4_DESC    => chan.next_desc_ptr,
            reg::CH_DESC_TABLE_PTR => chan.start_desc_ptr,
            reg::CH_DESC_0         => (chan.descriptors[0] & desc::DATA_MASK) as u32,
            reg::CH_DESC_1         => (chan.descriptors[1] & desc::DATA_MASK) as u32,
            reg::CH_DESC_2         => (chan.descriptors[2] & desc::DATA_MASK) as u32,
            reg::CH_DESC_3         => (chan.descriptors[3] & desc::DATA_MASK) as u32,
            reg::CH_FIFO_THRESHOLD => chan.fifo_threshold,
            reg::CH_FIFO_READ      => chan.fifo_gio_ptr,
            reg::CH_FIFO_WRITE     => chan.fifo_video_ptr,
            _ => {
                eprintln!("VINO: unknown channel read at ch_off {:#06x}", off);
                0
            }
        }
    }

    // ── Register write ────────────────────────────────────────────────────

    fn write_reg(&self, offset: u32, val: u32) {
        let mut st = self.state.lock();
        let off = offset & !4u32; // collapse 64-bit pair; see read_reg
        let irq = self.irq.lock().clone();
        dlog_dev!(LogModule::Vino, "VINO Write [{:#06x}] ({}) <- {:#010x}",
            off, vino_reg_name(off), val);

        if let Some((ch, ch_off)) = Self::decode_channel(off) {
            let mem = self.sys_mem.lock().clone();
            Self::write_channel_reg(&mut st, ch, ch_off, val, mem.as_ref());
            return;
        }

        match off {
            reg::REV_ID      => { /* read-only, ignore */ }
            reg::CONTROL     => {
                let started = Self::control_w(&mut st, &irq, val);
                drop(st);
                if started { self.wake.notify(); }
                return;
            }
            reg::INTR_STATUS => Self::intr_status_w(&mut st, &irq, val),
            reg::I2C_CONTROL => {
                let prev = st.i2c_ctrl;
                st.i2c_ctrl = val & i2c_ctrl::MASK;

                if prev & i2c_ctrl::NOT_IDLE != 0 && val & i2c_ctrl::NOT_IDLE == 0 {
                    // NOT_IDLE cleared → STOP condition
                    st.dmsd.i2c_stop();
                } else if val & i2c_ctrl::NOT_IDLE != 0 {
                    // Transfer request: execute one byte
                    if val & i2c_ctrl::READ != 0 {
                        // Read direction: fetch byte from SAA7191 → store in I2C_DATA
                        let byte = st.dmsd.i2c_read();
                        st.i2c_data = byte as u32;
                    } else {
                        // Write direction: push I2C_DATA byte to SAA7191
                        st.dmsd.i2c_write(st.i2c_data as u8);
                    }
                    // Transfer completes instantly (no real I2C bus timing)
                    st.i2c_ctrl &= !i2c_ctrl::XFER_BUSY;
                }
            }
            reg::I2C_DATA    => {
                st.i2c_data = val & i2c_data::MASK;
            }
            _ => {
                eprintln!("VINO: unknown write at offset {:#06x} = {:#010x}", offset, val);
            }
        }
    }

    fn write_channel_reg(st: &mut VinoState, ch: usize, off: u32, val: u32,
                         mem: Option<&Arc<dyn BusDevice>>) {
        let chan = &mut st.channels[ch];
        match off {
            reg::CH_ALPHA          => chan.alpha = val & 0xFF,
            reg::CH_CLIP_START     => chan.clip_start = val & clip::REG_MASK,
            reg::CH_CLIP_END       => chan.clip_end   = val & clip::REG_MASK,
            reg::CH_FRAME_RATE     => {
                chan.frame_rate = val & frame_rate::REG_MASK;
                // TODO: recompute frame-mask shifter
            }
            reg::CH_FIELD_COUNTER  => { /* read-only, ignore */ }
            reg::CH_LINE_SIZE      => chan.line_size    = val & 0x0FF8,
            reg::CH_LINE_COUNT     => chan.line_counter = val & 0x0FF8,
            reg::CH_PAGE_INDEX     => { Self::page_index_w(chan, val & 0x0FF8); }
            reg::CH_NEXT_4_DESC    => {
                let ptr = val & desc::PTR_MASK;
                if let Some(m) = mem {
                    Self::next_desc_w_with_mem(chan, ptr, m);
                } else {
                    Self::next_desc_w(chan, ptr);
                }
            }
            reg::CH_DESC_TABLE_PTR => chan.start_desc_ptr = val & desc::PTR_MASK,
            reg::CH_DESC_0         => {
                chan.descriptors[0] = (val as u64 & desc::DATA_MASK) | desc::VALID_BIT;
            }
            reg::CH_DESC_1         => {
                chan.descriptors[1] = (val as u64 & desc::DATA_MASK) | desc::VALID_BIT;
            }
            reg::CH_DESC_2         => {
                chan.descriptors[2] = (val as u64 & desc::DATA_MASK) | desc::VALID_BIT;
            }
            reg::CH_DESC_3         => {
                chan.descriptors[3] = (val as u64 & desc::DATA_MASK) | desc::VALID_BIT;
            }
            reg::CH_FIFO_THRESHOLD => chan.fifo_threshold = val & fifo::THRESHOLD_MASK,
            reg::CH_FIFO_READ      => { /* read-only, ignore */ }
            reg::CH_FIFO_WRITE     => { /* read-only, ignore */ }
            _ => {
                eprintln!("VINO: unknown channel write at ch_off {:#06x} = {:#010x}", off, val);
            }
        }
    }
}

// ─── Device start / stop ──────────────────────────────────────────────────────

impl Vino {
    pub fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) { return; }
        let vino = self.clone();
        *self.thread.lock() = Some(
            thread::Builder::new()
                .name("VINO-DMA".to_string())
                .spawn(move || vino.process_dma())
                .unwrap()
        );
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.wake.notify();
        if let Some(h) = self.thread.lock().take() {
            let _ = h.join();
        }
    }
}

// ─── BusDevice implementation ─────────────────────────────────────────────────
//
// Each VINO register is an 8-byte GIO64 slot; both words alias to the same
// register (bit 2 is masked in read_reg/write_reg, matching mc.rs `& !4`).

impl BusDevice for Vino {
    fn read32(&self, addr: u32) -> BusRead32 {
        let offset = addr.wrapping_sub(VINO_BASE);
        BusRead32::ok(self.read_reg(offset))
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        let offset = addr.wrapping_sub(VINO_BASE);
        self.write_reg(offset, val);
        BUS_OK
    }

    fn read64(&self, addr: u32) -> BusRead64 {
        // GIO64 double-word: high word at addr, low word at addr+4.
        let hi = { let _r = self.read32(addr); if _r.is_ok() { let v = _r.data; v } else { 0 } };
        let lo = { let _r = self.read32(addr + 4); if _r.is_ok() { let v = _r.data; v } else { 0 } };
        BusRead64::ok(((hi as u64) << 32) | lo as u64)
    }

    fn write64(&self, addr: u32, val: u64) -> u32 {
        self.write32(addr,     (val >> 32) as u32);
        self.write32(addr + 4, val as u32);
        BUS_OK
    }
}

// ─── Register name helper ─────────────────────────────────────────────────────

fn vino_reg_name(off: u32) -> &'static str {
    match off {
        reg::REV_ID      => "REV_ID",
        reg::CONTROL     => "CONTROL",
        reg::INTR_STATUS => "INTR_STATUS",
        reg::I2C_CONTROL => "I2C_CONTROL",
        reg::I2C_DATA    => "I2C_DATA",
        // Channel A
        o if o >= reg::CHA_BASE && o < reg::CHB_BASE => match o - reg::CHA_BASE {
            reg::CH_ALPHA          => "A_ALPHA",
            reg::CH_CLIP_START     => "A_CLIP_START",
            reg::CH_CLIP_END       => "A_CLIP_END",
            reg::CH_FRAME_RATE     => "A_FRAME_RATE",
            reg::CH_FIELD_COUNTER  => "A_FIELD_COUNTER",
            reg::CH_LINE_SIZE      => "A_LINE_SIZE",
            reg::CH_LINE_COUNT     => "A_LINE_COUNT",
            reg::CH_PAGE_INDEX     => "A_PAGE_INDEX",
            reg::CH_NEXT_4_DESC    => "A_NEXT_4_DESC",
            reg::CH_DESC_TABLE_PTR => "A_DESC_TABLE_PTR",
            reg::CH_DESC_0         => "A_DESC_0",
            reg::CH_DESC_1         => "A_DESC_1",
            reg::CH_DESC_2         => "A_DESC_2",
            reg::CH_DESC_3         => "A_DESC_3",
            reg::CH_FIFO_THRESHOLD => "A_FIFO_THRESHOLD",
            reg::CH_FIFO_READ      => "A_FIFO_READ",
            reg::CH_FIFO_WRITE     => "A_FIFO_WRITE",
            _                      => "A_?",
        },
        // Channel B
        o if o >= reg::CHB_BASE => match o - reg::CHB_BASE {
            reg::CH_ALPHA          => "B_ALPHA",
            reg::CH_CLIP_START     => "B_CLIP_START",
            reg::CH_CLIP_END       => "B_CLIP_END",
            reg::CH_FRAME_RATE     => "B_FRAME_RATE",
            reg::CH_FIELD_COUNTER  => "B_FIELD_COUNTER",
            reg::CH_LINE_SIZE      => "B_LINE_SIZE",
            reg::CH_LINE_COUNT     => "B_LINE_COUNT",
            reg::CH_PAGE_INDEX     => "B_PAGE_INDEX",
            reg::CH_NEXT_4_DESC    => "B_NEXT_4_DESC",
            reg::CH_DESC_TABLE_PTR => "B_DESC_TABLE_PTR",
            reg::CH_DESC_0         => "B_DESC_0",
            reg::CH_DESC_1         => "B_DESC_1",
            reg::CH_DESC_2         => "B_DESC_2",
            reg::CH_DESC_3         => "B_DESC_3",
            reg::CH_FIFO_THRESHOLD => "B_FIFO_THRESHOLD",
            reg::CH_FIFO_READ      => "B_FIFO_READ",
            reg::CH_FIFO_WRITE     => "B_FIFO_WRITE",
            _                      => "B_?",
        },
        _ => "?",
    }
}

// ─── Device trait (monitor commands) ─────────────────────────────────────────

impl Device for Vino {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) { Vino::stop(self); }
    fn start(&self) { Vino::start(self); }
    fn is_running(&self) -> bool { self.running.load(std::sync::atomic::Ordering::Relaxed) }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![(
            "vino".to_string(),
            "vino debug <on|off> | vino status".to_string(),
        )]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "vino" { return Err(format!("Unknown command: {}", cmd)); }
        let arg0 = args.get(0).copied().unwrap_or("");

        match arg0 {
            "debug" => {
                let arg1 = args.get(1).copied().unwrap_or("");
                match arg1 {
                    "on" => {
                        crate::devlog::devlog().enable(LogModule::Vino);
                        writeln!(writer, "VINO debug on").map_err(|e| e.to_string())?;
                    }
                    "off" => {
                        crate::devlog::devlog().disable(LogModule::Vino);
                        writeln!(writer, "VINO debug off").map_err(|e| e.to_string())?;
                    }
                    _ => return Err("Usage: vino debug <on|off>".to_string()),
                }
            }
            "status" => {
                let st = self.state.lock();
                let log = devlog_is_active(LogModule::Vino);

                writeln!(writer, "VINO Status  (debug {})", if log { "on" } else { "off" })
                    .map_err(|e| e.to_string())?;
                writeln!(writer, "  REV_ID      = {:#010x}  (chip_id={:#x} rev={})",
                    st.rev_id, (st.rev_id >> 4) & 0xF, st.rev_id & 0xF)
                    .map_err(|e| e.to_string())?;
                writeln!(writer, "  CONTROL     = {:#010x}", st.control)
                    .map_err(|e| e.to_string())?;
                writeln!(writer, "    CHA_DMA_EN={} CHB_DMA_EN={} ENDIAN_LITTLE={}",
                    (st.control & ctrl::CHA_DMA_EN != 0) as u8,
                    (st.control & ctrl::CHB_DMA_EN != 0) as u8,
                    (st.control & ctrl::ENDIAN_LITTLE != 0) as u8)
                    .map_err(|e| e.to_string())?;
                writeln!(writer, "  INTR_STATUS = {:#010x}", st.int_status)
                    .map_err(|e| e.to_string())?;
                writeln!(writer, "  I2C_CONTROL = {:#010x}  I2C_DATA = {:#010x}",
                    st.i2c_ctrl, st.i2c_data)
                    .map_err(|e| e.to_string())?;

                for (ch, name) in [(0usize, "A"), (1, "B")] {
                    let c = &st.channels[ch];
                    writeln!(writer, "\n  Channel {}:", name).map_err(|e| e.to_string())?;
                    writeln!(writer, "    alpha={:#04x}  clip_start={:#010x}  clip_end={:#010x}",
                        c.alpha, c.clip_start, c.clip_end)
                        .map_err(|e| e.to_string())?;
                    writeln!(writer, "    frame_rate={:#010x}  field_counter={}",
                        c.frame_rate, c.field_counter)
                        .map_err(|e| e.to_string())?;
                    writeln!(writer, "    line_size={:#06x}  line_counter={:#06x}  page_index={:#06x}",
                        c.line_size, c.line_counter, c.page_index)
                        .map_err(|e| e.to_string())?;
                    writeln!(writer, "    next_desc_ptr={:#010x}  start_desc_ptr={:#010x}",
                        c.next_desc_ptr, c.start_desc_ptr)
                        .map_err(|e| e.to_string())?;
                    for (i, d) in c.descriptors.iter().enumerate() {
                        let valid = d & desc::VALID_BIT != 0;
                        let stop  = d & desc::STOP_BIT  != 0;
                        let jump  = d & desc::JUMP_BIT  != 0;
                        let addr  = (*d as u32) & desc::PTR_MASK;
                        writeln!(writer, "    desc[{}] = {:#010x}  valid={} stop={} jump={}",
                            i, addr, valid as u8, stop as u8, jump as u8)
                            .map_err(|e| e.to_string())?;
                    }
                    writeln!(writer, "    fifo_threshold={:#06x}  decimation={}",
                        c.fifo_threshold, c.decimation)
                        .map_err(|e| e.to_string())?;
                }
            }
            _ => return Err("Usage: vino debug <on|off> | vino status".to_string()),
        }
        Ok(())
    }
}
