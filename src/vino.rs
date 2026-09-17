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
///   docs/vino/vino.pdf        — SGI VINO Design Spec 099-8937-001 v2.0
///   docs/vino/vino.{h,cpp}   — MAME reference implementation (Ryan Holtz)
///   irix/stand/arcs/ide/IP22/video/VINO/vinohw.h — IRIX diagnostic headers

use std::sync::Arc;
use std::time::Duration;
use parking_lot::{Mutex, Condvar};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device};
use crate::saa7191::Saa7191;
use crate::cdmc::Cdmc;
use crate::devlog::{LogModule, devlog_is_active};
use crate::video_source::{VideoSource, Field, FieldParity};

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
    pub const CDMC: u8 = 0x56;
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
    /// Physical address mask (bits [29:4], 16-byte aligned). The address field of
    /// a descriptor / descriptor-table pointer is only 30 bits wide: bits 31 and
    /// 30 are the STOP and JUMP control bits, NOT part of the address. The 6.5
    /// kernel encodes pointers as `JUMP_BIT | kvtophys(addr)` (see
    /// `vinoBuildJumpBugDAPS`), so masking with this — stripping bits 31/30 — is
    /// what recovers the real lomem address (e.g. 0x4861e000 → 0x0861e000). Indy
    /// RAM lives at 0x08000000..0x18000000, so a legitimate address never sets
    /// bits 31/30; there is no 0x40000000 RAM alias on the hardware.
    pub const PTR_MASK: u32     = 0x3FFF_FFF0;
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
    dmsd:       Saa7191,  // Philips SAA7191B on the VINO I2C bus (addr 0x8A/0x8B)
    cdmc:       Cdmc,     // SGI IndyCam controller on the same bus (addr 0x56/0x57)
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
            cdmc:       Cdmc::new(),
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
    /// D0 / composite (SAA7191 path) — defaults to black field generator.
    source_d0: Arc<Mutex<Option<Arc<dyn VideoSource>>>>,
    /// D1 / IndyCam (CDMC path) — set via set_source() from machine config.
    source_d1: Arc<Mutex<Option<Arc<dyn VideoSource>>>>,
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
            source_d0: Arc::new(Mutex::new(Some(Arc::new(
                crate::video_source::BlackSource::new(crate::video_source::VideoStandard::Ntsc),
            ) as Arc<dyn VideoSource>))),
            source_d1: Arc::new(Mutex::new(None)),
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

    /// Install the D1 (IndyCam / CDMC) video input source.
    pub fn set_source(&self, src: Arc<dyn VideoSource>) {
        *self.source_d1.lock() = Some(src);
        self.wake.notify();
    }

    /// Install the D0 (composite / SAA7191) video input source.
    pub fn set_source_d0(&self, src: Arc<dyn VideoSource>) {
        *self.source_d0.lock() = Some(src);
        self.wake.notify();
    }

    /// Read CDMC register file for pixel pipeline adjustments.
    pub fn cdmc_regs(&self) -> [u8; crate::cdmc::reg::COUNT] {
        self.state.lock().cdmc.regs_copy()
    }

    // ── Power-on reset ────────────────────────────────────────────────────

    pub fn power_on(&self) {
        let mut st = self.state.lock();
        st.dmsd.power_on(); // reset before overwriting the field
        st.cdmc.power_on();
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
        dlog_dev!(LogModule::Vino, "VINO: channel {} DMA enabled", if ch == 0 { 'A' } else { 'B' });
        // DMA thread is notified by control_w() after returning from here.
    }

    fn stop_channel(_st: &mut VinoState, ch: usize) {
        dlog_dev!(LogModule::Vino, "VINO: channel {} DMA disabled", if ch == 0 { 'A' } else { 'B' });
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
            // The 6.5 jump-bug descriptor chain (vinoBuildJumpBugDAPS) ends most
            // 4-descriptor groups with a JUMP whose encoded word is
            // `JUMP_BIT | kvtophys(next)` and whose target carries a +4 (sometimes
            // +8) low-bit offset — a workaround for the hardware's 4-at-a-time
            // descriptor-cache prefetch. PTR_MASK both strips the JUMP control bit
            // (bit 30) to recover the real lomem address and 16-byte-aligns it: the
            // real fetch is always 16-byte-group-aligned, so the +4/+8 offset must
            // be masked off; following it unaligned reads each next group 4 bytes
            // high, dropping the first data page of every group (~181 of 300 pages
            // reached) and scrambling the captured frame. Masking keeps the walk in
            // step so all 300 data pages land in order.
            let target = (chan.descriptors[0] as u32) & desc::PTR_MASK;
            Self::descriptor_fetch(chan, target, mem);
        }
    }

    // ── DMA: emit one dword to memory at the current descriptor offset ────

    /// Write one assembled dword to system memory at the current channel's
    /// DMA position, then advance `page_index` (handling 4 K rollover and
    /// interleave-mode line skips).  Returns false if DMA stopped — either
    /// the channel was disabled mid-flight, or the head descriptor had the
    /// STOP bit set (in which case the DESC interrupt is raised here).
    fn dma_emit_dword(&self, ch: usize, dword: u64, mem: &Arc<dyn BusDevice>) -> bool {
        let mut st = self.state.lock();

        let dma_en = [ctrl::CHA_DMA_EN, ctrl::CHB_DMA_EN][ch];
        if st.control & dma_en == 0 {
            return false;
        }

        let interleave = st.control & [ctrl::CHA_INTERLEAVE_EN, ctrl::CHB_INTERLEAVE_EN][ch] != 0;

        if st.channels[ch].descriptors[0] & desc::VALID_BIT != 0
            && st.channels[ch].descriptors[0] & desc::STOP_BIT  != 0
        {
            // Interlaced capture (IRIX 6.5): the kernel lays out ONE dense
            // descriptor chain spanning the whole frame buffer, and BOTH fields
            // traverse it to the same terminating STOP. Real VINO keeps DMA
            // running across both fields and raises end-of-descriptor (DESC) only
            // when the chain completes after the SECOND field. We model a
            // DMA-enable cycle as one interlaced frame: `field_counter` is 0 for
            // the first field of the cycle (reset in start_channel) and >=1
            // after. On the FIRST field, reaching STOP must NOT raise DESC or
            // disable DMA — otherwise the kernel restarts capture every field,
            // which re-sets its "first field of capture" flag (conn+0xb8) and
            // forces its field-parity counter even, so vinoEOD's completion check
            // never clears *(conn+0xc) and videod's vinoGetFrame is never woken.
            // Deferring completion to the second field lets the kernel's parity go
            // odd and the frame deliver. Both fields still render their own rows
            // into the shared buffer; only the DESC interrupt is deferred.
            // (Full derivation: rules/irix/vino-capture-on-6.5-progress.md cont.12.)
            //
            // 5.3 GATE: IRIX 5.3 capture is EOF-driven and page-steps NEXT_4_DESC
            // per field, so it never reaches a STOP descriptor here — this branch
            // never executes for 5.3 and its delivery path is untouched.
            if interleave && st.channels[ch].field_counter == 0 {
                return false;
            }
            let isr_desc = [isr::CHA_DESC, isr::CHB_DESC][ch];
            let new_status = st.int_status | isr_desc;
            let irq = self.irq.lock().clone();
            Self::raise_interrupt(&mut st, &irq, new_status);
            st.control &= !dma_en;
            return false;
        }

        let chan = &mut st.channels[ch];
        let desc_base  = (chan.descriptors[0] as u32) & desc::PTR_MASK as u32;
        let write_addr = desc_base | (chan.page_index & 0x0FF8);
        drop(st);

        mem.write64(write_addr, dword);

        let mut st = self.state.lock();
        let interleave = st.control & [ctrl::CHA_INTERLEAVE_EN, ctrl::CHB_INTERLEAVE_EN][ch] != 0;
        let chan = &mut st.channels[ch];

        let old_page = chan.page_index;
        chan.page_index = (chan.page_index + 8) & 0x0FFF;

        if interleave {
            chan.line_counter += 8;
            // CH_LINE_SIZE is encoded as "last dword's start offset within
            // the line" — i.e. one dword (8 bytes) short of the actual
            // stride. So an N-dword line has line_size = (N-1)*8, the
            // last dword writes when line_counter == line_size, and the
            // *next* dword (line_counter == line_size + 8) is the first
            // dword of the next interleaved row. Trigger on strict ">"
            // so we capture the last dword in this row before skipping —
            // not on ">=", which dropped the last dword and cascaded a
            // 2-pixel-per-row diagonal across the captured frame.
            if chan.line_counter > chan.line_size {
                chan.line_counter = 0;
                // Skip is the full row stride: (line_size + 8).
                let skip = chan.line_size.wrapping_add(8);
                let new_page = chan.page_index.wrapping_add(skip);
                chan.page_index = new_page & 0x0FFF;
                if chan.page_index < old_page || new_page >= 0x1000 {
                    Self::shift_descriptors(chan, mem);
                }
                return true;
            }
        }

        if chan.page_index < old_page {
            Self::shift_descriptors(chan, mem);
        }

        true
    }

    // ── Field pump: pull a field, clip/decimate/convert, DMA to memory ────

    /// Decode the packed pixel format from CONTROL bits for channel `ch`.
    fn channel_format(control: u32, ch: usize) -> PixelFormat {
        let (luma_only, color_rgb, dither) = if ch == 0 {
            (ctrl::CHA_LUMA_ONLY, ctrl::CHA_COLOR_SPACE_RGB, ctrl::CHA_DITHER_EN)
        } else {
            (ctrl::CHB_LUMA_ONLY, ctrl::CHB_COLOR_SPACE_RGB, ctrl::CHB_DITHER_EN)
        };
        if control & luma_only != 0 {
            PixelFormat::Y8
        } else if control & color_rgb != 0 {
            if control & dither != 0 { PixelFormat::Rgba8 } else { PixelFormat::Rgba32 }
        } else {
            PixelFormat::Yuv422
        }
    }

    /// Process one captured field for channel `ch`.  Applies the frame-rate
    /// mask (dropping unselected fields), then clips/decimates and converts
    /// to the channel's output format before pushing dwords through DMA.
    /// Raises the channel's end-of-field interrupt regardless of drop state.
    fn pump_field(&self, ch: usize, field: &Field, mem: &Arc<dyn BusDevice>) {
        let (clip_start, clip_end, format, dec_h_only, decimation, frame_rate, field_counter, interleave, line_size, start_desc_ptr)
            = {
            let st   = self.state.lock();
            let chan = &st.channels[ch];
            let dec_h_only_bit = if ch == 0 { ctrl::CHA_DECIMATE_HORIZ } else { ctrl::CHB_DECIMATE_HORIZ };
            let interleave_bit = if ch == 0 { ctrl::CHA_INTERLEAVE_EN } else { ctrl::CHB_INTERLEAVE_EN };
            (chan.clip_start, chan.clip_end,
             Self::channel_format(st.control, ch),
             st.control & dec_h_only_bit != 0,
             chan.decimation,
             chan.frame_rate,
             chan.field_counter,
             st.control & interleave_bit != 0,
             chan.line_size,
             chan.start_desc_ptr)
        };

        // Interlace placement: at the start of every field, position the
        // DMA cursor at the appropriate row offset within the destination
        // buffer. Even field writes rows 0, 2, 4, …; Odd field writes
        // rows 1, 3, 5, …. The per-line "skip one row" stride is handled
        // inside emit_byte's interleave branch; what we have to do here is
        // pick the *starting* row each time the field changes.
        //
        // Without this, Even field's writes advance page_index across the
        // whole buffer and Odd field continues from wherever Even left off
        // (out past the end), so the captured frame ends up with the Odd
        // field's rows untouched (all zeros). Visible as alternating
        // bright/black scanlines in the user-visible captured image.
        // Interlace placement: at the start of every field, rewind the
        // DMA cursor to the start of the descriptor chain and pick the
        // appropriate row offset within the frame buffer. Even field
        // starts at byte 0; Odd field starts one full row in
        // (= line_size + 8, since CH_LINE_SIZE is encoded as
        // actual_stride - 8 — see the interleave-skip code in
        // dma_write_dword). Without this reset the Odd field never
        // makes it into the buffer's odd-indexed rows; they stay zero
        // and the captured frame shows alternating bright/black
        // scanlines.
        //
        // KNOWN GAP: a 1-pixel-per-row diagonal artifact remains in
        // Even-field rows after this fix. The drift is consistent
        // (-1 col per output row from col 638 at row 0 toward col 0
        // at row 478) which suggests the kernel's frame buffer stride
        // or descriptor layout differs from what we infer from
        // line_size alone. Leaving as a follow-up — geometry and
        // colour are right; the artifact is a thin dark diagonal line
        // not a structural failure.
        if interleave && start_desc_ptr != 0 {
            let mut st = self.state.lock();
            let chan = &mut st.channels[ch];
            Self::descriptor_fetch(chan, start_desc_ptr, mem);
            chan.next_desc_ptr = start_desc_ptr.wrapping_add(16);
            chan.line_counter  = 0;
            chan.page_index    = match field.parity {
                FieldParity::Even => 0,
                FieldParity::Odd  => line_size.wrapping_add(8),
            };
        }

        let pal     = frame_rate & frame_rate::PAL != 0;
        let modulus = if pal { 10 } else { 12 };
        let mask    = (frame_rate >> frame_rate::MASK_SHIFT) & frame_rate::MASK_BITS;
        let drop    = mask != 0 && (mask >> (field_counter % modulus)) & 1 == 0;

        if !drop {
            let x_start = (clip_start >> clip::X_SHIFT) & clip::X_MASK;
            let x_end   = (clip_end   >> clip::X_SHIFT) & clip::X_MASK;
            let (y_shift, y_mask) = match field.parity {
                FieldParity::Even => (clip::YEVEN_SHIFT, clip::YEVEN_MASK),
                FieldParity::Odd  => (clip::YODD_SHIFT,  clip::YODD_MASK),
            };
            let y_start = (clip_start >> y_shift) & y_mask;
            let y_end   = (clip_end   >> y_shift) & y_mask;

            if x_end > x_start && y_end > y_start {
                let dec_x = decimation.max(1) as usize;
                let dec_y = if dec_h_only { 1 } else { dec_x };
                self.render_and_pump(ch, field, format,
                    dec_x, dec_y,
                    x_start, x_end, y_start, y_end, mem);
            }
        }

        let mut st = self.state.lock();
        st.channels[ch].field_counter = st.channels[ch].field_counter.wrapping_add(1);
        let isr_eof    = if ch == 0 { isr::CHA_EOF } else { isr::CHB_EOF };
        let new_status = st.int_status | isr_eof;
        let irq        = self.irq.lock().clone();
        Self::raise_interrupt(&mut st, &irq, new_status);
    }

    /// Walk the clipped rectangle in source coordinates with the configured
    /// decimation, sample UYVY from the field, convert to `format`, pack
    /// bytes MSB-first into 64-bit dwords, and stream them through DMA.
    /// In interleave mode each emitted output row is zero-padded out to
    /// `line_size + 8` (the kernel-allocated row stride in bytes), so
    /// `dma_emit_dword`'s row-skip trigger fires at the boundary the kernel
    /// expects — not at our shorter rendered line. Without this the source
    /// (e.g. 640 px NTSC) writing into a buffer the kernel sized for 768 px
    /// stride packs rows back-to-back and shears the captured image.
    fn render_and_pump(&self, ch: usize, field: &Field, format: PixelFormat,
                       dec_x: usize, dec_y: usize,
                       x_start: u32, x_end: u32, y_start: u32, y_end: u32,
                       mem: &Arc<dyn BusDevice>) {
        let src_w = field.width  as usize;
        let src_h = field.height as usize;
        let src   = &field.pixels;

        let x0 = (x_start as usize).min(src_w);
        let x1 = (x_end   as usize).min(src_w);
        let y0 = (y_start as usize).min(src_h);
        let y1 = (y_end   as usize).min(src_h);
        if x1 <= x0 || y1 <= y0 { return; }

        let (interleave, line_size) = {
            let st = self.state.lock();
            let bit = if ch == 0 { ctrl::CHA_INTERLEAVE_EN } else { ctrl::CHB_INTERLEAVE_EN };
            (st.control & bit != 0, st.channels[ch].line_size)
        };
        // In interleave mode pad each row out to the kernel-allocated stride;
        // outside interleave the descriptor chain handles linear layout itself
        // and any padding would just bloat the DMA stream.
        let target_line_bytes: u32 = if interleave { line_size + 8 } else { 0 };

        let mut accum: u64 = 0;
        let mut bytes_in: u32 = 0;
        let mut stopped = false;
        let mut line_bytes: u32;

        let mut y = y0;
        while y < y1 && !stopped {
            line_bytes = 0;
            let mut x = x0;
            while x < x1 && !stopped {
                let pair_x = x & !1;
                let i      = (y * src_w + pair_x) * 2;
                let u   = src[i    ];
                let y0p = src[i + 1];
                let v   = src[i + 2];
                let y1p = src[i + 3];
                let y_s = if x & 1 == 0 { y0p } else { y1p };

                match format {
                    PixelFormat::Y8 => {
                        emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, y_s);
                    }
                    PixelFormat::Yuv422 => {
                        // Emit a UYVY pair: U Y0 V Y1.  Pair with the next
                        // decimated pixel for Y1, then advance x past it.
                        let nx = x + dec_x;
                        let y_n = if nx < x1 {
                            let pair_nx = nx & !1;
                            let ni      = (y * src_w + pair_nx) * 2;
                            if nx & 1 == 0 { src[ni + 1] } else { src[ni + 3] }
                        } else {
                            y_s
                        };
                        emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, u);
                        emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, y_s);
                        emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, v);
                        emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, y_n);
                        x += dec_x;
                    }
                    PixelFormat::Rgba32 => {
                        let (r, g, b) = yuv_to_rgb(y_s, u, v);
                        // VINO 32-bit RGB lands in memory as A B G R (alpha high,
                        // then blue, green, red) — that's the order videod/the SGI
                        // imagelib reads back. Emitting A R G B instead swaps the
                        // red and blue channels (yellow↔cyan, red↔blue) in the
                        // captured frame. Bytes are packed MSB-first, so emit A,B,G,R.
                        emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, 0xFF);
                        emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, b);
                        emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, g);
                        emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, r);
                    }
                    PixelFormat::Rgba8 => {
                        let (r, g, b) = yuv_to_rgb(y_s, u, v);
                        // BGR 2:3:3 packed into one byte: BB GGG RRR.
                        let pix = ((b & 0xC0))            // B in bits [7:6]
                                | ((g & 0xE0) >> 2)       // G in bits [5:3]
                                | ((r & 0xE0) >> 5);      // R in bits [2:0]
                        emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, pix);
                    }
                }

                x += dec_x;
            }
            // Pad to row stride (interleave) or to dword boundary (otherwise).
            // The stride-pad path subsumes the dword pad whenever target is a
            // multiple of 8 (always true: line_size is masked to 0x0FF8).
            if target_line_bytes > 0 {
                while line_bytes < target_line_bytes && !stopped {
                    emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, 0);
                }
            } else {
                while bytes_in != 0 && !stopped {
                    emit_byte(self, ch, mem, &mut accum, &mut bytes_in, &mut stopped, &mut line_bytes, 0);
                }
            }
            y += dec_y;
        }
    }

    fn process_dma(&self) {
        loop {
            let active = {
                let st = self.state.lock();
                st.control & (ctrl::CHA_DMA_EN | ctrl::CHB_DMA_EN) != 0
            };
            if !active {
                self.wake.wait();
                if !self.running.load(Ordering::Relaxed) { return; }
                continue;
            }
            if !self.running.load(Ordering::Relaxed) { return; }

            let mem = match self.sys_mem.lock().clone() {
                Some(m) => m,
                None    => { thread::sleep(Duration::from_millis(10)); continue; }
            };

            let (control, a_en, b_en) = {
                let st = self.state.lock();
                (
                    st.control,
                    st.control & ctrl::CHA_DMA_EN != 0,
                    st.control & ctrl::CHB_DMA_EN != 0,
                )
            };

            let field_for = |ch: usize| -> Option<Field> {
                let select_d1 = if ch == 0 { ctrl::CHA_SELECT_D1 } else { ctrl::CHB_SELECT_D1 };
                let src = if control & select_d1 != 0 {
                    self.source_d1.lock().clone()?
                } else {
                    self.source_d0.lock().clone()?
                };
                Some(src.next_field())
            };

            if a_en {
                match field_for(0) {
                    Some(field) => self.pump_field(0, &field, &mem),
                    None => { thread::sleep(Duration::from_millis(10)); continue; }
                }
            }
            if b_en {
                match field_for(1) {
                    Some(field) => self.pump_field(1, &field, &mem),
                    None => { thread::sleep(Duration::from_millis(10)); continue; }
                }
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
            // Descriptor-table-pointer readback. The 6.5 kernel's buffer-completion
            // check (vino.o `0x77c0`) compares this against the buffer's recorded
            // field-boundary descriptor `*(bufentry+0x10)` = `base + FIELD_DESC_SPAN`
            // and the EOF/parity path (`0x7640` @ 0x7710) ABORTS the capture (so the
            // delivery fn `0x60b4` is skipped and no frame is ever handed to videod)
            // UNLESS, on the SECOND interlaced field, `0x77c0` returns 1 — which it
            // does only when this readback equals that boundary (or +0x10). On the
            // FIRST field it must read the base (so `0x77c0` returns 0 and the even
            // field doesn't abort either). `field_counter` is reset to 0 per
            // DMA-enable in start_channel and reaches 2 at the 2nd field's interrupt.
            // With this, vidtomem delivers a 640x480 frame on 6.5 (verified live,
            // cont. 15). FIELD_DESC_SPAN is the kernel's field-boundary offset for the
            // standard IndyCam capture (= rows-per-field 240 * 8); generalizing it for
            // other geometries is follow-up. 5.3 GATE: 5.3 capture is EOF-driven /
            // page-steps NEXT_4_DESC and uses neither this completion check nor a 2nd
            // interlaced field, so its readback stays at the base.
            reg::CH_DESC_TABLE_PTR => {
                const FIELD_DESC_SPAN: u32 = 0x780;
                if chan.field_counter >= 2 {
                    chan.start_desc_ptr.wrapping_add(FIELD_DESC_SPAN)
                } else {
                    chan.start_desc_ptr
                }
            }
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
                    // NOT_IDLE cleared → STOP condition.  Reset both devices
                    // since either (or neither) could have been the target.
                    st.dmsd.i2c_stop();
                    st.cdmc.i2c_stop();
                } else if val & i2c_ctrl::NOT_IDLE != 0
                    && (val & i2c_ctrl::READ != 0
                        || prev & i2c_ctrl::NOT_IDLE == 0)
                {
                    // Transfer request. Two cases:
                    //   (a) NOT_IDLE just transitioned clear→set (start of
                    //       transaction). Send the current I2C_DATA byte as
                    //       the slave-address byte.
                    //   (b) READ direction. The READ bit being set means the
                    //       caller wants to read the next byte from the slave
                    //       (driver writes NOT_IDLE|READ for every byte it
                    //       wants to receive). Always re-fire even if
                    //       NOT_IDLE was already set.
                    //
                    // Mid-write streaming is handled by the I2C_DATA write
                    // path below; we must not re-fire here when NOT_IDLE was
                    // already set in write direction, otherwise every kernel
                    // I2C_DATA write would be sent twice (once by I2C_DATA,
                    // once by the I2C_CONTROL poll/re-arm that follows).
                    if val & i2c_ctrl::READ != 0 {
                        let byte = if st.dmsd.is_active() {
                            st.dmsd.i2c_read()
                        } else if st.cdmc.is_active() {
                            st.cdmc.i2c_read()
                        } else {
                            0
                        };
                        st.i2c_data = byte as u32;
                    } else {
                        let data = st.i2c_data as u8;
                        let saa_active  = st.dmsd.is_active();
                        let cdmc_active = st.cdmc.is_active();
                        if saa_active  { st.dmsd.i2c_write(data); }
                        if cdmc_active { st.cdmc.i2c_write(data); }
                        if !saa_active && !cdmc_active {
                            st.dmsd.i2c_write(data);
                            st.cdmc.i2c_write(data);
                        }
                    }
                    // Transfer completes instantly (no real I2C bus timing)
                    st.i2c_ctrl &= !i2c_ctrl::XFER_BUSY;
                }
            }
            reg::I2C_DATA    => {
                st.i2c_data = val & i2c_data::MASK;
                // If the bus is currently active (NOT_IDLE set), each
                // I2C_DATA write triggers a byte transfer. IRIX 5.3 vino
                // driver writes I2C_CONTROL once at start (NOT_IDLE |
                // HOLD_BUS) and then streams bytes via I2C_DATA, polling
                // I2C_CONTROL.XFER_BUSY between writes. Without this, only
                // the very first byte (the one written immediately before
                // the I2C_CONTROL.NOT_IDLE write) ever reaches the slave.
                if st.i2c_ctrl & i2c_ctrl::NOT_IDLE != 0
                    && st.i2c_ctrl & i2c_ctrl::READ == 0
                {
                    let data = st.i2c_data as u8;
                    let saa_active  = st.dmsd.is_active();
                    let cdmc_active = st.cdmc.is_active();
                    if saa_active  { st.dmsd.i2c_write(data); }
                    if cdmc_active { st.cdmc.i2c_write(data); }
                    if !saa_active && !cdmc_active {
                        st.dmsd.i2c_write(data);
                        st.cdmc.i2c_write(data);
                    }
                    st.i2c_ctrl &= !i2c_ctrl::XFER_BUSY;
                }
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
                chan.field_counter = 0;
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
    // Byte/halfword accesses extract a sub-field of the underlying 32-bit
    // register. The default trait impl returns BusRead8::err() / err which
    // triggers a CPU bus error — and the IRIX vino driver does at least
    // one 16-bit access (offset 0x16, half of INTR_STATUS) during the
    // vidtomem path, escalating to a hard kernel panic
    // (`PANIC: KERNEL FAULT … Bad addr: 0xa0080016`). Routing
    // smaller-width accesses through read_reg keeps the bus quiet without
    // imposing semantics the real chip doesn't have.
    fn read8(&self, addr: u32) -> BusRead8 {
        let aligned = addr & !3;
        let shift = (3 - (addr & 3)) * 8;
        let w = self.read_reg(aligned.wrapping_sub(VINO_BASE));
        BusRead8::ok(((w >> shift) & 0xFF) as u8)
    }

    fn read16(&self, addr: u32) -> BusRead16 {
        let aligned = addr & !3;
        let shift = (2 - (addr & 2)) * 8;
        let w = self.read_reg(aligned.wrapping_sub(VINO_BASE));
        BusRead16::ok(((w >> shift) & 0xFFFF) as u16)
    }

    fn write8(&self, _addr: u32, _val: u8) -> u32 {
        // Sub-word writes to MMIO registers are atypical; quietly accept
        // them so the kernel doesn't panic. No partial-register update is
        // performed — full-register semantics is what the driver expects.
        BUS_OK
    }

    fn write16(&self, _addr: u32, _val: u16) -> u32 {
        BUS_OK
    }

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

// ─── Pixel pipeline helpers (free functions to keep render_and_pump tidy) ────

#[inline]
fn emit_byte(vino: &Vino, ch: usize, mem: &Arc<dyn BusDevice>,
             accum: &mut u64, bytes_in: &mut u32, stopped: &mut bool,
             line_bytes: &mut u32, b: u8) {
    if *stopped { return; }
    *accum = (*accum << 8) | (b as u64);
    *bytes_in += 1;
    *line_bytes += 1;
    if *bytes_in == 8 {
        if !vino.dma_emit_dword(ch, *accum, mem) {
            *stopped = true;
        }
        *accum = 0;
        *bytes_in = 0;
    }
}

/// BT.601 limited-range YCbCr → full-range RGB.  Fixed-point, no SIMD.
#[inline]
fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let yf = y as i32 - 16;
    let uf = u as i32 - 128;
    let vf = v as i32 - 128;
    let r = (298 * yf + 409 * vf + 128) >> 8;
    let g = (298 * yf - 100 * uf - 208 * vf + 128) >> 8;
    let b = (298 * yf + 516 * uf + 128) >> 8;
    (r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8)
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

                let d0_status = self.source_d0.lock()
                    .as_ref()
                    .map(|s| s.status())
                    .unwrap_or_else(|| "none".to_string());
                let d1_status = self.source_d1.lock()
                    .as_ref()
                    .map(|s| s.status())
                    .unwrap_or_else(|| "none".to_string());
                writeln!(writer, "  source D0 (composite): {}", d0_status).map_err(|e| e.to_string())?;
                writeln!(writer, "  source D1 (IndyCam):   {}", d1_status).map_err(|e| e.to_string())?;
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

// ─── Pixel pipeline tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{BusDevice, BUS_OK};
    use crate::video_source::{Field, FieldParity};

    /// Mock BusDevice that captures every write64 call.  We only care about
    /// 64-bit writes — that's all `dma_emit_dword` issues.
    struct MockMem { writes: Mutex<Vec<(u32, u64)>> }
    impl MockMem {
        fn new() -> Arc<Self> { Arc::new(Self { writes: Mutex::new(Vec::new()) }) }
        /// Sequential byte image, sorted by address, big-endian per dword.
        fn bytes(&self) -> Vec<u8> {
            let mut w = self.writes.lock().clone();
            w.sort_by_key(|(addr, _)| *addr);
            let mut out = Vec::with_capacity(w.len() * 8);
            for (_, val) in w { out.extend_from_slice(&val.to_be_bytes()); }
            out
        }
        fn dword_count(&self) -> usize { self.writes.lock().len() }
    }
    impl BusDevice for MockMem {
        fn write64(&self, addr: u32, val: u64) -> u32 {
            self.writes.lock().push((addr, val));
            BUS_OK
        }
    }

    /// UYVY field where each pair stores a deterministic pattern:
    ///   pix[i + 0] = 0x80 | pair_id   (U)
    ///   pix[i + 1] = 0x10 | pair_id   (Y0 — even-x luma)
    ///   pix[i + 2] = 0x40 | pair_id   (V)
    ///   pix[i + 3] = 0x20 | pair_id   (Y1 — odd-x luma)
    /// `pair_id` is just (line × pairs_per_line + pair) & 0x0F.
    fn make_field(w: u32, h: u32) -> Field {
        let mut pix = vec![0u8; (w * h * 2) as usize];
        let ppl = w / 2;
        for y in 0..h {
            for pair in 0..ppl {
                let pid = ((y * ppl + pair) & 0x0F) as u8;
                let i   = ((y * w + pair * 2) * 2) as usize;
                pix[i    ] = 0x80 | pid;
                pix[i + 1] = 0x10 | pid;
                pix[i + 2] = 0x40 | pid;
                pix[i + 3] = 0x20 | pid;
            }
        }
        Field { parity: FieldParity::Even, width: w, height: h, pixels: Arc::from(pix) }
    }

    /// Stand up a Vino with DMA enabled on channel A, a valid head descriptor,
    /// clip covering the full input field (even parity), and the chosen format
    /// + decimation set in CONTROL.  CHA_FIELD_INT_EN is set so EOF actually
    /// surfaces in `int_status` instead of being masked off.
    fn setup_vino(format: PixelFormat, dec: u32, dec_h_only: bool,
                  w: u32, h: u32, desc_base: u32) -> (Vino, Arc<MockMem>) {
        let vino = Vino::new();
        let mem  = MockMem::new();
        vino.set_phys(mem.clone());

        let mut control = ctrl::CHA_DMA_EN | ctrl::CHA_FIELD_INT_EN;
        match format {
            PixelFormat::Yuv422 => {}
            PixelFormat::Rgba32 => control |= ctrl::CHA_COLOR_SPACE_RGB,
            PixelFormat::Rgba8  => control |= ctrl::CHA_COLOR_SPACE_RGB | ctrl::CHA_DITHER_EN,
            PixelFormat::Y8     => control |= ctrl::CHA_LUMA_ONLY,
        }
        if dec > 1 {
            control |= ctrl::CHA_DECIMATE_EN
                     | ((dec - 1) << ctrl::CHA_DECIMATION_SHIFT);
            if dec_h_only { control |= ctrl::CHA_DECIMATE_HORIZ; }
        }

        {
            let mut st = vino.state.lock();
            st.control = control;
            let chan = &mut st.channels[0];
            chan.decimation     = dec;
            chan.descriptors[0] = (desc_base as u64 & desc::DATA_MASK) | desc::VALID_BIT;
            chan.page_index     = 0;
            chan.line_size      = 0;
            chan.clip_start     = 0;
            chan.clip_end       = (w & clip::X_MASK)
                                | ((h & clip::YEVEN_MASK) << clip::YEVEN_SHIFT)
                                | ((h & clip::YODD_MASK)  << clip::YODD_SHIFT);
            chan.frame_rate     = 0;
        }
        (vino, mem)
    }

    #[test]
    fn yuv422_full_field_is_byte_for_byte_passthrough() {
        let (vino, mem) = setup_vino(PixelFormat::Yuv422, 1, false, 4, 2, 0x1000);
        let field       = make_field(4, 2);
        let mem_dyn: Arc<dyn BusDevice> = mem.clone();

        vino.pump_field(0, &field, &mem_dyn);

        let bytes = mem.bytes();
        assert_eq!(bytes.len(), (field.width * field.height * 2) as usize);
        assert_eq!(&bytes[..], &field.pixels[..],
            "UYVY passthrough must copy input exactly");

        let st = vino.state.lock();
        assert_ne!(st.int_status & isr::CHA_EOF, 0, "CHA_EOF should fire");
        assert_eq!(st.channels[0].field_counter, 1);
    }

    #[test]
    fn y8_extracts_luma_byte_per_pixel() {
        let (vino, mem) = setup_vino(PixelFormat::Y8, 1, false, 8, 1, 0x2000);
        let field       = make_field(8, 1);
        let mem_dyn: Arc<dyn BusDevice> = mem.clone();

        vino.pump_field(0, &field, &mem_dyn);

        let bytes = mem.bytes();
        assert_eq!(bytes.len(), 8, "8 pixels → 1 dword → 8 bytes");
        for x in 0..8u32 {
            let pair_x = (x & !1) as usize;
            let i      = pair_x * 2;
            let expected = if x & 1 == 0 { field.pixels[i + 1] }
                           else          { field.pixels[i + 3] };
            assert_eq!(bytes[x as usize], expected,
                "Y8 byte at x={} should be the per-pixel luma", x);
        }
    }

    #[test]
    fn rgba32_emits_abgr_two_pixels_per_dword() {
        let (vino, mem) = setup_vino(PixelFormat::Rgba32, 1, false, 2, 1, 0x3000);
        let field       = make_field(2, 1);
        let mem_dyn: Arc<dyn BusDevice> = mem.clone();

        vino.pump_field(0, &field, &mem_dyn);

        let bytes = mem.bytes();
        assert_eq!(bytes.len(), 8, "2 pixels × 4 bytes each = one dword");
        // Alpha is always 0xFF in slots [0] and [4].
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[4], 0xFF);
        // VINO writes A B G R (blue before red); emitting A R G B swaps red/blue
        // in the captured frame (verified live — see render_and_pump).
        let (r0, g0, b0) = yuv_to_rgb(field.pixels[1], field.pixels[0], field.pixels[2]);
        let (r1, g1, b1) = yuv_to_rgb(field.pixels[3], field.pixels[0], field.pixels[2]);
        assert_eq!(&bytes[..],
                   &[0xFF, b0, g0, r0, 0xFF, b1, g1, r1][..],
                   "ABGR packing for the two pixels");
    }

    #[test]
    fn frame_rate_mask_drops_field_but_still_raises_eof() {
        let (vino, mem) = setup_vino(PixelFormat::Yuv422, 1, false, 4, 1, 0x4000);
        // mask = 0x0FFE (bits 11..1 set, bit 0 clear) → drop field 0 (counter % 12 = 0)
        {
            let mut st = vino.state.lock();
            st.channels[0].frame_rate    = 0x0FFE << frame_rate::MASK_SHIFT;
            st.channels[0].field_counter = 0;
        }
        let field = make_field(4, 1);
        let mem_dyn: Arc<dyn BusDevice> = mem.clone();

        vino.pump_field(0, &field, &mem_dyn);

        assert_eq!(mem.dword_count(), 0, "masked field should produce no DMA writes");
        let st = vino.state.lock();
        assert_ne!(st.int_status & isr::CHA_EOF, 0, "EOF must still fire on dropped fields");
        assert_eq!(st.channels[0].field_counter, 1, "field_counter advances even when dropped");
    }

    /// Interlaced (6.5) capture: hitting a STOP descriptor on the FIRST field of
    /// a DMA-enable cycle (field_counter == 0) must NOT raise DESC or disable DMA
    /// — the completion is deferred to the second field so the kernel's field
    /// pairing delivers the frame. The second field (field_counter >= 1) does
    /// raise DESC and disable DMA. See dma_emit_dword + cont.12 in the rules note.
    #[test]
    fn interleave_defers_desc_to_second_field() {
        let vino = Vino::new();
        let mem  = MockMem::new();
        vino.set_phys(mem.clone());
        let mem_dyn: Arc<dyn BusDevice> = mem.clone();

        // DMA + interleave on channel A, with DESC interrupts enabled so a raised
        // DESC actually surfaces in int_status.
        {
            let mut st = vino.state.lock();
            st.control = ctrl::CHA_DMA_EN | ctrl::CHA_INTERLEAVE_EN | ctrl::CHA_DESC_INT_EN;
            let chan = &mut st.channels[0];
            // Head descriptor carries the STOP bit (chain terminator).
            chan.descriptors[0] = (desc::STOP_BIT | 1) | desc::VALID_BIT;
            chan.field_counter  = 0; // first field of the cycle
        }

        // First field: STOP reached, but DESC deferred and DMA left enabled.
        assert!(!vino.dma_emit_dword(0, 0, &mem_dyn), "STOP still stops the field pump");
        {
            let st = vino.state.lock();
            assert_eq!(st.int_status & isr::CHA_DESC, 0,
                "DESC must NOT fire on the first interleaved field");
            assert_ne!(st.control & ctrl::CHA_DMA_EN, 0,
                "DMA must stay enabled across the first field");
        }

        // Second field of the cycle: now the STOP completes the frame.
        vino.state.lock().channels[0].field_counter = 1;
        assert!(!vino.dma_emit_dword(0, 0, &mem_dyn), "STOP stops the field pump");
        {
            let st = vino.state.lock();
            assert_ne!(st.int_status & isr::CHA_DESC, 0,
                "DESC must fire on the second interleaved field");
            assert_eq!(st.control & ctrl::CHA_DMA_EN, 0,
                "DMA is disabled once the frame completes");
        }
    }

    /// Non-interleaved capture (e.g. the IRIX 5.3 path is EOF-driven and never
    /// actually reaches a STOP here, but guard the gate anyway): a STOP on the
    /// first field still raises DESC and disables DMA — the deferral is
    /// interleave-only.
    #[test]
    fn non_interleave_stop_completes_immediately() {
        let vino = Vino::new();
        let mem  = MockMem::new();
        vino.set_phys(mem.clone());
        let mem_dyn: Arc<dyn BusDevice> = mem.clone();
        {
            let mut st = vino.state.lock();
            st.control = ctrl::CHA_DMA_EN | ctrl::CHA_DESC_INT_EN; // no INTERLEAVE
            let chan = &mut st.channels[0];
            chan.descriptors[0] = (desc::STOP_BIT | 1) | desc::VALID_BIT;
            chan.field_counter  = 0;
        }
        assert!(!vino.dma_emit_dword(0, 0, &mem_dyn));
        let st = vino.state.lock();
        assert_ne!(st.int_status & isr::CHA_DESC, 0,
            "non-interleaved STOP raises DESC even on the first field");
        assert_eq!(st.control & ctrl::CHA_DMA_EN, 0, "non-interleaved STOP disables DMA");
    }

    /// CH_DESC_TABLE_PTR readback: the 6.5 kernel's buffer-completion check needs
    /// this to equal the field-boundary descriptor (base + 0x780) on the SECOND
    /// interlaced field (field_counter >= 2) so it doesn't abort and delivery
    /// proceeds; on the first field it stays at the base. (reg 0x70 is the
    /// A_DESC_TABLE_PTR slot; read_reg masks bit 2 to collapse the 64-bit pair.)
    #[test]
    fn desc_table_ptr_advances_on_second_interlaced_field() {
        let vino = Vino::new();
        {
            let mut st = vino.state.lock();
            st.channels[0].start_desc_ptr = 0x0861_e000;
            st.channels[0].field_counter  = 1; // first field's interrupt
        }
        assert_eq!(vino.read_reg(reg::CHA_BASE + reg::CH_DESC_TABLE_PTR), 0x0861_e000,
            "first interlaced field reads the descriptor-table base");
        vino.state.lock().channels[0].field_counter = 2; // second field's interrupt
        assert_eq!(vino.read_reg(reg::CHA_BASE + reg::CH_DESC_TABLE_PTR), 0x0861_e780,
            "second interlaced field reads base + field-boundary span (0x780)");
    }

    /// Descriptor-pointer registers carry the kernel's bit-30 control flag: the
    /// 6.5 driver programs them as `JUMP_BIT | kvtophys(table)` (e.g. 0x4861e000).
    /// The address field is only bits [29:4], so PTR_MASK must strip bits 31/30 to
    /// recover the real lomem address (0x0861e000). This is what lets VINO DMA hit
    /// real RAM directly — there is no 0x40000000 RAM alias on the hardware, so
    /// the strip has to happen here at the source, not in the physical bus map.
    #[test]
    fn desc_pointer_registers_strip_bit30_control_flag() {
        let vino = Vino::new();
        let encoded = desc::JUMP_BIT as u32 | 0x0861_e000; // 0x4861e000
        vino.write_reg(reg::CHA_BASE + reg::CH_DESC_TABLE_PTR, encoded);
        vino.write_reg(reg::CHA_BASE + reg::CH_NEXT_4_DESC, encoded);
        let st = vino.state.lock();
        assert_eq!(st.channels[0].start_desc_ptr, 0x0861_e000,
            "DESC_TABLE_PTR must mask off bit 30 → real lomem address");
        assert_eq!(st.channels[0].next_desc_ptr, 0x0861_e000,
            "NEXT_4_DESC must mask off bit 30 → real lomem address");
    }

    // ─── I2C bus tests: SAA7191 + CDMC coexist on a shared bus ───────────

    /// Push one byte through the VINO I2C bridge by way of the I2C_CONTROL /
    /// I2C_DATA register pair, mirroring what an IRIX driver writes.
    fn i2c_byte_write(vino: &Vino, byte: u8) {
        vino.write_reg(reg::I2C_DATA, byte as u32);
        vino.write_reg(reg::I2C_CONTROL, i2c_ctrl::NOT_IDLE);
    }
    fn i2c_byte_read(vino: &Vino) -> u8 {
        vino.write_reg(reg::I2C_CONTROL, i2c_ctrl::NOT_IDLE | i2c_ctrl::READ);
        let st = vino.state.lock();
        st.i2c_data as u8
    }
    fn i2c_stop(vino: &Vino) {
        vino.write_reg(reg::I2C_CONTROL, 0);
    }

    /// Helper: read one byte from a device register via I2C.
    ///
    /// Protocol: START → write-addr → subaddr → REPEATED START → read-addr
    /// → READ → STOP. This matches what the real IRIX vino driver puts on
    /// the bus, which the CDMC / SAA7191 state machines recognise.
    fn i2c_read_reg(vino: &Vino, read_addr: u8, subaddr: u8) -> u8 {
        let write_addr = read_addr & !1;
        i2c_byte_write(vino, write_addr);
        i2c_byte_write(vino, subaddr);
        i2c_byte_write(vino, read_addr); // repeated-start re-addresses
        let v = i2c_byte_read(vino);
        i2c_stop(vino);
        v
    }

    /// Helper: write one byte to a device register via I2C.
    /// Protocol: START → write-addr → subaddr → data → STOP.
    fn i2c_write_reg(vino: &Vino, write_addr: u8, subaddr: u8, data: u8) {
        i2c_byte_write(vino, write_addr);
        i2c_byte_write(vino, subaddr);
        i2c_byte_write(vino, data);
        i2c_stop(vino);
    }

    #[test]
    fn cdmc_version_register_reads_as_identification_value() {
        let vino = Vino::new();
        let v = i2c_read_reg(&vino, 0x57, crate::cdmc::reg::VERSION);
        assert_eq!(v, crate::cdmc::reg::VERSION_VAL,
            "CDMC subaddress 0x00 should return the identification byte");
    }

    #[test]
    fn cdmc_register_write_then_read_round_trips() {
        let vino = Vino::new();
        i2c_write_reg(&vino, 0x56, crate::cdmc::reg::GAIN, 0x42);
        let v = i2c_read_reg(&vino, 0x57, crate::cdmc::reg::GAIN);
        assert_eq!(v, 0x42, "CDMC GAIN register write should round-trip");
    }

    #[test]
    fn saa7191_and_cdmc_dont_corrupt_each_other() {
        let vino = Vino::new();

        // Write CDMC GAIN = 0x77
        i2c_write_reg(&vino, 0x56, crate::cdmc::reg::GAIN, 0x77);
        {
            let st = vino.state.lock();
            assert!(!st.dmsd.is_active() && !st.cdmc.is_active(),
                "both devices back to idle after STOP");
        }

        // Address SAA7191 — must not touch CDMC state.
        i2c_write_reg(&vino, 0x8A, crate::saa7191::reg::HUEC, 0x55);

        // CDMC GAIN must still be 0x77.
        let v = i2c_read_reg(&vino, 0x57, crate::cdmc::reg::GAIN);
        assert_eq!(v, 0x77, "CDMC state must survive SAA7191 traffic");
    }

    #[test]
    fn horizontal_decimation_2x_halves_output_per_line() {
        // 8-pixel-wide input, horizontal decimation 2× → 4 pixels per output line.
        // YUV422 puts 4 pixels in one dword.  height = 1 → expect one dword total.
        let (vino, mem) = setup_vino(PixelFormat::Yuv422, 2, true, 8, 1, 0x5000);
        let field       = make_field(8, 1);
        let mem_dyn: Arc<dyn BusDevice> = mem.clone();

        vino.pump_field(0, &field, &mem_dyn);

        let bytes = mem.bytes();
        assert_eq!(bytes.len(), 8, "8 input pixels ÷ 2 = 4 output pixels = 1 dword");
        // Sampled pixels in source are x = 0, 2, 4, 6 (step by dec_x=2).
        // YUV422 emits U(x) Y(x) V(x) Y(x+2) for each iteration; iteration advances
        // x by 2*dec_x = 4.  So iterations at x=0 and x=4:
        //   it0: U=pix[0], Y=pix[1], V=pix[2], Y_next from x=2 → pair_x=2, i=4, even→pix[5]
        //   it1: U=pix[8], Y=pix[9], V=pix[10], Y_next from x=6 → pair_x=6, i=12, even→pix[13]
        let p = &field.pixels;
        assert_eq!(&bytes[..],
                   &[p[0], p[1], p[2], p[5], p[8], p[9], p[10], p[13]][..]);
    }
}
