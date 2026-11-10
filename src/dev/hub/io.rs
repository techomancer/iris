//! The Hub's I/O interface: the node's one XIO port. Its widget registers,
//! access and link control, the seven big-window translation entries and the
//! windows they steer, the two block transfer engines and the scratch
//! registers firmware uses as boot state.
//!
//! Offsets are from the base of the Hub window. The translation entries are
//! the decode: a big-window access takes the entry's widget and slice with one
//! atomic load, and an entry write changes where the next access lands, so no
//! mode flag is tested per access.
//!
//! # What is approximated here
//!
//! - The model completes every request before it returns, retries nothing and
//!   never fails its link, so IIO_WSTAT reads zero and the link log reads
//!   zero. The link reads up from reset because a link that reads down makes
//!   Linux abandon its XIO probe; the field's real reset value is not
//!   published.
//! - Access protection (IIO_ILAPR, the widget access pair, IIO_IIDEM), the
//!   interrupt destination and the crossbar credit are stored and enforced
//!   nowhere. What a refused request does is unknown.
//! - IIO_ILCSR bit 8 sends a warm-reset pulse to the far end of the link
//!   each time it is written set, and reads zero. That the bit sends the
//!   pattern is inferred from its name, and that it reads zero from the stock
//!   PROM's read-modify-write of the register. A link reset arriving from the
//!   far end changes nothing here and is logged.
//! - A big window aimed at an absent widget reads zero and drops writes, and
//!   logs the window; real behaviour is unknown, and guests disable windows by
//!   aiming them at a widget that is not there. A window whose entry selects
//!   I/O space is modelled only for entry 0 at slice 0, which the PROM uses to
//!   read a peer Hub's identity; it becomes an XIO I/O-space packet. Any other
//!   I/O-space entry reaches nothing.
//! - A peer's programmed I/O to the Hub's own widget number answers only a
//!   4-byte read of the widget identification word at offset 4, as the
//!   guest kernel issues it. Every other access fails.
//! - The block transfer engines run the whole transfer inside the store to
//!   IBCT that launches it, move 4096 lines (512 KiB) at a time, and work from
//!   reset with no enable written. Addresses keep bits 39:7. A zero-length
//!   launch completes at once; zero-fill does not read the source; poison mode
//!   copies, since the directory is not modelled; a read of IBCT returns the
//!   stored bits and stops nothing. The notification writes eight bytes.
//! - The Hub error interrupt is not modelled. A failed engine transfer latches
//!   busy and error, and only IIO_IECLR clears the error.
//!
//! Ported from hw/misc/sgi-hub.c in jamesbraid/qemu-sgi-origami.

use super::pi::MODIFY_SET;
use super::Effect;
use crate::devlog::LogModule;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Resettable, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock, Weak};

pub mod reg {
    pub const WIDGET_ID: u64 = 0x40_0000;
    pub const WIDGET_STATUS: u64 = 0x40_0008;
    pub const WIDGET_CONTROL: u64 = 0x40_0020;
    pub const LOCAL_ACCESS: u64 = 0x40_0100;
    pub const ACCESS_OVERRIDE: u64 = 0x40_0108;
    pub const OUTBOUND_ACCESS: u64 = 0x40_0110;
    pub const INBOUND_ACCESS: u64 = 0x40_0118;
    pub const DEVICE_ERR_MASK: u64 = 0x40_0120;
    pub const LINK_STATUS: u64 = 0x40_0128;
    pub const LLP_LOG: u64 = 0x40_0130;
    pub const INT_DEST: u64 = 0x40_0138;
    pub const GFX_INIT_0: u64 = 0x40_0140;
    pub const GFX_INIT_1: u64 = 0x40_0148;
    pub const SCRATCH_0: u64 = 0x40_0150;
    pub const SCRATCH_1: u64 = 0x40_0158;
    pub const BWIN_TABLE: u64 = 0x40_0160;
    pub const MISC_ENABLE: u64 = 0x40_01e8;
    pub const ERROR_CLEAR: u64 = 0x40_01f8;
    pub const PIO_DEALLOC: u64 = 0x40_0388;
    pub const CRB_DEALLOC: u64 = 0x40_0390;
    pub const IOQ_FIFO_DEPTH: u64 = 0x40_0398;
    pub const CRB_MGMT: u64 = 0x40_03a8;
    pub const CRB_CONTROL: u64 = 0x40_03b0;
    pub const XTALK_TIMEOUT: u64 = 0x40_03b8;
    pub const XTALK_PRESCALE: u64 = 0x40_03c0;
    pub const ICRB_C_BASE: u64 = 0x40_04f0;
    pub const BTE_BASE: u64 = 0x41_0000;
}
/// Where the block transfer engine `n`'s registers start.
pub const BTE_STRIDE: u64 = 0x1_0000;
pub mod bte {
    pub const LENGTH: u64 = 0x00;
    pub const SOURCE: u64 = 0x08;
    pub const DEST: u64 = 0x10;
    pub const CONTROL: u64 = 0x18;
    pub const NOTIFY: u64 = 0x20;
    pub const INTERRUPT: u64 = 0x28;
}

pub const START: u64 = 0x40_0000;
pub const END: u64 = 0x60_0000;
/// The Hub's own widget number on the XIO port.
pub const WIDGET: u8 = 1;
/// Interrupt vectors an interrupt-modify packet can name.
pub const VECTORS: u8 = 128;
pub const BIG_WINDOWS: usize = 7;
pub const BIG_WINDOW_SIZE: u64 = 512 << 20;

const ICRB_C_STRIDE: u64 = 0x20;
const ICRB_C_COUNT: u64 = 8;
const WIDGET_REVISION_SHIFT: u32 = 28;
const WIDGET_PART: u64 = 0xc101 << 12;
const WIDGET_CONTROL_RESET: u64 = 1 | (2 << 13);
const WIDGET_CONTROL_WRITABLE: u64 = 0x7f_ffff & !(1 << 4);
const WIDGET_ACCESS_MASK: u64 = 0xff01;
const INT_DEST_MASK: u64 = 0x1101_ff7f;
const GFX_INIT_MASK: u64 = 0x1f_ffff;
const SCRATCH_MASK: u64 = 0x0_f00f_11fff;
const TIMEOUT_MASK: u64 = 0xff;
const PRESCALE_MASK: u64 = 0xff_ffff;
const LINK_UP: u64 = 2 << 12;
const LINK_RESET: u64 = (0x3fc << 32) | (0x3ff << 16) | LINK_UP | 0xa;
const LINK_WARM_RESET: u64 = 1 << 8;
const LINK_WRITABLE: u64 = (0x3fc << 32) | (0x3ff << 16) | (1 << 10) | 0x3f;
const BWIN_MASK: u32 = 0x1fff;
const BWIN_IO_SPACE: u32 = 1 << 12;
const BWIN_WIDGET_SHIFT: u32 = 8;
const MISC_PRECISE_CLEAR: u64 = 1 << 12;
const MISC_SHUTDOWN: u64 = 0x111;
const IECLR_BTE: [u64; 2] = [1 << 17, 1 << 18];

const BTE_BUSY: u64 = 1 << 20;
const BTE_ERROR: u64 = 1 << 16;
const BTE_LINES: u64 = 0xffff;
const BTE_ADDRESS_MASK: u64 = 0xff_ffff_ff80;
const BTE_LINE: u64 = 128;
const BTE_POISON: u64 = 1 << 8;
const BTE_NOTIFY: u64 = 1 << 4;
const BTE_ZFILL: u64 = 1;
const BTE_CONTROL_MASK: u64 = BTE_POISON | BTE_NOTIFY | BTE_ZFILL;
const BTE_INT_LEVEL_SHIFT: u32 = 16;
const BTE_INT_LEVEL_MASK: u64 = 0x7f << BTE_INT_LEVEL_SHIFT;
const BTE_INT_NODE_MASK: u64 = 0x1ff;
const BTE_CHUNK_LINES: u64 = 4096;

/// What the I/O interface's engines and its XIO face reach by node number:
/// memory as a bus master sees it, and the interrupt-modify register of the
/// Hub programmed with a number. The Hub implements it over its claim map.
pub trait NodeFabric: Send + Sync {
    /// Read `buf.len()` bytes at the 40-bit `address`, its bits 39:32 naming
    /// the node. False when any byte could not be read.
    fn read(&self, address: u64, buf: &mut [u8]) -> bool;
    fn write(&self, address: u64, data: &[u8]) -> bool;
    /// Write `value` to the interrupt-modify register of the Hub programmed
    /// with `number`. False when no reachable Hub holds the number.
    fn interrupt(&self, number: u16, value: u64) -> bool;
}

fn unimp(what: &str, off: u64, value: Option<u64>) {
    match value {
        Some(v) => crate::dlog!(LogModule::Hub, "Hub IO: write of {v:#x} to {what} {off:#x}"),
        None => crate::dlog!(LogModule::Hub, "Hub IO: read of {what} {off:#x}"),
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Bte {
    length: u64,
    source: u64,
    dest: u64,
    control: u64,
    notify: u64,
    interrupt: u64,
}

struct State {
    widget_control: u64,
    link_status: u64,
    xtalk_timeout: u64,
    xtalk_prescale: u64,
    gfx_init: [u64; 2],
    local_access: u64,
    outbound_access: u64,
    inbound_access: u64,
    device_error_mask: u64,
    int_dest: u64,
    scratch: [u64; 2],
    bte: [Bte; 2],
}
impl State {
    fn new() -> Self {
        Self {
            widget_control: WIDGET_CONTROL_RESET,
            link_status: LINK_RESET,
            xtalk_timeout: 0,
            xtalk_prescale: 0,
            gfx_init: [0; 2],
            local_access: !0,
            outbound_access: 0,
            inbound_access: 0,
            device_error_mask: !0,
            int_dest: 0,
            scratch: [0; 2],
            bte: [Bte::default(); 2],
        }
    }
}

/// The lock-free half: what an access through a big window reads. The
/// translation entries are the decode table, written by a register store and
/// read by an access with one atomic load.
struct Windows {
    bwin: [AtomicU32; BIG_WINDOWS],
    widgets: [OnceLock<Arc<dyn BusDevice>>; 16],
    /// The peer's XIO packet face, for a window aimed at I/O space.
    packets: OnceLock<Arc<dyn BusDevice>>,
}

pub struct Io {
    widget_revision: u8,
    manufacturer: u16,
    state: Mutex<State>,
    windows: Arc<Windows>,
}

impl Io {
    /// `widget_revision` is the chip revision in bits 31:28 of the widget
    /// identification word and `manufacturer` its 11-bit manufacturer code.
    pub fn new(widget_revision: u8, manufacturer: u16) -> Self {
        assert!(widget_revision < 16 && manufacturer < 0x800);
        Self {
            widget_revision,
            manufacturer,
            state: Mutex::new(State::new()),
            windows: Arc::new(Windows {
                bwin: Default::default(),
                widgets: Default::default(),
                packets: OnceLock::new(),
            }),
        }
    }

    /// Put `dev` at crossbar widget `widget` (0 to 15) of the XIO space the
    /// big windows and small windows reach.
    pub fn connect_widget(&self, widget: u8, dev: Arc<dyn BusDevice>) {
        assert!(widget < 16);
        self.windows.widgets[usize::from(widget)].set(dev).unwrap_or_else(|_| panic!("widget {widget} is taken"));
    }

    /// The peer's packet face: an XIO address with bit 47 set is its I/O view
    /// and bits 51:48 name the destination widget.
    pub fn connect_packets(&self, dev: Arc<dyn BusDevice>) {
        self.windows.packets.set(dev).unwrap_or_else(|_| panic!("the XIO packet face is taken"));
    }


    /// The widget identification word.
    pub fn widget_id(&self) -> u64 {
        (u64::from(self.widget_revision) << WIDGET_REVISION_SHIFT) | WIDGET_PART | (u64::from(self.manufacturer) << 1) | 1
    }

    /// The big windows as one device of seven 512 MiB windows, addressed from
    /// the first. The machine places it at I/O-space offset 0x2000_0000.
    pub fn big_windows(&self) -> Arc<dyn BusDevice> {
        Arc::new(BigWindows(self.windows.clone()))
    }

    /// The crossbar widgets' small windows, 16 MiB each, addressed from
    /// widget 0's. Widget 1 is the Hub's own and is left to the caller.
    pub fn small_windows(&self) -> Arc<dyn BusDevice> {
        Arc::new(SmallWindows(self.windows.clone()))
    }

    /// The peer's request for this Hub's widget identification word. The one
    /// peer access with evidence is the guest kernel's 4-byte read at offset 4.
    pub fn widget_face_read32(&self, offset: u64) -> Option<u32> {
        (offset == 4).then(|| self.widget_id() as u32)
    }

    /// Scratch register `n`; the stock PROM keeps boot state here.
    pub fn scratch(&self, n: usize) -> u64 {
        self.state.lock().scratch[n]
    }

    /// The graphics node-widget map entry `n` (IIO_IGFX).
    pub fn gfx_init(&self, n: usize) -> u64 {
        self.state.lock().gfx_init[n]
    }

    fn bwin_entry(off: u64) -> Option<usize> {
        let rel = off.checked_sub(reg::BWIN_TABLE)?;
        (rel % 8 == 0 && rel / 8 < BIG_WINDOWS as u64).then_some((rel / 8) as usize)
    }
    fn icrb_c(off: u64) -> bool {
        off.checked_sub(reg::ICRB_C_BASE).is_some_and(|rel| rel % ICRB_C_STRIDE == 0 && rel / ICRB_C_STRIDE < ICRB_C_COUNT)
    }
    fn bte_entry(off: u64) -> Option<(usize, u64)> {
        let rel = off.checked_sub(reg::BTE_BASE)?;
        let (engine, reg) = ((rel / BTE_STRIDE) as usize, rel % BTE_STRIDE);
        (engine < 2 && reg <= bte::INTERRUPT && reg % 8 == 0).then_some((engine, reg))
    }

    /// A register's value without side effects, for the registers that store.
    pub fn peek(&self, off: u64) -> Option<u64> {
        if let Some(n) = Self::bwin_entry(off) {
            return Some(u64::from(self.windows.bwin[n].load(Ordering::Acquire)));
        }
        if let Some((engine, r)) = Self::bte_entry(off) {
            let b = self.state.lock().bte[engine];
            return Some(match r {
                bte::LENGTH => b.length,
                bte::SOURCE => b.source,
                bte::DEST => b.dest,
                bte::CONTROL => b.control,
                bte::NOTIFY => b.notify,
                _ => b.interrupt,
            });
        }
        let s = self.state.lock();
        Some(match off {
            reg::WIDGET_CONTROL => s.widget_control,
            reg::LOCAL_ACCESS => s.local_access,
            reg::OUTBOUND_ACCESS => s.outbound_access,
            reg::INBOUND_ACCESS => s.inbound_access,
            reg::DEVICE_ERR_MASK => s.device_error_mask,
            reg::LINK_STATUS => s.link_status,
            reg::INT_DEST => s.int_dest,
            reg::GFX_INIT_0 | reg::GFX_INIT_1 => s.gfx_init[((off - reg::GFX_INIT_0) / 8) as usize],
            reg::SCRATCH_0 | reg::SCRATCH_1 => s.scratch[((off - reg::SCRATCH_0) / 8) as usize],
            reg::XTALK_TIMEOUT => s.xtalk_timeout,
            reg::XTALK_PRESCALE => s.xtalk_prescale,
            _ => return None,
        })
    }

    pub fn read(&self, off: u64) -> u64 {
        if Self::icrb_c(off) {
            // No slot has ever tracked a transaction.
            return 0;
        }
        match off {
            reg::WIDGET_ID => self.widget_id(),
            // The model completes every request before it returns.
            reg::WIDGET_STATUS | reg::LLP_LOG | reg::ACCESS_OVERRIDE | reg::MISC_ENABLE => 0,
            reg::PIO_DEALLOC | reg::CRB_DEALLOC | reg::IOQ_FIFO_DEPTH | reg::CRB_MGMT | reg::CRB_CONTROL => 0,
            _ => match self.peek(off) {
                Some(v) => v,
                None => {
                    unimp("unimplemented or write-only register", off, None);
                    0
                }
            },
        }
    }

    /// A store from the bus. A control write that launches a block-transfer
    /// engine is returned for the Hub to run with [`Self::run_bte`], after this
    /// unit's lock is released.
    pub fn write(&self, off: u64, value: u64) -> Effect {
        if let Some(n) = Self::bwin_entry(off) {
            // Stored: OpenBSD's window allocator reads an entry straight back.
            let entry = value as u32 & BWIN_MASK;
            self.windows.bwin[n].store(entry, Ordering::Release);
            if entry & BWIN_IO_SPACE != 0 && (n != 0 || entry & 0x1f != 0) {
                crate::dlog!(LogModule::Hub, "Hub IO: unsupported I/O big window {n}, entry {entry:#x}");
            }
            return Effect::None;
        }
        if let Some((engine, r)) = Self::bte_entry(off) {
            return self.bte_write(engine, r, value);
        }
        if Self::icrb_c(off) {
            unimp("read-only incoming-request-buffer register", off, Some(value));
            return Effect::None;
        }
        let mut s = self.state.lock();
        match off {
            reg::WIDGET_ID => {
                crate::dlog!(LogModule::Hub, "Hub IO: write of {value:#x} to the read-only widget identification");
            }
            reg::WIDGET_STATUS => unimp("widget status (firmware clears it at power-on)", off, Some(value)),
            reg::WIDGET_CONTROL => {
                s.widget_control = (s.widget_control & !WIDGET_CONTROL_WRITABLE) | (value & WIDGET_CONTROL_WRITABLE)
            }
            reg::LOCAL_ACCESS => s.local_access = value,
            // The "SGIrules" password unlocks protection nothing enforces.
            reg::ACCESS_OVERRIDE => {}
            reg::OUTBOUND_ACCESS => s.outbound_access = value & WIDGET_ACCESS_MASK,
            reg::INBOUND_ACCESS => s.inbound_access = value & WIDGET_ACCESS_MASK,
            reg::DEVICE_ERR_MASK => s.device_error_mask = value,
            reg::LINK_STATUS => {
                // The link state and mode stay read-only: nothing may claim
                // the link down. Bit 8 is a one-shot warm reset that reads zero.
                s.link_status = (s.link_status & !LINK_WRITABLE) | (value & LINK_WRITABLE);
                drop(s);
                if value & LINK_WARM_RESET != 0 {
                    crate::dlog!(LogModule::Hub, "Hub IO: warm reset with nothing at the far end of the XIO link");
                }
            }
            reg::LLP_LOG => {} // closing a sampling interval clears zero counts
            reg::INT_DEST => s.int_dest = value & INT_DEST_MASK,
            reg::GFX_INIT_0 | reg::GFX_INIT_1 => s.gfx_init[((off - reg::GFX_INIT_0) / 8) as usize] = value & GFX_INIT_MASK,
            reg::SCRATCH_0 | reg::SCRATCH_1 => {
                let index = ((off - reg::SCRATCH_0) / 8) as usize;
                s.scratch[index] = value & SCRATCH_MASK;
            }
            reg::MISC_ENABLE => {
                if value & MISC_PRECISE_CLEAR != 0 {
                    unimp("precise-error latch clear (no such latch is modelled)", off, Some(value));
                }
                if value & MISC_SHUTDOWN != 0 {
                    unimp("shutdown-on-error bits (no unit here shuts down)", off, Some(value));
                }
            }
            reg::ERROR_CLEAR => {
                // Each bit clears a latch; only the engines' error bits exist.
                for (engine, bit) in IECLR_BTE.iter().enumerate() {
                    if value & bit != 0 {
                        s.bte[engine].length &= !BTE_ERROR;
                    }
                }
                if value & !(IECLR_BTE[0] | IECLR_BTE[1]) != 0 {
                    unimp("error clear (only the two engine latches exist)", off, Some(value));
                }
            }
            reg::PIO_DEALLOC | reg::CRB_DEALLOC | reg::CRB_MGMT | reg::CRB_CONTROL => {
                // Nothing outstanding: the free completes at once.
                unimp("I/O-queue register", off, Some(value));
            }
            reg::XTALK_TIMEOUT => s.xtalk_timeout = value & TIMEOUT_MASK,
            reg::XTALK_PRESCALE => s.xtalk_prescale = value & PRESCALE_MASK,
            _ => unimp("unimplemented register", off, Some(value)),
        }
        Effect::None
    }

    fn bte_address(engine: usize, what: &str, value: u64) -> u64 {
        if value & (BTE_LINE - 1) != 0 {
            crate::dlog!(LogModule::Hub, "Hub IO: BTE{engine} {what} address {value:#x} is not 128-byte aligned; bits 6:0 are not stored");
        }
        value & BTE_ADDRESS_MASK
    }

    fn bte_write(&self, engine: usize, r: u64, value: u64) -> Effect {
        let mut s = self.state.lock();
        let b = &mut s.bte[engine];
        match r {
            // A store arms or disarms but never absolves a latched error.
            bte::LENGTH => b.length = (b.length & BTE_ERROR) | (value & (BTE_BUSY | BTE_LINES)),
            bte::SOURCE => b.source = Self::bte_address(engine, "source", value),
            bte::DEST => b.dest = Self::bte_address(engine, "destination", value),
            bte::CONTROL => {
                b.control = value & BTE_CONTROL_MASK;
                // Launches if armed; busy clear does nothing.
                if b.length & BTE_BUSY != 0 {
                    return Effect::Bte(engine, *b);
                }
            }
            bte::NOTIFY => b.notify = Self::bte_address(engine, "notification", value),
            _ => b.interrupt = value & (BTE_INT_LEVEL_MASK | BTE_INT_NODE_MASK),
        }
        Effect::None
    }

    /// Run the transfer a control write launched through the node `fabric`,
    /// outside every unit lock.
    pub fn run_bte(&self, engine: usize, b: Bte, fabric: &dyn NodeFabric) {
        let lines = b.length & BTE_LINES;
        if b.control & BTE_POISON != 0 {
            crate::dlog!(LogModule::Hub, "Hub IO: BTE{engine} launched in poison mode; the copy runs but no directory state is poisoned");
        }
        if lines == 0 {
            crate::dlog!(LogModule::Hub, "Hub IO: BTE{engine} launched with a zero length");
        }
        // Chunked through one bounce buffer. A failed chunk is retried a line
        // at a time to latch the first unmoved line, and overlapping ranges go
        // a line at a time to keep the engine's forward propagation.
        let zfill = b.control & BTE_ZFILL != 0;
        let mut chunk = BTE_CHUNK_LINES;
        if !zfill && b.source < b.dest + lines * BTE_LINE && b.dest < b.source + lines * BTE_LINE {
            chunk = 1;
        }
        let mut buf = vec![0u8; (chunk.min(lines) * BTE_LINE) as usize];
        let mut done = 0;
        while done < lines {
            let run = (lines - done).min(chunk);
            let bytes = (run * BTE_LINE) as usize;
            let ok = (zfill || fabric.read(b.source + done * BTE_LINE, &mut buf[..bytes]))
                && fabric.write(b.dest + done * BTE_LINE, &buf[..bytes]);
            if !ok {
                if run > 1 {
                    chunk = 1;
                    continue;
                }
                // A failure latches error beside busy and keeps the unmoved
                // count; only IIO_IECLR absolves it, and neither completion runs.
                self.state.lock().bte[engine].length = BTE_BUSY | BTE_ERROR | (lines - done);
                crate::dlog!(LogModule::Hub, "Hub IO: BTE{engine} transfer failed at line {done} (src {:#x} dst {:#x})",
                    b.source + done * BTE_LINE, b.dest + done * BTE_LINE);
                return;
            }
            done += run;
        }
        let status = {
            let mut s = self.state.lock();
            // Busy and the count clear; a latched error survives.
            s.bte[engine].length &= BTE_ERROR;
            s.bte[engine].length
        };
        if b.control & BTE_NOTIFY != 0 {
            if !fabric.write(b.notify, &status.to_be_bytes()) {
                crate::dlog!(LogModule::Hub, "Hub IO: BTE{engine} notification address {:#x} reaches no memory", b.notify);
            }
        } else {
            // Completion arrives as a device interrupt through interrupt-modify
            // on the Hub holding the destination number.
            let level = (b.interrupt & BTE_INT_LEVEL_MASK) >> BTE_INT_LEVEL_SHIFT;
            let node = (b.interrupt & BTE_INT_NODE_MASK) as u16;
            if !fabric.interrupt(node, MODIFY_SET | level) {
                crate::dlog!(LogModule::Hub, "Hub IO: BTE{engine} completion interrupt aimed at node {node}, which no reachable Hub holds");
            }
        }
    }

    /// A link reset arriving from the crossbar. The specification says it
    /// clears the transmit buffer and the link protocol state and restarts
    /// width negotiation; the model has none of the three, so nothing changes.
    pub fn link_reset_in(&self) {
        crate::dlog!(LogModule::Hub, "Hub IO: inbound link reset; no transmit buffer or protocol state is represented");
    }
}

impl Io {
    /// Whether the register state is locked now, for the `locks` monitor command.
    pub fn is_locked(&self) -> bool {
        self.state.is_locked()
    }
}

impl Resettable for Io {
    fn power_on(&self) {
        *self.state.lock() = State::new();
        // Entries reset to zero with the windows enabled: a guest calls a zero
        // entry free, and zero spells widget 0, OpenBSD's only route there.
        for entry in &self.windows.bwin {
            entry.store(0, Ordering::Release);
        }
    }
}

impl BusDevice for Io {
    fn read64(&self, addr: u64) -> BusRead64 {
        BusRead64::ok(self.read(addr))
    }
    fn write64(&self, addr: u64, value: u64) -> u32 {
        let _ = self.write(addr, value);
        BUS_OK
    }
}

/// An access through one of the seven big windows, from the first's base.
struct BigWindows(Arc<Windows>);

impl BigWindows {
    /// The widget device and the address within it that `addr` reaches.
    fn target(&self, addr: u64) -> Option<(&dyn BusDevice, u64)> {
        let (window, offset) = ((addr / BIG_WINDOW_SIZE) as usize, addr % BIG_WINDOW_SIZE);
        let entry = self.0.bwin.get(window)?.load(Ordering::Acquire);
        let (widget, slice) = ((entry >> BWIN_WIDGET_SHIFT) & 15, u64::from(entry & 0x1f));
        if entry & BWIN_IO_SPACE != 0 {
            // The I/O view, modelled for entry 0 at slice 0 only: a packet
            // into the peer's I/O view, the widget in bits 51:48.
            if window != 0 || slice != 0 {
                return None;
            }
            let dev = self.0.packets.get()?;
            return Some((&**dev, (u64::from(widget) << 48) | (1 << 47) | offset));
        }
        let dev = self.0.widgets[widget as usize].get()?;
        Some((&**dev, (slice << 29) | offset))
    }
    fn missing(&self, addr: u64, value: Option<u64>) {
        crate::dlog!(LogModule::Hub, "Hub IO: {} through big window {} at {:#x} reaches no widget",
            if value.is_some() { "write" } else { "read" }, addr / BIG_WINDOW_SIZE, addr % BIG_WINDOW_SIZE);
    }
}

macro_rules! big_window_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            match self.target(a) {
                Some((dev, at)) => dev.$r(at),
                None if a / BIG_WINDOW_SIZE < BIG_WINDOWS as u64 => {
                    self.missing(a, None);
                    $out::ok(0)
                }
                None => $out::err(),
            }
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            match self.target(a) {
                Some((dev, at)) => dev.$w(at, v),
                None if a / BIG_WINDOW_SIZE < BIG_WINDOWS as u64 => {
                    self.missing(a, Some(v as u64));
                    BUS_OK
                }
                None => BUS_ERR,
            }
        }
    };
}
impl BusDevice for BigWindows {
    big_window_access!(read8, write8, u8, BusRead8);
    big_window_access!(read16, write16, u16, BusRead16);
    big_window_access!(read32, write32, u32, BusRead32);
    big_window_access!(read64, write64, u64, BusRead64);
}

/// A crossbar widget's 16 MiB small window, addressed as widget << 24 | offset.
struct SmallWindows(Arc<Windows>);

macro_rules! small_window_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            match self.0.widgets.get((a >> 24) as usize).and_then(|w| w.get()) {
                Some(dev) => dev.$r(a & 0xff_ffff),
                None => $out::err(),
            }
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            match self.0.widgets.get((a >> 24) as usize).and_then(|w| w.get()) {
                Some(dev) => dev.$w(a & 0xff_ffff, v),
                None => BUS_ERR,
            }
        }
    };
}
impl BusDevice for SmallWindows {
    small_window_access!(read8, write8, u8, BusRead8);
    small_window_access!(read16, write16, u16, BusRead16);
    small_window_access!(read32, write32, u32, BusRead32);
    small_window_access!(read64, write64, u64, BusRead64);
}

#[cfg(test)]
mod tests;
