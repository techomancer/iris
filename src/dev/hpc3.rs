use std::sync::{Arc, OnceLock};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::fs::File;
use std::io::Write as IoWrite;
use crate::devlog::{LogModule, devlog_mask};
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device, DmaClient, DmaStatus, Resettable, RxRoom, Saveable};
use crate::snapshot::{get_field, u32_slice_to_toml, load_u32_slice, toml_u32, toml_bool, hex_u32};
use crate::config::{AudioConfig, NetworkConfig, RtcOffset};
use crate::dev::eeprom_93c56::Eeprom93c56;
use crate::dev::ioc::Ioc;
use crate::dev::ds1x86::Ds1x86;
use crate::net::{GatewayConfig, Link};
use crate::dev::seeq8003::Seeq8003;
use crate::dev::wd33c93a::Wd33c93a;
use crate::irq::{InterruptController, IrqLine};
use crate::dev::ioc::IocInterrupt;
use crate::dev::hal2::Hal2;
use crate::hptimer::TimerManager;
use crate::exp::eval_const_expr;

pub const HPC3_BASE: u32 = 0x1FB80000;
pub const HPC3_SIZE: u32 = 0x00080000; // 512KB

// PBUS DMA Channels 0-7
pub const PBUS_DMA_STRIDE: u32 = 0x2000;
pub const PBUS_DMA_BP: u32 = 0x0000;
pub const PBUS_DMA_DP: u32 = 0x0004;
pub const PBUS_DMA_CTRL: u32 = 0x1000;

// SCSI Channels 0-1
pub const SCSI0_BASE: u32 = 0x10000;
pub const SCSI1_BASE: u32 = 0x12000;
pub const SCSI_CBP: u32 = 0x0000;
pub const SCSI_NBDP: u32 = 0x0004;
pub const SCSI_BC: u32 = 0x1000;
pub const SCSI_CTRL: u32 = 0x1004;
pub const SCSI_GIO: u32 = 0x1008;
pub const SCSI_DEV: u32 = 0x100C;
pub const SCSI_DMACFG: u32 = 0x1010;
pub const SCSI_PIOCFG: u32 = 0x1014;

// Ethernet
pub const ENET_RX_BASE: u32 = 0x14000;
pub const ENET_TX_BASE: u32 = 0x16000;
// ENET RX Offsets
pub const ENET_RX_CBP: u32 = 0x0000;
pub const ENET_RX_NBDP: u32 = 0x0004;
pub const ENET_RX_BC: u32 = 0x1000;
pub const ENET_RX_CTRL: u32 = 0x1004;
pub const ENET_RX_GIO: u32 = 0x1008;
pub const ENET_RX_DEV: u32 = 0x100C;
pub const ENET_RX_RESET: u32 = 0x1014;
pub const ENET_RX_DMACFG: u32 = 0x1018;
pub const ENET_RX_PIOCFG: u32 = 0x101C;
// ENET TX Offsets
pub const ENET_TX_CBP: u32 = 0x0000;
pub const ENET_TX_NBDP: u32 = 0x0004;
pub const ENET_TX_BC: u32 = 0x1000;
pub const ENET_TX_CTRL: u32 = 0x1004;
pub const ENET_TX_GIO: u32 = 0x1008;
pub const ENET_TX_DEV: u32 = 0x100C;

// Ethernet extra registers (beyond PDMA window, not in HPC3 spec but used by IRIX driver)
pub const ENET_CRBDP: u32 = 0x18000;   // current RX buffer descriptor pointer (HPC3-maintained)
pub const ENET_CPFXBDP: u32 = 0x1a000; // current/previous first TX buffer descriptor pointer
pub const ENET_PPFXBDP: u32 = 0x1a004; // previous/previous? first TX buffer descriptor pointer

// FIFO Areas
pub const SCSI0_FIFO_BASE: u32 = 0x28000;
pub const SCSI1_FIFO_BASE: u32 = 0x2A000;
pub const ENET_RX_FIFO_BASE: u32 = 0x2C000;
pub const ENET_TX_FIFO_BASE: u32 = 0x2E000;

// Misc
pub const MISC_BASE: u32 = 0x30000;
pub const MISC_INTSTAT: u32 = 0x0000;
pub const MISC_GIO_MISC: u32 = 0x0004;
pub const MISC_EEPROM_DATA: u32 = 0x0008;
pub const MISC_INTSTAT_BUG: u32 = 0x000C;
pub const MISC_GIO_BUS_ERROR: u32 = 0x0010;

// SCSI chip registers appear at two addresses due to HPC3 address line aliasing:
//   0x40000 (0x1fbc0000) — IRIX hpc3.h HPC3_SCSI_REG0 (wrong per spec, works on hw)
//   0x44000 (0x1fbc4000) — HPC3 spec / OpenBSD HPC3_SCSI0_DEVREGS (correct)
// Both map to the same WD33C93A chip.
pub const SCSI_REG_BASE: u32  = 0x40000;
pub const SCSI_REG_BASE2: u32 = 0x44000;
/// Fullhouse only: second WD33C93A register window (MAME's hpc3_device
/// maps controller 1 at 0x48000-0x4ffff, controller 0 at 0x40000-0x47fff).
/// Unlike SCSI_REG_BASE/SCSI_REG_BASE2, there is no second "spec alias" for
/// this controller — that dual-addressing was specific to controller 0's
/// historical IRIX-vs-spec address mismatch.
pub const SCSI1_REG_BASE: u32 = 0x48000;
pub const SEEQ_BASE: u32 = 0x54000;

// PBUS PIO
pub const PBUS_PIO_BASE: u32 = 0x58000;
pub const HAL2_BASE: u32 = 0x58000;

/// Returns the value a disabled (absent) HAL2 presents on reads.
/// REV (offset 0x20) returns 0xFFFF — not a valid chip version, so the IRIX
/// hal2 driver recognises "no chip" and skips init rather than spinning.
/// All other registers return 0: ISR.TSTATUS=0 (not busy), no spurious state.
fn hal2_absent_read(offset: u32) -> u16 {
    use crate::dev::hal2::HAL2_REV;
    if (offset & 0xF0) == HAL2_REV { 0xFFFF } else { 0x0000 }
}
pub const HPC3_IOC_BASE: u32 = 0x59800;
pub const PBUS_PIO_STRIDE: u32 = 0x400;
/// IP22 fullhouse only: INT2 interrupt registers at PBUS PIO channel 4.
/// Same register semantics as guinness's INT3 (HPC3_IOC_BASE / PIO channel
/// 6), just a different address — see `crate::dev::ioc::INT2_REG_COUNT`'s doc
/// comment. Registers are dword-indexed (`idx = offset >> 2`), matching the
/// packing every other PBUS PIO channel in this file already uses.
pub const HPC3_INT2_BASE: u32 = PBUS_PIO_BASE + 4 * PBUS_PIO_STRIDE;
pub const HPC3_INT2_SIZE: u32 = crate::dev::ioc::INT2_REG_COUNT * 4;

/// IP22 fullhouse only: Full House's "extended register" (PBUS PIO channel
/// 5). `kern/sys/hpc3.h`: "Address of Full House's extended register. PX at
/// 0x1fbd9400. (IP22 Only)" — sits right before `HPC3_EXT_IO_ADDR`
/// (`0x1fbd9900`, PIO channel 6 offset 0x100) in that header, but is itself
/// unimplemented here; not yet reverse-engineered, so accesses are only
/// logged (see `read8`/`write8`/`read16`/`write16`/`read32`/`write32`), not
/// backed by real register state.
pub const HPC3_EXT_PX_BASE: u32 = PBUS_PIO_BASE + 5 * PBUS_PIO_STRIDE;

// PBUS DMA Config
pub const PBUS_CFGDMA_BASE: u32 = 0x5C000;
pub const PBUS_CFGDMA_STRIDE: u32 = 0x200;

// PBUS PIO Config
pub const PBUS_CFGPIO_BASE: u32 = 0x5D000;
pub const PBUS_CFGPIO_STRIDE: u32 = 0x100;

// Other
pub const PBUS_PROM_WE: u32 = 0x5E000;
pub const PBUS_PROM_SWAP: u32 = 0x5E800;
pub const PBUS_GEN_OUT: u32 = 0x5F000;
pub const PBUS_BBRAM: u32 = 0x60000;

// PDMA Constants
pub const PDMA_DESC_CBP_OFFSET: u32 = 0x00;
pub const PDMA_DESC_BC_OFFSET: u32 = 0x04;
pub const PDMA_DESC_NBP_OFFSET: u32 = 0x08;
pub const PDMA_DESC_FILLER: u32 = 0x0C;
pub const PDMA_DESC_SIZE: u32 = 0x10;

pub const PDMA_DESC_EOX:  u32 = 0x80000000; // TX: end of chain / RX: end of ring
pub const PDMA_DESC_EOP:  u32 = 0x40000000; // TX: end of packet (EOXP) / RX: end of ring packet
pub const PDMA_DESC_XIE:  u32 = 0x20000000; // interrupt enable
pub const PDMA_DESC_ROWN: u32 = 0x00004000; // RX: owned by HPC3 (1=ready, 0=host owns)

// these are for writing
pub const PDMA_CTRL_LITTLE: u32 = 1u32 << 1;
pub const PDMA_CTRL_RECEIVE: u32 = 1u32 << 2;
pub const PDMA_CTRL_FLUSH: u32 = 1u32 << 3;
pub const PDMA_CTRL_CH_ACT: u32 = 1u32 << 4;
pub const PDMA_CTRL_CH_ACT_LD: u32 = 1u32 << 5;
// these are for reads
pub const PDMA_CTRL_INT: u32 = 1u32 << 0; // cleared after read
pub const PDMA_CTRL_ACT: u32 = 1u32 << 1; 

// HPC3 Interrupt Status Bits
pub const HPC3_INTSTAT_SCSI0_DEV: u32 = 1 << 0;
pub const HPC3_INTSTAT_SCSI0_DMA: u32 = 1 << 1;
pub const HPC3_INTSTAT_SCSI1_DEV: u32 = 1 << 2;
pub const HPC3_INTSTAT_SCSI1_DMA: u32 = 1 << 3;
pub const HPC3_INTSTAT_ENET_DEV: u32 = 1 << 4;
pub const HPC3_INTSTAT_ENET_RX_DMA: u32 = 1 << 5;
pub const HPC3_INTSTAT_ENET_TX_DMA: u32 = 1 << 6;

// Ethernet RX Control Register
pub const ENET_RX_CTRL_RBO: u32 = 0x800;     // HPC_RBO: receive buffer overflow
pub const ENET_RX_CTRL_AMASK: u32 = 0x400;   // HPC_STRCVDMA_MASK: active mask
pub const ENET_RX_CTRL_ACTIVE: u32 = 0x200;  // HPC_STRCVDMA: receive DMA started/active
pub const ENET_RX_CTRL_ENDIAN: u32 = 0x100;  // HPC_RCV_ENDIAN_LITTLE
pub const ENET_RX_CTRL_OLD_NEW: u32 = 0x80;  // SEQ_RS_OLD: old/new status
pub const ENET_RX_CTRL_LATE_RXDC: u32 = 0x40; // SEQ_RS_LATE_RXDC: late rx data collision (HPC-set, not from SEEQ)
pub const ENET_RX_CTRL_GOOD: u32 = 0x20;     // SEQ_RS_GOOD: good frame
pub const ENET_RX_CTRL_END: u32 = 0x10;      // SEQ_RS_END: end of frame
pub const ENET_RX_CTRL_SHORT: u32 = 0x08;    // SEQ_RS_SHORT: short frame
pub const ENET_RX_CTRL_DRBL: u32 = 0x04;     // SEQ_RS_DRBL: dribble error
pub const ENET_RX_CTRL_CRC: u32 = 0x02;      // SEQ_RS_CRC: CRC error
pub const ENET_RX_CTRL_OFLOW: u32 = 0x01;    // SEQ_RS_OFLOW: overflow error
// Bits [7,5:0] mirror SEEQ RX status; bit 6 (LATE_RXDC) is HPC-internal, preserved from chan.ctrl
pub const ENET_RX_CTRL_SEEQ_MASK: u32 = 0xBF;

// Ethernet RX Reset Register
pub const ENET_RX_RESET_CH_RESET: u32 = 0x01;
pub const ENET_RX_RESET_CLRINT: u32 = 0x02;
pub const ENET_RX_RESET_INTPEND: u32 = 0x02;
pub const ENET_RX_RESET_LOOPBACK: u32 = 0x04;

// Ethernet TX Control Register
pub const ENET_TX_CTRL_AMASK: u32 = 0x400;
pub const ENET_TX_CTRL_ACTIVE: u32 = 0x200;  // HPC_STTRDMA: xmit DMA started/active
pub const ENET_TX_CTRL_ENDIAN: u32 = 0x100;
pub const ENET_TX_CTRL_OLD:    u32 = 0x080;  // SEQ_XS_OLD: old/new status
pub const ENET_TX_CTRL_LC:     u32 = 0x010;  // SEQ_XS_LATE_COLL: late collision
pub const ENET_TX_CTRL_OK:     u32 = 0x008;  // SEQ_XS_SUCCESS: xmit success
pub const ENET_TX_CTRL_16TRY:  u32 = 0x004;  // SEQ_XS_16TRY: 16 retries (abort)
pub const ENET_TX_CTRL_COLL:   u32 = 0x002;  // SEQ_XS_COLL: collision
pub const ENET_TX_CTRL_UFLOW:  u32 = 0x001;  // SEQ_XS_UFLOW: underflow
// Bits [7:0] of tx_ctrl mirror all SEEQ TX status bits
pub const ENET_TX_CTRL_SEEQ_MASK: u32 = 0xFF;

// SCSI Control Register
pub const SCSI_CTRL_INT: u32 = 0x01;
pub const SCSI_CTRL_ENDIAN: u32 = 0x02;
pub const SCSI_CTRL_DIR: u32 = 0x04;
pub const SCSI_CTRL_FLUSH: u32 = 0x08;
pub const SCSI_CTRL_ACTIVE: u32 = 0x10;
pub const SCSI_CTRL_AMASK: u32 = 0x20;
pub const SCSI_CTRL_RESET: u32 = 0x40;
pub const SCSI_CTRL_PERR: u32 = 0x80;

// SCSI DMA Config
pub const SCSI_DMACFG_DMA16: u32 = 1 << 12;

// HPC3 PDMA Register Offsets
pub const HPC3_PDMA_CBP: u32 = 0x0000;
pub const HPC3_PDMA_NBDP: u32 = 0x0004;
pub const HPC3_PDMA_CTRL: u32 = 0x1000;

pub const HPC3_PDMA_CHAN_GENERIC: u32 = 7;
pub const HPC3_PDMA_CHAN_SCSI0: u32 = 8;
pub const HPC3_PDMA_CHAN_SCSI1: u32 = 9;
pub const HPC3_PDMA_CHAN_ENET_RX: u32 = 10;
pub const HPC3_PDMA_CHAN_ENET_TX: u32 = 11;

// PBUS DMA Config
pub const PBUS_DMACFG_DS16: u32      = 1 << 18; // Bit 18: ds_16 — 16-bit device
pub const PBUS_DMACFG_EVEN_HIGH: u32 = 1 << 19; // Bit 19: even_high — even bytes on high bus (15:8)

struct Hpc3State {
    intstat: u32,
    gio_misc: u32,
    eeprom_reg: u32,
    pbus_pio: [u32; 0x1000],
}

/// The HPC3's interrupt status (INTSTAT) as a controller: line `n` is
/// INTSTAT bit `n`. SCSI channel 0's chip and DMA bits are ORed onto the
/// IOC's SCSI0 line and channel 1's onto SCSI1; the SEEQ's goes to Ethernet.
struct Hpc3Interrupts {
    state: Arc<Mutex<Hpc3State>>,
    /// The IOC's SCSI0, SCSI1 and Ethernet lines.
    scsi: [IrqLine; 2],
    ethernet: IrqLine,
    /// The SCSI PDMA channels, so that a chip-INT ack can also drop the PDMA
    /// half of the shared SCSI line (see `set_line`).
    scsi_dma: [Arc<Mutex<PdmaChannel>>; 2],
}

impl Hpc3Interrupts {
    /// The line for one INTSTAT bit.
    fn line(self: &Arc<Self>, bit: u32) -> IrqLine {
        IrqLine::new(self.clone(), bit.trailing_zeros())
    }

    /// Set or clear `bit`, then drive SCSI channel `n`'s IOC line from both
    /// of its bits.
    fn update_scsi(&self, n: usize, bit: u32, level: bool) {
        let mut state = self.state.lock();
        if level { state.intstat |= bit; } else { state.intstat &= !bit; }
        self.scsi[n].set(state.intstat & SCSI_BITS[n] != 0);
    }
}

/// Each SCSI channel's chip and DMA bits, which share its IOC line.
const SCSI_BITS: [u32; 2] = [
    HPC3_INTSTAT_SCSI0_DEV | HPC3_INTSTAT_SCSI0_DMA,
    HPC3_INTSTAT_SCSI1_DEV | HPC3_INTSTAT_SCSI1_DMA,
];

impl InterruptController for Hpc3Interrupts {
    fn set_line(&self, line: u32, level: bool) {
        let bit = 1u32 << line;
        match bit {
            HPC3_INTSTAT_SCSI0_DMA => self.update_scsi(0, bit, level),
            HPC3_INTSTAT_SCSI1_DMA => self.update_scsi(1, bit, level),
            HPC3_INTSTAT_SCSI0_DEV | HPC3_INTSTAT_SCSI1_DEV => {
                let (n, dma_bit) = if bit == HPC3_INTSTAT_SCSI0_DEV {
                    (0, HPC3_INTSTAT_SCSI0_DMA)
                } else {
                    (1, HPC3_INTSTAT_SCSI1_DMA)
                };
                self.update_scsi(n, bit, level);
                if !level {
                    // When the chip-side INT deasserts, also drop any pending
                    // PDMA DMA-completion bit on the same SCSI line: on a real
                    // HPC3 the SCSI INT3 source is shared and the chip ack
                    // settles both halves. Without this, intstat[SCSI*_DMA]
                    // stays asserted forever when the IRIX miniroot's SCSI
                    // driver only acks via the chip path.
                    {
                        let mut c = self.scsi_dma[n].lock();
                        if c.ctrl & PDMA_CTRL_INT != 0 {
                            c.ctrl &= !PDMA_CTRL_INT;
                        }
                    }
                    let mut st = self.state.lock();
                    st.intstat &= !dma_bit;
                    let active = st.intstat & SCSI_BITS[n] != 0;
                    drop(st);
                    self.scsi[n].set(active);
                }
            }
            HPC3_INTSTAT_ENET_DEV => {
                {
                    let mut st = self.state.lock();
                    if level { st.intstat |= HPC3_INTSTAT_ENET_DEV; }
                    else     { st.intstat &= !HPC3_INTSTAT_ENET_DEV; }
                }
                dlog_dev!(LogModule::Hpc3, "ENET IRQ: level={}", level);
                // The SEEQ's own level, as it always was: not ORed with the
                // Ethernet DMA bits.
                self.ethernet.set(level);
            }
            _ => {}
        }
    }
}

struct PdmaChannel {
    id: usize,
    /// Device-side reads and writes since power-on (see `pdma status`).
    dev_reads: u64,
    dev_writes: u64,
    /// CPU reads of the CBP register: how often a driver polls the DMA
    /// position (a ring with no end-of-descriptor interrupts is refilled so).
    cbp_polls: u64,
    /// The last CBP reads: (microseconds since the first, value, bytes the
    /// device had read by then), for `pdma polls N`.
    cbp_log: std::collections::VecDeque<(u64, u32, u64)>,
    cbp_log_t0: Option<std::time::Instant>,
    /// Device reads when the guest last read CBP, and when.
    reads_at_poll: u64,
    last_poll: Option<std::time::Instant>,
    cbp: u32,
    nbdp: u32,
    bc: u32,
    ctrl: u32,
    gio: u32,
    dev: u32,
    dmacfg: u32,
    piocfg: u32,
    eox: bool,
    eop: bool,
    xie: bool,
    misc: u32,
    active_mask: u32,
    sys_mem: Option<Arc<dyn BusDevice>>,
    endian: bool,
    even_high: bool,    // PBUS only: even bytes on high bus lane (bit 19 of dmacfg)
    /// The channel's DMA-completion interrupt (SCSI channels only).
    irq: Option<IrqLine>,
    dump_enabled: Arc<AtomicU32>,
    dump_file: Option<File>,
    dump_transaction_id: u32,
    dump_is_write: bool,  // true = dma_read (host→device), false = dma_write (device→host)
    transaction_id: u32,
    bytes_transferred: usize,
    width_16: bool,
    // Enet-only extra registers (beyond the main PDMA window)
    // RX chan[10]: crbdp @ HPC3+0x18000
    // TX chan[11]: cpfxbdp @ HPC3+0x1a000, ppfxbdp @ HPC3+0x1a004
    crbdp:        u32,
    cpfxbdp:      u32,
    ppfxbdp:      u32,
    // TX: true when the next fetch starts a new packet (after EOXP or at chain start).
    // On fetch: if true, promote cpfxbdp→ppfxbdp and set cpfxbdp=nbdp, then clear.
    // Set back to true when an EOXP descriptor's transfer completes.
    tx_new_packet: bool,
    /// RX: descriptors the channel has written back, oldest first, until the
    /// driver re-arms them (see `rx_room`). Bounded by the largest ring.
    rx_filled: std::collections::VecDeque<u32>,
    // RX: last value returned from ENET_RX_CTRL read — suppress repeated debug prints
    last_rx_ctrl:  u32,
}

impl PdmaChannel {
    fn new(id: usize, dump_enabled: Arc<AtomicU32>) -> Self {
        Self {
            id,
            cbp: 0, nbdp: 0x80000000, bc: 0, ctrl: 0,
            gio: 0, dev: 0, dmacfg: 0, piocfg: 0,
            eox: false, eop: false, xie: false,
            misc: 0,
            active_mask: PDMA_CTRL_ACT,
            sys_mem: None,
            endian: false,
            even_high: false,
            irq: None,
            dump_enabled,
            dump_file: None,
            dump_transaction_id: 0,
            dump_is_write: false,
            transaction_id: 0,
            bytes_transferred: 0,
            dev_reads: 0,
            dev_writes: 0,
            cbp_polls: 0,
            cbp_log: std::collections::VecDeque::new(),
            cbp_log_t0: None,
            reads_at_poll: 0,
            last_poll: None,
            width_16: false,
            crbdp: 0, cpfxbdp: 0, ppfxbdp: 0, tx_new_packet: true, rx_filled: std::collections::VecDeque::new(), last_rx_ctrl: 0xFFFFFFFF,
        }
    }

    fn is_active(&self) -> bool {
        (self.ctrl & self.active_mask) != 0
    }

    /// True if this channel's bit is set in the pdma log mask.
    fn log_active(&self) -> bool {
        (devlog_mask(LogModule::Pdma) >> self.id) & 1 != 0
    }

fn start_transaction(&mut self) {
        self.transaction_id += 1;
        self.bytes_transferred = 0;
    }

    fn set_active(&mut self, active: bool) {
        if active {
            if !self.is_active() {
                self.start_transaction();
                crate::pf_begin!(crate::perfetto::events::PDMA_ACTIVE[self.id], self.nbdp);
            }
            self.ctrl |= self.active_mask;
        } else {
            if self.is_active() {
                if self.log_active() {
                    dlog_dev!(LogModule::Pdma, "PDMA[{}]: Channel Deactivated. Transferred {:x} bytes. CTRL={:08x}",
                        self.id, self.bytes_transferred, self.ctrl);
                }
                crate::pf_end!(crate::perfetto::events::PDMA_ACTIVE[self.id]);
            }
            self.ctrl &= !self.active_mask;
        }
    }

    /// Perfetto: this channel raised its completion interrupt.
    #[inline]
    fn trace_int(&self) {
        crate::pf_instant!(crate::perfetto::events::PDMA_INT[self.id]);
    }

    fn fetch_descriptor(&mut self) {
        let mut nbdp = self.nbdp;

        // Loop to handle 0-byte descriptors (links/markers) immediately
        loop {
            if let Some(mem) = &self.sys_mem {
                // Read descriptor: 3 words
                // Word 0: Buffer Address (PADDR)
                // Word 1: Byte Count & Flags (CNTINFO)
                // Word 2: Next Descriptor (PNEXT)
                let w_addr = { let _r = mem.read32((nbdp + PDMA_DESC_CBP_OFFSET) as u64); if _r.is_ok() { let d = _r.data; d } else { 0 } };
                let w_cnt = { let _r = mem.read32((nbdp + PDMA_DESC_BC_OFFSET) as u64); if _r.is_ok() { let d = _r.data; d } else { 0 } };
                let w_next = { let _r = mem.read32((nbdp + PDMA_DESC_NBP_OFFSET) as u64); if _r.is_ok() { let d = _r.data; d } else { 0 } };

                // Track current descriptor pointer for HPC3 writeback at interrupt time.
                // RX (ch10): crbdp = address of descriptor currently being filled.
                // TX (ch11): cpfxbdp/ppfxbdp promoted only at packet boundaries (EOXP).
                self.crbdp = nbdp;
                if self.id == 11 && self.tx_new_packet {
                    self.ppfxbdp = self.cpfxbdp;
                    self.cpfxbdp = nbdp;
                    self.tx_new_packet = false;
                }
                self.cbp = w_addr;
                self.bc = w_cnt; // Save full descriptor value (count + flags)
                self.eox  = (w_cnt & PDMA_DESC_EOX)  != 0;
                // TX (ch11): HPC3 sets bit 28 in BC register to indicate it has sampled EOX.
                if self.id == 11 && self.eox { self.bc |= ENET_BC_EOX_SAMPLED; }
                self.eop  = (w_cnt & PDMA_DESC_EOP)  != 0;
                self.xie  = (w_cnt & PDMA_DESC_XIE)  != 0;
                self.nbdp = w_next;
                
                if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[{}]: Fetched desc@{:08x}: CBP={:08x} BC={:08x} EOX={} XIE={} Next={:08x}",
                    self.id, nbdp, self.cbp, self.bc, self.eox, self.xie, self.nbdp); }

                // If byte count is 0, handle it immediately (it's a link or EOX marker)
                if (self.bc & 0x3FFF) == 0 {
                    if self.eox {
                        self.set_active(false);
                        if self.xie {
                            if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[{}]: Transfer Complete (EOX), Interrupting", self.id); }
                            self.ctrl |= PDMA_CTRL_INT; // Set interrupt pending
                            self.trace_int();
                            if let Some(irq) = &self.irq {
                                irq.set(true);
                            }
                        }
                        break; // Done
                    } else {
                        // Link descriptor (0 bytes, not EOX) -> fetch next immediately
                        nbdp = self.nbdp;
                        continue;
                    }
                }
                crate::pf_instant!(crate::perfetto::events::PDMA_DESC[self.id], self.bc);
                break; // Valid data descriptor loaded
            } else {
                break; // No memory attached
            }
        }
    }

    /// RX: what the channel can take now, for the SEEQ's host-side hold.
    ///
    /// A frame that would fill the descriptor carrying EOX (read from memory)
    /// stops the channel. If the driver still has filled descriptors to reap,
    /// it is about to re-arm them and move EOX on, and a stop now can land in
    /// the middle of its interrupt pass, where IRIX takes a stop for one it
    /// did not expect and resets the interface. A descriptor counts as reaped
    /// once its bit 14 is set again, which is how every driver re-arms one.
    fn rx_room(&mut self) -> RxRoom {
        if !self.is_active() { return RxRoom::Stopped; }
        let Some(mem) = self.sys_mem.as_ref() else { return RxRoom::Room };
        let word1 = |d: u32| mem.read32((d + PDMA_DESC_BC_OFFSET) as u64).data;
        if word1(self.crbdp) & PDMA_DESC_EOX == 0 { return RxRoom::Room; }
        let current = self.crbdp;
        self.rx_filled.retain(|&d| d != current && word1(d) & PDMA_DESC_ROWN == 0);
        if self.rx_filled.is_empty() { RxRoom::Room } else { RxRoom::Last }
    }

    fn get_name(&self) -> String {
        match self.id {
            0..=7 => format!("pbus_ch{}", self.id),
            8 => "scsi0".to_string(),
            9 => "scsi1".to_string(),
            10 => "enet_rx".to_string(),
            11 => "enet_tx".to_string(),
            _ => format!("unknown_{}", self.id),
        }
    }

    fn handle_dump(&mut self, addr: u32, data: &[u8], is_write: bool) {
        if (self.dump_enabled.load(Ordering::Relaxed) >> self.id) & 1 == 0 {
            if self.dump_file.is_some() {
                self.dump_file = None;
            }
            return;
        }

        if self.dump_file.is_some() && (self.transaction_id != self.dump_transaction_id || self.dump_is_write != is_write) {
            self.dump_file = None;
        }

        if self.dump_file.is_none() {
            let dir = if is_write { "w" } else { "r" };
            let name = format!("{}_{}_{:08x}_{}.bin", self.get_name(), dir, addr, self.transaction_id);
            if let Ok(file) = File::create(&name) {
                self.dump_file = Some(file);
                self.dump_transaction_id = self.transaction_id;
                self.dump_is_write = is_write;
                eprintln!("Created PDMA dump file: {}", name);
            } else {
                eprintln!("Failed to create PDMA dump file: {}", name);
                return;
            }
        }

        if let Some(file) = &mut self.dump_file {
            let _ = file.write_all(data);
        }
    }

    fn dma_read(&mut self) -> Option<(u32, DmaStatus)> {
        if !self.is_active() { return None; }
        self.dev_reads += 1;

        // PBUS DMA (channels 0-7) reads 32-bit words; 16-bit samples occupy
        // bits 23:8, while 8-bit transfers use bits 31:24.
        if self.id < 8 {
            let addr = self.cbp;
            let mem_val = if let Some(mem) = &self.sys_mem {
                { let _r = mem.read32(addr as u64); if _r.is_ok() { _r.data } else { return None } }
            } else {
                return None;
            };

            let val = if self.width_16 {
                // SGI audio DMA convention: the producer (ADPCM decoder, sine generator, etc.)
                // stores a signed 16-bit sample as `sample << 8` into a 32-bit int, placing
                // the sample in bits 23:8 with bits 7:0 zero-padded.  Shift right by 8 to
                // recover the full 16-bit value.  even_high (bit 19 of dmacfg) is a physical
                // PBUS byte-lane hint; it does not change the in-memory word layout.
                (mem_val >> 8) as u16 as u32
            } else {
                (mem_val >> 24) as u8 as u32
            };

            self.handle_dump(addr, &mem_val.to_be_bytes(), true);
            let st = self.advance(4, false);
            return Some((val, st));
        }

        let addr = self.cbp;
        let step = if self.width_16 { 2 } else { 1 };
        let swap = self.endian;

        let val = if let Some(mem) = &self.sys_mem {
            if self.width_16 {
                let _r = mem.read16(addr as u64);
                if _r.is_ok() {
                    let v = if swap { _r.data.swap_bytes() } else { _r.data };
                    self.handle_dump(addr, &v.to_be_bytes(), true);
                    Some(v as u32)
                } else { None }
            } else {
                let _r = mem.read8(addr as u64);
                if _r.is_ok() {
                    self.handle_dump(addr, &[_r.data], true);
                    Some(_r.data as u32)
                } else { None }
            }
        } else {
            None
        };

        if let Some(v) = val {
            let st = self.advance(step, false);
            Some((v, st))
        } else {
            None
        }
    }

    fn dma_write(&mut self, val: u32, eop: bool) -> DmaStatus {
        self.dev_writes += 1;
        if !self.is_active() {
            dlog_dev!(LogModule::Pdma, "PDMA[{}]: dma_write refused — channel not active (CBP={:08x} BC={:08x})", self.id, self.cbp, self.bc);
            return DmaStatus(DmaStatus::NOT_ACTIVE);
        }
        // RX (ch10): no ownership check. Bit 14 of word 1 is reserved in the
        // HPC3 specification; drivers use it as an ownership flag only because
        // the count write-back clears it. HPC3 fills whatever descriptor the
        // chain reaches until EOX (inferred, strong: no register or sequence
        // in the specification or any driver suggests it refuses one).

        // PBUS DMA (Channels 0-7) always operates on 32-bit words
        if self.id < 8 {
            let addr = self.cbp;
            let mem_val = if self.width_16 {
                let v = val as u16;
                let v = if self.endian { v.swap_bytes() } else { v };
                (v as u32) << 16
            } else {
                let v = val as u8;
                (v as u32) << 24
            };

            if let Some(mem) = &self.sys_mem {
                mem.write32(addr as u64, mem_val);
                self.handle_dump(addr, &mem_val.to_be_bytes(), false);
            }
            return self.advance(4, eop);
        }

        let addr = self.cbp;
        let step = if self.width_16 { 2 } else { 1 };
        let swap = self.endian;

        if let Some(mem) = &self.sys_mem {
            if self.width_16 {
                let v = if swap { (val as u16).swap_bytes() } else { val as u16 };
                mem.write16(addr as u64, v);
                self.handle_dump(addr, &v.to_be_bytes(), false);
            } else {
                mem.write8(addr as u64, val as u8);
                self.handle_dump(addr, &[val as u8], false);
            }
        }

        self.advance(step, eop)
    }

    /// Move past `step` bytes; at the end of a descriptor, write it back
    /// (Ethernet), then stop at EOX or fetch the next one.
    fn advance(&mut self, step: u32, caller_eop: bool) -> DmaStatus {
        self.cbp = self.cbp.wrapping_add(step);
        self.bytes_transferred += step as usize;
        let count = self.bc & 0x3FFF;
        if count >= step {
            self.bc -= step;
        } else {
            self.bc &= !0x3FFF;
        }
        // Propagate caller-supplied EOP regardless of bc
        let mut status = if caller_eop { DmaStatus(DmaStatus::EOP) } else { DmaStatus::ok() };

        let mut bc_done = (self.bc & 0x3FFF) == 0;

        // RX (ch10): a frame longer than its buffer sets RBO and stops the
        // channel with an interrupt (established, HPC3 specification); one
        // frame never spans descriptors. The rest of the frame is refused.
        // Whether the descriptor is written back is unknown; it is left as
        // armed, so a driver restarting at its first armed descriptor
        // reuses it.
        if self.id == 10 && bc_done && !caller_eop {
            self.ctrl |= ENET_RX_CTRL_RBO;
            self.set_active(false);
            if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[10]: receive buffer overflow at desc {:08x}", self.crbdp); }
            self.trace_int();
            return DmaStatus(DmaStatus::OVERFLOW | DmaStatus::IRQ);
        }

        // xie fires on every descriptor completion: either caller signals EOP, or bc hits zero.
        // These are orthogonal to EOX.
        let mut irq = self.xie && (caller_eop || bc_done);

        // RX (ch10): the frame is in; hand the descriptor back to the host by
        // writing the remaining count with ROWN clear. This has to land before
        // the channel stops at EOX or moves on below. A driver that finds the
        // channel stopped restarts it at the first descriptor it still sees as
        // the HPC's (sgiseeq's rx_maybe_restart, sq_rxintr). Stopped with the
        // write still pending, that is the descriptor just filled, and the next
        // frame lands on top of this one.
        //
        // The order (data and status byte, then word 1, then ACTIVE clear,
        // then the interrupt the SEEQ thread raises) is inferred from the HPC3
        // specification: the write-back happens once the last byte is in
        // memory, ACTIVE clears on completion of the descriptor's transfer.
        // The drivers read the status from the frame buffer and wait on the
        // interrupt, so nothing depends on rx_stat landing with the descriptor.
        if self.id == 10 && caller_eop {
            if self.crbdp != 0 {
                // EOX is taken from memory as the frame completes, not from the
                // copy fetched when the previous descriptor finished. Model
                // choice: the specification does not say when HPC3 samples it.
                // Drivers move EOX forward while the channel runs (IRIX sets it
                // on the new last descriptor, then clears it on the old one),
                // and IRIX restarts cleanly only when the channel stops at the
                // descriptor that carries EOX in memory; stopping on a stale
                // copy names a descriptor it no longer expects, and it resets
                // the interface.
                if let Some(mem) = &self.sys_mem {
                    let w = mem.read32((self.crbdp + PDMA_DESC_BC_OFFSET) as u64);
                    if w.is_ok() { self.eox = w.data & PDMA_DESC_EOX != 0; }
                }
                let rem = (self.bc & 0x3FFF) as u16;
                if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[10]: RX writeback crbdp={:08x}+6 ← rem_bc={:04x}", self.crbdp, rem); }
                if let Some(mem) = &self.sys_mem {
                    mem.write16((self.crbdp + 6) as u64, rem);
                }
                if self.rx_filled.len() == RX_FILLED_MAX { self.rx_filled.pop_front(); }
                self.rx_filled.push_back(self.crbdp);
            }
            bc_done = true;
        }

        // SCSI TX (ch8/9): caller_eop means the device has finished sending all its data.
        // Force bc_done so the descriptor completes even if fewer bytes were written than
        // the host allocated (e.g. MODE_SENSE response shorter than allocation length).
        // This fires EOX → set_active(false), clearing WDSC_DMA_ACTIVE for the next transfer.
        if (self.id == 8 || self.id == 9) && caller_eop {
            bc_done = true;
        }

        // Byte count reached zero — end of this descriptor.
        if bc_done {
            if self.eop {
                status |= DmaStatus(DmaStatus::EOP);
                if self.id == 11 { self.tx_new_packet = true; }
            }
            // TX (ch11): mark the packet sent before the channel stops, for the
            // same reason as RX above (and inferred in the same way): a driver
            // restarts a stopped channel at the first descriptor without TXD
            // (sgiseeq's kick_tx, sq_txring_hpc3), which would send this
            // packet again. sq_txring_hpc3 reads the descriptor status before
            // the channel's ACTIVE bit, which only works if the mark lands first.
            if self.id == 11 && self.eop && self.cpfxbdp != 0 {
                if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[11]: TX writeback cpfxbdp={:08x}+6 ← TXD", self.cpfxbdp); }
                if let Some(mem) = &self.sys_mem {
                    mem.write16((self.cpfxbdp + 6) as u64, ENET_BC_TXD);
                }
            }
            if self.eox {
                status |= DmaStatus(DmaStatus::EOX);
                self.set_active(false);
                // RX: stopping at EOX interrupts whether or not the descriptor
                // has XIE (established, HPC3 specification: the second listed
                // receive interrupt condition).
                if self.id == 10 { irq = true; }
                if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[{}]: Transfer Complete (EOX). CBP={:08x} NBDP={:08x} BC={:08x} CTRL={:08x}",
                    self.id, self.cbp, self.nbdp, self.bc, self.ctrl); }
            } else {
                self.fetch_descriptor();
            }
        }

        if irq {
            self.trace_int();
            status |= DmaStatus(DmaStatus::IRQ);
            // For SCSI/PBUS channels (callback installed): set ctrl INT flag and notify.
            // For enet channels (no callback): IRQ bit in status is the signal; don't touch ctrl.
            if let Some(irq) = &self.irq {
                self.ctrl |= PDMA_CTRL_INT;
                irq.set(true);
            }
            if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[{}]: Interrupting (xie caller_eop={} bc_done={})", self.id, caller_eop, bc_done); }
        }
        status
    }
}

struct PdmaClientImpl {
    channel: Arc<Mutex<PdmaChannel>>,
}

impl DmaClient for PdmaClientImpl {
    fn read(&self) -> Option<(u32, DmaStatus)> {
        self.channel.lock().dma_read()
    }
    fn read_ahead_of_poll(&self) -> Option<u64> {
        let c = self.channel.lock();
        // A guest that has not read CBP for a while is not steering by it.
        match c.last_poll {
            Some(t) if t.elapsed() < std::time::Duration::from_millis(50) =>
                Some(c.dev_reads.saturating_sub(c.reads_at_poll)),
            _ => None,
        }
    }
    fn write(&self, val: u32, eop: bool) -> DmaStatus {
        self.channel.lock().dma_write(val, eop)
    }
    fn rx_room(&self) -> RxRoom {
        let mut c = self.channel.lock();
        if c.id == HPC3_PDMA_CHAN_ENET_RX as usize { c.rx_room() } else { RxRoom::Room }
    }
    fn overflow(&self) {
        let mut c = self.channel.lock();
        // A SEEQ receive FIFO overflow stops the receive channel at whatever
        // descriptor is current, not at EOX (established, HPC3 specification:
        // one of the three receive interrupt causes that clear ACTIVE).
        if c.id == HPC3_PDMA_CHAN_ENET_RX as usize {
            c.set_active(false);
        }
    }
}

trait PdmaChannelOps: Send + Sync {
    fn read(&self, chan: &mut PdmaChannel, reg: u32) -> u32;
    fn write(&self, chan: &mut PdmaChannel, reg: u32, val: u32);

    fn read_dmacfg(&self, chan: &mut PdmaChannel) -> u32 { chan.dmacfg }
    fn write_dmacfg(&self, chan: &mut PdmaChannel, val: u32) { 
        chan.dmacfg = val;
        // Default implementation, override for specific channels
    }
    fn read_piocfg(&self, chan: &mut PdmaChannel) -> u32 { chan.piocfg }
    fn write_piocfg(&self, chan: &mut PdmaChannel, val: u32) { chan.piocfg = val; }
}

struct PbusDmaOps;
impl PdmaChannelOps for PbusDmaOps {
    fn read(&self, chan: &mut PdmaChannel, reg: u32) -> u32 {
        match reg {
            HPC3_PDMA_CBP => {
                chan.cbp_polls += 1;
                chan.reads_at_poll = chan.dev_reads;
                chan.last_poll = Some(std::time::Instant::now());
                let t0 = *chan.cbp_log_t0.get_or_insert_with(std::time::Instant::now);
                if chan.cbp_log.len() == 64 { chan.cbp_log.pop_front(); }
                let entry = (t0.elapsed().as_micros() as u64, chan.cbp, chan.dev_reads * 4);
                chan.cbp_log.push_back(entry);
                chan.cbp
            }
            HPC3_PDMA_NBDP => chan.nbdp,
            HPC3_PDMA_CTRL => {
                let val = chan.ctrl;
                if (chan.ctrl & PDMA_CTRL_INT) != 0 {
                    chan.ctrl &= !PDMA_CTRL_INT;
                    if let Some(irq) = &chan.irq {
                        irq.set(false);
                    }
                }
                val
            }
            _ => 0
        }
    }
    fn write(&self, chan: &mut PdmaChannel, reg: u32, val: u32) {
        match reg {
            HPC3_PDMA_CBP => chan.cbp = val,
            HPC3_PDMA_NBDP => chan.nbdp = val,
            HPC3_PDMA_CTRL => {
                chan.endian = (val & PDMA_CTRL_LITTLE) != 0;
                if (val & PDMA_CTRL_CH_ACT_LD) != 0 {
                    let enable = (val & PDMA_CTRL_CH_ACT) != 0;
                    if enable {
                        if !chan.is_active() {
                            chan.set_active(true);
                            chan.fetch_descriptor();
                        }
                    } else {
                        chan.set_active(false);
                    }
                }
            }
            _ => {}
        }
    }
    fn write_dmacfg(&self, chan: &mut PdmaChannel, val: u32) {
        chan.dmacfg = val;
        chan.width_16  = (val & PBUS_DMACFG_DS16)      != 0;
        chan.even_high = (val & PBUS_DMACFG_EVEN_HIGH)  != 0;
    }
}

struct ScsiDmaOps {
    wd: Arc<OnceLock<Arc<Wd33c93a>>>,
}
impl PdmaChannelOps for ScsiDmaOps {
    fn read(&self, chan: &mut PdmaChannel, reg: u32) -> u32 {
        match reg {
            HPC3_PDMA_CBP => chan.cbp,
            HPC3_PDMA_NBDP => chan.nbdp,
            SCSI_BC => chan.bc,
            SCSI_CTRL => {
                let val = chan.ctrl;
                if (chan.ctrl & SCSI_CTRL_INT) != 0 {
                    chan.ctrl &= !SCSI_CTRL_INT;
                    if let Some(irq) = &chan.irq {
                        irq.set(false);
                    }
                }
                val
            }
            SCSI_GIO => chan.gio,
            SCSI_DEV => chan.dev,
            SCSI_DMACFG => chan.dmacfg,
            SCSI_PIOCFG => chan.piocfg,
            _ => 0
        }
    }
    fn write(&self, chan: &mut PdmaChannel, reg: u32, val: u32) {
        match reg {
            HPC3_PDMA_CBP => chan.cbp = val,
            HPC3_PDMA_NBDP => chan.nbdp = val,
            SCSI_BC => {
                chan.bc = val;
                chan.eox = (val & PDMA_DESC_EOX) != 0;
                chan.xie = (val & PDMA_DESC_XIE) != 0;
            }
            SCSI_CTRL => {
                chan.endian = (val & SCSI_CTRL_ENDIAN) != 0;

                let was_active = chan.is_active();
                let prev_reset = (chan.ctrl & SCSI_CTRL_RESET) != 0;
                // Update control register, preserving the active bit for now
                chan.ctrl = (val & !SCSI_CTRL_ACTIVE) | (chan.ctrl & SCSI_CTRL_ACTIVE);

                let mask_active = (val & SCSI_CTRL_AMASK) != 0;
                let reset = (val & SCSI_CTRL_RESET) != 0;
                let mut should_be_active = if mask_active { was_active } else { (val & SCSI_CTRL_ACTIVE) != 0 };

                if reset { should_be_active = false; }

                // Falling edge of RESET: pulse the WD33C93A chip reset.
                // After reset the chip sets ASR.INT; OpenBSD wdsc_match() polls for this.
                if prev_reset && !reset {
                    if let Some(wd) = self.wd.get() { wd.power_on(); }
                }

                chan.set_active(should_be_active);

                if !was_active && should_be_active {
                    chan.fetch_descriptor();
                }

                if (val & SCSI_CTRL_FLUSH) != 0 {
                    // Flush: drain FIFO to memory and terminate DMA.
                    // In emulation the FIFO doesn't exist, so just stop the channel.
                    // NOTE: previously we raised PDMA_CTRL_INT here if XIE was set
                    // in the current descriptor, but the IRIX 6.5 miniroot SCSI
                    // driver doesn't expect an IRQ from its own FLUSH (teardown)
                    // — it acks the prior real IRQ and writes FLUSH to clean up.
                    // Firing again on FLUSH leaves the bit stuck → IRQ storm.
                    chan.set_active(false);
                    chan.ctrl &= !(SCSI_CTRL_ACTIVE | SCSI_CTRL_FLUSH);
                }
            }
            SCSI_GIO => chan.gio = val,
            SCSI_DEV => chan.dev = val,
            SCSI_DMACFG => chan.dmacfg = val,
            SCSI_PIOCFG => chan.piocfg = val,
            _ => {}
        }
    }
    fn write_dmacfg(&self, chan: &mut PdmaChannel, val: u32) {
        chan.dmacfg = val;
        chan.width_16 = (val & SCSI_DMACFG_DMA16) != 0;
    }
}

// TX descriptor done flag (BC_TXD): written to cpfxbdp+6 on successful TX
const ENET_BC_TXD: u16 = 0x8000;
// RX: the most descriptors `rx_filled` remembers. The drivers' rings are at
// most 64 long. It is not saved in snapshots: after a load it only makes the
// first `Last` answer lax.
const RX_FILLED_MAX: usize = 64;
// BC register bit 28: HPC3 has sampled EOX from the current descriptor
const ENET_BC_EOX_SAMPLED: u32 = 0x10000000;

struct EnetRxDmaOps {
    seeq: Arc<OnceLock<Arc<Seeq8003>>>,
}
impl PdmaChannelOps for EnetRxDmaOps {
    fn read(&self, chan: &mut PdmaChannel, reg: u32) -> u32 {
        match reg {
            ENET_RX_CBP => chan.cbp,
            ENET_RX_NBDP => chan.nbdp,
            ENET_RX_BC => chan.bc,
            ENET_RX_CTRL => {
                // Mirror SEEQ RX status into low 8 bits (read-only snapshot, no side-effects)
                let seeq_st = self.seeq.get()
                    .map(|s| s.get_rx_status() as u32)
                    .unwrap_or(0);
                let val = (chan.ctrl & !ENET_RX_CTRL_SEEQ_MASK) | (seeq_st & ENET_RX_CTRL_SEEQ_MASK);
                // Model choice: a read clears RBO. The specification says both
                // that a write of 1 and that a read of this register clear it,
                // and no driver clears it explicitly, so only the read keeps a
                // driver from seeing it, and its interrupt, forever.
                chan.ctrl &= !ENET_RX_CTRL_RBO;
                // Drivers read this after re-arming their ring: a frame the
                // SEEQ holds for room may now fit.
                if let Some(s) = self.seeq.get() { s.kick_if_holding(); }
                if val != chan.last_rx_ctrl {
                    dlog_dev!(LogModule::Hpc3, "PDMA[{}]: ENET_RX_CTRL read → {:08x} (ctrl={:08x} active={} seeq_st={:02x})",
                        chan.id, val, chan.ctrl, chan.is_active(), seeq_st);
                    chan.last_rx_ctrl = val;
                }
                val
            }
            ENET_RX_GIO => chan.gio,
            ENET_RX_DEV => chan.dev,
            ENET_RX_DMACFG => chan.dmacfg,
            ENET_RX_PIOCFG => chan.piocfg,
            ENET_RX_RESET => {
                // INTPEND (bit 1) comes from SeeqState; other bits (CH_RESET etc.) from misc.
                let intpend = self.seeq.get()
                    .map(|s| s.is_interrupt_pending())
                    .unwrap_or(false);
                (chan.misc & !ENET_RX_RESET_INTPEND)
                    | if intpend { ENET_RX_RESET_INTPEND } else { 0 }
            }
            _ => 0
        }
    }
    fn write(&self, chan: &mut PdmaChannel, reg: u32, val: u32) {
        match reg {
            ENET_RX_CBP => chan.cbp = val,
            ENET_RX_NBDP => chan.nbdp = val,
            ENET_RX_BC => {
                chan.bc = val;
                chan.eox = (val & PDMA_DESC_EOX) != 0;
                chan.xie = (val & PDMA_DESC_XIE) != 0;
            }
            ENET_RX_CTRL => {
                chan.endian = (val & ENET_RX_CTRL_ENDIAN) != 0;

                let was_active = chan.is_active();
                chan.ctrl = (val & !ENET_RX_CTRL_ACTIVE) | (chan.ctrl & ENET_RX_CTRL_ACTIVE);

                let mask_active = (val & ENET_RX_CTRL_AMASK) == 0;
                let should_be_active = if mask_active { (val & ENET_RX_CTRL_ACTIVE) != 0 } else { was_active };

                dlog_dev!(LogModule::Hpc3, "PDMA[{}]: ENET_RX_CTRL write val={:08x} was_active={} should_be_active={} ch_reset={}",
                    chan.id, val, was_active, should_be_active, (chan.misc & ENET_RX_RESET_CH_RESET) != 0);

                chan.set_active(should_be_active);

                if !was_active && should_be_active {
                    if (chan.misc & ENET_RX_RESET_CH_RESET) == 0 {
                        chan.fetch_descriptor();
                    }
                    // Kick enet thread so any queued RX frames are delivered promptly
                    if let Some(seeq) = self.seeq.get() { seeq.kick_rx(); }
                }
            }
            ENET_RX_GIO => chan.gio = val,
            ENET_RX_DEV => chan.dev = val,
            // ENET_RX_RESET handled at Hpc3 level (needs both channels + seeq)
            ENET_RX_DMACFG => chan.dmacfg = val,
            ENET_RX_PIOCFG => chan.piocfg = val,
            _ => {}
        }
    }
}

struct EnetTxDmaOps {
    seeq: Arc<OnceLock<Arc<Seeq8003>>>,
}
impl PdmaChannelOps for EnetTxDmaOps {
    fn read(&self, chan: &mut PdmaChannel, reg: u32) -> u32 {
        match reg {
            ENET_TX_CBP => chan.cbp,
            ENET_TX_NBDP => chan.nbdp,
            ENET_TX_BC => chan.bc | if chan.eox { ENET_BC_EOX_SAMPLED } else { 0 },
            ENET_TX_CTRL => {
                // Mirror SEEQ TX status into low 8 bits (read-only snapshot, no side-effects)
                let seeq_st = self.seeq.get()
                    .map(|s| s.get_tx_status() as u32)
                    .unwrap_or(0);
                let val = (chan.ctrl & !ENET_TX_CTRL_SEEQ_MASK) | (seeq_st & ENET_TX_CTRL_SEEQ_MASK);
                dlog_dev!(LogModule::Hpc3, "PDMA[{}]: ENET_TX_CTRL read → {:08x} (ctrl={:08x} active={} seeq_st={:02x})",
                    chan.id, val, chan.ctrl, chan.is_active(), seeq_st);
                val
            }
            ENET_TX_GIO => chan.gio,
            ENET_TX_DEV => chan.dev,
            _ => 0
        }
    }
    fn write(&self, chan: &mut PdmaChannel, reg: u32, val: u32) {
        match reg {
            ENET_TX_CBP => chan.cbp = val,
            ENET_TX_NBDP => chan.nbdp = val,
            ENET_TX_BC => {
                chan.bc = val;
                chan.eox = (val & PDMA_DESC_EOX) != 0;
                chan.xie = (val & PDMA_DESC_XIE) != 0;
            }
            ENET_TX_CTRL => {
                chan.endian = (val & ENET_TX_CTRL_ENDIAN) != 0;

                let was_active = chan.is_active();
                chan.ctrl = (val & !ENET_TX_CTRL_ACTIVE) | (chan.ctrl & ENET_TX_CTRL_ACTIVE);

                let mask_active = (val & ENET_TX_CTRL_AMASK) == 0;
                let should_be_active = if mask_active { (val & ENET_RX_CTRL_ACTIVE) != 0 } else { was_active };

                dlog_dev!(LogModule::Hpc3, "PDMA[{}]: ENET_TX_CTRL write val={:08x} was_active={} should_be_active={}",
                    chan.id, val, was_active, should_be_active);

                chan.set_active(should_be_active);

                if !was_active && should_be_active {
                    chan.tx_new_packet = true; // first descriptor of new chain starts a new packet
                    chan.fetch_descriptor();
                    // Wake the enet thread immediately to drain this TX data
                    if let Some(seeq) = self.seeq.get() {
                        seeq.kick_tx();
                    }
                }
            }
            ENET_TX_GIO => chan.gio = val,
            ENET_TX_DEV => chan.dev = val,
            _ => {}
        }
    }
}

#[derive(Clone)]
pub struct Hpc3 {
    state: Arc<Mutex<Hpc3State>>,
    ioc: Ioc,
    rtc: Arc<Ds1x86>,
    eeprom: Arc<Mutex<Eeprom93c56>>,
    seeq: Arc<Seeq8003>,
    pdma_channels: Vec<Arc<Mutex<PdmaChannel>>>,
    pdma_ops: Vec<Arc<dyn PdmaChannelOps>>,
    scsi_dev: Arc<Wd33c93a>,
    /// Fullhouse (Indigo2) only — second, independent WD33C93A. `None` on
    /// guinness (Indy). See `SCSI1_REG_BASE`'s doc comment.
    scsi_dev1: Option<Arc<Wd33c93a>>,
    hal2: Option<Arc<Hal2>>,
    pdma_dump: Arc<AtomicU32>,
    /// The machine's gateway settings, kept so a DaynaPort SCSI target can be
    /// given its own `NatEngine` sharing the backend selection (NAT vs PCAP)
    /// and the NFS export, but with its own subnet and MAC.
    net_base: GatewayConfig,
    /// Indy (Guinness) vs Indigo2 (fullhouse). Fullhouse exposes the INT2
    /// interrupt registers at PBUS PIO channel 4 (see HPC3_INT2_BASE);
    /// guinness does not have anything mapped there.
    guinness: bool,
}

impl Hpc3 {
    /// The empty SCSI buses of a board: one controller on an Indy (guinness),
    /// two on an Indigo2. A machine fills them from `[scsi.N]` and gives them
    /// to [`Self::with_net`].
    pub fn new_scsi_buses(guinness: bool) -> Vec<crate::scsi::ScsiBus> {
        (0..if guinness { 1 } else { 2 }).map(|_| Wd33c93a::new_bus()).collect()
    }

    /// The bus `controller` names among a board's `buses`, or why there is none.
    pub fn route_scsi(buses: &[crate::scsi::ScsiBus], controller: u8) -> Result<&crate::scsi::ScsiBus, String> {
        match controller {
            c if (c as usize) < buses.len() => Ok(&buses[c as usize]),
            1 => Err("controller 1 (this machine has only controller 0)".to_string()),
            c => Err(format!("controller {c} (valid: 0, 1)")),
        }
    }

    /// `no_audio` skips HAL2 audio init (used by `--noaudio` and also by full
    /// `--headless`, which can't run audio in CI).
    /// `nvram_path` is the on-disk NVRAM file (loaded at startup, default save
    /// target for `iris-ci rtc-save`); `rtc_offset` shifts the RTC's
    /// host-time seed (`[rtc_offset]`).
    pub fn with_net(eeprom: Arc<Mutex<Eeprom93c56>>, ioc: Ioc, guinness: bool, heartbeat: Arc<AtomicU64>, net: NetworkConfig, no_audio: bool, audio: AudioConfig, nvram_path: String, rtc_offset: RtcOffset, scsi_deferred_int: bool, scsi_buses: Vec<crate::scsi::ScsiBus>) -> Self {
        let mut scsi_buses = scsi_buses.into_iter();
        let gateway_cfg = GatewayConfig::from_network(&net);
        let rtc = Arc::new(Ds1x86::new(8192, nvram_path, rtc_offset));
        let pdma_dump = Arc::new(AtomicU32::new(0));
        
        let state = Arc::new(Mutex::new(Hpc3State {
            intstat: 0,
            gio_misc: 0,
            eeprom_reg: 0,
            pbus_pio: [0; 0x1000],
        }));

        // Shared OnceLock so ScsiDmaOps can call power_on() on the WD33C93A reset falling edge.
        let scsi_wd_lock: Arc<OnceLock<Arc<Wd33c93a>>> = Arc::new(OnceLock::new());
        // Fullhouse only: second, independent WD33C93A on PDMA channel 9 /
        // register window SCSI1_REG_BASE. Indy (guinness) has no second
        // physical chip, so channel 9 there just keeps pointing at
        // controller 0's lock (unreachable in practice — nothing maps to
        // its register window on guinness).
        let scsi1_wd_lock: Arc<OnceLock<Arc<Wd33c93a>>> = Arc::new(OnceLock::new());

        // Shared OnceLock so EnetRx/TxDmaOps can pull SEEQ status on CTRL read.
        // Populated after seeq creation below.
        let enet_seeq_lock: Arc<OnceLock<Arc<Seeq8003>>> = Arc::new(OnceLock::new());

        let mut pdma_channels = Vec::new();
        let mut pdma_ops: Vec<Arc<dyn PdmaChannelOps>> = Vec::new();
        let mut dma_clients: Vec<Arc<dyn DmaClient>> = Vec::new();
        for i in 0..12 {
            let mut chan = PdmaChannel::new(i, pdma_dump.clone());
            if i == HPC3_PDMA_CHAN_SCSI0 as usize || i == HPC3_PDMA_CHAN_SCSI1 as usize {
                chan.active_mask = SCSI_CTRL_ACTIVE; // 0x10 for SCSI
            } else if i == HPC3_PDMA_CHAN_ENET_RX as usize || i == HPC3_PDMA_CHAN_ENET_TX as usize {
                chan.active_mask = ENET_TX_CTRL_ACTIVE; // 0x200 for enet RX/TX
            }
            
            pdma_channels.push(Arc::new(Mutex::new(chan)));
            dma_clients.push(Arc::new(PdmaClientImpl { channel: pdma_channels.last().unwrap().clone() }));
            if i <= HPC3_PDMA_CHAN_GENERIC as usize {
                pdma_ops.push(Arc::new(PbusDmaOps));
            } else if i == HPC3_PDMA_CHAN_SCSI0 as usize {
                pdma_ops.push(Arc::new(ScsiDmaOps { wd: scsi_wd_lock.clone() }));
            } else if i == HPC3_PDMA_CHAN_SCSI1 as usize {
                pdma_ops.push(Arc::new(ScsiDmaOps {
                    wd: if guinness { scsi_wd_lock.clone() } else { scsi1_wd_lock.clone() },
                }));
            } else if i == HPC3_PDMA_CHAN_ENET_RX as usize {
                pdma_ops.push(Arc::new(EnetRxDmaOps { seeq: enet_seeq_lock.clone() }));
            } else {
                pdma_ops.push(Arc::new(EnetTxDmaOps { seeq: enet_seeq_lock.clone() }));
            }
        }

        let enet_rx_dma = Arc::new(PdmaClientImpl { channel: pdma_channels[10].clone() });
        let enet_tx_dma = Arc::new(PdmaClientImpl { channel: pdma_channels[11].clone() });

        // The interrupt status, and its lines: each SCSI PDMA channel's
        // completion, each WD33C93A, the SEEQ. Enet channels 10/11 have no
        // DMA completion line: the SEEQ drives the interrupt.
        let irqs = Arc::new(Hpc3Interrupts {
            state: state.clone(),
            scsi: [ioc.irq_line(IocInterrupt::Scsi0), ioc.irq_line(IocInterrupt::Scsi1)],
            ethernet: ioc.irq_line(IocInterrupt::Ethernet),
            scsi_dma: [pdma_channels[HPC3_PDMA_CHAN_SCSI0 as usize].clone(), pdma_channels[HPC3_PDMA_CHAN_SCSI1 as usize].clone()],
        });
        pdma_channels[HPC3_PDMA_CHAN_SCSI0 as usize].lock().irq = Some(irqs.line(HPC3_INTSTAT_SCSI0_DMA));
        pdma_channels[HPC3_PDMA_CHAN_SCSI1 as usize].lock().irq = Some(irqs.line(HPC3_INTSTAT_SCSI1_DMA));
        let seeq_irq = irqs.line(HPC3_INTSTAT_ENET_DEV);
        let net_base = gateway_cfg.clone();
        let link = Link::with_gateway(gateway_cfg.clone());
        let seeq = Arc::new(Seeq8003::with_config(Some(seeq_irq), Some(enet_rx_dma), Some(enet_tx_dma), gateway_cfg, heartbeat.clone()));
        match link {
            Ok(Some(link)) => seeq.attach(link),
            Ok(None) => {}
            Err(e) => eprintln!("iris: network: {e}; Ethernet left unplugged"),
        }
        // Publish seeq to both the DMA ops (CTRL reads) and the irq (status checks in set_interrupt)
        let _ = enet_seeq_lock.set(seeq.clone());
        
        let scsi0_dma = Arc::new(PdmaClientImpl { channel: pdma_channels[8].clone() });
        // A chip-INT ack (the kernel reads SCSI_STATUS) also drops any
        // lingering PDMA INT on the same line; see Hpc3Interrupts.
        let scsi0_irq = irqs.line(HPC3_INTSTAT_SCSI0_DEV);

        let scsi_dev = Arc::new(Wd33c93a::new_with_id(Some(scsi0_dma), Some(scsi0_irq), heartbeat.clone(), scsi_deferred_int, 0,
            scsi_buses.next().expect("a board has a SCSI bus")));
        let _ = scsi_wd_lock.set(scsi_dev.clone());

        // Fullhouse's second WD33C93A — see scsi1_wd_lock's doc comment above.
        let scsi_dev1 = if !guinness {
            let scsi1_dma = Arc::new(PdmaClientImpl { channel: pdma_channels[9].clone() });
            let scsi1_irq = irqs.line(HPC3_INTSTAT_SCSI1_DEV);
            let dev = Arc::new(Wd33c93a::new_with_id(Some(scsi1_dma), Some(scsi1_irq), heartbeat.clone(), scsi_deferred_int, 1,
                scsi_buses.next().expect("a fullhouse board has a second SCSI bus")));
            let _ = scsi1_wd_lock.set(dev.clone());
            Some(dev)
        } else {
            None
        };

        let hal2 = if no_audio { None } else { Some(Arc::new(Hal2::with_audio(dma_clients[0..8].to_vec(), audio))) };

        Self {
            state,
            ioc,
            rtc,
            eeprom,
            seeq,
            pdma_channels,
            pdma_ops,
            scsi_dev,
            scsi_dev1,
            hal2,
            pdma_dump,
            net_base,
            guinness,
        }
    }

    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        if let Some(hal2) = &self.hal2 { hal2.set_timer_manager(tm); }
    }

    pub fn set_phys(&self, mem: Arc<dyn BusDevice>) {
        for chan in &self.pdma_channels {
            chan.lock().sys_mem = Some(mem.clone());
        }
    }

    /// Pick controller 0 or 1's chip. `controller == 1` on guinness (Indy,
    /// no second chip) is a config-validation error — see
    /// `MachineConfig::validate` — so this panics rather than silently
    /// falling back, to catch that class of bug immediately instead of
    /// quietly wiring a disk to the wrong (or a nonexistent) controller.
    fn scsi_controller(&self, controller: u8) -> &Arc<Wd33c93a> {
        match controller {
            0 => &self.scsi_dev,
            1 => self.scsi_dev1.as_ref().expect("scsi controller 1 requested but not present (guinness profile?)"),
            _ => panic!("invalid SCSI controller {} (valid: 0, 1)", controller),
        }
    }

    /// Attach a DaynaPort SCSI/Link (SCSI-attached Ethernet) at `id`.
    ///
    /// Each DaynaPort gets its **own** `NatEngine` on its own subnet, separate
    /// from the onboard SEEQ's — so `dp0` and `ec0` land on different networks
    /// and traffic through the DaynaPort is unmistakable. Backend selection
    /// (NAT vs PCAP) and the NFS export are inherited from `[network]`/`[nfs]`;
    /// host port forwards are **not**, since only one engine can own a host
    /// listening port.
    pub fn add_scsi_daynaport(&self, controller: u8, id: usize, params: crate::config::DaynaportParams) -> std::io::Result<()> {
        let gateway = GatewayConfig {
            // A distinct gateway MAC per target: same 02:00:DE:AD prefix as the
            // SEEQ's, then DA ("Dayna") and the SCSI id.
            gateway_mac: [0x02, 0x00, 0xDE, 0xAD, 0xDA, id as u8],
            gateway_ip:  params.subnet.gateway_ip,
            client_ip:   params.subnet.client_ip,
            netmask:     params.subnet.netmask,
            port_forwards: vec![],
            ..self.net_base.clone()
        };
        self.scsi_controller(controller).add_daynaport(id, params.mac, gateway)
    }

    pub fn ioc(&self) -> &Ioc {
        &self.ioc
    }

    pub fn rtc(&self) -> &Arc<Ds1x86> {
        &self.rtc
    }

    pub fn eeprom(&self) -> &Arc<Mutex<Eeprom93c56>> {
        &self.eeprom
    }

    pub fn seeq(&self) -> &Arc<Seeq8003> {
        &self.seeq
    }

    pub fn hal2(&self) -> Option<&Arc<Hal2>> {
        self.hal2.as_ref()
    }

    pub fn scsi(&self) -> &Arc<Wd33c93a> {
        &self.scsi_dev
    }

    /// Fullhouse (Indigo2) only — `None` on guinness (Indy).
    pub fn scsi1(&self) -> Option<&Arc<Wd33c93a>> {
        self.scsi_dev1.as_ref()
    }

    pub fn register_locks(&self) {
        use crate::locks::register_lock_fn;
        let state = self.state.clone();
        register_lock_fn("hpc3::state",   move || state.is_locked());
        let eeprom = self.eeprom.clone();
        register_lock_fn("hpc3::eeprom",  move || eeprom.is_locked());
        for (i, chan) in self.pdma_channels.iter().enumerate() {
            let chan = chan.clone();
            register_lock_fn(format!("hpc3::pdma_channels[{}]", i), move || chan.is_locked());
        }
        // Delegate to child components
        self.seeq.register_locks();
        self.scsi_dev.register_locks();
        if let Some(dev) = &self.scsi_dev1 { dev.register_locks(); }
        if let Some(hal2) = &self.hal2 { hal2.register_locks(); }
        self.ioc.register_locks();
    }
}

impl Device for Hpc3 {
    fn step(&self, _cycles: u64) {
        // DMA is driven by clients and peripheral workers, not CPU cycles.
    }

    fn stop(&self) {
        self.seeq.stop();
        self.scsi_dev.stop();
        if let Some(dev) = &self.scsi_dev1 { dev.stop(); }
        self.rtc.stop();
        self.ioc.stop();
        if let Some(hal2) = &self.hal2 { hal2.stop(); }
        if !self.guinness { self.eeprom.lock().stop(); }
    }

    fn start(&self) {
        if let Some(hal2) = &self.hal2 { hal2.start(); }
        self.ioc.start();
        self.rtc.start();
        self.scsi_dev.start();
        if let Some(dev) = &self.scsi_dev1 { dev.start(); }
        self.seeq.start();
    }
    fn is_running(&self) -> bool { self.ioc.is_running() }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        let mut cmds = vec![("hpc3".to_string(), "HPC3 commands: hpc3 status".to_string())];
        cmds.push(("pdma".to_string(), "PDMA commands: pdma status | pdma chain <addr> | pdma dump <on|off|hal|scsi|enet|MASK> [DEV]".to_string()));
        cmds.extend(self.ioc.register_commands());
        cmds.extend(self.rtc.register_commands());
        cmds.push(("nveeprom".to_string(), "NVRAM EEPROM commands (93CS56 @ 0x1fbb0008, stores env vars + MAC @ words 0x7D-0x7F — NOT the CPU/MC chip, see `eeprom`): nveeprom <on|off> | nveeprom dump | nveeprom r <word> | nveeprom w <word> <val> | nveeprom save [file]".to_string()));
        cmds.extend(self.seeq.register_commands());
        cmds.extend(self.scsi_dev.register_commands());
        // "scsi"/"cow" above already cover controller 0; scsi0 is an
        // explicit alias for the same, and scsi1 (fullhouse only) targets
        // the second controller — see execute_command's dispatch.
        cmds.push(("scsi0".to_string(), "Alias for `scsi` (controller 0)".to_string()));
        if self.scsi_dev1.is_some() {
            cmds.push(("scsi1".to_string(), "Second SCSI controller (fullhouse only) — same subcommands as `scsi`".to_string()));
        }
        if let Some(hal2) = &self.hal2 { cmds.extend(hal2.register_commands()); }
        cmds
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        if cmd == "hpc3" {
            if args.first().copied() != Some("status") {
                return Err("Usage: hpc3 status".to_string());
            }
            let s = self.state.lock();
            let intstat_names: &[(u32, &str)] = &[
                (HPC3_INTSTAT_SCSI0_DEV, "SCSI0_DEV"),
                (HPC3_INTSTAT_SCSI0_DMA, "SCSI0_DMA"),
                (HPC3_INTSTAT_SCSI1_DEV, "SCSI1_DEV"),
                (HPC3_INTSTAT_SCSI1_DMA, "SCSI1_DMA"),
                (HPC3_INTSTAT_ENET_DEV,  "ENET_DEV"),
                (HPC3_INTSTAT_ENET_RX_DMA, "ENET_RX_DMA"),
                (HPC3_INTSTAT_ENET_TX_DMA, "ENET_TX_DMA"),
            ];
            let mut names = Vec::new();
            for (b, n) in intstat_names { if s.intstat & b != 0 { names.push(*n); } }
            let names_s = if names.is_empty() { "-".into() } else { names.join("|") };
            let _ = writeln!(writer, "HPC3 MISC state:");
            let _ = writeln!(writer, "  intstat   = {:08x}  [{}]", s.intstat, names_s);
            let _ = writeln!(writer, "  gio_misc  = {:08x}", s.gio_misc);
            let _ = writeln!(writer, "  eeprom    = {:08x}", s.eeprom_reg);
            return Ok(());
        }

        if cmd == "pdma" {
            if args.is_empty() {
                return Err("Usage: pdma <dump|status|chain> ...".to_string());
            }
            match args[0] {
                "dump" => {
                    let mask: u32 = match args.get(1).map(|s| *s) {
                        Some("on")   => 0xFFFF,
                        Some("off")  => 0x0000,
                        Some("hal")  => 0x00FF,
                        Some("scsi") => 0x0300,
                        Some("enet") => 0x0C00,
                        Some(s) => {
                            eval_const_expr(s).map(|v| v as u32)
                                .map_err(|e| format!("pdma dump: {}", e))?
                        }
                        None => return Err("Usage: pdma dump <on|off|hal|scsi|enet|MASK>".to_string()),
                    };
                    self.pdma_dump.store(mask, Ordering::Relaxed);
                    writeln!(writer, "PDMA dump mask = 0x{:04x}", mask).unwrap();
                    return Ok(());
                }
                "status" => {
                    writeln!(writer, "PDMA Channels:").unwrap();
                    for (i, chan) in self.pdma_channels.iter().enumerate() {
                        let c = chan.lock();
                        let type_str = if i <= 7 { "Generic" } else if i == 8 { "SCSI0" } else if i == 9 { "SCSI1" } else if i == 10 { "ENET RX" } else { "ENET TX" };
                        let dir_str = if i == 8 || i == 9 {
                            if (c.ctrl & SCSI_CTRL_DIR) != 0 { " DIR=OUT" } else { " DIR=IN" }
                        } else { "" };
                        writeln!(writer, "  [{:2}] {:8}: Active={} CBP={:08x} NBDP={:08x} BC={:08x} CRBDP={:08x} Endian={}{} CTRL={:02x} reads={} writes={} cbp_polls={}",
                            i, type_str, c.is_active(), c.cbp, c.nbdp, c.bc, c.crbdp, if c.endian { "Little" } else { "Big" }, dir_str, c.ctrl, c.dev_reads, c.dev_writes, c.cbp_polls).unwrap();
                    }
                    return Ok(());
                }
                "polls" => {
                    let ch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                    let c = self.pdma_channels.get(ch).ok_or("pdma polls: no such channel")?.lock();
                    writeln!(writer, "last CBP reads on channel {} (us, CBP, bytes read by the device):", ch).unwrap();
                    for (t, v, b) in c.cbp_log.iter() {
                        writeln!(writer, "  {:>12} {:08x} {}", t, v, b).unwrap();
                    }
                    return Ok(());
                }
                "chain" => {
                    let addr_str = args.get(1).ok_or_else(|| "Usage: pdma chain <addr>".to_string())?;
                    let mut addr = eval_const_expr(addr_str).map(|v| v as u32)
                        .map_err(|e| format!("pdma chain: {}", e))?;
                    // Find any channel that has sys_mem set
                    let mem_opt = self.pdma_channels.iter()
                        .find_map(|c| c.lock().sys_mem.clone());
                    let mem = mem_opt.ok_or_else(|| "pdma chain: no memory attached".to_string())?;
                    writeln!(writer, "PDMA descriptor chain from {:08x}:", addr).unwrap();
                    let mut idx = 0usize;
                    loop {
                        let cbp  = { let _r = mem.read32((addr + PDMA_DESC_CBP_OFFSET) as u64); if _r.is_ok() { let d = _r.data; d } else { break } };
                        let bc   = { let _r = mem.read32((addr + PDMA_DESC_BC_OFFSET) as u64); if _r.is_ok() { let d = _r.data; d } else { break } };
                        let nbdp = { let _r = mem.read32((addr + PDMA_DESC_NBP_OFFSET) as u64); if _r.is_ok() { let d = _r.data; d } else { break } };
                        let eox  = (bc & PDMA_DESC_EOX) != 0;
                        let eop  = (bc & PDMA_DESC_EOP) != 0;
                        let xie  = (bc & PDMA_DESC_XIE) != 0;
                        let rown = (bc & PDMA_DESC_ROWN) != 0;
                        let count = bc & 0x3FFF;
                        writeln!(writer, "  [{:3}] @{:08x}: CBP={:08x} BC={:08x} (cnt={:5} EOX={} EOP={} XIE={} ROWN={}) NBDP={:08x}",
                            idx, addr, cbp, bc, count, eox as u8, eop as u8, xie as u8, rown as u8, nbdp).unwrap();
                        if eox || nbdp == 0 {
                            break;
                        }
                        addr = nbdp;
                        idx += 1;
                        if idx > 1024 {
                            writeln!(writer, "  (truncated after 1024 descriptors)").unwrap();
                            break;
                        }
                    }
                    return Ok(());
                }
                _ => return Err("Usage: pdma <dump|status|chain> ...".to_string()),
            }
        }
        
        if cmd == "ioc" || cmd == "serial" || cmd == "pit" || cmd == "ps2" {
             return self.ioc.execute_command(cmd, args, writer);
        }
        if cmd == "seeq" || cmd == "net" {
             return self.seeq.execute_command(cmd, args, writer);
        }
        if cmd == "scsi" || cmd == "scsi0" || cmd == "cow" {
             return self.scsi_dev.execute_command(cmd, args, writer);
        }
        if cmd == "scsi1" {
            return match &self.scsi_dev1 {
                Some(dev) => dev.execute_command(cmd, args, writer),
                None => Err("scsi1: not available (guinness/Indy profile has a single SCSI controller)".to_string()),
            };
        }
        if cmd == "hal2" {
            if let Some(hal2) = &self.hal2 {
                return hal2.execute_command(cmd, args, writer);
            }
            let _ = writeln!(writer, "hal2: not available in headless mode");
            return Ok(());
        }
        if cmd == "rtc" {
             return self.rtc.execute_command(cmd, args, writer);
        }
        if cmd == "nveeprom" {
            if args.is_empty() {
                return Err("Usage: nveeprom <on|off|dump|r|w|save> ...".to_string());
            }
            match args[0] {
                "on" | "1" | "off" | "0" => {
                    let debug = matches!(args[0], "on" | "1");
                    self.eeprom.lock().set_debug(debug);
                    writeln!(writer, "NVRAM EEPROM debug {}", if debug { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }
                "dump" => {
                    let eeprom = self.eeprom.lock();
                    for (i, chunk) in eeprom.get_data().chunks(8).enumerate() {
                        let mut line = format!("  {:02X}:", i * 8);
                        for word in chunk { line.push_str(&format!(" {:04X}", word)); }
                        writeln!(writer, "{}", line).unwrap();
                    }
                    return Ok(());
                }
                "r" => {
                    let addr_str = args.get(1).ok_or_else(|| "Usage: nveeprom r <word 0-127>".to_string())?;
                    let addr: usize = usize::from_str_radix(addr_str.trim_start_matches("0x"), 16)
                        .or_else(|_| addr_str.parse())
                        .map_err(|_| format!("nveeprom r: \"{}\" is not a number", addr_str))?;
                    let eeprom = self.eeprom.lock();
                    let data = eeprom.get_data();
                    if addr >= data.len() {
                        return Err(format!("nveeprom r: word {} out of range (0-{})", addr, data.len() - 1));
                    }
                    writeln!(writer, "{:02X}: {:04X}", addr, data[addr]).unwrap();
                    return Ok(());
                }
                "w" => {
                    let addr_str = args.get(1).ok_or_else(|| "Usage: nveeprom w <word 0-127> <val>".to_string())?;
                    let val_str = args.get(2).ok_or_else(|| "Usage: nveeprom w <word 0-127> <val>".to_string())?;
                    let addr: usize = usize::from_str_radix(addr_str.trim_start_matches("0x"), 16)
                        .or_else(|_| addr_str.parse())
                        .map_err(|_| format!("nveeprom w: \"{}\" is not a number", addr_str))?;
                    let val: u16 = u16::from_str_radix(val_str.trim_start_matches("0x"), 16)
                        .or_else(|_| val_str.parse())
                        .map_err(|_| format!("nveeprom w: \"{}\" is not a number", val_str))?;
                    if addr >= 128 {
                        return Err(format!("nveeprom w: word {} out of range (0-127)", addr));
                    }
                    self.eeprom.lock().set_word(addr, val);
                    writeln!(writer, "{:02X}: {:04X}", addr, val).unwrap();
                    return Ok(());
                }
                "save" => {
                    let eeprom = self.eeprom.lock();
                    let filename = match args.get(1) {
                        Some(f) => f.to_string(),
                        None => match eeprom.path() {
                            Some(p) => p.to_string(),
                            None => return Err("nveeprom save: no default path (pass a filename)".to_string()),
                        },
                    };
                    match eeprom.save(&filename) {
                        Ok(_) => { writeln!(writer, "Saved NVRAM EEPROM to {}", filename).unwrap(); return Ok(()); },
                        Err(e) => return Err(format!("Failed to save NVRAM EEPROM: {}", e)),
                    }
                }
                _ => return Err("Usage: nveeprom <on|off|dump|r|w|save> ...".to_string()),
            }
        }
        Err("Command not found".to_string())
    }
}

impl BusDevice for Hpc3 {
    fn read8(&self, addr: u64) -> BusRead8 {
        let offset = (addr - HPC3_BASE as u64) as u32;

        // IP28's PROM copies the PBUS control bitfield with byte loads.
        if offset < 8 * PBUS_DMA_STRIDE {
            let idx = (offset / PBUS_DMA_STRIDE) as usize;
            let reg = (offset % PBUS_DMA_STRIDE) & !3;
            let lane = offset & 3;
            // Status/interrupt flags are in the low byte. Reading another
            // lane must not acknowledge an interrupt through PbusDmaOps::read.
            if reg == HPC3_PDMA_CTRL && lane != 3 { return BusRead8::ok(0); }
            let val = self.pdma_ops[idx].read(&mut self.pdma_channels[idx].lock(), reg);
            return BusRead8::ok((val >> ((3 - lane) * 8)) as u8);
        }

        // INT2 (fullhouse only, PBUS PIO channel 4) — forwarded to Ioc's
        // BusDevice::read8, which detects the INT2 window itself.
        if !self.guinness && (HPC3_INT2_BASE..HPC3_INT2_BASE + HPC3_INT2_SIZE).contains(&offset) {
            return self.ioc.read8(addr);
        }

        // IOC (0x59800 - 0x598FF) - forward 8-bit access directly to IOC
        if (HPC3_IOC_BASE..HPC3_IOC_BASE + 0x104).contains(&offset) {
            return self.ioc.read8(addr);
        }

        // SCSI Registers — two aliases: 0x40000 (IRIX) and 0x44000 (HPC3 spec/OpenBSD)
        if (SCSI_REG_BASE..SCSI_REG_BASE + 8).contains(&offset)
            || (SCSI_REG_BASE2..SCSI_REG_BASE2 + 8).contains(&offset) {
            let base = if offset >= SCSI_REG_BASE2 { SCSI_REG_BASE2 } else { SCSI_REG_BASE };
            let idx = (offset - base) >> 2;
            return self.scsi_dev.read(idx);
        }

        // Fullhouse only: second WD33C93A at SCSI1_REG_BASE
        if !self.guinness && (SCSI1_REG_BASE..SCSI1_REG_BASE + 8).contains(&offset) {
            let idx = (offset - SCSI1_REG_BASE) >> 2;
            if let Some(dev) = &self.scsi_dev1 {
                return dev.read(idx);
            }
            return BusRead8::ok(0);
        }

        // SEEQ8003 Ethernet Controller (0x54000 - 0x5401F) - 8-bit device
        if (SEEQ_BASE..SEEQ_BASE + 0x20).contains(&offset) {
            let idx = (offset - SEEQ_BASE) >> 2;
            return self.seeq.read(idx);
        }

        // FIFOs (0x28000 - 0x2FFFF)
        if (SCSI0_FIFO_BASE..MISC_BASE).contains(&offset) {
            if offset < SCSI1_FIFO_BASE {
                // SCSI0 FIFO
                return BusRead8::ok(0); // Placeholder
            } else if offset < ENET_RX_FIFO_BASE {
                // SCSI1 FIFO
                return BusRead8::ok(0); // Placeholder
            } else if offset < ENET_TX_FIFO_BASE {
                // ENET RX FIFO
                return BusRead8::ok(0); // Placeholder
            } else {
                // ENET TX FIFO (Write Only)
                return BusRead8::ok(0);
            }
        }

        // PBUS BBRAM (RTC) - 8-bit access with sparse packing
        // RTC range: 0x60000-0x7ffff (128KB for 32K RTC, or 0x60000-0x67fff for 8K RTC)
        // Sparse packing: one byte per dword, only bottom byte lane is valid (offset & 3 == 3)
        if (PBUS_BBRAM..PBUS_BBRAM + 0x20000).contains(&offset) {
            let rtc_offset = (offset - PBUS_BBRAM) as u32;
            // Check if this is the valid byte lane (bottom byte of dword in big-endian)
            if (rtc_offset & 3) != 3 {
                return BusRead8::ok(0);
            }
            // Sparse decode: addr/4 gives actual byte index in RTC
            let byte_index = rtc_offset >> 2;
            return self.rtc.read8(byte_index as u64);
        }

        // MISC_EEPROM_DATA, byte lane (PROM bit-bangs this register 8 bits
        // at a time, not just via 32-bit access — offset & 3 == 3 is the
        // bottom byte of the big-endian word, same convention as RTC above).
        if (MISC_BASE..MISC_BASE + 0x1000).contains(&offset) && offset - MISC_BASE == MISC_EEPROM_DATA + 3 {
            let state = self.state.lock();
            let mut val = state.eeprom_reg as u8;
            if self.eeprom.lock().get_do() {
                val |= 1 << 4;
            } else {
                val &= !(1 << 4);
            }
            return BusRead8::ok(val);
        }

        let state = self.state.lock();

        // PBUS PIO (0x58000 - 0x5BFFF)
        if (PBUS_PIO_BASE..PBUS_CFGDMA_BASE).contains(&offset) {
            let channel = (offset - PBUS_PIO_BASE) / PBUS_PIO_STRIDE;
            dlog_dev!(LogModule::Hpc3, "HPC3: Read8 PBUS PIO Channel {} (offset {:05x})", channel, offset);
            let idx = ((offset - PBUS_PIO_BASE) >> 2) as usize;
            if idx < state.pbus_pio.len() {
                return BusRead8::ok(state.pbus_pio[idx] as u8);
            }
            return BusRead8::ok(0);
        }

        // All other registers require 32-bit access
        dlog_dev!(LogModule::Hpc3, "HPC3: Unexpected read8 at offset {:05x} (addr {:08x})", offset, addr);
        BusRead8::ok(0)
    }

    fn write8(&self, addr: u64, val: u8) -> u32 {
        let offset = (addr - HPC3_BASE as u64) as u32;

        // IP28's PROM emits four sb instructions for its DMA control struct.
        // Use the addressed big-endian byte lane, without reading CTRL back:
        // its read status and write strobes have different bit assignments.
        if offset < 8 * PBUS_DMA_STRIDE {
            let idx = (offset / PBUS_DMA_STRIDE) as usize;
            let reg = (offset % PBUS_DMA_STRIDE) & !3;
            let lane = offset & 3;
            let shift = (3 - lane) * 8;
            let mut chan = self.pdma_channels[idx].lock();
            match reg {
                HPC3_PDMA_CBP => chan.cbp = (chan.cbp & !(0xff << shift)) | ((val as u32) << shift),
                HPC3_PDMA_NBDP => chan.nbdp = (chan.nbdp & !(0xff << shift)) | ((val as u32) << shift),
                HPC3_PDMA_CTRL if lane == 3 => self.pdma_ops[idx].write(&mut chan, reg, val as u32),
                // Upper CTRL lanes configure the FIFO, which is not modeled.
                _ => {}
            }
            return BUS_OK;
        }

        // INT2 (fullhouse only, PBUS PIO channel 4) — forwarded to Ioc's
        // BusDevice::write8, which detects the INT2 window itself.
        if !self.guinness && (HPC3_INT2_BASE..HPC3_INT2_BASE + HPC3_INT2_SIZE).contains(&offset) {
            return self.ioc.write8(addr, val);
        }

        // IOC (0x59800 - 0x598FF) - forward 8-bit access directly to IOC
        if (HPC3_IOC_BASE..HPC3_IOC_BASE + 0x104).contains(&offset) {
            return self.ioc.write8(addr, val);
        }

        // SCSI Registers — two aliases: 0x40000 (IRIX) and 0x44000 (HPC3 spec/OpenBSD)
        if (SCSI_REG_BASE..SCSI_REG_BASE + 8).contains(&offset)
            || (SCSI_REG_BASE2..SCSI_REG_BASE2 + 8).contains(&offset) {
            let base = if offset >= SCSI_REG_BASE2 { SCSI_REG_BASE2 } else { SCSI_REG_BASE };
            let idx = (offset - base) >> 2;
            return self.scsi_dev.write(idx, val);
        }

        // Fullhouse only: second WD33C93A at SCSI1_REG_BASE
        if !self.guinness && (SCSI1_REG_BASE..SCSI1_REG_BASE + 8).contains(&offset) {
            let idx = (offset - SCSI1_REG_BASE) >> 2;
            if let Some(dev) = &self.scsi_dev1 {
                return dev.write(idx, val);
            }
            return BUS_OK;
        }

        // SEEQ8003 Ethernet Controller (0x54000 - 0x5401F) - 8-bit device
        if (SEEQ_BASE..SEEQ_BASE + 0x20).contains(&offset) {
            let idx = (offset - SEEQ_BASE) >> 2;
            return self.seeq.write(idx, val);
        }

        // FIFOs (0x28000 - 0x2FFFF)
        if (SCSI0_FIFO_BASE..MISC_BASE).contains(&offset) {
            if offset < SCSI1_FIFO_BASE {
                // SCSI0 FIFO
            } else if offset < ENET_RX_FIFO_BASE {
                // SCSI1 FIFO
            } else if offset < ENET_TX_FIFO_BASE {
                // ENET RX FIFO (Read Only) - no-op
            } else {
                // ENET TX FIFO - DMA only, no PIO path
            }
            return BUS_OK;
        }

        // PBUS BBRAM (RTC) - 8-bit access with sparse packing
        if (PBUS_BBRAM..PBUS_BBRAM + 0x20000).contains(&offset) {
            let rtc_offset = (offset - PBUS_BBRAM) as u32;
            // Check if this is the valid byte lane (bottom byte of dword in big-endian)
            if (rtc_offset & 3) != 3 {
                return BUS_OK; // Ignore writes to invalid byte lanes
            }
            // Sparse decode: addr/4 gives actual byte index in RTC
            let byte_index = rtc_offset >> 2;
            return self.rtc.write8(byte_index as u64, val);
        }

        // MISC_EEPROM_DATA, byte lane — see matching comment in read8.
        if (MISC_BASE..MISC_BASE + 0x1000).contains(&offset) && offset - MISC_BASE == MISC_EEPROM_DATA + 3 {
            let mut state = self.state.lock();
            state.eeprom_reg = (state.eeprom_reg & !0xFF) | val as u32;
            let mut eeprom = self.eeprom.lock();
            eeprom.set_cs((val & (1 << 1)) != 0);
            eeprom.set_di((val & (1 << 3)) != 0);
            eeprom.set_sk((val & (1 << 2)) != 0);
            return BUS_OK;
        }

        let mut state = self.state.lock();

        // PBUS PIO (0x58000 - 0x5BFFF)
        if (PBUS_PIO_BASE..PBUS_CFGDMA_BASE).contains(&offset) {
            let channel = (offset - PBUS_PIO_BASE) / PBUS_PIO_STRIDE;
            dlog_dev!(LogModule::Hpc3, "HPC3: Write8 PBUS PIO Channel {} (offset {:05x}) val {:02x}", channel, offset, val);
            let idx = ((offset - PBUS_PIO_BASE) >> 2) as usize;
            if idx < state.pbus_pio.len() {
                state.pbus_pio[idx] = val as u32;
            }
            return BUS_OK;
        }

        // All other registers require 32-bit access
        dlog_dev!(LogModule::Hpc3, "HPC3: Unexpected write8 at offset {:05x} (addr {:08x}) val={:02x}", offset, addr, val);
        BUS_OK
    }

    fn read32(&self, addr: u64) -> BusRead32 {
        let offset = (addr - HPC3_BASE as u64) as u32;

        // INT2 (fullhouse only, PBUS PIO channel 4) - should not use 32-bit access, but allow for legacy
        if !self.guinness && (HPC3_INT2_BASE..HPC3_INT2_BASE + HPC3_INT2_SIZE).contains(&offset) {
            let r = self.ioc.read8(addr);
            return if r.is_ok() { BusRead32::ok(r.data as u32) } else { BusRead32 { status: r.status, data: 0 } };
        }

        // IOC (0x59800 - 0x598FF) - should not use 32-bit access, but allow for legacy
        if (HPC3_IOC_BASE..HPC3_IOC_BASE + 0x104).contains(&offset) {
            return self.ioc.read32(addr);
        }

        // PBUS DMA (0-7), SCSI (0-1), Ethernet RX/TX
        if offset < 0x18000 {
            let idx = (offset / 0x2000) as usize;
            let reg = offset % 0x2000;
            let val = self.pdma_ops[idx].read(&mut self.pdma_channels[idx].lock(), reg);
            dlog_dev!(LogModule::Hpc3, "HPC3: Read PDMA addr {:08x} = {:08x}", addr, val);
//            if idx == 0 {
                //eprintln!("HPC3: PDMA[0] read reg={:04x} val={:08x}", reg, val);
            //}
            return BusRead32::ok(val);
        }

        // Enet extra registers (0x18000-0x1a007): crbdp, cpfxbdp, ppfxbdp
        match offset {
            ENET_CRBDP   => return BusRead32::ok(self.pdma_channels[10].lock().crbdp),
            ENET_CPFXBDP => return BusRead32::ok(self.pdma_channels[11].lock().cpfxbdp),
            ENET_PPFXBDP => return BusRead32::ok(self.pdma_channels[11].lock().ppfxbdp),
            _ => {}
        }

        // FIFOs (0x28000 - 0x2FFFF) - these should use 8-bit access but allow 32-bit for legacy
        if (SCSI0_FIFO_BASE..MISC_BASE).contains(&offset) {
            return BusRead32::ok(0); // Placeholder
        }

        // SCSI Registers — two aliases: 0x40000 (IRIX) and 0x44000 (HPC3 spec/OpenBSD);
        // 0x48000 is fullhouse's second controller (SCSI1_REG_BASE)
        if (SCSI_REG_BASE..SCSI_REG_BASE + 8).contains(&offset)
            || (SCSI_REG_BASE2..SCSI_REG_BASE2 + 8).contains(&offset)
            || (!self.guinness && (SCSI1_REG_BASE..SCSI1_REG_BASE + 8).contains(&offset)) {
            let r = self.read8(addr);
            return if r.is_ok() { BusRead32::ok(r.data as u32) } else { BusRead32 { status: r.status, data: 0 } };
        }

        // SEEQ8003 Ethernet Controller (0x54000 - 0x5401F) - 8-bit device, convert to 32-bit
        if (SEEQ_BASE..SEEQ_BASE + 0x20).contains(&offset) {
            let r = self.read8(addr);
            return if r.is_ok() { BusRead32::ok(r.data as u32) } else { BusRead32 { status: r.status, data: 0 } };
        }

        // Misc Registers (0x30000 - 0x30014)
        if (MISC_BASE..MISC_BASE + 0x1000).contains(&offset) {
            let state = self.state.lock();
            match offset - MISC_BASE {
                MISC_INTSTAT => return BusRead32::ok(state.intstat),
                MISC_GIO_MISC => return BusRead32::ok(state.gio_misc),
                MISC_EEPROM_DATA => {
                    let mut val = state.eeprom_reg;
                    if self.eeprom.lock().get_do() {
                        val |= 1 << 4;
                    } else {
                        val &= !(1 << 4);
                    }
                    return BusRead32::ok(val);
                }
                MISC_INTSTAT_BUG => return BusRead32::ok(state.intstat), // Mirror?
                MISC_GIO_BUS_ERROR => {
                    dlog_dev!(LogModule::Hpc3, "HPC3: Read GIO_BUS_ERROR at {:08x}", addr);
                    return BusRead32::ok(0);
                }
                _ => {
                    dlog_dev!(LogModule::Hpc3, "HPC3: Read Misc addr {:08x}", addr);
                    return BusRead32::ok(0);
                }
            }
        }

        // PBUS PIO Config
        if (PBUS_CFGPIO_BASE..PBUS_PROM_WE).contains(&offset) {
            let idx = (offset - PBUS_CFGPIO_BASE) / PBUS_CFGPIO_STRIDE;
            dlog_dev!(LogModule::Hpc3, "HPC3: Read PBUS PIO Config[{}] at {:08x}", idx, addr);
            return BusRead32::ok(self.pdma_ops[idx as usize].read_piocfg(&mut self.pdma_channels[idx as usize].lock()));
        }

        // PBUS DMA Config
        if (PBUS_CFGDMA_BASE..PBUS_CFGPIO_BASE).contains(&offset) {
            let idx = (offset - PBUS_CFGDMA_BASE) / PBUS_CFGDMA_STRIDE;
            dlog_dev!(LogModule::Hpc3, "HPC3: Read PBUS DMA Config[{}] at {:08x}", idx, addr);
            return BusRead32::ok(self.pdma_ops[idx as usize].read_dmacfg(&mut self.pdma_channels[idx as usize].lock()));
        }

        // Channel 0: HAL2 (0x58000 - 0x583FF)
        if (HAL2_BASE..HAL2_BASE + 0x400).contains(&offset) {
            if let Some(hal2) = &self.hal2 {
                let r = hal2.read(offset - HAL2_BASE);
                return if r.is_ok() { BusRead32::ok(r.data as u32) } else { BusRead32 { status: r.status, data: 0 } };
            }
            return BusRead32::ok(hal2_absent_read(offset - HAL2_BASE) as u32);
        }

        // PBUS PIO (0x58000 - 0x5BFFF)
        if (PBUS_PIO_BASE..PBUS_CFGDMA_BASE).contains(&offset) {
            let state = self.state.lock();
            let channel = (offset - PBUS_PIO_BASE) / PBUS_PIO_STRIDE;
            dlog_dev!(LogModule::Hpc3, "HPC3: Read32 PBUS PIO Channel {} (offset {:05x})", channel, offset);
            let idx = ((offset - PBUS_PIO_BASE) >> 2) as usize;
            if idx < state.pbus_pio.len() {
                return BusRead32::ok(state.pbus_pio[idx]);
            }
            return BusRead32::ok(0);
        }

        // PBUS BBRAM (RTC) - 32-bit access with sparse packing
        // RTC range: 0x60000-0x7ffff (128KB for 32K RTC, or 0x60000-0x67fff for 8K RTC)
        // Sparse packing: one byte per dword in bottom byte lane
        if (PBUS_BBRAM..PBUS_BBRAM + 0x20000).contains(&offset) {
            let rtc_offset = (offset - PBUS_BBRAM) as u32;
            // For 32-bit reads, only the bottom byte is valid
            let byte_index = rtc_offset >> 2;
            let r = self.rtc.read8(byte_index as u64);
            return if r.is_ok() { BusRead32::ok(r.data as u32) } else { BusRead32 { status: r.status, data: 0 } };
        }

        dlog_dev!(LogModule::Hpc3, "HPC3: Read addr {:08x}", addr);
        BusRead32::ok(0)
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        let offset = (addr - HPC3_BASE as u64) as u32;

        // INT2 (fullhouse only, PBUS PIO channel 4) - should not use 32-bit access, but allow for legacy
        if !self.guinness && (HPC3_INT2_BASE..HPC3_INT2_BASE + HPC3_INT2_SIZE).contains(&offset) {
            return self.ioc.write8(addr, (val & 0xFF) as u8);
        }

        // IOC (0x59800 - 0x598FF) - should not use 32-bit access, but allow for legacy
        if (HPC3_IOC_BASE..HPC3_IOC_BASE + 0x104).contains(&offset) {
            return self.ioc.write32(addr, val);
        }

        // ENET_RX_RESET (0x15014) — handled at Hpc3 level to access both channels + seeq
        if offset == ENET_RX_BASE + ENET_RX_RESET {
            let mut rx = self.pdma_channels[10].lock();
            let old_reset = (rx.misc & ENET_RX_RESET_CH_RESET) != 0;
            let new_reset = (val   & ENET_RX_RESET_CH_RESET) != 0;
            // CLRINT is a write strobe — strip it; INTPEND lives in SeeqState, not misc.
            rx.misc = val & !(ENET_RX_RESET_CLRINT | ENET_RX_RESET_INTPEND);
            dlog_dev!(LogModule::Hpc3, "ENET_RX_RESET: write val={:08x} old_reset={} new_reset={} rx_active={} clrint={}",
                val, old_reset, new_reset, rx.is_active(), (val & ENET_RX_RESET_CLRINT) != 0);
            if new_reset && !old_reset {
                // Rising edge: deactivate both channels and assert SEEQ reset
                rx.set_active(false);
                drop(rx);
                self.pdma_channels[11].lock().set_active(false);
                self.seeq.assert_reset();
            } else if !new_reset && old_reset {
                // Falling edge: deassert SEEQ reset (clears SEEQ registers)
                drop(rx);
                self.seeq.deassert_reset();
            } else {
                drop(rx);
            }
            if (val & ENET_RX_RESET_CLRINT) != 0 {
                self.seeq.reset_interrupt();
                // CLRINT clears every Ethernet interrupt except RBO
                // (established, HPC3 specification).
                if self.pdma_channels[10].lock().ctrl & ENET_RX_CTRL_RBO != 0 {
                    self.seeq.raise_dma_interrupt();
                }
            }
            return BUS_OK;
        }

        // PBUS DMA (0-7), SCSI (0-1), Ethernet RX/TX
        if offset < 0x18000 {
            dlog_dev!(LogModule::Hpc3, "HPC3: Write PDMA addr {:08x} = {:08x}", addr, val);
            let idx = (offset / 0x2000) as usize;
            let reg = offset % 0x2000;
            self.pdma_ops[idx].write(&mut self.pdma_channels[idx].lock(), reg, val);
            return BUS_OK;
        }

        // Enet extra registers (0x18000-0x1a007): crbdp, cpfxbdp, ppfxbdp
        match offset {
            ENET_CRBDP   => { self.pdma_channels[10].lock().crbdp   = val; return BUS_OK; }
            ENET_CPFXBDP => { self.pdma_channels[11].lock().cpfxbdp = val; return BUS_OK; }
            ENET_PPFXBDP => { self.pdma_channels[11].lock().ppfxbdp = val; return BUS_OK; }
            _ => {}
        }

        // FIFOs (0x28000 - 0x2FFFF) - these should use 8-bit access
        if (SCSI0_FIFO_BASE..MISC_BASE).contains(&offset) {
            // Fall back to writing low byte
            return self.write8(addr, val as u8);
        }

        // SCSI Registers — two aliases: 0x40000 (IRIX) and 0x44000 (HPC3 spec/OpenBSD);
        // 0x48000 is fullhouse's second controller (SCSI1_REG_BASE)
        if (SCSI_REG_BASE..SCSI_REG_BASE + 8).contains(&offset)
            || (SCSI_REG_BASE2..SCSI_REG_BASE2 + 8).contains(&offset)
            || (!self.guinness && (SCSI1_REG_BASE..SCSI1_REG_BASE + 8).contains(&offset)) {
            return self.write8(addr, val as u8);
        }

        // SEEQ8003 Ethernet Controller (0x54000 - 0x5401F) - 8-bit device
        if (SEEQ_BASE..SEEQ_BASE + 0x20).contains(&offset) {
            return self.write8(addr, val as u8);
        }

        // Misc Registers
        if (MISC_BASE..MISC_BASE + 0x1000).contains(&offset) {
            let reg_off = offset - MISC_BASE;
            match reg_off {
                MISC_GIO_MISC => {
                    self.state.lock().gio_misc = val;
                    dlog_dev!(LogModule::Hpc3, "HPC3: GIO_MISC ({:08x}) = {:08x}", addr, val);
                }
                MISC_EEPROM_DATA => {
                    let mut state = self.state.lock();
                    state.eeprom_reg = val;
                    let mut eeprom = self.eeprom.lock();
                    // Bit 1: CS
                    eeprom.set_cs((val & (1 << 1)) != 0);
                    // Bit 3: DATO (Data to EEPROM)
                    eeprom.set_di((val & (1 << 3)) != 0);
                    // Bit 2: CLK
                    eeprom.set_sk((val & (1 << 2)) != 0);
                }
                MISC_INTSTAT | MISC_INTSTAT_BUG => {
                    // W1C — writing 1 to a bit clears it.  For SCSI/ENET DMA
                    // bits, also clear the per-channel PDMA INT flag and call
                    // set_dma_interrupt(false) so the IOC line drops.
                    let cleared = {
                        let mut state = self.state.lock();
                        let prev = state.intstat;
                        state.intstat &= !val;
                        prev & val
                    };
                    for (bit, ch_idx) in &[
                        (HPC3_INTSTAT_SCSI0_DMA, HPC3_PDMA_CHAN_SCSI0 as usize),
                        (HPC3_INTSTAT_SCSI1_DMA, HPC3_PDMA_CHAN_SCSI1 as usize),
                        (HPC3_INTSTAT_ENET_RX_DMA, HPC3_PDMA_CHAN_ENET_RX as usize),
                        (HPC3_INTSTAT_ENET_TX_DMA, HPC3_PDMA_CHAN_ENET_TX as usize),
                    ] {
                        if cleared & *bit != 0 {
                            let mut chan = self.pdma_channels[*ch_idx].lock();
                            if chan.ctrl & PDMA_CTRL_INT != 0 {
                                chan.ctrl &= !PDMA_CTRL_INT;
                                if let Some(irq) = &chan.irq {
                                    irq.set(false);
                                }
                            }
                        }
                    }
                    dlog_dev!(LogModule::Hpc3, "HPC3: MISC_INTSTAT W1C val={:08x} cleared={:08x}", val, cleared);
                }
                _ => {}
            }
            return BUS_OK;
        }

        // PBUS PIO Config
        if (PBUS_CFGPIO_BASE..PBUS_PROM_WE).contains(&offset) {
            let idx = (offset - PBUS_CFGPIO_BASE) / PBUS_CFGPIO_STRIDE;
            dlog_dev!(LogModule::Hpc3, "HPC3: PBUS PIO Config[{}] ({:08x}) = {:08x}", idx, addr, val);
            self.pdma_ops[idx as usize].write_piocfg(&mut self.pdma_channels[idx as usize].lock(), val);
            return BUS_OK;
        }

        // PBUS DMA Config
        if (PBUS_CFGDMA_BASE..PBUS_CFGPIO_BASE).contains(&offset) {
            let idx = (offset - PBUS_CFGDMA_BASE) / PBUS_CFGDMA_STRIDE;
            dlog_dev!(LogModule::Hpc3, "HPC3: PBUS DMA Config[{}] ({:08x}) = {:08x}", idx, addr, val);
            self.pdma_ops[idx as usize].write_dmacfg(&mut self.pdma_channels[idx as usize].lock(), val);
            return BUS_OK;
        }

        // Channel 0: HAL2 (0x58000 - 0x583FF)
        if (HAL2_BASE..HAL2_BASE + 0x400).contains(&offset) {
            if let Some(hal2) = &self.hal2 {
                return hal2.write(offset - HAL2_BASE, val as u16);
            }
            return BUS_OK;
        }

        // PBUS PIO (0x58000 - 0x5BFFF)
        if (PBUS_PIO_BASE..PBUS_CFGDMA_BASE).contains(&offset) {
            let mut state = self.state.lock();
            let channel = (offset - PBUS_PIO_BASE) / PBUS_PIO_STRIDE;
            dlog_dev!(LogModule::Hpc3, "HPC3: Write32 PBUS PIO Channel {} (offset {:05x}) val {:08x}", channel, offset, val);
            let idx = ((offset - PBUS_PIO_BASE) >> 2) as usize;
            if idx < state.pbus_pio.len() {
                state.pbus_pio[idx] = val;
            }
            return BUS_OK;
        }

        // PBUS BBRAM (RTC) - 32-bit access with sparse packing
        // RTC range: 0x60000-0x7ffff (128KB for 32K RTC, or 0x60000-0x67fff for 8K RTC)
        // Sparse packing: one byte per dword in bottom byte lane
        if (PBUS_BBRAM..PBUS_BBRAM + 0x20000).contains(&offset) {
            let rtc_offset = (offset - PBUS_BBRAM) as u32;
            // For 32-bit writes, extract the bottom byte
            let byte_index = rtc_offset >> 2;
            let byte_val = (val & 0xFF) as u8;
            return self.rtc.write8(byte_index as u64, byte_val);
        }

        // Log other writes
        dlog_dev!(LogModule::Hpc3, "HPC3: Write addr {:08x} = {:08x}", addr, val);

        BUS_OK
    }

    fn read16(&self, addr: u64) -> BusRead16 {
        let offset = (addr - HPC3_BASE as u64) as u32;

        // HAL2 registers are 16-bit; forward directly
        if (HAL2_BASE..HAL2_BASE + 0x400).contains(&offset) {
            if let Some(hal2) = &self.hal2 {
                return hal2.read(offset - HAL2_BASE);
            }
            return BusRead16::ok(hal2_absent_read(offset - HAL2_BASE));
        }

        // IOC EXT_IO (fullhouse only, 0x59900 = HPC3_IOC_BASE + 0x100)
        if !self.guinness && (HPC3_IOC_BASE..HPC3_IOC_BASE + 0x104).contains(&offset) {
            return self.ioc.read16(addr);
        }

        BusRead16::ok(0)
    }

    fn write16(&self, addr: u64, val: u16) -> u32 {
        let offset = (addr - HPC3_BASE as u64) as u32;

        // HAL2 registers are 16-bit; forward directly
        if (HAL2_BASE..HAL2_BASE + 0x400).contains(&offset) {
            if let Some(hal2) = &self.hal2 {
                return hal2.write(offset - HAL2_BASE, val);
            }
            return BUS_OK;
        }

        // IOC EXT_IO (fullhouse only, 0x59900 = HPC3_IOC_BASE + 0x100)
        if !self.guinness && (HPC3_IOC_BASE..HPC3_IOC_BASE + 0x104).contains(&offset) {
            return self.ioc.write16(addr, val);
        }

        BUS_OK
    }
}

// ============================================================================
// Resettable + Saveable for Hpc3
// ============================================================================

impl Resettable for Hpc3 {
    fn power_on(&self) {
        {
            let mut state = self.state.lock();
            state.intstat = 0;
            state.gio_misc = 0;
            state.eeprom_reg = 0;
            state.pbus_pio = [0; 0x1000];
        }
        // Reset all DMA channels to their power-on state.
        // We preserve the configuration fields (active_mask, sys_mem, irq, etc.)
        // that are wired at construction time, and only reset the transfer-state registers.
        for chan_arc in &self.pdma_channels {
            let mut chan = chan_arc.lock();
            chan.cbp = 0;
            chan.nbdp = 0x80000000;
            chan.bc = 0;
            chan.ctrl = 0;
            chan.gio = 0;
            chan.dev = 0;
            chan.eox = false;
            chan.eop = false;
            chan.xie = false;
            chan.crbdp = 0;
            chan.cpfxbdp = 0;
            chan.ppfxbdp = 0;
            chan.tx_new_packet = true;
            chan.rx_filled.clear();
            chan.last_rx_ctrl = 0xFFFFFFFF;
            chan.transaction_id = 0;
            chan.bytes_transferred = 0;
            chan.dump_file = None;
        }
    }
}

/// Save one PdmaChannel's transfer-state registers to a TOML table.
fn save_pdma_channel(chan: &PdmaChannel) -> toml::Value {
    let mut tbl = toml::map::Map::new();
    tbl.insert("cbp".into(),    hex_u32(chan.cbp));
    tbl.insert("nbdp".into(),   hex_u32(chan.nbdp));
    tbl.insert("bc".into(),     hex_u32(chan.bc));
    tbl.insert("ctrl".into(),   hex_u32(chan.ctrl));
    tbl.insert("gio".into(),    hex_u32(chan.gio));
    tbl.insert("dev".into(),    hex_u32(chan.dev));
    tbl.insert("dmacfg".into(), hex_u32(chan.dmacfg));
    tbl.insert("piocfg".into(), hex_u32(chan.piocfg));
    tbl.insert("crbdp".into(),   hex_u32(chan.crbdp));
    tbl.insert("cpfxbdp".into(), hex_u32(chan.cpfxbdp));
    tbl.insert("ppfxbdp".into(), hex_u32(chan.ppfxbdp));
    tbl.insert("tx_new_packet".into(), toml::Value::Boolean(chan.tx_new_packet));
    // Latched. `fetch_descriptor` does derive eox/eop/xie from `bc`, but a
    // raw BC register write updates only eox and xie and leaves eop stale, so
    // `bc` alone does not reconstruct them.
    //
    // The other three are not derivable at all. `PDMA_CTRL_LITTLE` and
    // `PDMA_CTRL_ACT` are both `1 << 1`, and `PbusDmaOps::write` folds the
    // written CTRL value into `endian` without ever storing `ctrl`, so on
    // channels 0-7 a derivation of `endian` from `ctrl` is just `is_active()`.
    // `width_16` has the same problem on 8-9: the `SCSI_DMACFG` register write
    // path updates `dmacfg` alone, only `write_dmacfg` also updates `width_16`.
    tbl.insert("eox".into(),       toml::Value::Boolean(chan.eox));
    tbl.insert("eop".into(),       toml::Value::Boolean(chan.eop));
    tbl.insert("xie".into(),       toml::Value::Boolean(chan.xie));
    tbl.insert("width_16".into(),  toml::Value::Boolean(chan.width_16));
    tbl.insert("even_high".into(), toml::Value::Boolean(chan.even_high));
    tbl.insert("endian".into(),    toml::Value::Boolean(chan.endian));
    toml::Value::Table(tbl)
}

/// Restore one PdmaChannel's transfer-state registers from a TOML table.
///
/// The six latched booleans are newer than the record, so a snapshot taken
/// before them has no such keys and each field keeps the value it already has:
/// `false` for `eox`/`eop`/`xie`, which `power_on` clears ahead of the
/// restore, and the pre-restore run's value for `width_16`/`even_high`/`endian`,
/// which `power_on` leaves alone.
fn load_pdma_channel(chan: &mut PdmaChannel, v: &toml::Value) {
    macro_rules! ldu32 { ($f:ident) => {
        if let Some(x) = get_field(v, stringify!($f)) { chan.$f = toml_u32(x).unwrap_or(chan.$f); }
    }}
    macro_rules! ldbool { ($f:ident) => {
        if let Some(x) = get_field(v, stringify!($f)) { chan.$f = toml_bool(x).unwrap_or(chan.$f); }
    }}
    ldu32!(cbp); ldu32!(nbdp); ldu32!(bc); ldu32!(ctrl);
    ldu32!(gio); ldu32!(dev); ldu32!(dmacfg); ldu32!(piocfg);
    ldu32!(crbdp); ldu32!(cpfxbdp); ldu32!(ppfxbdp);
    if let Some(x) = get_field(v, "tx_new_packet") { chan.tx_new_packet = toml_bool(x).unwrap_or(true); }
    ldbool!(eox); ldbool!(eop); ldbool!(xie);
    ldbool!(width_16); ldbool!(even_high); ldbool!(endian);
}

impl Saveable for Hpc3 {
    fn save_state(&self) -> toml::Value {
        let state = self.state.lock();
        let mut tbl = toml::map::Map::new();

        tbl.insert("intstat".into(),    hex_u32(state.intstat));
        tbl.insert("gio_misc".into(),   hex_u32(state.gio_misc));
        tbl.insert("eeprom_reg".into(), hex_u32(state.eeprom_reg));
        tbl.insert("pbus_pio".into(), u32_slice_to_toml(&state.pbus_pio));

        let chans: Vec<toml::Value> = self.pdma_channels.iter().map(|c| {
            save_pdma_channel(&c.lock())
        }).collect();
        tbl.insert("pdma_channels".into(), toml::Value::Array(chans));

        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut state = self.state.lock();
        if let Some(x) = get_field(v, "intstat")    { state.intstat    = toml_u32(x).unwrap_or(0); }
        if let Some(x) = get_field(v, "gio_misc")   { state.gio_misc   = toml_u32(x).unwrap_or(0); }
        if let Some(x) = get_field(v, "eeprom_reg") { state.eeprom_reg = toml_u32(x).unwrap_or(0); }
        if let Some(r) = get_field(v, "pbus_pio")   { load_u32_slice(r, &mut state.pbus_pio); }
        drop(state);

        if let Some(toml::Value::Array(chans)) = get_field(v, "pdma_channels") {
            for (i, cv) in chans.iter().enumerate() {
                if i >= self.pdma_channels.len() { break; }
                load_pdma_channel(&mut self.pdma_channels[i].lock(), cv);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// INTSTAT as a controller: each SCSI channel's chip and DMA bits are
    /// ORed onto its IOC line; a chip-INT ack also settles the paired PDMA
    /// half; the SEEQ's level goes to Ethernet as it is.
    #[test]
    fn intstat_ors_scsi_halves_and_a_chip_ack_settles_the_dma_half() {
        use crate::irq::LevelProbe;
        let (s0, s0_line) = LevelProbe::line();
        let (s1, s1_line) = LevelProbe::line();
        let (en, en_line) = LevelProbe::line();
        let dump = Arc::new(AtomicU32::new(0));
        let chans = [
            Arc::new(Mutex::new(PdmaChannel::new(8, dump.clone()))),
            Arc::new(Mutex::new(PdmaChannel::new(9, dump))),
        ];
        let state = Arc::new(Mutex::new(Hpc3State { intstat: 0, gio_misc: 0, eeprom_reg: 0, pbus_pio: [0; 0x1000] }));
        let irqs = Arc::new(Hpc3Interrupts {
            state: state.clone(),
            scsi: [s0_line, s1_line],
            ethernet: en_line,
            scsi_dma: chans.clone(),
        });
        let (dev0, dma0) = (irqs.line(HPC3_INTSTAT_SCSI0_DEV), irqs.line(HPC3_INTSTAT_SCSI0_DMA));

        dma0.set(true);
        assert!(s0.level() && !s1.level());
        dev0.set(true);
        dma0.set(false);
        assert!(s0.level(), "the chip half still holds the line");

        dma0.set(true);
        chans[0].lock().ctrl |= PDMA_CTRL_INT;
        dev0.set(false);
        assert!(!s0.level(), "the chip ack settles both halves");
        assert_eq!(state.lock().intstat & SCSI_BITS[0], 0);
        assert_eq!(chans[0].lock().ctrl & PDMA_CTRL_INT, 0);

        irqs.line(HPC3_INTSTAT_SCSI1_DMA).set(true);
        assert!(s1.level() && !s0.level());

        state.lock().intstat |= HPC3_INTSTAT_ENET_RX_DMA;
        let enet = irqs.line(HPC3_INTSTAT_ENET_DEV);
        enet.set(true);
        assert!(en.level());
        enet.set(false);
        assert!(!en.level(), "the SEEQ's own level, not ORed with the DMA bits");
    }

    #[test]
    fn stop_saves_both_battery_backed_devices() {
        let dir = std::env::temp_dir().join(format!("iris-hpc3-stop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let nvram = dir.join("nvram.bin").to_str().unwrap().to_owned();
        let nveeprom = dir.join("nveeprom.bin").to_str().unwrap().to_owned();
        let ee = Arc::new(Mutex::new(Eeprom93c56::with_path(LogModule::Nveeprom, nveeprom.clone())));
        let hpc = Hpc3::with_net(
            ee.clone(), Ioc::new(true), false, Arc::new(AtomicU64::new(0)),
            NetworkConfig::default(), true, AudioConfig::default(), nvram.clone(),
            RtcOffset::default(), true, Hpc3::new_scsi_buses(false),
        );
        hpc.rtc.write8(500, 0x5a);
        ee.lock().set_word(0x74, 0x3530);
        hpc.stop();
        drop(hpc);
        drop(ee);
        let rtc = Ds1x86::new(8192, nvram.clone(), RtcOffset::default());
        assert_eq!(rtc.read8(500).data, 0x5a);
        assert_eq!(std::fs::read(nveeprom).unwrap()[232..234], [b'5', b'0']);
        assert_eq!(std::fs::metadata(nvram).unwrap().len(), 8192);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// One channel per ops group: PBUS, SCSI, enet RX, enet TX. All twelve go
    /// through the same record, so a group-specific field such as `even_high`
    /// still has to survive on every channel.
    const SAMPLED: [usize; 4] = [0, 8, 10, 11];

    fn hpc3_for_test() -> Hpc3 {
        Hpc3::with_net(
            Arc::new(Mutex::new(Eeprom93c56::new())),
            Ioc::new(true),
            true,
            Arc::new(AtomicU64::new(0)),
            NetworkConfig::default(),
            true,
            AudioConfig::default(),
            // Empty path: nothing on disk to load, and nothing written back.
            String::new(),
            RtcOffset::default(),
            true,
            Hpc3::new_scsi_buses(true),
        )
    }

    /// Bus address of a register in PBUS DMA channel 1 (Codec A).
    fn ch1(off: u32) -> u64 {
        (HPC3_BASE + PBUS_DMA_STRIDE + off) as u64
    }

    #[test]
    fn ip28_byte_control_writes_start_and_drain_audio_dma() {
        struct TuneMemory;
        impl BusDevice for TuneMemory {
            fn read32(&self, addr: u64) -> BusRead32 {
                match addr {
                    0x1000 => BusRead32::ok(0x2000),
                    0x1004 => BusRead32::ok(0x8000_0004), // one sample, EOX
                    0x1008 => BusRead32::ok(0),
                    0x2000 => BusRead32::ok(0x0012_3400), // PROM: sample << 8
                    _ => BusRead32::err(),
                }
            }
        }
        let hpc = hpc3_for_test();
        hpc.pdma_channels[1].lock().sys_mem = Some(Arc::new(TuneMemory));
        hpc.write32((HPC3_BASE + PBUS_CFGDMA_BASE + PBUS_CFGDMA_STRIDE) as u64, PBUS_DMACFG_DS16);
        hpc.write32(ch1(PBUS_DMA_DP), 0x1000);
        // InitiateDMA's four byte stores, with ACT/ACT_LD in the final lane.
        for (lane, byte) in [0x07, 0x04, 0x0b, 0x70].into_iter().enumerate() {
            assert_eq!(hpc.write8(ch1(PBUS_DMA_CTRL + lane as u32), byte), BUS_OK);
            assert_eq!(hpc.pdma_channels[1].lock().is_active(), lane == 3);
        }
        assert_eq!(hpc.read8(ch1(PBUS_DMA_CTRL + 3)).data & 2, 2);
        let (sample, status) = hpc.pdma_channels[1].lock().dma_read().expect("audio DMA sample");
        assert_eq!(sample, 0x1234);
        assert!(!status.refused());
        assert_eq!(hpc.read8(ch1(PBUS_DMA_CTRL + 3)).data & 2, 0);
        assert_eq!(hpc.read32(ch1(PBUS_DMA_BP)).data, 0x2004);
    }

    #[test]
    fn pbus_byte_lanes_preserve_addresses_and_control_side_effects() {
        let hpc = hpc3_for_test();
        for reg in [PBUS_DMA_BP, PBUS_DMA_DP] {
            hpc.write32(ch1(reg), 0x1234_5678);
            hpc.write8(ch1(reg + 1), 0xab);
            assert_eq!(hpc.read32(ch1(reg)).data, 0x12ab_5678);
            for (lane, byte) in [0x12, 0xab, 0x56, 0x78].into_iter().enumerate() {
                assert_eq!(hpc.read8(ch1(reg + lane as u32)).data, byte);
            }
        }
        hpc.write8(ch1(PBUS_DMA_CTRL + 3), PDMA_CTRL_LITTLE as u8);
        hpc.pdma_channels[1].lock().ctrl |= PDMA_CTRL_INT;
        for lane in 0..3 {
            hpc.write8(ch1(PBUS_DMA_CTRL + lane), 0xff);
            assert_eq!(hpc.read8(ch1(PBUS_DMA_CTRL + lane)).data, 0);
            let c = hpc.pdma_channels[1].lock();
            assert!(c.endian);
            assert_ne!(c.ctrl & PDMA_CTRL_INT, 0);
        }
        assert_eq!(hpc.read8(ch1(PBUS_DMA_CTRL + 3)).data & 1, 1);
        assert_eq!(hpc.pdma_channels[1].lock().ctrl & PDMA_CTRL_INT, 0);
    }

    fn set_latched_flags(hpc3: &Hpc3) {
        for &i in SAMPLED.iter() {
            let mut c = hpc3.pdma_channels[i].lock();
            c.eox = true;
            c.eop = true;
            c.xie = true;
            c.width_16 = true;
            c.even_high = true;
            c.endian = true;
        }
    }

    #[test]
    fn save_load_round_trip() {
        let src = hpc3_for_test();
        {
            let mut st = src.state.lock();
            st.intstat = HPC3_INTSTAT_SCSI0_DMA | HPC3_INTSTAT_ENET_RX_DMA;
            st.gio_misc = 0x0000_0003;
            st.eeprom_reg = 0x0000_00a5;
            st.pbus_pio[0] = 0xdead_beef;
            st.pbus_pio[0xfff] = 0x1234_5678;
        }
        for (n, &i) in SAMPLED.iter().enumerate() {
            let mut c = src.pdma_channels[i].lock();
            let b = 0x1000_0000u32 * (n as u32 + 1);
            c.cbp = b | 0x40;
            c.nbdp = b | 0x80;
            c.bc = 0x0000_0200;
            c.ctrl = 0x0000_0011;
            c.gio = b | 0xc0;
            c.dev = b | 0xd0;
            c.dmacfg = PBUS_DMACFG_DS16 | PBUS_DMACFG_EVEN_HIGH;
            c.piocfg = 0x0000_1357;
            c.crbdp = b | 0x100;
            c.cpfxbdp = b | 0x110;
            c.ppfxbdp = b | 0x120;
            c.tx_new_packet = false;
        }
        set_latched_flags(&src);
        let v1 = src.save_state();

        let dst = hpc3_for_test();
        dst.load_state(&v1).expect("load_state");
        let v2 = dst.save_state();

        assert_eq!(v1, v2, "Hpc3 save_state mismatch after load_state round-trip");
    }

    /// The round trip above passes even when a field is dropped from both
    /// halves, which is how these seven went missing. Restoring into a device
    /// whose flags sit at their defaults is the direction that notices.
    #[test]
    fn latched_flags_reach_a_fresh_device() {
        let src = hpc3_for_test();
        set_latched_flags(&src);
        let saved = src.save_state();

        let dst = hpc3_for_test();
        // The real restore path calls this before load_state.
        dst.power_on();
        for &i in SAMPLED.iter() {
            let c = dst.pdma_channels[i].lock();
            assert!(!c.eox && !c.eop && !c.xie, "chan {} latches not at default", i);
            assert!(!c.width_16 && !c.even_high && !c.endian, "chan {} config not at default", i);
        }

        dst.load_state(&saved).expect("load_state");

        for &i in SAMPLED.iter() {
            let c = dst.pdma_channels[i].lock();
            assert!(c.eox, "chan {} lost eox", i);
            assert!(c.eop, "chan {} lost eop", i);
            assert!(c.xie, "chan {} lost xie", i);
            assert!(c.width_16, "chan {} lost width_16", i);
            assert!(c.even_high, "chan {} lost even_high", i);
            assert!(c.endian, "chan {} lost endian", i);
        }
    }
}

#[cfg(test)]
mod enet_rx_chain_tests {
    use super::*;

    fn hpc3_for_test() -> Hpc3 {
        Hpc3::with_net(
            Arc::new(Mutex::new(Eeprom93c56::new())),
            Ioc::new(true),
            true,
            Arc::new(AtomicU64::new(0)),
            NetworkConfig::default(),
            true,
            AudioConfig::default(),
            String::new(),
            RtcOffset::default(),
            true,
            Hpc3::new_scsi_buses(true),
        )
    }

    /// Guest memory for descriptor tests, big-endian like the guest's.
    struct Ram(Mutex<Vec<u8>>);
    impl BusDevice for Ram {
        fn read8(&self, a: u64) -> BusRead8 { BusRead8::ok(self.0.lock()[a as usize]) }
        fn write8(&self, a: u64, v: u8) -> u32 { self.0.lock()[a as usize] = v; BUS_OK }
        fn read32(&self, a: u64) -> BusRead32 {
            let m = self.0.lock();
            BusRead32::ok(u32::from_be_bytes(m[a as usize..a as usize + 4].try_into().unwrap()))
        }
        fn write16(&self, a: u64, v: u16) -> u32 {
            self.0.lock()[a as usize..a as usize + 2].copy_from_slice(&v.to_be_bytes());
            BUS_OK
        }
        fn write32(&self, a: u64, v: u32) -> u32 {
            self.0.lock()[a as usize..a as usize + 4].copy_from_slice(&v.to_be_bytes());
            BUS_OK
        }
    }

    /// One descriptor at 0x100 with its buffer at 0x1000 and the chain's end
    /// flag, so the channel stops after it.
    fn one_descriptor(flags: u32) -> (Hpc3, Arc<Ram>) {
        let hpc3 = hpc3_for_test();
        let ram = Arc::new(Ram(Mutex::new(vec![0; 0x2000])));
        hpc3.set_phys(ram.clone());
        ram.write32(0x100, 0x1000);
        ram.write32(0x104, PDMA_DESC_EOX | PDMA_DESC_EOP | PDMA_DESC_XIE | flags | 0x600);
        ram.write32(0x108, 0x100);
        (hpc3, ram)
    }

    fn reg(off: u32) -> u64 { (HPC3_BASE + off) as u64 }

    /// Receive descriptor `i` of a ring: at 0x100 + 0x10*i, its buffer at
    /// 0x1000 + 0x800*i, with count 0x600 (1536).
    fn desc(i: u32) -> u32 { 0x100 + 0x10 * i }
    fn buf(i: u32) -> u32 { 0x1000 + 0x800 * i }
    const RING_COUNT: u32 = 0x600;
    /// What a driver writes into word 1 when it arms a receive descriptor:
    /// one frame per descriptor, interrupt after it, bit 14 set.
    const ARMED: u32 = PDMA_DESC_EOP | PDMA_DESC_XIE | PDMA_DESC_ROWN | RING_COUNT;

    /// `n` descriptors linked in a circle, all armed, EOX on `eox` (0-based),
    /// and the receive channel started on the first.
    fn ring(n: u32, eox: u32) -> (Hpc3, Arc<Ram>) {
        ring_armed(n, eox, ARMED)
    }

    /// As `ring`, with each descriptor's word 1 `armed` instead.
    fn ring_armed(n: u32, eox: u32, armed: u32) -> (Hpc3, Arc<Ram>) {
        let hpc3 = hpc3_for_test();
        let ram = Arc::new(Ram(Mutex::new(vec![0; 0x8000])));
        hpc3.set_phys(ram.clone());
        for i in 0..n {
            ram.write32(desc(i) as u64, buf(i));
            ram.write32((desc(i) + 4) as u64, armed | if i == eox { PDMA_DESC_EOX } else { 0 });
            ram.write32((desc(i) + 8) as u64, desc((i + 1) % n));
        }
        hpc3.write32(reg(ENET_RX_BASE + ENET_RX_NBDP), desc(0));
        hpc3.write32(reg(ENET_RX_BASE + ENET_RX_CTRL), ENET_RX_CTRL_ACTIVE);
        (hpc3, ram)
    }

    /// Put one frame of `len` bytes, every byte `fill`, through the receive
    /// channel the way the SEEQ does: two pad bytes, the frame, the status
    /// byte. Returns the statuses of all the writes ORed together.
    fn deliver(hpc3: &Hpc3, len: usize, fill: u8) -> DmaStatus {
        let rx = PdmaClientImpl { channel: hpc3.pdma_channels[10].clone() };
        let mut st = rx.write(0, false) | rx.write(0, false);
        for _ in 0..len {
            st |= rx.write(fill as u32, false);
        }
        st | rx.write(0x30, true)
    }

    fn word1(ram: &Ram, i: u32) -> u32 { ram.read32((desc(i) + 4) as u64).data }
    fn rx_active(hpc3: &Hpc3) -> bool {
        hpc3.read32(reg(ENET_RX_BASE + ENET_RX_CTRL)).data & ENET_RX_CTRL_ACTIVE != 0
    }

    /// The channel stops at the descriptor carrying EOX and clears ACTIVE;
    /// it does not wrap round the ring onto descriptors it filled. Frames
    /// sent after the stop go nowhere.
    #[test]
    fn receive_stops_at_eox_and_does_not_wrap() {
        let (hpc3, ram) = ring(4, 2);
        for f in 1..=5u8 {
            deliver(&hpc3, 60, f);
        }
        for i in 0..3 {
            assert_eq!(word1(&ram, i) & (PDMA_DESC_ROWN | 0x3fff), RING_COUNT - 63, "descriptor {} not written back", i + 1);
        }
        assert_eq!(word1(&ram, 3), ARMED, "the descriptor after EOX was touched");
        assert!(!rx_active(&hpc3));
        assert_eq!(ram.read8((buf(0) + 2) as u64).data, 1, "a frame after the stop overwrote the first");
        assert_eq!(ram.read8((buf(3) + 2) as u64).data, 0, "a frame after the stop reached the fourth buffer");
    }

    /// A driver moves EOX forward while the channel runs: it sets EOX on its
    /// new last descriptor, then clears it on the old one. A channel already
    /// sitting on the old one when that happens follows the chain on, and
    /// stops at the new last descriptor instead.
    #[test]
    fn receive_follows_eox_moved_while_running() {
        let (hpc3, ram) = ring(4, 2);
        deliver(&hpc3, 60, 1);
        deliver(&hpc3, 60, 2);
        // The driver reaps and re-arms the first two; the second is now last.
        for i in 0..2 {
            ram.write32((desc(i) + 4) as u64, ARMED | if i == 1 { PDMA_DESC_EOX } else { 0 });
        }
        ram.write32((desc(2) + 4) as u64, ARMED);
        deliver(&hpc3, 60, 3);
        assert!(rx_active(&hpc3), "the channel stopped at a descriptor whose EOX the driver had cleared");
        deliver(&hpc3, 60, 4);
        deliver(&hpc3, 60, 5);
        assert!(rx_active(&hpc3));
        deliver(&hpc3, 60, 6);
        assert!(!rx_active(&hpc3), "the channel did not stop at the new last descriptor");
        assert_eq!(hpc3.read32(reg(ENET_CRBDP)).data, desc(1));
    }

    /// Reaching EOX interrupts even without XIE; an ordinary descriptor
    /// without XIE does not.
    #[test]
    fn receive_stop_at_eox_interrupts_without_xie() {
        let (hpc3, _ram) = ring_armed(2, 1, ARMED & !PDMA_DESC_XIE);
        assert!(!deliver(&hpc3, 60, 1).irq());
        let st = deliver(&hpc3, 60, 2);
        assert!(!rx_active(&hpc3));
        assert!(st.irq(), "the stop at EOX raised no interrupt");
    }

    /// A frame longer than its buffer sets RBO and stops the channel with an
    /// interrupt, without spilling into the next descriptor. CLRINT leaves
    /// that interrupt pending; reading the control register clears RBO.
    #[test]
    fn a_frame_longer_than_its_buffer_sets_rbo_and_stops() {
        let (hpc3, ram) = ring_armed(2, 1, (ARMED & !0x3fff) | 40);
        let st = deliver(&hpc3, 60, 1);
        assert!(st.irq(), "RBO raised no interrupt");
        assert_eq!(hpc3.pdma_channels[10].lock().ctrl & ENET_RX_CTRL_RBO, ENET_RX_CTRL_RBO);
        assert!(!hpc3.pdma_channels[10].lock().is_active());
        assert_eq!(word1(&ram, 1), (ARMED & !0x3fff) | 40 | PDMA_DESC_EOX, "the frame spilled into the next descriptor");
        assert_eq!(ram.read8((buf(1) + 2) as u64).data, 0);

        let reset = reg(ENET_RX_BASE + ENET_RX_RESET);
        hpc3.seeq().raise_dma_interrupt();
        hpc3.write32(reset, ENET_RX_RESET_CLRINT);
        assert_eq!(hpc3.read32(reset).data & ENET_RX_RESET_INTPEND, ENET_RX_RESET_INTPEND, "CLRINT cleared RBO");
        let ctrl = hpc3.read32(reg(ENET_RX_BASE + ENET_RX_CTRL)).data;
        assert_eq!(ctrl & ENET_RX_CTRL_RBO, ENET_RX_CTRL_RBO);
        assert_eq!(hpc3.read32(reg(ENET_RX_BASE + ENET_RX_CTRL)).data & ENET_RX_CTRL_RBO, 0, "a read left RBO set");
        hpc3.write32(reset, ENET_RX_RESET_CLRINT);
        assert_eq!(hpc3.read32(reset).data & ENET_RX_RESET_INTPEND, 0);
    }

    /// A SEEQ receive FIFO overflow stops the channel where it is, with the
    /// interrupt and the overflow bit in the receive control register; the
    /// descriptors after it are untouched.
    #[test]
    fn a_seeq_receive_overflow_stops_the_channel() {
        let (hpc3, ram) = ring(4, 3);
        hpc3.write32(reg(SEEQ_BASE + 4 * 6), 0xbf); // receive command: every interrupt, station+broadcast
        deliver(&hpc3, 60, 1);
        hpc3.seeq().rx_overflow();
        assert!(!rx_active(&hpc3), "the channel kept receiving after an overflow");
        assert_eq!(hpc3.read32(reg(ENET_RX_BASE + ENET_RX_RESET)).data & ENET_RX_RESET_INTPEND, ENET_RX_RESET_INTPEND);
        assert_eq!(hpc3.read32(reg(ENET_RX_BASE + ENET_RX_CTRL)).data & ENET_RX_CTRL_OFLOW, ENET_RX_CTRL_OFLOW);
        assert_eq!(word1(&ram, 2), ARMED);
        assert_eq!(word1(&ram, 3), ARMED | PDMA_DESC_EOX);
    }

    /// The receive hold's view of the ring: room while the next descriptor
    /// does not end the chain; the last descriptor withheld while the driver
    /// still has filled descriptors to reap; room again once it has re-armed
    /// them (bit 14 set), even if it left EOX where it was.
    #[test]
    fn rx_room_withholds_the_last_descriptor_until_the_driver_catches_up() {
        let (hpc3, ram) = ring(4, 2);
        let rx = PdmaClientImpl { channel: hpc3.pdma_channels[10].clone() };
        assert_eq!(rx.rx_room(), RxRoom::Room);
        deliver(&hpc3, 60, 1);
        deliver(&hpc3, 60, 2);
        assert_eq!(rx.rx_room(), RxRoom::Last);
        ram.write32((desc(0) + 4) as u64, ARMED);
        assert_eq!(rx.rx_room(), RxRoom::Last, "one filled descriptor is still to be reaped");
        ram.write32((desc(1) + 4) as u64, ARMED);
        assert_eq!(rx.rx_room(), RxRoom::Room);
        deliver(&hpc3, 60, 3);
        assert_eq!(rx.rx_room(), RxRoom::Stopped);
    }

    /// A ring of one descriptor always has room for its one frame: nothing
    /// else is there for the driver to reap first. After the frame the channel
    /// is stopped. The driver restarts it on the same descriptor without
    /// re-arming it (bit 14 still clear): that descriptor is the one the next
    /// frame fills, not one left to reap, so there is room, not `Last`.
    #[test]
    fn rx_room_on_a_one_descriptor_ring() {
        let (hpc3, _ram) = ring(1, 0);
        let rx = PdmaClientImpl { channel: hpc3.pdma_channels[10].clone() };
        assert_eq!(rx.rx_room(), RxRoom::Room);
        deliver(&hpc3, 60, 1);
        assert_eq!(rx.rx_room(), RxRoom::Stopped);
        hpc3.write32(reg(ENET_RX_BASE + ENET_RX_NBDP), desc(0));
        hpc3.write32(reg(ENET_RX_BASE + ENET_RX_CTRL), ENET_RX_CTRL_ACTIVE);
        assert_eq!(rx.rx_room(), RxRoom::Room);
    }

    /// A driver's read of the receive control register, which it does after
    /// re-arming, wakes a SEEQ that is holding a frame for room, and only
    /// then.
    #[test]
    fn a_receive_control_read_wakes_a_holding_seeq() {
        let (hpc3, _ram) = ring(2, 1);
        let (link, _sent, _inject) = crate::net::Link::loopback();
        let wake = link.rx_wake();
        hpc3.seeq().attach(link);
        let ctrl = reg(ENET_RX_BASE + ENET_RX_CTRL);
        hpc3.read32(ctrl);
        assert!(!wake.consume(), "woke a SEEQ that was not holding");
        hpc3.seeq().hold_for_test();
        hpc3.read32(ctrl);
        assert!(wake.consume(), "the read did not wake the holding SEEQ");
    }

    /// Bit 14 of word 1 is the drivers' ownership flag, not the channel's:
    /// a descriptor whose bit 14 is clear is filled like any other.
    #[test]
    fn receive_fills_a_descriptor_whose_bit_14_is_clear() {
        let (hpc3, ram) = ring(4, 3);
        ram.write32((desc(1) + 4) as u64, ARMED & !PDMA_DESC_ROWN);
        deliver(&hpc3, 60, 1);
        deliver(&hpc3, 60, 2);
        assert!(rx_active(&hpc3), "the channel stopped at the descriptor with bit 14 clear");
        assert_eq!(word1(&ram, 1) & 0x3fff, RING_COUNT - 63);
        assert_eq!(ram.read8((buf(1) + 2) as u64).data, 2);
    }

    /// A driver that finds the receive channel stopped restarts it at the
    /// first descriptor it still sees as the HPC's (sgiseeq's
    /// rx_maybe_restart, sq_rxintr). The descriptor store has to be part of
    /// the same channel operation as the stop, so that by the time the
    /// channel reads as stopped the filled descriptor is the host's, with the
    /// remaining count written back. The channel lock hides the order inside
    /// that operation from a reader; this pins that the store is part of it.
    #[test]
    fn a_stopped_receive_channel_has_handed_its_frame_back() {
        let (hpc3, ram) = one_descriptor(PDMA_DESC_ROWN);
        hpc3.write32(reg(ENET_RX_BASE + ENET_RX_NBDP), 0x100);
        hpc3.write32(reg(ENET_RX_BASE + ENET_RX_CTRL), ENET_RX_CTRL_ACTIVE);

        // What the SEEQ sends: two pad bytes, a 60-byte frame, the status byte.
        let rx = PdmaClientImpl { channel: hpc3.pdma_channels[10].clone() };
        for _ in 0..62 {
            assert!(!rx.write(0, false).refused());
        }
        rx.write(0x20, true);

        assert_eq!(hpc3.read32(reg(ENET_RX_BASE + ENET_RX_CTRL)).data & ENET_RX_CTRL_ACTIVE, 0);
        let bc = ram.read32(0x104).data;
        assert_eq!(bc & PDMA_DESC_ROWN, 0, "the channel stopped, but the descriptor it filled still reads as the HPC's");
        assert_eq!(bc & 0x3fff, 0x600 - 63);
    }

    /// The same for transmit: a driver restarts a stopped transmit channel at
    /// the first descriptor not yet marked sent (sgiseeq's kick_tx,
    /// sq_txring_hpc3), so a packet whose mark is still pending would go twice.
    #[test]
    fn a_stopped_transmit_channel_has_marked_its_packet_sent() {
        let (hpc3, ram) = one_descriptor(0);
        hpc3.write32(reg(ENET_TX_BASE + ENET_TX_NBDP), 0x100);
        hpc3.write32(reg(ENET_TX_BASE + ENET_TX_CTRL), ENET_TX_CTRL_ACTIVE);

        let tx = PdmaClientImpl { channel: hpc3.pdma_channels[11].clone() };
        let mut sent = 0;
        while let Some((_, st)) = tx.read() {
            sent += 1;
            if st.eop() { break; }
        }

        assert_eq!(sent, 0x600);
        assert_eq!(hpc3.read32(reg(ENET_TX_BASE + ENET_TX_CTRL)).data & ENET_TX_CTRL_ACTIVE, 0);
        assert_ne!(ram.read32(0x104).data & ENET_BC_TXD as u32, 0, "the channel stopped before marking the packet sent");
    }

    /// A target for controller 1 on an Indy, which has one, is an error that
    /// names the machine; an Indigo2 has the second bus.
    #[test]
    fn the_second_scsi_controller_exists_on_an_indigo2_only() {
        let indy = Hpc3::new_scsi_buses(true);
        assert!(Hpc3::route_scsi(&indy, 0).is_ok());
        let e = Hpc3::route_scsi(&indy, 1).err().unwrap();
        assert!(e.contains("only controller 0"), "{e}");
        let indigo2 = Hpc3::new_scsi_buses(false);
        assert!(Hpc3::route_scsi(&indigo2, 1).is_ok());
        assert!(Hpc3::route_scsi(&indigo2, 2).err().unwrap().contains("valid: 0, 1"));
    }
}
