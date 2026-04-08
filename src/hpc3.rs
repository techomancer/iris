use std::sync::{Arc, OnceLock};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::fs::File;
use std::io::Write as IoWrite;
use crate::devlog::{LogModule, devlog_mask};
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device, DmaClient, DmaStatus, Resettable, Saveable};
use crate::snapshot::{get_field, u32_slice_to_toml, load_u32_slice, toml_u32, toml_bool, hex_u32};
use crate::config::NfsConfig;
use crate::eeprom_93c56::Eeprom93c56;
use crate::ioc::Ioc;
use crate::ds1x86::Ds1x86;
use crate::net::GatewayConfig;
use crate::seeq8003::{Seeq8003, SeeqCallback};
use crate::wd33c93a::{Wd33c93a, ScsiCallback};
use crate::ioc::IocInterrupt;
use crate::hal2::Hal2;
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

// SCSI Registers
pub const SCSI_REG_BASE: u32 = 0x40000;
pub const SEEQ_BASE: u32 = 0x54000;

// PBUS PIO
pub const PBUS_PIO_BASE: u32 = 0x58000;
pub const HAL2_BASE: u32 = 0x58000;

/// Returns the value a disabled (absent) HAL2 presents on reads.
/// REV (offset 0x20) returns 0xFFFF — not a valid chip version, so the IRIX
/// hal2 driver recognises "no chip" and skips init rather than spinning.
/// All other registers return 0: ISR.TSTATUS=0 (not busy), no spurious state.
fn hal2_absent_read(offset: u32) -> u16 {
    use crate::hal2::HAL2_REV;
    if (offset & 0xF0) == HAL2_REV { 0xFFFF } else { 0x0000 }
}
pub const HPC3_IOC_BASE: u32 = 0x59800;
pub const PBUS_PIO_STRIDE: u32 = 0x400;

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

pub trait PdmaCallback: Send + Sync {
    fn set_dma_interrupt(&self, level: bool);
}


struct Hpc3Irq {
    state: Arc<Mutex<Hpc3State>>,
    ioc: Ioc,
    bit: u32,
    ioc_line: IocInterrupt,
}

impl Hpc3Irq {
    fn update(&self, level: bool) {
        let mut state = self.state.lock();
        if level { state.intstat |= self.bit; } else { state.intstat &= !self.bit; }
        
        // Determine IOC line state based on all contributors
        let active = match self.ioc_line {
            IocInterrupt::Scsi0 => (state.intstat & (HPC3_INTSTAT_SCSI0_DEV | HPC3_INTSTAT_SCSI0_DMA)) != 0,
            IocInterrupt::Scsi1 => (state.intstat & (HPC3_INTSTAT_SCSI1_DEV | HPC3_INTSTAT_SCSI1_DMA)) != 0,
            IocInterrupt::Ethernet => (state.intstat & (HPC3_INTSTAT_ENET_DEV | HPC3_INTSTAT_ENET_RX_DMA | HPC3_INTSTAT_ENET_TX_DMA)) != 0,
            _ => false,
        };
        self.ioc.set_interrupt(self.ioc_line, active);
    }
}


impl ScsiCallback for Hpc3Irq {
    fn set_interrupt(&self, level: bool) {
        self.update(level);
    }
}

impl PdmaCallback for Hpc3Irq {
    fn set_dma_interrupt(&self, level: bool) {
        self.update(level);
    }
}

struct PdmaChannel {
    id: usize,
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
    callback: Option<Arc<dyn PdmaCallback>>,
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
    // RX: ROWN bit from current descriptor (host must set 1 before handing to HPC3)
    rown:          bool,
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
            callback: None,
            dump_enabled,
            dump_file: None,
            dump_transaction_id: 0,
            dump_is_write: false,
            transaction_id: 0,
            bytes_transferred: 0,
            width_16: false,
            crbdp: 0, cpfxbdp: 0, ppfxbdp: 0, tx_new_packet: true, rown: false, last_rx_ctrl: 0xFFFFFFFF,
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
            }
            self.ctrl |= self.active_mask;
        } else {
            if self.is_active() && self.log_active() {
                dlog_dev!(LogModule::Pdma, "PDMA[{}]: Channel Deactivated. Transferred {:x} bytes. CTRL={:08x}",
                    self.id, self.bytes_transferred, self.ctrl);
            }
            self.ctrl &= !self.active_mask;
        }
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
                let w_addr = { let _r = mem.read32(nbdp + PDMA_DESC_CBP_OFFSET); if _r.is_ok() { let d = _r.data; d } else { 0 } };
                let w_cnt = { let _r = mem.read32(nbdp + PDMA_DESC_BC_OFFSET); if _r.is_ok() { let d = _r.data; d } else { 0 } };
                let w_next = { let _r = mem.read32(nbdp + PDMA_DESC_NBP_OFFSET); if _r.is_ok() { let d = _r.data; d } else { 0 } };

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
                self.rown = (w_cnt & PDMA_DESC_ROWN) != 0; // RX only: host hands to HPC3 with ROWN=1
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
                            if let Some(cb) = &self.callback {
                                cb.set_dma_interrupt(true);
                            }
                        }
                        break; // Done
                    } else {
                        // Link descriptor (0 bytes, not EOX) -> fetch next immediately
                        nbdp = self.nbdp;
                        continue;
                    }
                }
                break; // Valid data descriptor loaded
            } else {
                break; // No memory attached
            }
        }
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

    fn dma_read(&mut self) -> Option<(u32, DmaStatus, Option<(u32, u16)>)> {
        if !self.is_active() { return None; }

        // PBUS DMA (Channels 0-7) always operates on 32-bit words
        // but only uses the most significant 8 or 16 bits.
        if self.id < 8 {
            let addr = self.cbp;
            let mem_val = if let Some(mem) = &self.sys_mem {
                { let _r = mem.read32(addr); if _r.is_ok() { _r.data } else { return None } }
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
            let (st, wb) = self.advance(4, false);
            return Some((val, st, wb));
        }

        let addr = self.cbp;
        let step = if self.width_16 { 2 } else { 1 };
        let swap = self.endian;

        let val = if let Some(mem) = &self.sys_mem {
            if self.width_16 {
                let _r = mem.read16(addr);
                if _r.is_ok() {
                    let v = if swap { _r.data.swap_bytes() } else { _r.data };
                    self.handle_dump(addr, &v.to_be_bytes(), true);
                    Some(v as u32)
                } else { None }
            } else {
                let _r = mem.read8(addr);
                if _r.is_ok() {
                    self.handle_dump(addr, &[_r.data], true);
                    Some(_r.data as u32)
                } else { None }
            }
        } else {
            None
        };

        if let Some(v) = val {
            let (st, wb) = self.advance(step, false);
            Some((v, st, wb))
        } else {
            None
        }
    }

    fn dma_write(&mut self, val: u32, eop: bool) -> (DmaStatus, Option<(u32, u16)>) {
        if !self.is_active() {
            if self.id >= 10 && self.log_active() {
                dlog_dev!(LogModule::Pdma, "PDMA[{}]: dma_write refused — channel not active", self.id);
            }
            return (DmaStatus(DmaStatus::NOT_ACTIVE), None);
        }
        // RX channel (id=10): respect ROWN — only write if HPC3 owns the descriptor
        if self.id == 10 && !self.rown {
            if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[{}]: dma_write refused — ROWN=0 (host owns descriptor, cbp={:08x})", self.id, self.cbp); }
            return (DmaStatus(DmaStatus::ROWN), None);
        }

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
                mem.write32(addr, mem_val);
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
                mem.write16(addr, v);
                self.handle_dump(addr, &v.to_be_bytes(), false);
            } else {
                mem.write8(addr, val as u8);
                self.handle_dump(addr, &[val as u8], false);
            }
        }

        self.advance(step, eop)
    }

    /// Returns (status, writeback).
    /// writeback: Some((addr, val16)) is a deferred memory write to be applied by the caller
    /// under a higher-level lock (e.g. SeeqState) for atomicity. Only set for enet channels.
    fn advance(&mut self, step: u32, caller_eop: bool) -> (DmaStatus, Option<(u32, u16)>) {
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
        let mut writeback: Option<(u32, u16)> = None;

        // xie fires on every descriptor completion: either caller signals EOP, or bc hits zero.
        // These are orthogonal to EOX.
        let irq = self.xie && (caller_eop || bc_done);

        // RX (ch10) writeback: return remaining bc and crbdp+6 address to caller.
        // Caller (Seeq8003) will write this under SeeqState lock for atomicity.
        if self.id == 10 && caller_eop {
            if self.crbdp != 0 {
                let rem = (self.bc & 0x3FFF) as u16;
                if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[10]: RX writeback deferred crbdp={:08x}+6 ← rem_bc={:04x}", self.crbdp, rem); }
                writeback = Some((self.crbdp + 6, rem));
            }
            bc_done = true;
        }

        // Byte count reached zero — end of this descriptor.
        if bc_done {
            if self.eop {
                status |= DmaStatus(DmaStatus::EOP);
                if self.id == 11 { self.tx_new_packet = true; }
            }
            // TX (ch11) writeback: return cpfxbdp+6 address to caller.
            // Caller (Seeq8003) will write ENET_BC_TXD under SeeqState lock for atomicity.
            if self.id == 11 && self.eop && self.cpfxbdp != 0 {
                if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[11]: TX writeback deferred cpfxbdp={:08x}+6 ← TXD", self.cpfxbdp); }
                writeback = Some((self.cpfxbdp + 6, ENET_BC_TXD));
            }
            if self.eox {
                status |= DmaStatus(DmaStatus::EOX);
                self.set_active(false);
                if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[{}]: Transfer Complete (EOX). CBP={:08x} NBDP={:08x} BC={:08x} CTRL={:08x}",
                    self.id, self.cbp, self.nbdp, self.bc, self.ctrl); }
            } else {
                self.fetch_descriptor();
            }
        }

        if irq {
            status |= DmaStatus(DmaStatus::IRQ);
            // For SCSI/PBUS channels (callback installed): set ctrl INT flag and notify.
            // For enet channels (no callback): IRQ bit in status is the signal; don't touch ctrl.
            if let Some(cb) = &self.callback {
                self.ctrl |= PDMA_CTRL_INT;
                cb.set_dma_interrupt(true);
            }
            if self.log_active() { dlog_dev!(LogModule::Pdma, "PDMA[{}]: Interrupting (xie caller_eop={} bc_done={})", self.id, caller_eop, bc_done); }
        }
        (status, writeback)
    }
}

struct PdmaClientImpl {
    channel: Arc<Mutex<PdmaChannel>>,
}

impl DmaClient for PdmaClientImpl {
    fn read(&self) -> Option<(u32, DmaStatus, Option<(u32, u16)>)> {
        self.channel.lock().dma_read()
    }
    fn write(&self, val: u32, eop: bool) -> (DmaStatus, Option<(u32, u16)>) {
        self.channel.lock().dma_write(val, eop)
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
            HPC3_PDMA_CBP => chan.cbp,
            HPC3_PDMA_NBDP => chan.nbdp,
            HPC3_PDMA_CTRL => {
                let val = chan.ctrl;
                if (chan.ctrl & PDMA_CTRL_INT) != 0 {
                    chan.ctrl &= !PDMA_CTRL_INT;
                    if let Some(cb) = &chan.callback {
                        cb.set_dma_interrupt(false);
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

struct ScsiDmaOps;
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
                    if let Some(cb) = &chan.callback {
                        cb.set_dma_interrupt(false);
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
                // Update control register, preserving the active bit for now
                chan.ctrl = (val & !SCSI_CTRL_ACTIVE) | (chan.ctrl & SCSI_CTRL_ACTIVE);

                let mask_active = (val & SCSI_CTRL_AMASK) != 0;
                let reset = (val & SCSI_CTRL_RESET) != 0;
                let mut should_be_active = if mask_active { was_active } else { (val & SCSI_CTRL_ACTIVE) != 0 };

                if reset { should_be_active = false; }

                chan.set_active(should_be_active);

                if !was_active && should_be_active {
                    chan.fetch_descriptor();
                }

                if (val & SCSI_CTRL_FLUSH) != 0 {
                    // Flush: drain FIFO to memory and terminate DMA.
                    // In emulation the FIFO doesn't exist, so just stop the channel.
                    // Raise DMA interrupt if XIE is set in the current descriptor.
                    if chan.xie {
                        chan.ctrl |= PDMA_CTRL_INT;
                        if let Some(cb) = &chan.callback {
                            cb.set_dma_interrupt(true);
                        }
                    }
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
// BC register bit 28: HPC3 has sampled EOX from the current descriptor
const ENET_BC_EOX_SAMPLED: u32 = 0x10000000;

/// Ethernet SEEQ interrupt callback.
/// Raises/lowers the IOC Ethernet line and updates intstat.
/// INTPEND (bit 1 of ENET_RX_RESET register) is mirrored in enet_intpend (AtomicBool)
/// so it can be read lock-free by ENET_RX_RESET register reads (which hold the channel lock).
struct EnetSeeqIrq {
    hpc3_state: Arc<Mutex<Hpc3State>>,
    ioc:        Ioc,
}
impl SeeqCallback for EnetSeeqIrq {
    fn set_interrupt(&self, level: bool) {
        {
            let mut st = self.hpc3_state.lock();
            if level { st.intstat |= HPC3_INTSTAT_ENET_DEV; }
            else     { st.intstat &= !HPC3_INTSTAT_ENET_DEV; }
        }
        dlog_dev!(LogModule::Hpc3, "ENET IRQ: level={}", level);
        self.ioc.set_interrupt(IocInterrupt::Ethernet, level);
    }
}

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
    hal2: Option<Arc<Hal2>>,
    pdma_dump: Arc<AtomicU32>,
    guinness: bool,
}

impl Hpc3 {
    pub fn new(eeprom: Arc<Mutex<Eeprom93c56>>, ioc: Ioc, guinness: bool, heartbeat: Arc<AtomicU64>) -> Self {
        Self::with_nfs(eeprom, ioc, guinness, heartbeat, None, vec![], false)
    }

    pub fn with_nfs(eeprom: Arc<Mutex<Eeprom93c56>>, ioc: Ioc, guinness: bool, heartbeat: Arc<AtomicU64>, nfs: Option<NfsConfig>, port_forwards: Vec<crate::config::PortForwardConfig>, headless: bool) -> Self {
        let rtc = Arc::new(Ds1x86::new(8192));
        let pdma_dump = Arc::new(AtomicU32::new(0));
        
        let state = Arc::new(Mutex::new(Hpc3State {
            intstat: 0,
            gio_misc: 0,
            eeprom_reg: 0,
            pbus_pio: [0; 0x1000],
        }));

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
            
            // Setup DMA interrupts
            if i == HPC3_PDMA_CHAN_SCSI0 as usize {
                chan.callback = Some(Arc::new(Hpc3Irq {
                    state: state.clone(), ioc: ioc.clone(), bit: HPC3_INTSTAT_SCSI0_DMA, ioc_line: IocInterrupt::Scsi0
                }));
            } else if i == HPC3_PDMA_CHAN_SCSI1 as usize {
                chan.callback = Some(Arc::new(Hpc3Irq {
                    state: state.clone(), ioc: ioc.clone(), bit: HPC3_INTSTAT_SCSI1_DMA, ioc_line: IocInterrupt::Scsi1
                }));
            }
            // Enet channels 10/11: no DMA completion callback — interrupt is driven by SEEQ via EnetSeeqIrq
            pdma_channels.push(Arc::new(Mutex::new(chan)));
            dma_clients.push(Arc::new(PdmaClientImpl { channel: pdma_channels.last().unwrap().clone() }));
            if i <= HPC3_PDMA_CHAN_GENERIC as usize {
                pdma_ops.push(Arc::new(PbusDmaOps));
            } else if i <= HPC3_PDMA_CHAN_SCSI1 as usize {
                pdma_ops.push(Arc::new(ScsiDmaOps));
            } else if i == HPC3_PDMA_CHAN_ENET_RX as usize {
                pdma_ops.push(Arc::new(EnetRxDmaOps { seeq: enet_seeq_lock.clone() }));
            } else {
                pdma_ops.push(Arc::new(EnetTxDmaOps { seeq: enet_seeq_lock.clone() }));
            }
        }

        let enet_rx_dma = Arc::new(PdmaClientImpl { channel: pdma_channels[10].clone() });
        let enet_tx_dma = Arc::new(PdmaClientImpl { channel: pdma_channels[11].clone() });

        let seeq_irq = Arc::new(EnetSeeqIrq {
            hpc3_state: state.clone(),
            ioc:        ioc.clone(),
        });
        let gateway_cfg = GatewayConfig { nfs, port_forwards, ..GatewayConfig::default() };
        let seeq = Arc::new(Seeq8003::with_config(Some(seeq_irq), Some(enet_rx_dma), Some(enet_tx_dma), gateway_cfg, heartbeat.clone()));
        // Publish seeq to both the DMA ops (CTRL reads) and the irq (status checks in set_interrupt)
        let _ = enet_seeq_lock.set(seeq.clone());
        
        let scsi0_dma = Arc::new(PdmaClientImpl { channel: pdma_channels[8].clone() });
        let scsi0_irq = Arc::new(Hpc3Irq {
            state: state.clone(), ioc: ioc.clone(), bit: HPC3_INTSTAT_SCSI0_DEV, ioc_line: IocInterrupt::Scsi0
        });

        let scsi_dev = Arc::new(Wd33c93a::new(Some(scsi0_dma), Some(scsi0_irq), heartbeat.clone()));
        
        let hal2 = if headless { None } else { Some(Arc::new(Hal2::new(dma_clients[0..8].to_vec()))) };

        Self {
            state,
            ioc,
            rtc,
            eeprom,
            seeq,
            pdma_channels,
            pdma_ops,
            scsi_dev,
            hal2,
            pdma_dump,
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
        self.seeq.set_phys(mem);
    }

    pub fn add_scsi_device(&self, id: usize, path: &str, is_cdrom: bool, discs: Vec<String>, overlay: bool) -> std::io::Result<()> {
        self.scsi_dev.add_device(id, path, is_cdrom, discs, overlay)
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
        if let Some(hal2) = &self.hal2 { hal2.register_locks(); }
        self.ioc.register_locks();
    }
}

impl Device for Hpc3 {
    fn step(&self, _cycles: u64) {
        // TODO: Implement DMA stepping
    }

    fn stop(&self) {
        self.seeq.stop();
        self.scsi_dev.stop();
        self.rtc.stop();
        self.ioc.stop();
        if let Some(hal2) = &self.hal2 { hal2.stop(); }
    }

    fn start(&self) {
        if let Some(hal2) = &self.hal2 { hal2.start(); }
        self.ioc.start();
        self.rtc.start();
        self.scsi_dev.start();
        self.seeq.start();
    }
    fn is_running(&self) -> bool { self.ioc.is_running() }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        let mut cmds = vec![("hpc3".to_string(), "HPC3 commands: hpc3 status".to_string())];
        cmds.push(("pdma".to_string(), "PDMA commands: pdma status | pdma chain <addr> | pdma dump <on|off|hal|scsi|enet|MASK> [DEV]".to_string()));
        cmds.extend(self.ioc.register_commands());
        cmds.extend(self.rtc.register_commands());
        cmds.extend(self.seeq.register_commands());
        cmds.extend(self.scsi_dev.register_commands());
        if let Some(hal2) = &self.hal2 { cmds.extend(hal2.register_commands()); }
        cmds
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        if cmd == "hpc3" {
            return Err("Usage: hpc3 status".to_string());
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
                        writeln!(writer, "  [{:2}] {:8}: Active={} CBP={:08x} NBDP={:08x} BC={:08x} CRBDP={:08x} Endian={}",
                            i, type_str, c.is_active(), c.cbp, c.nbdp, c.bc, c.crbdp, if c.endian { "Little" } else { "Big" }).unwrap();
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
                        let cbp  = { let _r = mem.read32(addr + PDMA_DESC_CBP_OFFSET); if _r.is_ok() { let d = _r.data; d } else { break } };
                        let bc   = { let _r = mem.read32(addr + PDMA_DESC_BC_OFFSET); if _r.is_ok() { let d = _r.data; d } else { break } };
                        let nbdp = { let _r = mem.read32(addr + PDMA_DESC_NBP_OFFSET); if _r.is_ok() { let d = _r.data; d } else { break } };
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
        if cmd == "scsi" || cmd == "cow" {
             return self.scsi_dev.execute_command(cmd, args, writer);
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
        
        Err("Command not found".to_string())
    }
}

impl BusDevice for Hpc3 {
    fn read8(&self, addr: u32) -> BusRead8 {
        let offset = addr - HPC3_BASE;

        // IOC (0x59800 - 0x598FF) - forward 8-bit access directly to IOC
        if (HPC3_IOC_BASE..HPC3_IOC_BASE + 0x100).contains(&offset) {
            return self.ioc.read8(addr);
        }

        // SCSI Registers (0x40000 - 0x40007) - these are 8-bit devices
        if (SCSI_REG_BASE..SCSI_REG_BASE + 8).contains(&offset) {
            let idx = (offset - SCSI_REG_BASE) >> 2;
            return self.scsi_dev.read(idx);
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
            return self.rtc.read8(byte_index);
        }

        // All other registers require 32-bit access
        dlog_dev!(LogModule::Hpc3, "HPC3: Unexpected read8 at offset {:05x} (addr {:08x})", offset, addr);
        BusRead8::ok(0)
    }

    fn write8(&self, addr: u32, val: u8) -> u32 {
        let offset = addr - HPC3_BASE;

        // IOC (0x59800 - 0x598FF) - forward 8-bit access directly to IOC
        if (HPC3_IOC_BASE..HPC3_IOC_BASE + 0x100).contains(&offset) {
            return self.ioc.write8(addr, val);
        }

        // SCSI Registers (0x40000 - 0x40007) - 8-bit devices
        if (SCSI_REG_BASE..SCSI_REG_BASE + 8).contains(&offset) {
            let idx = (offset - SCSI_REG_BASE) >> 2;
            return self.scsi_dev.write(idx, val);
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
            return self.rtc.write8(byte_index, val);
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

    fn read32(&self, addr: u32) -> BusRead32 {
        let offset = addr - HPC3_BASE;

        // IOC (0x59800 - 0x598FF) - should not use 32-bit access, but allow for legacy
        if (HPC3_IOC_BASE..HPC3_IOC_BASE + 0x100).contains(&offset) {
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

        // SCSI Registers (0x40000 - 0x40007) - 8-bit devices, convert to 32-bit
        if (SCSI_REG_BASE..SCSI_REG_BASE + 8).contains(&offset) {
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
            let r = self.rtc.read8(byte_index);
            return if r.is_ok() { BusRead32::ok(r.data as u32) } else { BusRead32 { status: r.status, data: 0 } };
        }

        dlog_dev!(LogModule::Hpc3, "HPC3: Read addr {:08x}", addr);
        BusRead32::ok(0)
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        let offset = addr - HPC3_BASE;

        // IOC (0x59800 - 0x598FF) - should not use 32-bit access, but allow for legacy
        if (HPC3_IOC_BASE..HPC3_IOC_BASE + 0x100).contains(&offset) {
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

        // SCSI Registers (0x40000 - 0x40007) - 8-bit devices
        if (SCSI_REG_BASE..SCSI_REG_BASE + 8).contains(&offset) {
            return self.write8(addr, val as u8);
        }

        // SEEQ8003 Ethernet Controller (0x54000 - 0x5401F) - 8-bit device
        if (SEEQ_BASE..SEEQ_BASE + 0x20).contains(&offset) {
            return self.write8(addr, val as u8);
        }

        // Misc Registers
        if (MISC_BASE..MISC_BASE + 0x1000).contains(&offset) {
            let mut state = self.state.lock();
            match offset - MISC_BASE {
                MISC_GIO_MISC => {
                    state.gio_misc = val;
                    dlog_dev!(LogModule::Hpc3, "HPC3: GIO_MISC ({:08x}) = {:08x}", addr, val);
                }
                MISC_EEPROM_DATA => {
                    state.eeprom_reg = val;
                    let mut eeprom = self.eeprom.lock();
                    // Bit 1: CS
                    eeprom.set_cs((val & (1 << 1)) != 0);
                    // Bit 3: DATO (Data to EEPROM)
                    eeprom.set_di((val & (1 << 3)) != 0);
                    // Bit 2: CLK
                    eeprom.set_sk((val & (1 << 2)) != 0);
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
            return self.rtc.write8(byte_index, byte_val);
        }

        // Log other writes
        dlog_dev!(LogModule::Hpc3, "HPC3: Write addr {:08x} = {:08x}", addr, val);

        BUS_OK
    }

    fn read16(&self, addr: u32) -> BusRead16 {
        let offset = addr - HPC3_BASE;

        // HAL2 registers are 16-bit; forward directly
        if (HAL2_BASE..HAL2_BASE + 0x400).contains(&offset) {
            if let Some(hal2) = &self.hal2 {
                return hal2.read(offset - HAL2_BASE);
            }
            return BusRead16::ok(hal2_absent_read(offset - HAL2_BASE));
        }

        BusRead16::ok(0)
    }

    fn write16(&self, addr: u32, val: u16) -> u32 {
        let offset = addr - HPC3_BASE;

        // HAL2 registers are 16-bit; forward directly
        if (HAL2_BASE..HAL2_BASE + 0x400).contains(&offset) {
            if let Some(hal2) = &self.hal2 {
                return hal2.write(offset - HAL2_BASE, val);
            }
            return BUS_OK;
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
        // We preserve the configuration fields (active_mask, sys_mem, callback, etc.)
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
            chan.rown = false;
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
    toml::Value::Table(tbl)
}

/// Restore one PdmaChannel's transfer-state registers from a TOML table.
fn load_pdma_channel(chan: &mut PdmaChannel, v: &toml::Value) {
    macro_rules! ldu32 { ($f:ident) => {
        if let Some(x) = get_field(v, stringify!($f)) { chan.$f = toml_u32(x).unwrap_or(chan.$f); }
    }}
    ldu32!(cbp); ldu32!(nbdp); ldu32!(bc); ldu32!(ctrl);
    ldu32!(gio); ldu32!(dev); ldu32!(dmacfg); ldu32!(piocfg);
    ldu32!(crbdp); ldu32!(cpfxbdp); ldu32!(ppfxbdp);
    if let Some(x) = get_field(v, "tx_new_packet") { chan.tx_new_packet = toml_bool(x).unwrap_or(true); }
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
