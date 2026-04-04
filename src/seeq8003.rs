// SEEQ 8003 / 80C03 Ethernet Data Link Controller emulation
//
// Acts as a minimal NAT gateway/router at 192.168.0.1 for the emulated machine.
//
// RX DMA buffer layout (per IRIX if_ec2.c / seeq.h):
//   [2 bytes pad][ethernet frame][1 byte SEEQ status]
// The status byte (SEQ_RS_GOOD etc.) is appended as the last DMA byte,
// not read from a register on a per-frame basis.
//
// Register model: the SEEQ status registers (rx_stat/tx_stat) hold device-level
// OLD/NEW status for interrupt acknowledgement only.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use crate::devlog::LogModule;
use parking_lot::{Condvar, Mutex};
use std::thread;
use std::time::Duration;

use rtrb::RingBuffer;

use crate::net::{eth_summary, mac_str, GatewayConfig, NatControl, NatEngine};
use crate::traits::{BusDevice, BusStatus, Device, DmaClient, DmaStatus, Resettable, Saveable};
use crate::snapshot::{get_field, toml_u8, u8_slice_to_toml, load_u8_slice, hex_u8};

// ── Register offsets (A2:A0) ──────────────────────────────────────────────────
pub const SEEQ_STATION_ADDR_0: u32 = 0x00;
pub const SEEQ_STATION_ADDR_1: u32 = 0x01;
pub const SEEQ_STATION_ADDR_2: u32 = 0x02;
pub const SEEQ_STATION_ADDR_3: u32 = 0x03;
pub const SEEQ_STATION_ADDR_4: u32 = 0x04;
pub const SEEQ_STATION_ADDR_5: u32 = 0x05;
pub const SEEQ_RX_REG: u32 = 0x06; // Read: RX Status  Write: RX Command
pub const SEEQ_TX_REG: u32 = 0x07; // Read: TX Status  Write: TX Command

// ── RX Command bits (written to SEEQ_RX_REG) ─────────────────────────────────
// From seeq.h: SEQ_RC_*
pub mod rx_cmd {
    pub const INTOVERFLOW: u8 = 0x01; // interrupt on overflow
    pub const INTCRC:      u8 = 0x02; // interrupt on CRC error
    pub const INTDRIBBLE:  u8 = 0x04; // interrupt on dribble error
    pub const INTSHORT:    u8 = 0x08; // interrupt on short frame
    pub const INTEOF:      u8 = 0x10; // interrupt on good end-of-frame
    // bits 5 unused
    pub const MATCH1:      u8 = 0x40; // match mode bit 1
    pub const MATCH0:      u8 = 0x80; // match mode bit 0

    pub const MATCH_MASK:       u8 = MATCH1 | MATCH0;
    pub const MATCH_DISABLE:    u8 = 0x00; // receiver disabled
    pub const MATCH_PROMISCUOUS:u8 = 0x40; // SEQ_RC_RALL: receive all
    pub const MATCH_STA_BCAST:  u8 = 0x80; // SEQ_RC_RSB: station + broadcast
    pub const MATCH_STA_MULTI:  u8 = 0xC0; // SEQ_RC_RSMB: station + broadcast + multicast
}

// ── RX Status bits (read from SEEQ_RX_REG) ───────────────────────────────────
// From seeq.h: SEQ_RS_*
pub mod rx_stat {
    pub const OLD:      u8 = 0x80; // 1 = status already read (stale), 0 = new
    pub const GOOD:     u8 = 0x20; // good frame received (SEQ_RS_GOOD)
    pub const END:      u8 = 0x10; // end of frame seen   (SEQ_RS_END)
    pub const SHORT:    u8 = 0x08; // short frame         (SEQ_RS_SHORT)
    pub const DRIBBLE:  u8 = 0x04; // dribble error       (SEQ_RS_DRBL)
    pub const CRC:      u8 = 0x02; // CRC error           (SEQ_RS_CRC)
    pub const OVERFLOW: u8 = 0x01; // overflow            (SEQ_RS_OFLOW)
}

// ── TX Command bits (written to SEEQ_TX_REG) ─────────────────────────────────
// From seeq.h: SEQ_XC_* (bank select in bits 7:5)
pub mod tx_cmd {
    pub const BANK_MASK:  u8 = 0x60; // bits 6:5 = register bank select
    pub const BANK0:      u8 = 0x00; // SEQ_XC_REGBANK0: station addr
    pub const BANK1:      u8 = 0x20; // SEQ_XC_REGBANK1: multicast lsb
    pub const BANK2:      u8 = 0x40; // SEQ_XC_REGBANK2: multicast msb, ctl
    pub const INTGOOD:    u8 = 0x08; // SEQ_XC_INTGOOD:  interrupt on successful tx
    pub const INT16TRY:   u8 = 0x04; // SEQ_XC_INT16TRY: interrupt on 16 retries
    pub const INTCOLL:    u8 = 0x02; // SEQ_XC_INTCOLL:  interrupt on collision
    pub const INTUFLOW:   u8 = 0x01; // SEQ_XC_INTUFLOW: interrupt on underflow
}

// ── TX Status bits (read from SEEQ_TX_REG) ───────────────────────────────────
// From seeq.h: SEQ_XS_*
pub mod tx_stat {
    pub const OLD:     u8 = 0x80; // 1 = status already read (stale), 0 = new
    pub const LATECOL: u8 = 0x10; // late collision
    pub const SUCCESS: u8 = 0x08; // successful transmission (SEQ_XS_SUCCESS)
    pub const R16:     u8 = 0x04; // 16 retries (SEQ_XS_16TRY)
    pub const COLL:    u8 = 0x02; // collision   (SEQ_XS_COLL)
    pub const UFLOW:   u8 = 0x01; // underflow   (SEQ_XS_UFLOW)
}

// ── Status byte appended to RX DMA buffer (SEQ_RS_GOOD etc.) ─────────────────
// This is the byte the SEEQ appends at the end of the DMA'd frame data.
// IRIX reads it as: rstat = *((caddr_t)eh + rlen)
const RX_STATUS_GOOD: u8 = rx_stat::GOOD | rx_stat::END; // 0x30


// ── 80C03 bank-select bits in tx_cmd ──────────────────────────────────────────
// Written to SEEQ_TX_REG; selects which registers are exposed at addrs 0-5.
pub mod bank {
    pub const MASK: u8 = 0x60;
    pub const B0:   u8 = 0x00; // Station address (write) / collision counters (read)
    pub const B1:   u8 = 0x20; // Multicast hash LSB
    pub const B2:   u8 = 0x40; // Multicast hash MSB + control register
}

// ── 80C03 read-mode register layout (regs 0-5, all banks read-mode is the same) ──
// reg 0-1: coll_xmit[0-1]  (TX collision counter, per-packet)
// reg 2-3: coll_total[0-1] (total collision counter)
// reg 4:   fill / unused
// reg 5:   flags — SEQ_XS_NO_SQE(0x01) | SEQ_XS_NO_CARRIER(0x02)
//
// coll_xmit[0] == 0  → driver detects SGI EDLC (80C03)
// flags = SEQ_XS_NO_SQE (bit 0) → carrier present, no SQE heartbeat (normal 10base-T)
pub mod edlc_flags {
    pub const NO_SQE:     u8 = 0x01; // SQE absent (normal for 10base-T)
    pub const NO_CARRIER: u8 = 0x02; // No carrier detected (fault)
}

// ── Internal register state ───────────────────────────────────────────────────
pub struct SeeqState {
    pub rx_cmd:   u8,
    pub rx_stat:  u8,
    pub tx_cmd:   u8,
    pub tx_stat:  u8,
    station_addr: [u8; 6],
    // 80C03 bank 1: multicast hash filter (48-bit, 6 bytes)
    mcast_lsb:    [u8; 6],
    // 80C03 bank 2: multicast hash MSB (2 bytes), pktgap, ctl
    mcast_msb:    [u8; 2],
    pktgap:       u8,
    ctl:          u8,
    /// Interrupt pending — set when an unacknowledged interrupt is outstanding.
    pub intpend:  bool,
    /// Monotonic lock counter — incremented on every mutex acquisition.
    /// Prefix all debug messages with this to correlate across threads.
    pub ts:       u32,
}

// ── Interrupt callback ────────────────────────────────────────────────────────
pub trait SeeqCallback: Send + Sync {
    /// Raise or lower the interrupt line. Called by Seeq8003 with SeeqState lock held.
    /// Implementor must not re-acquire SeeqState lock.
    fn set_interrupt(&self, level: bool);
}

// ── Ring-buffer capacity (number of frames) ───────────────────────────────────
const CHAN_CAPACITY: usize = 32;

// ── Main device struct ────────────────────────────────────────────────────────
pub struct Seeq8003 {
    state:    Arc<Mutex<SeeqState>>,
    callback: Option<Arc<dyn SeeqCallback>>,
    rx_dma:   Option<Arc<dyn DmaClient>>,
    tx_dma:   Option<Arc<dyn DmaClient>>,
    sys_mem:  Mutex<Option<Arc<dyn BusDevice>>>,
    in_reset: Arc<AtomicBool>, // CH_RESET edge — lock-free check in pump thread
    config:   GatewayConfig,
    running:  Arc<AtomicBool>,
    nat_ctl:  Arc<NatControl>,
    // TX: enet thread → NAT thread (outbound frames from guest)
    tx_prod:  Mutex<Option<rtrb::Producer<Vec<u8>>>>,
    tx_cons:  Mutex<Option<rtrb::Consumer<Vec<u8>>>>,
    // RX: NAT thread → enet thread (inbound frames to guest)
    rx_prod:  Mutex<Option<rtrb::Producer<Vec<u8>>>>,
    rx_cons:  Mutex<Option<rtrb::Consumer<Vec<u8>>>>,
    // tx_wake: enet signals NAT when it pushes a TX frame
    tx_wake:  Arc<(Mutex<()>, Condvar)>,
    // rx_wake: NAT signals enet when it pushes an RX frame
    rx_wake:  Arc<(Mutex<()>, Condvar)>,
    /// Activity heartbeat — shared with Rex3 display thread.
    heartbeat: Arc<AtomicU64>,
}

/// Lock a SeeqState mutex and bump the timestamp in one step.
macro_rules! lock_state {
    ($mutex:expr) => {{
        let mut st = $mutex.lock();
        st.ts = st.ts.wrapping_add(1);
        st
    }};
}

/// Result of a TX pump: applied under SeeqState lock.
enum TxPumpResult {
    Nothing,
    Sent { dma_irq: bool, writeback: Option<(u32, u16)> },
}

/// Result of an RX pump: applied under SeeqState lock.
enum RxPumpResult {
    Nothing,
    Refused,
    Delivered { dma_irq: bool, writeback: Option<(u32, u16)>, frame_len: usize },
}

// ── Seeq8003 ─────────────────────────────────────────────────────────────────
impl Seeq8003 {
    pub fn new(callback: Option<Arc<dyn SeeqCallback>>,
               rx_dma: Option<Arc<dyn DmaClient>>,
               tx_dma: Option<Arc<dyn DmaClient>>,
               heartbeat: Arc<AtomicU64>) -> Self {
        Self::with_config(callback, rx_dma, tx_dma, GatewayConfig::default(), heartbeat)
    }

    pub fn with_config(callback: Option<Arc<dyn SeeqCallback>>,
                       rx_dma: Option<Arc<dyn DmaClient>>,
                       tx_dma: Option<Arc<dyn DmaClient>>,
                       config: GatewayConfig,
                       heartbeat: Arc<AtomicU64>) -> Self {
        let (tx_prod, tx_cons) = RingBuffer::new(CHAN_CAPACITY);
        let (rx_prod, rx_cons) = RingBuffer::new(CHAN_CAPACITY);
        Self {
            state: Arc::new(Mutex::new(SeeqState {
                rx_cmd:       0,
                rx_stat:      rx_stat::OLD, // initially stale / no new status
                tx_cmd:       0,
                tx_stat:      tx_stat::OLD | tx_stat::SUCCESS, // ready
                station_addr: [0; 6],
                mcast_lsb:    [0; 6],
                mcast_msb:    [0; 2],
                pktgap:       0,
                ctl:          0,
                intpend:      false,
                ts:           0,
            })),
            callback, rx_dma, tx_dma,
            sys_mem: Mutex::new(None),
            in_reset: Arc::new(AtomicBool::new(false)),
            config,
            running:  Arc::new(AtomicBool::new(false)),
            nat_ctl:  NatControl::new(),
            tx_prod: Mutex::new(Some(tx_prod)),
            tx_cons: Mutex::new(Some(tx_cons)),
            rx_prod: Mutex::new(Some(rx_prod)),
            rx_cons: Mutex::new(Some(rx_cons)),
            tx_wake: Arc::new((Mutex::new(()), Condvar::new())),
            rx_wake: Arc::new((Mutex::new(()), Condvar::new())),
            heartbeat,
        }
    }

    fn raise_interrupt(state: &mut SeeqState, dma_irq: bool, callback: &Option<Arc<dyn SeeqCallback>>) {
        // Interrupt when status is NEW (OLD=0) and the corresponding IRQ enable bit is set.
        let rx_irq = (state.rx_stat & rx_stat::OLD) == 0 && {
            let enabled = state.rx_cmd & 0x1f;
            let status  = state.rx_stat & 0x1f;
            enabled & status != 0
        };
        let tx_irq = (state.tx_stat & tx_stat::OLD) == 0 && {
            let enabled = state.tx_cmd & 0x0f;
            let status  = state.tx_stat & 0x0f;
            enabled & status != 0
        };
        // dma_irq: DMA channel raised its own IRQ (xie), independent of SEEQ status.
        let level = tx_irq || rx_irq || dma_irq;
        if level && !state.intpend {
            state.intpend = true;
            if let Some(cb) = callback { cb.set_interrupt(true); }
        } else if !level && state.intpend {
            state.intpend = false;
            if let Some(cb) = callback { cb.set_interrupt(false); }
        }
    }

    /// Deassert the interrupt line. Called when the driver writes CLRINT.
    /// Mark both status registers as OLD so raise_interrupt won't immediately re-raise.
    pub fn reset_interrupt(&self) {
        let mut st = lock_state!(self.state);
        st.rx_stat |= rx_stat::OLD;
        st.tx_stat |= tx_stat::OLD;
        st.intpend = false;
        dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=reset_interrupt(CLRINT) rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
        if let Some(ref cb) = self.callback { cb.set_interrupt(false); }
    }

    /// Returns true if an unacknowledged interrupt is pending.
    pub fn is_interrupt_pending(&self) -> bool {
        lock_state!(self.state).intpend
    }

    fn address_filter_state(st: &SeeqState, frame: &[u8]) -> bool {
        Self::address_filter(st.rx_cmd, &st.station_addr, frame)
    }

    fn address_filter(rx_cmd: u8, station_addr: &[u8; 6], frame: &[u8]) -> bool {
        if frame.len() < 6 { return false; }
        const BCAST: [u8; 6] = [0xff; 6];
        let dst = &frame[0..6];
        match rx_cmd & rx_cmd::MATCH_MASK {
            rx_cmd::MATCH_DISABLE     => false,
            rx_cmd::MATCH_PROMISCUOUS => true,
            rx_cmd::MATCH_STA_BCAST  =>
                dst == BCAST || dst == station_addr.as_ref(),
            rx_cmd::MATCH_STA_MULTI  =>
                dst == BCAST || dst == station_addr.as_ref() || dst[0] & 1 != 0,
            _ => dst == BCAST || dst == station_addr.as_ref(),
        }
    }

    /// Drain TX DMA into a frame and push it to the NAT thread.
    /// Runs WITHOUT SeeqState lock. Returns a TxPumpResult to be applied under SeeqState lock.
    fn pump_tx(tx_dma: &Arc<dyn DmaClient>,
               tx_prod: &mut rtrb::Producer<Vec<u8>>,
               tx_wake: &Arc<(Mutex<()>, Condvar)>,
               in_reset: &Arc<AtomicBool>) -> TxPumpResult {
        if in_reset.load(Ordering::Relaxed) { return TxPumpResult::Nothing; }
        let mut frame = Vec::with_capacity(1518);
        let mut got_eop = false;
        let mut dma_st = DmaStatus::ok();
        let mut writeback: Option<(u32, u16)> = None;
        while let Some((b, dst, wb)) = tx_dma.read() {
            frame.push(b as u8);
            dma_st = dst;
            if let Some(w) = wb { writeback = Some(w); }
            if dst.eop() { got_eop = true; break; }
        }
        if frame.is_empty() || !got_eop { return TxPumpResult::Nothing; }

        dlog_dev!(LogModule::Seeq, "SEEQ TX {} dma_st={:#x}", eth_summary(&frame), dma_st.0);

        let _ = tx_prod.push(frame);
        tx_wake.1.notify_one();

        TxPumpResult::Sent { dma_irq: dma_st.irq(), writeback }
    }

    /// Inject one pending RX frame into guest RAM via HPC3 RX DMA.
    /// Runs WITHOUT SeeqState lock. Address filter uses snapshots taken before this call.
    /// Returns an RxPumpResult to be applied under SeeqState lock.
    fn pump_rx(rx_dma: &Arc<dyn DmaClient>,
               rx_cons: &mut rtrb::Consumer<Vec<u8>>,
               rx_cmd_snap: u8,
               station_addr_snap: [u8; 6],
               in_reset: &Arc<AtomicBool>) -> RxPumpResult {
        if in_reset.load(Ordering::Relaxed) {
            dlog_dev!(LogModule::Seeq, "SEEQ pump_rx: in reset, dropping frame");
            return RxPumpResult::Nothing;
        }
        let frame = match rx_cons.peek() {
            Ok(f) => f,
            Err(_) => return RxPumpResult::Nothing,
        };

        dlog_dev!(LogModule::Seeq, "SEEQ pump_rx: got frame {} bytes", frame.len());

        if !Self::address_filter(rx_cmd_snap, &station_addr_snap, frame) {
            dlog_dev!(LogModule::Seeq, "SEEQ pump_rx: address filter dropped {}", eth_summary(frame));
            let _ = rx_cons.pop(); // discard filtered frame
            return RxPumpResult::Nothing;
        }

        dlog_dev!(LogModule::Seeq, "SEEQ RX {} → guest DMA", eth_summary(frame));

        let (dst, _) = rx_dma.write(0, false); // pad[0]
        if dst.refused() {
            dlog_dev!(LogModule::Seeq, "SEEQ pump_rx: DMA not ready ({:#x}), will retry: {}", dst.0, eth_summary(frame));
            return RxPumpResult::Refused; // frame stays in rx_cons for next iteration
        }
        // DMA is committed (pad[0] written); pop the frame now and finish writing it.
        let frame = rx_cons.pop().expect("peek succeeded so pop must succeed");
        let _ = rx_dma.write(0, false); // pad[1]
        for b in &frame {
            let _ = rx_dma.write(*b as u32, false);
        }
        // Status byte — eop=true so DmaStatus::EOP is set; also catches EOX/IRQ if chain ended.
        // This write triggers the RX writeback (crbdp+6) inside advance(), returned as writeback.
        let (dma_st, writeback) = rx_dma.write(RX_STATUS_GOOD as u32, true);

        dlog_dev!(LogModule::Seeq, "SEEQ pump_rx: frame delivered ({} bytes to DMA) dma_st={:#x} irq={} wb={:?}",
                  2 + frame.len() + 1, dma_st.0, dma_st.irq(), writeback);

        RxPumpResult::Delivered { dma_irq: dma_st.irq(), writeback, frame_len: frame.len() }
    }

    /// Return current RX status byte without side-effects (no OLD-marking).
    /// Used by HPC3 to mirror SEEQ status into rx_ctrl[7:0].
    pub fn get_rx_status(&self) -> u8 {
        lock_state!(self.state).rx_stat
    }

    /// Return current TX status byte without side-effects (no OLD-marking).
    /// Used by HPC3 to mirror SEEQ status into tx_ctrl[7:0].
    pub fn get_tx_status(&self) -> u8 {
        lock_state!(self.state).tx_stat
    }

    /// Set the system memory bus device for descriptor writebacks.
    /// Must be called before start(). Called from Hpc3::set_phys().
    pub fn set_phys(&self, mem: Arc<dyn BusDevice>) {
        *self.sys_mem.lock() = Some(mem);
    }

    /// Wake the enet thread immediately to drain pending TX data.
    /// Call this when the TX DMA channel is activated with a descriptor ready.
    pub fn kick_tx(&self) {
        self.rx_wake.1.notify_one();
    }

    /// Wake the enet thread to deliver any pending RX frames.
    /// Call this when the RX DMA channel is activated so queued frames aren't lost.
    pub fn kick_rx(&self) {
        self.rx_wake.1.notify_one();
    }

    /// Assert CH_RESET (rising edge): hold device in reset state.
    /// Does not clear registers — deassert_reset does that
    pub fn assert_reset(&self) {
        self.in_reset.store(true, Ordering::Relaxed);
    }

    /// Deassert CH_RESET (falling edge): clear registers.
    /// Clears command registers, marks both status registers as OLD (stale),
    /// re-evaluates the interrupt line.
    /// Does NOT touch station_addr, NAT connections, rx_queue, or tx_frame.
    pub fn deassert_reset(&self) {
        self.in_reset.store(false, Ordering::Relaxed);
        let mut st = lock_state!(self.state);
        st.rx_stat = rx_stat::OLD;
        st.tx_stat = tx_stat::OLD;
        st.rx_cmd  = 0;
        st.tx_cmd  = 0;
        st.ctl     = 0;
        dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=deassert_reset rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
        Self::raise_interrupt(&mut st, false, &self.callback);
    }

    /// Immediate reset (legacy): assert then immediately deassert.
    pub fn reset(&self) {
        self.assert_reset();
        self.deassert_reset();
    }

    pub fn read(&self, addr: u32) -> BusStatus {
        let mut st = lock_state!(self.state);
        let val = match addr {
            SEEQ_STATION_ADDR_0..=SEEQ_STATION_ADDR_5 => {
                // 80C03: regs 0-5 are read-only collision counters / flags regardless of bank.
                // reg 0-1: coll_xmit (0 → driver identifies this as SGI 80C03 EDLC)
                // reg 2-3: coll_total (0)
                // reg 4:   fill (0)
                // reg 5:   flags — NO_SQE set (carrier present, no SQE = normal 10base-T)
                let idx = (addr - SEEQ_STATION_ADDR_0) as usize;
                if idx == 5 { edlc_flags::NO_SQE } else { 0x00 }
            }
            SEEQ_RX_REG => {
                let v = st.rx_stat;
                st.rx_stat |= rx_stat::OLD; // mark as read / stale
                dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=read_rx_reg rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
                Self::raise_interrupt(&mut st, false, &self.callback);
                v
            }
            SEEQ_TX_REG => {
                let v = st.tx_stat;
                st.tx_stat |= tx_stat::OLD;
                dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=read_tx_reg rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
                Self::raise_interrupt(&mut st, false, &self.callback);
                v
            }
            _ => 0,
        };
        dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ R[{:x}] -> {:02x}", st.ts, addr, val);
        BusStatus::Data8(val)
    }

    pub fn write(&self, addr: u32, val: u8) -> BusStatus {
        let mut st = lock_state!(self.state);
        dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ W[{:x}] <- {:02x}", st.ts, addr, val);
        match addr {
            SEEQ_STATION_ADDR_0..=SEEQ_STATION_ADDR_5 => {
                // 80C03: bank select in tx_cmd bits 6:5 determines what regs 0-5 address.
                let idx = (addr - SEEQ_STATION_ADDR_0) as usize;
                match st.tx_cmd & bank::MASK {
                    bank::B0 => {
                        st.station_addr[idx] = val;
                        if idx == 5 {
                            dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ station addr: {}", st.ts, mac_str(&st.station_addr));
                        }
                    }
                    bank::B1 => {
                        st.mcast_lsb[idx] = val;
                    }
                    bank::B2 => {
                        // bank 2: regs 0-1=mcast_msb, reg 2=pktgap, reg 3=ctl, regs 4-5=unused
                        match idx {
                            0 | 1 => st.mcast_msb[idx] = val,
                            2     => st.pktgap = val,
                            3     => st.ctl = val,
                            _     => {}
                        }
                    }
                    _ => {}
                }
            }
            SEEQ_RX_REG => {
                st.rx_cmd = val;
                dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=write_rx_reg rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
                Self::raise_interrupt(&mut st, false, &self.callback);
            }
            SEEQ_TX_REG => {
                st.tx_cmd = val;
                dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=write_tx_reg rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
                Self::raise_interrupt(&mut st, false, &self.callback);
            }
            _ => {}
        }
        BusStatus::Ready
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_lock_fn;
        let me = self.clone(); register_lock_fn("seeq::state",             move || me.state.is_locked());
        let me = self.clone(); register_lock_fn("seeq::sys_mem",           move || me.sys_mem.is_locked());
        let me = self.clone(); register_lock_fn("seeq::tx_prod",           move || me.tx_prod.is_locked());
        let me = self.clone(); register_lock_fn("seeq::tx_cons",           move || me.tx_cons.is_locked());
        let me = self.clone(); register_lock_fn("seeq::rx_prod",           move || me.rx_prod.is_locked());
        let me = self.clone(); register_lock_fn("seeq::rx_cons",           move || me.rx_cons.is_locked());
        let me = self.clone(); register_lock_fn("seeq::tx_wake",           move || me.tx_wake.0.is_locked());
        let me = self.clone(); register_lock_fn("seeq::rx_wake",           move || me.rx_wake.0.is_locked());
        let me = self.clone(); register_lock_fn("seeq::nat_ctl::snapshot", move || me.nat_ctl.snapshot.is_locked());
    }
}

// ── Device ────────────────────────────────────────────────────────────────────
impl Device for Seeq8003 {
    fn step(&self, _cycles: u64) {
        // DMA pumps run in the seeq-enet thread; nothing to do here.
    }

    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) { return; }

        // Take the channel endpoints — each is moved into exactly one thread.
        let tx_prod = self.tx_prod.lock().take().expect("tx_prod already taken");
        let tx_cons = self.tx_cons.lock().take().expect("tx_cons already taken");
        let rx_prod = self.rx_prod.lock().take().expect("rx_prod already taken");
        let rx_cons = self.rx_cons.lock().take().expect("rx_cons already taken");

        // ── seeq-nat thread: NAT engine ──────────────────────────────────────
        let config      = self.config.clone();
        let running_nat = self.running.clone();
        let tx_wake_nat = self.tx_wake.clone();
        let rx_wake_nat = self.rx_wake.clone();
        let nat_ctl     = self.nat_ctl.clone();
        thread::Builder::new().name("seeq-nat".into()).spawn(move || {
            NatEngine::new(config, tx_cons, rx_prod,
                           rx_wake_nat, tx_wake_nat,
                           running_nat, nat_ctl).run();
        }).expect("seeq-nat spawn");

        // ── seeq-enet thread: DMA pump loop ───────────────────────────────────
        // Polls TX DMA, pushes frames to NAT; receives frames from NAT, writes to RX DMA.
        // Waits on rx_wake (signalled by NAT when it enqueues an RX frame) with a 1ms
        // timeout so TX DMA is also polled periodically.
        //
        // Lock ordering — to avoid deadlock:
        //   CPU thread: rx_chan/tx_chan lock → SeeqState lock (CTRL register reads)
        //   Enet thread: DMA ops (chan lock, released per-op) → SeeqState lock (apply results)
        // The pumps run WITHOUT SeeqState lock; we take it once after to apply results atomically.
        let running_enet  = self.running.clone();
        let in_reset_enet = self.in_reset.clone();
        let tx_dma        = self.tx_dma.clone();
        let rx_dma        = self.rx_dma.clone();
        let sys_mem_enet: Option<Arc<dyn BusDevice>> = self.sys_mem.lock().clone();
        let state_enet    = self.state.clone();
        let callback      = self.callback.clone();
        let rx_wake_enet  = self.rx_wake.clone();
        let tx_wake_enet  = self.tx_wake.clone();
        let heartbeat_enet = self.heartbeat.clone();
        thread::Builder::new().name("seeq-enet".into()).spawn(move || {
            let mut tx_prod = tx_prod;
            let mut rx_cons = rx_cons;
            while running_enet.load(Ordering::Relaxed) {
                // Wait for an RX frame from NAT, or 1ms timeout to poll TX DMA
                {
                    let (lock, cvar) = &*rx_wake_enet;
                    let mut guard = lock.lock();
                    let _ = cvar.wait_for(&mut guard, Duration::from_millis(1));
                }

                // Snapshot rx_cmd and station_addr for address filtering outside state lock.
                let (rx_cmd_snap, station_addr_snap) = {
                    let st = lock_state!(state_enet);
                    (st.rx_cmd, st.station_addr)
                };

                // Run DMA pumps WITHOUT holding SeeqState lock.
                // Each DMA call takes the channel lock internally and releases it.
                // This breaks the ABBA deadlock:
                //   Old: SeeqState → chan lock  (enet thread)
                //   Old: chan lock → SeeqState  (CPU thread reading CTRL)
                let tx_result = if let Some(ref dma) = tx_dma {
                    Self::pump_tx(dma, &mut tx_prod, &tx_wake_enet, &in_reset_enet)
                } else { TxPumpResult::Nothing };

                let rx_result = if let Some(ref dma) = rx_dma {
                    Self::pump_rx(dma, &mut rx_cons, rx_cmd_snap, station_addr_snap, &in_reset_enet)
                } else { RxPumpResult::Nothing };

                // Now take SeeqState lock once to apply all results atomically.
                let mut st = lock_state!(state_enet);
                let mut dma_irq = false;

                let tx_activity = !matches!(tx_result, TxPumpResult::Nothing);
                match tx_result {
                    TxPumpResult::Sent { dma_irq: irq, writeback } => {
                        dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ TX complete dma_irq={}", st.ts, irq);
                        st.tx_stat = tx_stat::SUCCESS;
                        if let Some((addr, val)) = writeback {
                            if let Some(ref mem) = sys_mem_enet {
                                dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ TX writeback {:08x}+6 ← {:04x}", st.ts, addr - 6, val);
                                mem.write16(addr, val);
                            }
                        }
                        dma_irq |= irq;
                    }
                    TxPumpResult::Nothing => {}
                }

                let rx_activity = !matches!(rx_result, RxPumpResult::Nothing);
                match rx_result {
                    RxPumpResult::Delivered { dma_irq: irq, writeback, frame_len } => {
                        dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ RX delivered {} bytes dma_irq={}", st.ts, frame_len, irq);
                        st.rx_stat = rx_stat::GOOD | rx_stat::END;
                        if let Some((addr, val)) = writeback {
                            if let Some(ref mem) = sys_mem_enet {
                                dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ RX writeback {:08x}+6 ← {:04x}", st.ts, addr - 6, val);
                                mem.write16(addr, val);
                            }
                        }
                        dma_irq |= irq;
                    }
                    RxPumpResult::Refused => {
                        dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ RX DMA refused, retrying next tick", st.ts);
                    }
                    RxPumpResult::Nothing => {}
                }

                // Only call raise_interrupt if something actually happened this iteration.
                // Calling it unconditionally on every idle loop spin is wasteful and can
                // re-raise a stale interrupt that the driver already cleared via CLRINT.
                // Signal heartbeat activity bits for the display thread.
                if tx_activity { heartbeat_enet.fetch_or(crate::rex3::Rex3::HB_ENET_TX, Ordering::Relaxed); }
                if rx_activity { heartbeat_enet.fetch_or(crate::rex3::Rex3::HB_ENET_RX, Ordering::Relaxed); }

                let did_something = tx_activity || rx_activity || dma_irq;
                if did_something {
                    dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=enet_thread dma_irq={} rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, dma_irq, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
                    Self::raise_interrupt(&mut st, dma_irq, &callback);
                }
            }
        }).expect("seeq-enet spawn");
    }

    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        // The seeq-nat and seeq-enet threads own the ring-buffer endpoints.
        // They will exit on the next loop iteration once `running` is false,
        // but we don't join them here (Device::stop is &self, not &mut self).
        // Allocate fresh ring buffers now so the next start() can take them.
        // Any in-flight frames are discarded, which is fine — we're stopping.
        let (tx_prod, tx_cons) = RingBuffer::new(CHAN_CAPACITY);
        let (rx_prod, rx_cons) = RingBuffer::new(CHAN_CAPACITY);
        *self.tx_prod.lock() = Some(tx_prod);
        *self.tx_cons.lock() = Some(tx_cons);
        *self.rx_prod.lock() = Some(rx_prod);
        *self.rx_cons.lock() = Some(rx_cons);
    }

    fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
    fn get_clock(&self) -> u64   { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("seeq".into(), "seeq status".into()),
            ("net".into(),  "net status [tcp|udp|icmp|all] | net debug [tcp|udp|icmp] <on|off> [DEV]".into()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn Write + Send>) -> Result<(), String> {
        match cmd {
            "seeq" => {
                match args.first().copied() {
                    Some("status") | None => {
                        let st = lock_state!(self.state);
                        writeln!(w, "Station MAC : {}", mac_str(&st.station_addr)).ok();
                        writeln!(w, "Gateway MAC : {}", mac_str(&self.config.gateway_mac)).ok();
                        writeln!(w, "Gateway IP  : {}", self.config.gateway_ip).ok();
                        writeln!(w, "Client IP   : {}", self.config.client_ip).ok();
                        writeln!(w, "Netmask     : {}", self.config.netmask).ok();
                        writeln!(w, "rx_cmd={:#04x} rx_stat={:#04x} tx_cmd={:#04x} tx_stat={:#04x}",
                                 st.rx_cmd, st.rx_stat, st.tx_cmd, st.tx_stat).ok();
                        writeln!(w, "threads: {}", if self.is_running() { "running" } else { "stopped" }).ok();
                    }
                    _ => return Err("usage: seeq status".into()),
                }
            }
            "net" => {
                match args.first().copied() {
                    Some("status") | None => {
                        // net status [tcp|udp|icmp|all]
                        let proto = args.get(1).copied().unwrap_or("all");
                        let snap = self.nat_ctl.snapshot.lock();
                        let show_tcp  = matches!(proto, "tcp"  | "all");
                        let show_udp  = matches!(proto, "udp"  | "all");
                        let show_icmp = matches!(proto, "icmp" | "all");

                        if show_tcp {
                            writeln!(w, "TCP NAT ({} entries):", snap.tcp.len()).ok();
                            if snap.tcp.is_empty() {
                                writeln!(w, "  (none)").ok();
                            }
                            for e in &snap.tcp {
                                let in_flight = e.server_seq.wrapping_sub(e.server_seq_acked);
                                writeln!(w, "  client:{} → {}:{} age={}s{}",
                                         e.client_port, e.remote_ip, e.remote_port, e.age_secs,
                                         if e.fin_wait { " [fin_wait]" } else { "" }).ok();
                                writeln!(w, "    srv_seq={:#010x} srv_acked={:#010x} in_flight={} cli_win={} cli_seq={:#010x} rtx={}/{}B",
                                         e.server_seq, e.server_seq_acked, in_flight, e.client_win, e.client_seq,
                                         e.rtx_count, e.rtx_bytes).ok();
                            }
                        }
                        if show_udp {
                            writeln!(w, "UDP NAT ({} entries):", snap.udp.len()).ok();
                            if snap.udp.is_empty() {
                                writeln!(w, "  (none)").ok();
                            }
                            for e in &snap.udp {
                                writeln!(w, "  client:{} → {}:{} age={}s",
                                         e.client_port, e.remote_ip, e.remote_port, e.age_secs).ok();
                            }
                        }
                        if show_icmp {
                            writeln!(w, "ICMP NAT ({} entries):", snap.icmp.len()).ok();
                            if snap.icmp.is_empty() {
                                writeln!(w, "  (none)").ok();
                            }
                            for e in &snap.icmp {
                                writeln!(w, "  ident={} → {} age={}s",
                                         e.ident, e.remote_ip, e.age_secs).ok();
                            }
                        }
                    }
                    Some("debug") => {
                        let proto = args.get(1).copied().unwrap_or("");
                        let onoff = args.get(2).copied().unwrap_or("");
                        let enable = match onoff {
                            "on"  | "1" => true,
                            "off" | "0" => false,
                            _ => return Err("usage: net debug [tcp|udp|icmp] [on|off]".into()),
                        };
                        match proto {
                            "tcp"  => { self.nat_ctl.debug_tcp.store(enable,  std::sync::atomic::Ordering::Relaxed);
                                        writeln!(w, "TCP debug {}", if enable { "on" } else { "off" }).ok(); }
                            "udp"  => { self.nat_ctl.debug_udp.store(enable,  std::sync::atomic::Ordering::Relaxed);
                                        writeln!(w, "UDP debug {}", if enable { "on" } else { "off" }).ok(); }
                            "icmp" => { self.nat_ctl.debug_icmp.store(enable, std::sync::atomic::Ordering::Relaxed);
                                        writeln!(w, "ICMP debug {}", if enable { "on" } else { "off" }).ok(); }
                            _ => return Err("usage: net debug [tcp|udp|icmp] [on|off]".into()),
                        }
                    }
                    _ => return Err("usage: net status [tcp|udp|icmp|all] | net debug [tcp|udp|icmp] [on|off]".into()),
                }
            }
            _ => return Err("not found".into()),
        }
        Ok(())
    }
}

// ============================================================================
// Resettable + Saveable for Seeq8003
// ============================================================================

impl Resettable for Seeq8003 {
    /// Reset SEEQ registers and request NAT flush.
    /// Must be called with threads stopped.
    fn power_on(&self) {
        // Clear/reset the hardware registers via the existing reset() method.
        self.reset();
        // Signal the NAT thread to flush all TCP/UDP/ICMP tables on its next loop.
        self.nat_ctl.reset_nat.store(true, Ordering::Release);
    }
}

impl Saveable for Seeq8003 {
    fn save_state(&self) -> toml::Value {
        let st = self.state.lock();
        let mut tbl = toml::map::Map::new();
        tbl.insert("station_addr".into(), u8_slice_to_toml(&st.station_addr));
        tbl.insert("rx_cmd".into(),  hex_u8(st.rx_cmd));
        tbl.insert("rx_stat".into(), hex_u8(st.rx_stat));
        tbl.insert("tx_cmd".into(),  hex_u8(st.tx_cmd));
        tbl.insert("tx_stat".into(), hex_u8(st.tx_stat));
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut st = self.state.lock();
        if let Some(r) = get_field(v, "station_addr") { load_u8_slice(r, &mut st.station_addr); }
        if let Some(x) = get_field(v, "rx_cmd")  { if let Some(n) = toml_u8(x) { st.rx_cmd  = n; } }
        if let Some(x) = get_field(v, "rx_stat") { if let Some(n) = toml_u8(x) { st.rx_stat = n; } }
        if let Some(x) = get_field(v, "tx_cmd")  { if let Some(n) = toml_u8(x) { st.tx_cmd  = n; } }
        if let Some(x) = get_field(v, "tx_stat") { if let Some(n) = toml_u8(x) { st.tx_stat = n; } }
        Ok(())
    }
}
