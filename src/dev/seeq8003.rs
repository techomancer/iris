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

use crate::irq::IrqLine;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use crate::devlog::LogModule;
use parking_lot::Mutex;
use std::thread;
use std::time::Duration;


use crate::net::{eth_summary, mac_str, GatewayConfig, Link, NatControl, Rings};
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device, DmaClient, DmaStatus, Resettable, Saveable};
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
    /// Diagnostic tallies for the receive pump, printed by `seeq status`.
    /// Which of these stops moving at a stall says whether we are refusing
    /// frames, have none to give, or are delivering into a guest that has
    /// stopped reaping them.
    rx_delivered: u64,
    rx_refused: u64,
    rx_nothing: u64,
    /// Frames the address filter dropped, and of those, how many arrived
    /// while the receiver was off (rx_cmd match mode 0, as during the
    /// driver's reset of the interface). Nothing else reports either.
    rx_filtered: u64,
    rx_disabled: u64,
    /// Monotonic lock counter — incremented on every mutex acquisition.
    /// Prefix all debug messages with this to correlate across threads.
    pub ts:       u32,
}


// ── Ring-buffer capacity (number of frames) ───────────────────────────────────
const CHAN_CAPACITY: usize = 256;

// Max RX frames the enet thread drains from the ring per loop iteration before
// yielding to TX/interrupt work. Lets it clear a burst of address-filtered
// chatter (a busy LAN in promiscuous PCAP mode floods the ring with frames the
// guest rejects) in one pass, so chatter can't pin the ring full and starve the
// guest's own frames. Bounded so TX and interrupt servicing still run.
const RX_DRAIN_CAP: usize = 256;

// ── Main device struct ────────────────────────────────────────────────────────
pub struct Seeq8003 {
    state:    Arc<Mutex<SeeqState>>,
    /// The interrupt line. Driven with SeeqState's lock held, so whatever it
    /// leads into must not take that lock again.
    irq: Option<IrqLine>,
    rx_dma:   Option<Arc<dyn DmaClient>>,
    tx_dma:   Option<Arc<dyn DmaClient>>,
    sys_mem:  Mutex<Option<Arc<dyn BusDevice>>>,
    in_reset: Arc<AtomicBool>, // CH_RESET edge — lock-free check in pump thread
    config:   GatewayConfig,
    running:  Arc<AtomicBool>,
    /// The network backend: see [`Net`].
    net:      Mutex<Net>,
    /// Activity heartbeat — shared with Rex3 display thread.
    heartbeat: Arc<AtomicU64>,
}

/// The network backend behind the Ethernet. Each `start` runs the link
/// attached for it and attaches a fresh one for the next, so a restarted
/// machine gets new rings and a new backend.
struct Net {
    /// The link the next `start` runs.
    next: Option<Link>,
    /// The flag the running backend watches.
    live: Option<Arc<AtomicBool>>,
    /// enet waits on this for frames from the backend; `kick_*` notifies it.
    rx_wake: Option<Arc<crate::net::Wake>>,
    frames: Option<Arc<AtomicU64>>,
    /// The backend's controls, which every link of the run shares. A SEEQ
    /// with no link has a set of its own, so the monitor's `net` commands work.
    nat: Arc<NatControl>,
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
    Filtered, // frame rejected by address_filter (not for the guest) and discarded
    Refused,
    Delivered { dma_irq: bool, writeback: Option<(u32, u16)>, frame_len: usize },
}

// ── Seeq8003 ─────────────────────────────────────────────────────────────────
impl Seeq8003 {
    pub fn new(irq: Option<IrqLine>,
               rx_dma: Option<Arc<dyn DmaClient>>,
               tx_dma: Option<Arc<dyn DmaClient>>,
               heartbeat: Arc<AtomicU64>) -> Self {
        Self::with_config(irq, rx_dma, tx_dma, GatewayConfig::default(), heartbeat)
    }

    pub fn with_config(irq: Option<IrqLine>,
                       rx_dma: Option<Arc<dyn DmaClient>>,
                       tx_dma: Option<Arc<dyn DmaClient>>,
                       config: GatewayConfig,
                       heartbeat: Arc<AtomicU64>) -> Self {
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
                rx_delivered: 0,
                rx_refused: 0,
                rx_nothing: 0,
                rx_filtered: 0,
                rx_disabled: 0,
                ts:           0,
            })),
            irq, rx_dma, tx_dma,
            sys_mem: Mutex::new(None),
            in_reset: Arc::new(AtomicBool::new(false)),
            config,
            running:  Arc::new(AtomicBool::new(false)),
            net:      Mutex::new(Net { next: None, live: None, rx_wake: None, frames: None, nat: NatControl::new() }),
            heartbeat,
        }
    }

    /// Connect to the backend `[network]` selected. It runs from `start` to
    /// `stop`. Attach before the first `start`: a SEEQ with no link has no
    /// network.
    pub fn attach(&self, link: Link) {
        let mut net = self.net.lock();
        net.nat = link.nat();
        net.rx_wake = Some(link.rx_wake());
        net.frames = Some(link.frames());
        net.next = Some(link);
    }

    fn raise_interrupt(state: &mut SeeqState, dma_irq: bool, irq: &Option<IrqLine>) {
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
            if let Some(irq) = irq { irq.set(true); }
        } else if !level && state.intpend {
            state.intpend = false;
            if let Some(irq) = irq { irq.set(false); }
        }
    }

    /// Shared NAT control/stats handle (debug toggles, table reset, and the
    /// guest-frame counter the GUI's network indicator samples).
    pub fn nat_control(&self) -> Arc<NatControl> {
        self.net.lock().nat.clone()
    }

    /// Guest frames the backend has processed (see `Workstation::net_frames`).
    pub fn frames(&self) -> Option<Arc<AtomicU64>> {
        self.net.lock().frames.clone()
    }

    /// NAT addresses this gateway hands the guest: (client_ip, gateway_ip,
    /// netmask) — i.e. what the guest's ec0 should be configured as.
    pub fn gateway_addrs(&self) -> (std::net::Ipv4Addr, std::net::Ipv4Addr, std::net::Ipv4Addr) {
        (self.config.client_ip, self.config.gateway_ip, self.config.netmask)
    }

    /// Deassert the interrupt line. Called when the driver writes CLRINT.
    /// Mark both status registers as OLD so raise_interrupt won't immediately re-raise.
    pub fn reset_interrupt(&self) {
        let mut st = lock_state!(self.state);
        st.rx_stat |= rx_stat::OLD;
        st.tx_stat |= tx_stat::OLD;
        st.intpend = false;
        dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=reset_interrupt(CLRINT) rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
        if let Some(ref irq) = self.irq { irq.set(false); }
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
               tx_wake: &Arc<crate::net::Wake>,
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

        crate::pf_instant!(crate::perfetto::events::ENET_TX, frame.len());
        let _ = tx_prod.push(frame);
        tx_wake.notify();

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
            crate::pf_instant!(crate::perfetto::events::ENET_RX_FILTERED);
            return RxPumpResult::Filtered;
        }

        dlog_dev!(LogModule::Seeq, "SEEQ RX {} → guest DMA", eth_summary(frame));

        let (dst, _) = rx_dma.write(0, false); // pad[0]
        if dst.refused() {
            dlog_dev!(LogModule::Seeq, "SEEQ pump_rx: DMA not ready ({:#x}), will retry: {}", dst.0, eth_summary(frame));
            crate::pf_instant!(crate::perfetto::events::ENET_RX_REFUSED);
            return RxPumpResult::Refused; // frame stays in rx_cons for next iteration
        }
        // DMA is committed (pad[0] written); pop the frame now and finish writing it.
        let frame = rx_cons.pop().expect("peek succeeded so pop must succeed");
        crate::pf_instant!(crate::perfetto::events::ENET_RX, frame.len());
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
        self.kick_enet();
    }

    /// Wake the enet thread to deliver any pending RX frames.
    /// Call this when the RX DMA channel is activated so queued frames aren't lost.
    pub fn kick_rx(&self) {
        self.kick_enet();
    }

    fn kick_enet(&self) {
        if let Some(w) = self.net.lock().rx_wake.as_ref() {
            w.notify();
        }
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
        Self::raise_interrupt(&mut st, false, &self.irq);
    }

    /// Immediate reset (legacy): assert then immediately deassert.
    pub fn reset(&self) {
        self.assert_reset();
        self.deassert_reset();
    }

    pub fn read(&self, addr: u32) -> BusRead8 {
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
                Self::raise_interrupt(&mut st, false, &self.irq);
                v
            }
            SEEQ_TX_REG => {
                let v = st.tx_stat;
                st.tx_stat |= tx_stat::OLD;
                dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=read_tx_reg rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
                Self::raise_interrupt(&mut st, false, &self.irq);
                v
            }
            _ => 0,
        };
        dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ R[{:x}] -> {:02x}", st.ts, addr, val);
        BusRead8::ok(val)
    }

    pub fn write(&self, addr: u32, val: u8) -> u32 {
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
                Self::raise_interrupt(&mut st, false, &self.irq);
            }
            SEEQ_TX_REG => {
                st.tx_cmd = val;
                dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=write_tx_reg rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
                Self::raise_interrupt(&mut st, false, &self.irq);
            }
            _ => {}
        }
        BUS_OK
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_lock_fn;
        let me = self.clone(); register_lock_fn("seeq::state",             move || me.state.is_locked());
        let me = self.clone(); register_lock_fn("seeq::sys_mem",           move || me.sys_mem.is_locked());
        let me = self.clone(); register_lock_fn("seeq::net",               move || me.net.is_locked());
        let me = self.clone(); register_lock_fn("seeq::nat_ctl::snapshot", move || me.net.try_lock().map_or(false, |n| n.nat.snapshot.is_locked()));
    }
}

// ── Device ────────────────────────────────────────────────────────────────────
impl Device for Seeq8003 {
    fn step(&self, _cycles: u64) {
        // DMA pumps run in the seeq-enet thread; nothing to do here.
    }

    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) { return; }

        // The link for this run, and a fresh one for the next.
        let mut link = {
            let mut net = self.net.lock();
            let Some(link) = net.next.take() else {
                self.running.store(false, Ordering::SeqCst);
                return;
            };
            match link.renew() {
                Ok(next) => net.next = Some(next),
                Err(e) => eprintln!("iris: network: {e}; the Ethernet stays down after this run"),
            }
            net.live = Some(link.running());
            link
        };

        // ── seeq-net thread: the network backend ─────────────────────────────
        eprintln!("iris: networking backend = {}", link.describe());
        link.spawn("seeq-net");
        let Rings { to_net: tx_prod, from_net: rx_cons, tx_wake: tx_wake_enet, rx_wake: rx_wake_enet } = link.into_rings();

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
        let irq      = self.irq.clone();
        let heartbeat_enet = self.heartbeat.clone();
        thread::Builder::new().name("seeq-enet".into()).spawn(move || {
            let mut tx_prod = tx_prod;
            let mut rx_cons = rx_cons;
            // Whether the last pass moved a frame. While frames are moving we go
            // straight round again: a wake-up that arrived while we were busy is
            // lost (the condvar has no memory), so waiting here would pace a
            // burst at one frame per timeout.
            let mut busy = false;
            // How long an idle pass waits for a wake before polling TX DMA.
            // Backs off from 1ms to ENET_POLL_MAX while nothing moves so an
            // idle network costs ~20 wakeups/s instead of 1000. The real wake
            // paths — kick_tx from HPC3 when the CPU arms the TX channel, and
            // rx_wake from NAT when a frame arrives — never lose a notify
            // (Wake keeps it pending for the next wait), so the timeout is a
            // safety net for a re-armed descriptor chain that arrives while
            // the channel stays active. Any notify or activity drops it back
            // to 1ms, so bursts still drain at speed.
            const ENET_POLL_MAX: Duration = Duration::from_millis(50);
            let mut poll = Duration::from_millis(1);
            while running_enet.load(Ordering::Relaxed) {
                let mut notified = false;
                // Wait for an RX frame from NAT (or a TX kick), or the poll
                // timeout to check TX DMA.
                if !busy {
                    notified = rx_wake_enet.wait(poll);
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

                // Drain the RX ring: discard leading chatter (frames the guest's
                // address_filter rejects) cheaply so a busy LAN can't pin the ring
                // full and starve the guest's own frames, then handle at most one
                // for-guest frame this iteration (one delivery per interrupt).
                // Bounded by RX_DRAIN_CAP so TX and interrupt servicing still run.
                let mut filtered = 0;
                let rx_result = if let Some(ref dma) = rx_dma {
                    let mut r = RxPumpResult::Nothing;
                    for _ in 0..RX_DRAIN_CAP {
                        match Self::pump_rx(dma, &mut rx_cons, rx_cmd_snap, station_addr_snap, &in_reset_enet) {
                            RxPumpResult::Filtered => { filtered += 1; continue }   // chatter discarded; keep draining
                            other => { r = other; break; }        // Delivered / Refused / Nothing
                        }
                    }
                    r
                } else { RxPumpResult::Nothing };

                // Now take SeeqState lock once to apply all results atomically.
                let mut st = lock_state!(state_enet);
                let mut dma_irq = false;
                st.rx_filtered += filtered;
                if rx_cmd_snap & rx_cmd::MATCH_MASK == rx_cmd::MATCH_DISABLE {
                    st.rx_disabled += filtered;
                }

                let tx_activity = !matches!(tx_result, TxPumpResult::Nothing);
                let tx_moved = matches!(tx_result, TxPumpResult::Sent { .. });
                match tx_result {
                    TxPumpResult::Sent { dma_irq: irq, writeback } => {
                        dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ TX complete dma_irq={}", st.ts, irq);
                        st.tx_stat = tx_stat::SUCCESS;
                        if let Some((addr, val)) = writeback {
                            if let Some(ref mem) = sys_mem_enet {
                                dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ TX writeback {:08x}+6 ← {:04x}", st.ts, addr - 6, val);
                                mem.write16(addr as u64, val);
                            }
                        }
                        dma_irq |= irq;
                    }
                    TxPumpResult::Nothing => {}
                }

                let rx_activity = !matches!(rx_result, RxPumpResult::Nothing);
                let rx_moved = matches!(rx_result, RxPumpResult::Delivered { .. });
                match rx_result {
                    RxPumpResult::Delivered { dma_irq: irq, writeback, frame_len } => {
                        dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ RX delivered {} bytes dma_irq={}", st.ts, frame_len, irq);
                        st.rx_delivered += 1;
                        st.rx_stat = rx_stat::GOOD | rx_stat::END;
                        if let Some((addr, val)) = writeback {
                            if let Some(ref mem) = sys_mem_enet {
                                dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ RX writeback {:08x}+6 ← {:04x}", st.ts, addr - 6, val);
                                mem.write16(addr as u64, val);
                            }
                        }
                        dma_irq |= irq;
                    }
                    RxPumpResult::Refused => {
                        dlog_dev!(LogModule::Seeq, "[ts={}] SEEQ RX DMA refused, retrying next tick", st.ts);
                        st.rx_refused += 1;
                    }
                    // Filtered never reaches here (the drain loop consumes it), but
                    // keep the match exhaustive over RxPumpResult.
                    RxPumpResult::Nothing | RxPumpResult::Filtered => { st.rx_nothing += 1; }
                }

                // Only call raise_interrupt if something actually happened this iteration.
                // Calling it unconditionally on every idle loop spin is wasteful and can
                // re-raise a stale interrupt that the driver already cleared via CLRINT.
                // Signal heartbeat activity bits for the display thread.
                if tx_activity { heartbeat_enet.fetch_or(crate::dev::ng1::rex3::Rex3::HB_ENET_TX, Ordering::Relaxed); }
                if rx_activity { heartbeat_enet.fetch_or(crate::dev::ng1::rex3::Rex3::HB_ENET_RX, Ordering::Relaxed); }

                let did_something = tx_activity || rx_activity || dma_irq;
                busy = tx_moved || rx_moved;
                // Idle-poll backoff (see the wait above): reset while the
                // guest or the network is active, grow while idle.
                if notified || did_something {
                    poll = Duration::from_millis(1);
                } else if poll < ENET_POLL_MAX {
                    poll = (poll * 2).min(ENET_POLL_MAX);
                }
                if did_something {
                    dlog_dev!(LogModule::Seeq, "[ts={}] raise_interrupt caller=enet_thread dma_irq={} rx_stat={:02x} rx_cmd={:02x} tx_stat={:02x} tx_cmd={:02x}", st.ts, dma_irq, st.rx_stat, st.rx_cmd, st.tx_stat, st.tx_cmd);
                    Self::raise_interrupt(&mut st, dma_irq, &irq);
                }
            }
        }).expect("seeq-enet spawn");
    }

    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        // The seeq-nat and seeq-enet threads own the ring-buffer endpoints.
        // They will exit on the next loop iteration once `running` is false,
        // but we don't join them here (Device::stop is &self, not &mut self).
        // The next start() runs a fresh link. Any in-flight frames are
        // discarded, which is fine — we're stopping.
        if let Some(live) = self.net.lock().live.take() {
            live.store(false, Ordering::SeqCst);
        }
    }

    fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
    fn get_clock(&self) -> u64   { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("seeq".into(), "seeq status".into()),
            ("net".into(),  "net status [tcp|udp|icmp|all] | net debug [tcp|udp|icmp] <on|off> [DEV] | net interfaces".into()),
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
                        writeln!(w, "intpend={} rx_delivered={} rx_refused={} rx_nothing={}",
                                 st.intpend, st.rx_delivered, st.rx_refused, st.rx_nothing).ok();
                        writeln!(w, "rx_filtered={} (receiver off: {})", st.rx_filtered, st.rx_disabled).ok();
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
                        let ctl = self.nat_control();
                        let snap = ctl.snapshot.lock();
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
                            "tcp"  => { self.nat_control().debug_tcp.store(enable,  std::sync::atomic::Ordering::Relaxed);
                                        writeln!(w, "TCP debug {}", if enable { "on" } else { "off" }).ok(); }
                            "udp"  => { self.nat_control().debug_udp.store(enable,  std::sync::atomic::Ordering::Relaxed);
                                        writeln!(w, "UDP debug {}", if enable { "on" } else { "off" }).ok(); }
                            "icmp" => { self.nat_control().debug_icmp.store(enable, std::sync::atomic::Ordering::Relaxed);
                                        writeln!(w, "ICMP debug {}", if enable { "on" } else { "off" }).ok(); }
                            _ => return Err("usage: net debug [tcp|udp|icmp] [on|off]".into()),
                        }
                    }
                    Some("interfaces") | Some("ifaces") => {
                        // List host interfaces available for PCAP bridging.
                        #[cfg(feature = "pcap")]
                        {
                            write!(w, "{}", crate::net::net_pcap::format_interfaces()).ok();
                        }
                        #[cfg(not(feature = "pcap"))]
                        {
                            writeln!(w, "PCAP support not built in (rebuild with --features pcap).").ok();
                        }
                    }
                    _ => return Err("usage: net status [tcp|udp|icmp|all] | net debug [tcp|udp|icmp] [on|off] | net interfaces".into()),
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
        self.nat_control().reset_nat.store(true, Ordering::Release);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 1.7 round-trip: a fresh Seeq loaded from a captured save_state
    /// must re-serialize byte-identically. Mutates station_addr and the four
    /// rx/tx command/status registers.
    #[test]
    fn save_load_round_trip() {
        let src = Seeq8003::new(None, None, None, Arc::new(AtomicU64::new(0)));
        {
            let mut st = src.state.lock();
            st.station_addr = [0x08, 0x00, 0x69, 0x12, 0x34, 0x56];
            st.rx_cmd  = 0x18;
            st.rx_stat = 0xa1;
            st.tx_cmd  = 0x40;
            st.tx_stat = 0x82;
        }
        let v1 = src.save_state();

        let dst = Seeq8003::new(None, None, None, Arc::new(AtomicU64::new(0)));
        dst.load_state(&v1).expect("load_state");
        let v2 = dst.save_state();

        assert_eq!(v1, v2, "Seeq8003 save_state mismatch after load_state round-trip");
    }

    /// A receive channel that takes every frame and counts them.
    #[derive(Default)]
    struct Ring { filled: Mutex<usize> }

    impl DmaClient for Ring {
        fn read(&self) -> Option<(u32, DmaStatus, Option<(u32, u16)>)> { None }
        fn write(&self, _val: u32, eop: bool) -> (DmaStatus, Option<(u32, u16)>) {
            if eop { *self.filled.lock() += 1; }
            (DmaStatus::ok(), None)
        }
    }

    fn settle<F: Fn() -> bool>(done: F) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !done() && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        thread::sleep(Duration::from_millis(50));
    }

    /// Frames the address filter drops while the receiver is off are
    /// counted, since nothing else reports them.
    #[test]
    fn frames_dropped_with_the_receiver_off_are_counted() {
        let ring = Arc::new(Ring::default());
        let seeq = Seeq8003::new(None, Some(ring.clone()), None, Arc::new(AtomicU64::new(0)));
        let (link, _sent, mut inject) = Link::loopback();
        let wake = link.rx_wake();
        seeq.attach(link);
        for _ in 0..3 {
            inject.push(vec![0xff; 60]).unwrap();
        }
        seeq.start();
        wake.notify();
        settle(|| inject.slots() == CHAN_CAPACITY);
        seeq.stop();
        let st = seeq.state.lock();
        assert_eq!((st.rx_filtered, st.rx_disabled, *ring.filled.lock()), (3, 3, 0));
    }
}
