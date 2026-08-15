// DaynaPort SCSI/Link (DP0801 / DP0802) — a SCSI-attached Ethernet adapter.
//
// The card presents as a SCSI **type 3 (Processor)** target and moves Ethernet
// frames with five vendor-specific 6-byte CDBs (0x08 READ, 0x09 RETRIEVE
// STATS, 0x0A WRITE, 0x0C SET INTERFACE MODE, 0x0E ENABLE/DISABLE). Modern
// re-implementations (BlueSCSI V2, ZuluSCSI, PiSCSI, SCSI2SD) speak the same
// protocol, which is how vintage SGI/Mac/Atari machines get networking over
// nothing but a SCSI bus. Reference: SLINKCMD.TXT (Roger Burrows, rev 1.20).
//
// ── Where this sits in IRIS ──────────────────────────────────────────────────
//
// This is the same shape as `seeq8003` with a different front end: the device
// owns two `rtrb` rings and hands the far ends to a `NatEngine` (or PcapEngine)
// running on its own `daynaN-nat` thread. Instead of DMA descriptor rings driven
// by MMIO registers, the front end is five SCSI CDBs:
//
//     WRITE(6)  → push a frame to `tx_prod`   (guest → world)
//     READ(6)   → pop frames from `rx_cons`   (world → guest), each wrapped in
//                 a 6-byte DaynaPort record header
//
// The whole NAT stack — DHCP, DNS, ICMP, TCP/UDP, NFS, `NatControl` telemetry —
// comes along unchanged.
//
// ── Concurrency ──────────────────────────────────────────────────────────────
//
// `request()` runs on the WD33C93A worker thread with `Wd33c93aState` locked.
// Everything it touches is either owned outright (`&mut self`) or lock-free
// (the rtrb rings, the atomics), so it never blocks and never calls back up
// into the controller — see HACKING.md on per-device concurrency.
//
// ── Endianness ───────────────────────────────────────────────────────────────
//
// Every multi-byte protocol field is big-endian *on the wire* and is built with
// explicit `to_be_bytes()` / shifts here at the protocol edge. Nothing in this
// file byte-swaps a host integer.

use std::io::Error as IoError;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use parking_lot::{Condvar, Mutex};
use rtrb::RingBuffer;

use crate::devlog::LogModule;
use crate::net::{eth_summary, mac_str, GatewayConfig, NatControl, NatEngine};
use crate::scsi::{ScsiRequest, ScsiResponse};

// ── Command set ──────────────────────────────────────────────────────────────
// NB: 0x08 and 0x0A collide with SCSI READ(6)/WRITE(6). A DaynaPort target is
// dispatched *before* the storage opcodes in `ScsiDevice::request`, so these
// never reach the disk path.
pub mod dp_cmd {
    /// Receive queued packet(s). Device → host.
    pub const READ: u8 = 0x08;
    /// Retrieve statistics (MAC address + counters). Device → host.
    pub const RETRIEVE_STATS: u8 = 0x09;
    /// Transmit one packet. Host → device.
    pub const WRITE: u8 = 0x0a;
    /// Set interface mode (broadcast/multicast filtering). No data.
    pub const SET_IFACE_MODE: u8 = 0x0c;
    /// Enable / disable the interface. No data.
    pub const ENABLE: u8 = 0x0e;
}

/// Per-record header on a READ response: 2-byte length + 4-byte flags.
const RX_HDR_LEN: usize = 6;
/// Trailing CRC the protocol counts in `pktlen` but nobody validates. The
/// driver strips `pktlen - 4` bytes and discards these, so zeros are fine —
/// but they must physically be there or every frame arrives 4 bytes short.
const CRC_LEN: usize = 4;

/// READ record flag: more packets are still queued in the device; the host
/// should issue another READ immediately rather than waiting for its next poll.
const FLAG_MORE: u32 = 0x0000_0010;
/// READ CDB byte 5 bit 6: pack as many whole frames as fit into one response.
/// Bit 7 is undocumented; every implementation accepts the driver's `0xC0`.
const READ_FLAG_MULTI: u8 = 0x40;

/// SET INTERFACE MODE byte 4: receive broadcasts.
const MODE_BROADCAST: u8 = 0x04;
/// ENABLE/DISABLE byte 5: 0x80 = enable, 0x00 = disable.
const ENABLE_ON: u8 = 0x80;

/// RETRIEVE STATS response length: 6-byte MAC + 12 bytes of counters.
const STATS_LEN: usize = 18;

/// Upper bound on a single READ response, so a bogus CDB can't make us
/// allocate wildly. The IRIX driver asks for 3072.
const MAX_READ_LEN: usize = 16 * 1024;

/// Frames in flight per direction. Same depth as `seeq8003`.
const CHAN_CAPACITY: usize = 256;

/// Smallest thing that can plausibly be an Ethernet frame (dst+src+type).
const MIN_FRAME: usize = 14;
/// Largest frame we will hand to the backend (1500 MTU + 14 header + 4 VLAN).
const MAX_FRAME: usize = 1518;

/// Default NAT subnet for a DaynaPort target. Deliberately *not* the onboard
/// SEEQ's 192.168.0.0/24: each target gets its own `NatEngine`, and putting
/// `dp0` on a different subnet from `ec0` proves traffic really went through
/// the DaynaPort. Override per target with `subnet = "..."` in `[scsi.N]`.
pub const DEFAULT_SUBNET: &str = "192.168.10.0/24";

/// Default MAC for target `id`: the real DaynaPort `00:80:19` OUI, then
/// `44:50` ("DP") and the SCSI target id, so two targets never collide.
/// Deliberately distinct from the IRIX driver's `00:80:19:00:00:NN`
/// placeholder, so a MAC read via RETRIEVE STATS is visibly different from one
/// the driver made up.
pub fn default_mac(id: usize) -> [u8; 6] {
    [0x00, 0x80, 0x19, 0x44, 0x50, id as u8]
}

#[derive(Default, Clone, Copy)]
struct DpStats {
    tx_frames:  u64,
    tx_dropped: u64,
    rx_frames:  u64,
    rx_filtered: u64,
}

pub struct DaynaPort {
    /// SCSI id this target answers on — used for the default MAC and thread name.
    target_id: usize,
    mac: [u8; 6],
    /// Set by ENABLE/DISABLE (0x0E). While false, RX returns empty responses
    /// and TX frames are dropped.
    enabled: bool,
    /// Set by SET INTERFACE MODE (0x0C) byte 4 bit 2.
    broadcast: bool,
    config: GatewayConfig,
    running: Arc<AtomicBool>,
    nat_ctl: Arc<NatControl>,
    /// Activity heartbeat shared with the display thread (lights the status
    /// bar's network indicators, same as the onboard SEEQ).
    heartbeat: Arc<AtomicU64>,

    /// guest → world. Filled by WRITE(6), drained by the NAT thread.
    tx_prod: rtrb::Producer<Vec<u8>>,
    /// world → guest. Filled by the NAT thread, drained by READ(6).
    rx_cons: rtrb::Consumer<Vec<u8>>,
    /// The far ends, held until `start()` moves them into the NAT thread.
    nat_ends: Option<(rtrb::Consumer<Vec<u8>>, rtrb::Producer<Vec<u8>>)>,
    /// Signalled by us when a TX frame is queued (the NAT thread waits on it).
    tx_wake: Arc<(Mutex<()>, Condvar)>,
    /// Signalled by the NAT thread when an RX frame is queued. Nothing waits on
    /// it here — the driver polls READ every 10 ms — but `NatEngine` needs it.
    rx_wake: Arc<(Mutex<()>, Condvar)>,

    stats: DpStats,
    /// Sense data served on REQUEST SENSE, set by the last CHECK CONDITION.
    pending_sense: [u8; 18],
}

impl DaynaPort {
    pub fn new(target_id: usize, mac: [u8; 6], config: GatewayConfig,
               heartbeat: Arc<AtomicU64>) -> Self {
        let (tx_prod, tx_cons) = RingBuffer::new(CHAN_CAPACITY);
        let (rx_prod, rx_cons) = RingBuffer::new(CHAN_CAPACITY);
        Self {
            target_id,
            mac,
            enabled: false,
            broadcast: false,
            config,
            running: Arc::new(AtomicBool::new(false)),
            nat_ctl: NatControl::new(),
            heartbeat,
            tx_prod,
            rx_cons,
            nat_ends: Some((tx_cons, rx_prod)),
            tx_wake: Arc::new((Mutex::new(()), Condvar::new())),
            rx_wake: Arc::new((Mutex::new(()), Condvar::new())),
            stats: DpStats::default(),
            pending_sense: [0u8; 18],
        }
    }

    pub fn mac(&self) -> [u8; 6] { self.mac }
    pub fn target_id(&self) -> usize { self.target_id }

    /// Shared NAT control/stats handle (debug toggles, table reset, and the
    /// guest-frame counter the GUI's network indicator samples).
    pub fn nat_control(&self) -> Arc<NatControl> { self.nat_ctl.clone() }

    /// NAT addresses this target hands the guest: (client_ip, gateway_ip,
    /// netmask) — i.e. what the guest's `dp0` should be configured as.
    pub fn gateway_addrs(&self) -> (std::net::Ipv4Addr, std::net::Ipv4Addr, std::net::Ipv4Addr) {
        (self.config.client_ip, self.config.gateway_ip, self.config.netmask)
    }

    /// Spawn the backend thread. Mirrors `seeq8003::Device::start`, including
    /// the PCAP fallback message, so a DaynaPort can be bridged too.
    pub fn start(&mut self) {
        if self.running.swap(true, Ordering::SeqCst) { return; }
        let Some((tx_cons, rx_prod)) = self.nat_ends.take() else {
            self.running.store(false, Ordering::SeqCst);
            return;
        };
        let config      = self.config.clone();
        let running_nat = self.running.clone();
        let tx_wake_nat = self.tx_wake.clone();
        let rx_wake_nat = self.rx_wake.clone();
        let nat_ctl     = self.nat_ctl.clone();
        let id          = self.target_id;
        let name        = format!("dayna{}-nat", id);
        thread::Builder::new().name(name).spawn(move || {
            #[cfg(feature = "pcap")]
            {
                use crate::net::NetBackend;
                if config.mode == crate::config::NetMode::Pcap {
                    eprintln!("iris: DaynaPort {} backend = PCAP (bridged)", id);
                    let mut engine = crate::net_pcap::PcapEngine::new(
                        config, tx_cons, rx_prod,
                        rx_wake_nat, tx_wake_nat,
                        running_nat, nat_ctl);
                    engine.run();
                    return;
                }
                eprintln!("iris: DaynaPort {} backend = NAT (software gateway)", id);
                NatEngine::new(config, tx_cons, rx_prod,
                               rx_wake_nat, tx_wake_nat,
                               running_nat, nat_ctl).run();
            }
            #[cfg(not(feature = "pcap"))]
            {
                if config.mode == crate::config::NetMode::Pcap {
                    eprintln!("iris: DaynaPort {}: [network] mode = \"pcap\" requested but this \
                               build lacks --features pcap; falling back to NAT gateway.", id);
                } else {
                    eprintln!("iris: DaynaPort {} backend = NAT (software gateway)", id);
                }
                NatEngine::new(config, tx_cons, rx_prod,
                               rx_wake_nat, tx_wake_nat,
                               running_nat, nat_ctl).run();
            }
        }).expect("dayna-nat spawn");
    }

    /// Stop the backend thread. The thread owns its ring endpoints and exits on
    /// its next loop iteration; fresh rings are allocated here so a later
    /// `start()` works. In-flight frames are discarded — the same answer
    /// `seeq8003::stop` gives.
    pub fn stop(&mut self) {
        if !self.running.swap(false, Ordering::SeqCst) { return; }
        self.tx_wake.1.notify_all();
        let (tx_prod, tx_cons) = RingBuffer::new(CHAN_CAPACITY);
        let (rx_prod, rx_cons) = RingBuffer::new(CHAN_CAPACITY);
        self.tx_prod  = tx_prod;
        self.rx_cons  = rx_cons;
        self.nat_ends = Some((tx_cons, rx_prod));
    }

    /// Machine reset: interface off, queued frames dropped, NAT tables flushed
    /// on the backend thread's next iteration (as `Seeq8003::power_on` does).
    pub fn power_on(&mut self) {
        self.enabled   = false;
        self.broadcast = false;
        self.drain_rx();
        self.stats = DpStats::default();
        self.nat_ctl.reset_nat.store(true, Ordering::Release);
    }

    /// Human-readable status for `scsi dayna`.
    pub fn status_lines(&self) -> Vec<String> {
        let (client, gw, mask) = self.gateway_addrs();
        vec![
            format!("SCSI ID {}: DaynaPort SCSI/Link", self.target_id),
            format!("  MAC        : {}", mac_str(&self.mac)),
            format!("  Gateway MAC: {}", mac_str(&self.config.gateway_mac)),
            format!("  Gateway IP : {}  client {}  netmask {}", gw, client, mask),
            format!("  State      : {}  broadcast={}  backend={}",
                    if self.enabled { "enabled" } else { "disabled" },
                    self.broadcast,
                    if self.running.load(Ordering::Relaxed) { "running" } else { "stopped" }),
            format!("  TX         : {} frames, {} dropped (ring full)",
                    self.stats.tx_frames, self.stats.tx_dropped),
            format!("  RX         : {} frames, {} filtered, {} queued",
                    self.stats.rx_frames, self.stats.rx_filtered, self.rx_cons.slots()),
        ]
    }

    // ── SCSI command dispatch ────────────────────────────────────────────────

    /// Execute one CDB. Never blocks: the driver polls READ every 10 ms and a
    /// blocking read would wedge the interface.
    pub fn request(&mut self, req: &ScsiRequest) -> Result<ScsiResponse, IoError> {
        let cdb = &req.cdb;
        if cdb.len() < 6 {
            return Ok(self.check_condition(0x05, 0x20, 0x00)); // Invalid command
        }
        let resp = match cdb[0] {
            crate::scsi::scsi_cmd::TEST_UNIT_READY => good(),
            crate::scsi::scsi_cmd::REQUEST_SENSE   => self.exec_request_sense(cdb),
            crate::scsi::scsi_cmd::INQUIRY         => self.exec_inquiry(cdb),
            dp_cmd::READ            => self.exec_read(cdb),
            dp_cmd::RETRIEVE_STATS  => self.exec_retrieve_stats(cdb),
            dp_cmd::WRITE           => self.exec_write(cdb, req.data_in.as_ref()),
            dp_cmd::SET_IFACE_MODE  => self.exec_set_iface_mode(cdb),
            dp_cmd::ENABLE          => self.exec_enable(cdb),
            other => {
                dlog_dev!(LogModule::Net, "DaynaPort {}: unsupported command {:02x} cdb={:02x?}",
                          self.target_id, other, cdb);
                self.check_condition(0x05, 0x20, 0x00) // Illegal Request: Invalid Command
            }
        };
        Ok(resp)
    }

    /// INQUIRY — identify as a DaynaPort SCSI/Link. The IRIX driver matches on
    /// the `"Dayna"` / `"SCSI/Link"` prefixes only, but Mac and Atari drivers
    /// are pickier, so emit the full padded strings.
    fn exec_inquiry(&self, cdb: &[u8]) -> ScsiResponse {
        let alloc_len = cdb[4] as usize;
        let lun = (cdb[1] >> 5) & 0x7;
        let mut data = vec![0u8; 36];
        if lun == 0 {
            data[0] = 0x03; // Processor device
            data[1] = 0x00; // not removable
            data[2] = 0x02; // ANSI SCSI-2
            data[3] = 0x02; // SCSI-2 response format
            data[4] = 31;   // additional length (36 - 5)
            data[8..16].copy_from_slice(b"Dayna   ");
            data[16..32].copy_from_slice(b"SCSI/Link       ");
            data[32..36].copy_from_slice(b"1.4a");
        } else {
            // SPC-2 8.2.5: PQ=011b, PDT=0x1F. data[4] must still report the standard
            // 31-byte additional length or Linux's scsi_scan.c computes response_len
            // = data[4]+5 = 5, clamps inquiry_len to 5, and logs "INQUIRY result too
            // short (5), using 36" even though the short response was intentional.
            data[0] = 0x7F; // LUN not present
            data[2] = 0x02; // ANSI SCSI-2
            data[3] = 0x02; // SCSI-2 response format
            data[4] = 31;   // additional length (36 - 5)
        }
        data.truncate(alloc_len.min(data.len()));
        ScsiResponse { status: 0x00, data }
    }

    fn exec_request_sense(&mut self, cdb: &[u8]) -> ScsiResponse {
        let alloc_len = cdb[4] as usize;
        let sense = self.pending_sense;
        self.pending_sense = [0u8; 18];
        self.pending_sense[0] = 0x70;
        let data = sense[..sense.len().min(alloc_len.max(18))].to_vec();
        ScsiResponse { status: 0x00, data }
    }

    /// 0x08 READ — hand the host every queued frame that fits.
    ///
    /// Response layout, records back to back:
    ///
    /// ```text
    ///  off  size    field
    ///   0    2      pktlen, BIG-ENDIAN — frame length INCLUDING the 4-byte
    ///                trailing CRC, EXCLUDING this 6-byte header
    ///   2    4      flags, BIG-ENDIAN — 0x10 = more still queued, 0 = last
    ///   6  pktlen   the Ethernet frame, then 4 CRC bytes
    /// ```
    ///
    /// A `pktlen` of 0 means "nothing (more) here", so an idle device answers
    /// with six zero bytes rather than stalling.
    fn exec_read(&mut self, cdb: &[u8]) -> ScsiResponse {
        let want = ((((cdb[3] as usize) << 8) | cdb[4] as usize)).min(MAX_READ_LEN);
        // Bit 6 = multi-packet. Anything else falls back to one frame per READ
        // rather than erroring (byte 5 is only partly documented).
        let multi = (cdb[5] & READ_FLAG_MULTI) != 0;

        let mut records: Vec<Vec<u8>> = Vec::new();
        let mut used = 0usize;
        while self.enabled {
            let frame_len = match self.rx_cons.peek() {
                Ok(f) => f.len(),
                Err(_) => break,
            };
            // Never emit a record that would overrun the requested transfer
            // length — the driver bounds-checks and silently drops the tail.
            // Leave the frame queued for the next READ instead… unless it
            // would not fit in an *empty* response either, in which case
            // leaving it queued would stall the ring forever: drop it.
            if used + RX_HDR_LEN + frame_len + CRC_LEN > want {
                if used > 0 || RX_HDR_LEN + frame_len + CRC_LEN <= want { break; }
                let _ = self.rx_cons.pop();
                self.stats.rx_filtered += 1;
                continue;
            }
            let frame = self.rx_cons.pop().expect("peek succeeded so pop must succeed");
            if !self.accepts(&frame) {
                self.stats.rx_filtered += 1;
                continue;
            }
            dlog_dev!(LogModule::Net, "DaynaPort {} RX {}", self.target_id, eth_summary(&frame));
            used += RX_HDR_LEN + frame.len() + CRC_LEN;
            records.push(frame);
            if !multi { break; }
        }

        // MORE on the last record tells the driver to issue another READ right
        // away instead of waiting for its next 10 ms tick; on any earlier record
        // it is what makes the driver keep parsing this response at all.
        let more_queued = !self.rx_cons.is_empty();
        let n = records.len();
        let mut data = Vec::with_capacity(used.max(RX_HDR_LEN));
        for (i, frame) in records.iter().enumerate() {
            let pktlen = (frame.len() + CRC_LEN) as u16;
            let flags: u32 = if i + 1 < n || more_queued { FLAG_MORE } else { 0 };
            data.extend_from_slice(&pktlen.to_be_bytes());
            data.extend_from_slice(&flags.to_be_bytes());
            data.extend_from_slice(frame);
            data.extend_from_slice(&[0u8; CRC_LEN]); // CRC placeholder; driver discards it
        }
        if n > 0 {
            self.stats.rx_frames += n as u64;
            self.heartbeat.fetch_or(crate::rex3::Rex3::HB_ENET_RX, Ordering::Relaxed);
        } else {
            // Nothing queued (or the interface is disabled): a zero pktlen is
            // how the driver is told to stop parsing. Even that never exceeds
            // the requested transfer length.
            data.resize(RX_HDR_LEN.min(want), 0);
        }
        ScsiResponse { status: 0x00, data }
    }

    /// 0x09 RETRIEVE STATS — 6-byte MAC first, then counters. The IRIX driver
    /// asks for 18 bytes and reads the MAC out of the head of it.
    fn exec_retrieve_stats(&mut self, cdb: &[u8]) -> ScsiResponse {
        let alloc_len = if cdb[4] == 0 { STATS_LEN } else { cdb[4] as usize };
        let mut data = vec![0u8; STATS_LEN];
        data[..6].copy_from_slice(&self.mac);
        // Bytes 6..18 are packet/error counters; the driver ignores them.
        data[6..10].copy_from_slice(&(self.stats.rx_frames as u32).to_be_bytes());
        data[10..14].copy_from_slice(&(self.stats.tx_frames as u32).to_be_bytes());
        data.truncate(alloc_len.min(STATS_LEN));
        ScsiResponse { status: 0x00, data }
    }

    /// 0x0A WRITE — one Ethernet frame, no CRC appended, length in CDB 3..4.
    ///
    /// A full ring drops the frame and still reports GOOD: the driver has no
    /// retry path, so failing the command only makes things worse.
    fn exec_write(&mut self, cdb: &[u8], data_in: Option<&Vec<u8>>) -> ScsiResponse {
        let len = ((cdb[3] as usize) << 8) | cdb[4] as usize;
        let Some(buf) = data_in else { return good(); };
        let n = len.min(buf.len()).min(MAX_FRAME);
        if n < MIN_FRAME || !self.enabled {
            dlog_dev!(LogModule::Net, "DaynaPort {} TX dropped (len={} enabled={})",
                      self.target_id, n, self.enabled);
            return good();
        }
        let frame = buf[..n].to_vec();
        dlog_dev!(LogModule::Net, "DaynaPort {} TX {}", self.target_id, eth_summary(&frame));
        if self.tx_prod.push(frame).is_err() {
            self.stats.tx_dropped += 1;
        } else {
            self.stats.tx_frames += 1;
            self.tx_wake.1.notify_one();
            self.heartbeat.fetch_or(crate::rex3::Rex3::HB_ENET_TX, Ordering::Relaxed);
        }
        good()
    }

    /// 0x0C SET INTERFACE MODE — byte 4 bit 2 requests broadcast reception.
    /// The driver sends this after every enable and on `SIOCADDMULTI`.
    fn exec_set_iface_mode(&mut self, cdb: &[u8]) -> ScsiResponse {
        self.broadcast = (cdb[4] & MODE_BROADCAST) != 0;
        dlog_dev!(LogModule::Net, "DaynaPort {}: set mode flags={:02x} broadcast={}",
                  self.target_id, cdb[4], self.broadcast);
        good()
    }

    /// 0x0E ENABLE/DISABLE — byte 5 `0x80` enables. Enabling starts from a
    /// clean slate: anything the backend queued while the interface was down is
    /// discarded, exactly as a card that was not listening would have missed it.
    fn exec_enable(&mut self, cdb: &[u8]) -> ScsiResponse {
        let on = (cdb[5] & ENABLE_ON) != 0;
        if on && !self.enabled {
            self.drain_rx();
        }
        self.enabled = on;
        dlog_dev!(LogModule::Net, "DaynaPort {}: {}", self.target_id,
                  if on { "enabled" } else { "disabled" });
        good()
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Address filter. A NAT backend only ever sends us frames for the guest or
    /// for broadcast, so this is close to a no-op there — but a PCAP-bridged
    /// DaynaPort sees the whole LAN, and without broadcast the guest never sees
    /// an ARP reply and nothing works.
    fn accepts(&self, frame: &[u8]) -> bool {
        if frame.len() < MIN_FRAME { return false; }
        let dst = &frame[0..6];
        if dst == self.mac { return true; }
        const BCAST: [u8; 6] = [0xff; 6];
        if dst == BCAST { return self.broadcast; }
        // Multicast (group bit set) rides along with broadcast: the driver only
        // ever asks for 0x04, and IRIX filters multicast itself.
        if dst[0] & 1 != 0 { return self.broadcast; }
        false
    }

    fn drain_rx(&mut self) {
        while self.rx_cons.pop().is_ok() {}
    }

    fn check_condition(&mut self, key: u8, asc: u8, ascq: u8) -> ScsiResponse {
        self.pending_sense = [0u8; 18];
        self.pending_sense[0] = 0x70; // current error
        self.pending_sense[2] = key;
        self.pending_sense[7] = 10;   // additional length
        self.pending_sense[12] = asc;
        self.pending_sense[13] = ascq;
        ScsiResponse { status: 0x02, data: vec![] }
    }

    /// Test hook: take the backend ends so a test can drive both sides of the
    /// rings without spawning a `NatEngine`.
    #[cfg(test)]
    fn take_nat_ends(&mut self) -> (rtrb::Consumer<Vec<u8>>, rtrb::Producer<Vec<u8>>) {
        self.nat_ends.take().expect("nat ends already taken")
    }
}

fn good() -> ScsiResponse {
    ScsiResponse { status: 0x00, data: vec![] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scsi::{ScsiDataLength, ScsiRequest};

    fn dp() -> (DaynaPort, rtrb::Consumer<Vec<u8>>, rtrb::Producer<Vec<u8>>) {
        let mut d = DaynaPort::new(3, default_mac(3), GatewayConfig::default(),
                                   Arc::new(AtomicU64::new(0)));
        let (tx_cons, rx_prod) = d.take_nat_ends();
        (d, tx_cons, rx_prod)
    }

    fn cdb6(b: [u8; 6]) -> ScsiRequest {
        ScsiRequest { cdb: b.to_vec(), data_len: ScsiDataLength::Unlimited, data_in: None }
    }

    fn frame(dst: [u8; 6], len: usize) -> Vec<u8> {
        let mut f = vec![0xAAu8; len];
        f[0..6].copy_from_slice(&dst);
        f
    }

    fn enable(d: &mut DaynaPort) {
        d.request(&cdb6([dp_cmd::ENABLE, 0, 0, 0, 0, 0x80])).unwrap();
        d.request(&cdb6([dp_cmd::SET_IFACE_MODE, 0, 0, 0, 0x04, 0x80])).unwrap();
    }

    fn read_cdb(len: usize) -> ScsiRequest {
        cdb6([dp_cmd::READ, 0, 0, (len >> 8) as u8, len as u8, 0xC0])
    }

    #[test]
    fn inquiry_identifies_as_dayna_processor() {
        let (mut d, _tx, _rx) = dp();
        let r = d.request(&cdb6([0x12, 0, 0, 0, 36, 0])).unwrap();
        assert_eq!(r.status, 0x00);
        assert_eq!(r.data.len(), 36);
        assert_eq!(r.data[0], 0x03, "must be SCSI type 3 (Processor)");
        assert_eq!(&r.data[8..13], b"Dayna");
        assert_eq!(&r.data[8..16], b"Dayna   ");
        assert_eq!(&r.data[16..25], b"SCSI/Link");
        assert_eq!(&r.data[16..32], b"SCSI/Link       ");
        assert_eq!(r.data[4], 31);
    }

    #[test]
    fn inquiry_truncates_to_allocation_length() {
        let (mut d, _tx, _rx) = dp();
        let r = d.request(&cdb6([0x12, 0, 0, 0, 5, 0])).unwrap();
        assert_eq!(r.data.len(), 5);
    }

    #[test]
    fn inquiry_reports_no_device_on_nonzero_lun() {
        let (mut d, _tx, _rx) = dp();
        let r = d.request(&cdb6([0x12, 0x20, 0, 0, 36, 0])).unwrap();
        assert_eq!(r.data[0], 0x7F);
    }

    #[test]
    fn retrieve_stats_returns_mac_first() {
        let (mut d, _tx, _rx) = dp();
        let r = d.request(&cdb6([dp_cmd::RETRIEVE_STATS, 0, 0, 0, 18, 0])).unwrap();
        assert_eq!(r.data.len(), 18);
        assert_eq!(&r.data[..6], &default_mac(3));
    }

    #[test]
    fn idle_read_returns_zero_pktlen_not_a_stall() {
        let (mut d, _tx, _rx) = dp();
        enable(&mut d);
        let r = d.request(&read_cdb(3072)).unwrap();
        assert_eq!(r.status, 0x00);
        assert_eq!(r.data.len(), 6);
        assert_eq!(&r.data[..2], &[0, 0], "pktlen 0 = no more records");
    }

    #[test]
    fn read_record_header_counts_the_crc() {
        let (mut d, _tx, mut rx) = dp();
        enable(&mut d);
        let f = frame(default_mac(3), 64);
        rx.push(f.clone()).unwrap();

        let r = d.request(&read_cdb(3072)).unwrap();
        let pktlen = ((r.data[0] as usize) << 8) | r.data[1] as usize;
        assert_eq!(pktlen, f.len() + 4, "pktlen must include the 4 CRC bytes");
        let flags = u32::from_be_bytes([r.data[2], r.data[3], r.data[4], r.data[5]]);
        assert_eq!(flags, 0, "only record and nothing queued → last record");
        assert_eq!(&r.data[6..6 + f.len()], &f[..]);
        assert_eq!(r.data.len(), 6 + pktlen, "payload must physically carry the CRC bytes");
    }

    #[test]
    fn multi_packet_sets_more_on_every_record_but_the_last() {
        let (mut d, _tx, mut rx) = dp();
        enable(&mut d);
        rx.push(frame(default_mac(3), 100)).unwrap();
        rx.push(frame(default_mac(3), 200)).unwrap();

        let r = d.request(&read_cdb(3072)).unwrap();
        // record 0
        let len0 = ((r.data[0] as usize) << 8) | r.data[1] as usize;
        assert_eq!(len0, 104);
        assert_eq!(u32::from_be_bytes([r.data[2], r.data[3], r.data[4], r.data[5]]), FLAG_MORE);
        // record 1
        let off = 6 + len0;
        let len1 = ((r.data[off] as usize) << 8) | r.data[off + 1] as usize;
        assert_eq!(len1, 204);
        assert_eq!(u32::from_be_bytes([r.data[off + 2], r.data[off + 3],
                                       r.data[off + 4], r.data[off + 5]]), 0);
        assert_eq!(r.data.len(), 6 + len0 + 6 + len1);
    }

    #[test]
    fn single_packet_mode_leaves_the_rest_queued_and_flags_more() {
        let (mut d, _tx, mut rx) = dp();
        enable(&mut d);
        rx.push(frame(default_mac(3), 100)).unwrap();
        rx.push(frame(default_mac(3), 100)).unwrap();

        // byte 5 without bit 6 → one frame per READ
        let r = d.request(&cdb6([dp_cmd::READ, 0, 0, 0x0C, 0x00, 0x00])).unwrap();
        assert_eq!(r.data.len(), 6 + 104);
        assert_eq!(u32::from_be_bytes([r.data[2], r.data[3], r.data[4], r.data[5]]), FLAG_MORE,
                   "device still has a frame queued → MORE");
    }

    #[test]
    fn read_never_overruns_the_requested_transfer_length() {
        let (mut d, _tx, mut rx) = dp();
        enable(&mut d);
        rx.push(frame(default_mac(3), 1514)).unwrap();
        rx.push(frame(default_mac(3), 1514)).unwrap();
        rx.push(frame(default_mac(3), 1514)).unwrap();

        let r = d.request(&read_cdb(3072)).unwrap();
        assert!(r.data.len() <= 3072, "response {} > requested 3072", r.data.len());
        // 2 × (6 + 1514 + 4) = 3048 fits; a third would not.
        assert_eq!(r.data.len(), 2 * (6 + 1518));
        // The straggler stays queued, so the last record must say MORE.
        let off = 6 + 1518;
        assert_eq!(u32::from_be_bytes([r.data[off + 2], r.data[off + 3],
                                       r.data[off + 4], r.data[off + 5]]), FLAG_MORE);
    }

    /// A frame too big for even an empty response must be dropped, not left to
    /// block the head of the ring on every subsequent READ.
    #[test]
    fn oversize_frame_is_dropped_rather_than_stalling_the_queue() {
        let (mut d, _tx, mut rx) = dp();
        enable(&mut d);
        rx.push(frame(default_mac(3), 1514)).unwrap();
        rx.push(frame(default_mac(3), 64)).unwrap();

        // A 128-byte ask can never carry the 1514-byte frame.
        let r = d.request(&read_cdb(128)).unwrap();
        assert_eq!(r.data.len(), 6 + 68, "the small frame behind it must get through");
    }

    #[test]
    fn read_filters_by_address_and_honours_broadcast() {
        let (mut d, _tx, mut rx) = dp();
        enable(&mut d);
        rx.push(frame([0x00, 0x11, 0x22, 0x33, 0x44, 0x55], 64)).unwrap(); // not ours
        rx.push(frame([0xff; 6], 64)).unwrap();                            // broadcast
        let r = d.request(&read_cdb(3072)).unwrap();
        assert_eq!(r.data.len(), 6 + 68, "only the broadcast frame survives");

        // Same again with broadcast reception turned off.
        d.request(&cdb6([dp_cmd::SET_IFACE_MODE, 0, 0, 0, 0x00, 0x80])).unwrap();
        rx.push(frame([0xff; 6], 64)).unwrap();
        let r = d.request(&read_cdb(3072)).unwrap();
        assert_eq!(r.data.len(), 6);
        assert_eq!(&r.data[..2], &[0, 0]);
    }

    #[test]
    fn disabled_interface_reads_empty_and_drops_transmits() {
        let (mut d, mut tx, mut rx) = dp();
        rx.push(frame(default_mac(3), 64)).unwrap();
        let r = d.request(&read_cdb(3072)).unwrap();
        assert_eq!(&r.data[..2], &[0, 0]);

        let f = frame([0xff; 6], 64);
        let w = ScsiRequest {
            cdb: vec![dp_cmd::WRITE, 0, 0, 0, 64, 0],
            data_len: ScsiDataLength::Unlimited,
            data_in: Some(f),
        };
        assert_eq!(d.request(&w).unwrap().status, 0x00, "TX while down still reports GOOD");
        assert!(tx.pop().is_err(), "nothing should reach the backend");
    }

    #[test]
    fn enable_discards_frames_queued_while_down() {
        let (mut d, _tx, mut rx) = dp();
        rx.push(frame(default_mac(3), 64)).unwrap();
        enable(&mut d);
        let r = d.request(&read_cdb(3072)).unwrap();
        assert_eq!(&r.data[..2], &[0, 0], "stale frame must not be delivered");
    }

    #[test]
    fn write_pushes_the_frame_verbatim() {
        let (mut d, mut tx, _rx) = dp();
        enable(&mut d);
        let mut f = frame([0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 100);
        f[6..12].copy_from_slice(&default_mac(3));
        let w = ScsiRequest {
            cdb: vec![dp_cmd::WRITE, 0, 0, (100 >> 8) as u8, 100u8, 0],
            data_len: ScsiDataLength::Unlimited,
            data_in: Some(f.clone()),
        };
        assert_eq!(d.request(&w).unwrap().status, 0x00);
        assert_eq!(tx.pop().unwrap(), f, "frame must go out byte-for-byte, no CRC appended");
    }

    #[test]
    fn write_uses_the_cdb_length_not_the_buffer_length() {
        let (mut d, mut tx, _rx) = dp();
        enable(&mut d);
        // Buffer padded past the frame (DMA rounds up); CDB says 60 bytes.
        let mut buf = frame([0xff; 6], 60);
        buf.extend_from_slice(&[0u8; 40]);
        let w = ScsiRequest {
            cdb: vec![dp_cmd::WRITE, 0, 0, 0, 60, 0],
            data_len: ScsiDataLength::Unlimited,
            data_in: Some(buf),
        };
        d.request(&w).unwrap();
        assert_eq!(tx.pop().unwrap().len(), 60);
    }

    /// The whole point of the device-kind dispatch: READ(6)/WRITE(6) on a
    /// DaynaPort are packet RX/TX, and must never reach the storage opcodes.
    #[test]
    fn scsi_device_routes_read6_to_the_packet_path_not_the_disk_path() {
        let mut d = DaynaPort::new(3, default_mac(3), GatewayConfig::default(),
                                   Arc::new(AtomicU64::new(0)));
        let (_tx, mut rx) = d.take_nat_ends();
        let mut dev = crate::scsi::ScsiDevice::new_daynaport(d);
        assert!(dev.is_daynaport());
        assert!(!dev.is_cdrom());

        dev.request(&cdb6([dp_cmd::ENABLE, 0, 0, 0, 0, 0x80])).unwrap();
        dev.request(&cdb6([dp_cmd::SET_IFACE_MODE, 0, 0, 0, 0x04, 0x80])).unwrap();
        rx.push(frame(default_mac(3), 64)).unwrap();

        // Same CDB a disk would read 12 blocks of 512 bytes for.
        let r = dev.request(&read_cdb(3072)).unwrap();
        assert_eq!(r.status, 0x00);
        assert_eq!(r.data.len(), 6 + 68, "must be one 64-byte frame + header + CRC");

        // INQUIRY still identifies the DaynaPort, not "IRIS EMUL DISK".
        let inq = dev.request(&cdb6([0x12, 0, 0, 0, 36, 0])).unwrap();
        assert_eq!(inq.data[0], 0x03);
        assert_eq!(&inq.data[8..13], b"Dayna");

        // And no storage command is answered.
        let cap = dev.request(&ScsiRequest {
            cdb: vec![0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            data_len: ScsiDataLength::Unlimited,
            data_in: None,
        }).unwrap();
        assert_eq!(cap.status, 0x02, "READ CAPACITY must not be answered");
    }

    #[test]
    fn unsupported_command_reports_illegal_request() {
        let (mut d, _tx, _rx) = dp();
        // READ CAPACITY — a storage command a DaynaPort must never answer.
        let r = d.request(&ScsiRequest {
            cdb: vec![0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            data_len: ScsiDataLength::Unlimited,
            data_in: None,
        }).unwrap();
        assert_eq!(r.status, 0x02);
        let sense = d.request(&cdb6([0x03, 0, 0, 0, 18, 0])).unwrap();
        assert_eq!(sense.data[2], 0x05, "ILLEGAL REQUEST");
        assert_eq!(sense.data[12], 0x20, "invalid command operation code");
    }
}
