//! The SGI O2's Ethernet: MACE's MAC-110 ("mec"), at MACE + 0x280000, on
//! CRIME interrupt 3.
//!
//! What it implements is the contract NetBSD's `mec` driver relies on, as its
//! source shows it (see `clanker/devices/ip32-o2-bringup-log.md`):
//!
//! - Registers are 64 bits, big-endian. INT_STATUS's low byte is
//!   write-one-to-clear; bits 16-24 read back the transmit read pointer.
//! - Transmit: a ring of 64 descriptors of 128 bytes at TX_RING_BASE. Word 0
//!   is the command (frame length - 1, where the inline data starts, which of
//!   three pointers follow, interrupt-on-send); inline data runs to the end of
//!   the descriptor, then each pointer's bytes (address in bits 3-31, length -
//!   1 in bits 32-47). Writing TX_RING_PTR sends everything up to it; each
//!   descriptor's word 0 then becomes its status (sent, success, length) and
//!   the read pointer advances past it. Running dry latches TX_EMPTY.
//! - Receive: the driver hands 4 KB buffers over by writing their addresses
//!   to the MCL FIFO. A frame goes into the oldest: two bytes of padding at
//!   offset 32, the frame at 34, then its CRC; the status word (length with
//!   the CRC, received, broadcast/multicast) at offset 0 is written last, and
//!   RX_THRESHOLD is latched.
//! - The PHY: a Quality Semiconductor QS6612 at MII address 8, as on a real
//!   O2, autonegotiated to 100 Mbit full duplex. Every MII operation is done
//!   by the time the driver looks, so the busy bit never reads set.
//!
//! Frames go to and come from a network backend over two rings, the way the
//! Indy's SEEQ 8003 meets IRIS's NAT gateway: `start_nat` runs one.

use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use crate::snapshot::{get_field, hex_u32, hex_u64, load_u16_slice, toml_u32, toml_u64, u16_slice_to_toml};

pub mod reg {
    pub const MAC_CONTROL: u32 = 0x00;
    pub const INT_STATUS: u32 = 0x08;
    pub const DMA_CONTROL: u32 = 0x10;
    pub const TIMER: u32 = 0x18;
    pub const TX_ALIAS: u32 = 0x20;
    pub const RX_ALIAS: u32 = 0x28;
    pub const TX_RING_PTR: u32 = 0x30;
    pub const RX_FIFO: u32 = 0x40;
    pub const PHY_DATA: u32 = 0x60;
    pub const PHY_ADDRESS: u32 = 0x68;
    pub const PHY_READ_INITIATE: u32 = 0x70;
    pub const STATION: u32 = 0xa0;
    pub const STATION_ALT: u32 = 0xa8;
    pub const MULTICAST: u32 = 0xb0;
    pub const TX_RING_BASE: u32 = 0xb8;
    pub const MCL_RX_FIFO: u32 = 0x100;
}

pub mod bits {
    pub const MAC_CORE_RESET: u64 = 0x1;
    pub const MAC_FILTER_MASK: u64 = 0x60;
    pub const MAC_FILTER_MATCHMULTI: u64 = 0x20;
    pub const MAC_FILTER_ALLMULTI: u64 = 0x40;
    pub const MAC_FILTER_PROMISC: u64 = 0x60;

    pub const INT_TX_EMPTY: u64 = 0x01;
    pub const INT_TX_PACKET_SENT: u64 = 0x02;
    pub const INT_RX_THRESHOLD: u64 = 0x20;
    pub const INT_RX_FIFO_UNDERFLOW: u64 = 0x40;

    pub const DMA_TX_INT_ENABLE: u64 = 0x1;
    pub const DMA_TX_DMA_ENABLE: u64 = 0x2;
    pub const DMA_TX_RING_SIZE_SHIFT: u32 = 2;
    pub const DMA_RX_INT_ENABLE: u64 = 0x200;
    pub const DMA_RX_INT_THRESHOLD_SHIFT: u32 = 4;
    pub const DMA_RX_DMA_OFFSET_SHIFT: u32 = 12;
    pub const DMA_RX_DMA_ENABLE: u64 = 0x8000;

    pub const TXCMD_LEN_MASK: u64 = 0xffff;
    pub const TXCMD_BUFSTART_SHIFT: u32 = 16;
    pub const TXCMD_TXINT: u64 = 1 << 24;
    pub const TXCMD_PTR1: u64 = 1 << 25;

    pub const TXSTAT_SUCCESS: u64 = 1 << 23;
    pub const TXSTAT_SENT: u64 = 1 << 63;

    pub const RXSTAT_MCAST: u64 = 1 << 19;
    pub const RXSTAT_BCAST: u64 = 1 << 20;
    pub const RXSTAT_MATCHMAC: u64 = 1 << 26;
    pub const RXSTAT_SEQ_SHIFT: u32 = 27;
    pub const RXSTAT_RECEIVED: u64 = 1 << 63;

    pub const PHY_BUSY: u64 = 0x1_0000;
}

/// Ring and buffer geometry the driver fixes. The transmit ring is
/// `TX_RING_LEN << DMA_CONTROL[3:2]` descriptors: NetBSD and OpenBSD leave
/// the field 0 (64), IRIX sets 1 and runs a ring of 128; the pointers are
/// nine bits, so 3 would be 512.
pub const TX_RING_LEN: u32 = 64;
pub const TX_DESC_SIZE: u32 = 128;
pub const RX_FIFO_DEPTH: usize = 16;
pub const RX_BUF_SIZE: usize = 4096;
/// Two bytes the chip puts in front of a received frame, so its IP header
/// lands on a 4-byte boundary.
const RX_PAD: usize = 2;
/// The shortest frame on the wire, without its CRC.
const MIN_FRAME: usize = 60;

/// The PHY: one, at the MII address a real O2's has.
pub const PHY_ADDR: u32 = 8;

/// A QS6612, autonegotiated to 100 Mbit full duplex.
struct Phy {
    regs: [u16; 32],
}

impl Phy {
    const BMCR: usize = 0;
    const BMSR: usize = 1;
    const ANAR: usize = 4;
    /// After a reset: autonegotiation on, 100 Mbit, full duplex.
    const BMCR_DEFAULT: u16 = 0x3100;
    const BMCR_RESET: u16 = 0x8000;
    const BMCR_STARTNEG: u16 = 0x0200;

    fn new() -> Self {
        let mut regs = [0u16; 32];
        regs[Self::BMCR] = Self::BMCR_DEFAULT;
        // 100TX FD/HD, 10 FD/HD, autonegotiation complete, able to
        // autonegotiate, link up, extended registers.
        regs[Self::BMSR] = 0x782d;
        // QS6612: OUI 0x00068a, model 0.
        regs[2] = 0x0181;
        regs[3] = 0x4401;
        regs[Self::ANAR] = 0x01e1;
        // The partner offers the same.
        regs[5] = 0x45e1;
        regs[6] = 0x0001;
        // QS6612 PHY control: the negotiated mode, 100TX full duplex.
        regs[31] = 0x1018;
        Self { regs }
    }

    fn read(&self, r: usize) -> u16 {
        self.regs[r & 31]
    }

    fn write(&mut self, r: usize, v: u16) {
        let r = r & 31;
        match r {
            Self::BMCR => {
                if v & Self::BMCR_RESET != 0 {
                    *self = Self::new();
                } else {
                    // Starting a negotiation finishes at once.
                    self.regs[r] = v & !Self::BMCR_STARTNEG;
                }
            }
            // Status, identity and the partner's word are read-only.
            1 | 2 | 3 | 5 | 6 | 31 => {}
            _ => self.regs[r] = v,
        }
    }
}

struct State {
    mac_control: u64,
    /// INT_STATUS bits 0-7, latched until written back.
    int_status: u64,
    dma_control: u64,
    tx_ring_base: u64,
    tx_read: u32,
    tx_write: u32,
    station: u64,
    multicast: u64,
    /// Buffers handed over and not yet filled, oldest first.
    mcl_fifo: VecDeque<u32>,
    /// Bits 27-31 of each received frame's status.
    rx_seq: u64,
    /// Receive buffers filled so far, mod 32: the MCL FIFO read pointer,
    /// which INT_STATUS reports in bits 8-12. OpenBSD's driver takes the
    /// buffers up to it from there instead of polling each one's status, so
    /// without it every frame is delivered and none is ever seen.
    rx_mcl_read: u32,
    phy: Phy,
    phy_address: u64,
    phy_data: u64,
    /// Anything else written, read back as written.
    other: std::collections::HashMap<u32, u64>,
    tx_frames: u64,
    rx_frames: u64,
    rx_dropped: u64,
}

impl State {
    fn new() -> Self {
        Self {
            mac_control: 0,
            int_status: 0,
            dma_control: 0,
            tx_ring_base: 0,
            tx_read: 0,
            tx_write: 0,
            station: 0,
            multicast: 0,
            mcl_fifo: VecDeque::new(),
            rx_seq: 0,
            rx_mcl_read: 0,
            phy: Phy::new(),
            phy_address: 0,
            phy_data: 0,
            other: Default::default(),
            tx_frames: 0,
            rx_frames: 0,
            rx_dropped: 0,
        }
    }

    /// A core reset, or DMA turned off: both rings start again from empty.
    /// The station address survives -- the driver writes it once, before its
    /// first reset.
    fn tx_ring_len(&self) -> u32 {
        TX_RING_LEN << ((self.dma_control >> bits::DMA_TX_RING_SIZE_SHIFT) & 3)
    }

    fn reset_rings(&mut self) {
        self.rx_mcl_read = 0;
        self.tx_read = 0;
        self.tx_write = 0;
        self.mcl_fifo.clear();
        self.int_status = 0;
    }
}

/// Where frames go: a backend's two rings.
struct Link {
    to_net: rtrb::Producer<Vec<u8>>,
    from_net: rtrb::Consumer<Vec<u8>>,
    /// Told when a frame has been put on `to_net`.
    tx_wake: Option<Arc<crate::net::Wake>>,
}

pub struct Mec {
    ram: Arc<crate::dev::crime_mem::O2Memory>,
    st: Mutex<State>,
    link: Mutex<Option<Link>>,
    /// CRIME's Ethernet input.
    irq: std::sync::OnceLock<crate::irq::IrqLine>,
}

impl Mec {
    pub fn new(ram: Arc<crate::dev::crime_mem::O2Memory>) -> Self {
        Self { ram, st: Mutex::new(State::new()), link: Mutex::new(None), irq: std::sync::OnceLock::new() }
    }

    /// Wire the chip's interrupt to `line`.
    pub fn connect_irq(&self, line: crate::irq::IrqLine) {
        let _ = self.irq.set(line);
        self.update_irq();
    }

    /// Drive the interrupt line to the chip's state. Called after anything
    /// that can change it: a register write, frames received.
    fn update_irq(&self) {
        if let Some(l) = self.irq.get() {
            l.set(self.interrupting());
        }
    }

    /// Connect to a backend: frames sent go on `to_net`, frames on `from_net`
    /// are received.
    pub fn attach(&self, to_net: rtrb::Producer<Vec<u8>>, from_net: rtrb::Consumer<Vec<u8>>, tx_wake: Option<Arc<crate::net::Wake>>) {
        *self.link.lock().unwrap() = Some(Link { to_net, from_net, tx_wake });
    }

    /// Run IRIS's NAT gateway behind this interface, on a thread of its own:
    /// DHCP, DNS and the port forwards in `config`, as the Indy's SEEQ 8003
    /// has them.
    /// Returns the NAT's controls.
    pub fn start_nat(&self, config: crate::net::GatewayConfig) -> Arc<crate::net::NatControl> {
        let (to_net, tx_cons) = rtrb::RingBuffer::new(256);
        let (rx_prod, from_net) = rtrb::RingBuffer::new(256);
        let rx_wake = crate::net::Wake::new();
        let tx_wake = crate::net::Wake::new();
        let running = Arc::new(AtomicBool::new(true));
        let ctl = crate::net::NatControl::new();
        let controls = ctl.clone();
        let tw = tx_wake.clone();
        std::thread::Builder::new()
            .name("ip32-mec-nat".into())
            .spawn(move || {
                crate::net::NatEngine::new(config, tx_cons, rx_prod, rx_wake, tw, running, ctl).run();
            })
            .expect("mec NAT thread");
        self.attach(to_net, from_net, Some(tx_wake));
        controls
    }

    /// Frames sent and received so far, and received ones dropped for want of
    /// a buffer.
    pub fn counts(&self) -> (u64, u64, u64) {
        let s = self.st.lock().unwrap();
        (s.tx_frames, s.rx_frames, s.rx_dropped)
    }

    /// The interrupt line: something latched that is enabled.
    pub fn interrupting(&self) -> bool {
        let s = self.st.lock().unwrap();
        let tx = s.int_status & (bits::INT_TX_EMPTY | bits::INT_TX_PACKET_SENT) != 0
            && s.dma_control & bits::DMA_TX_INT_ENABLE != 0;
        let rx = s.int_status & (bits::INT_RX_THRESHOLD | bits::INT_RX_FIFO_UNDERFLOW) != 0
            && s.dma_control & bits::DMA_RX_INT_ENABLE != 0;
        tx || rx
    }

    /// Run `f` over `[addr, addr + len)` a page at a time, with where each
    /// page is on the bus, or None where its bus address is not memory.
    fn each_page(&self, addr: u64, len: usize, mut f: impl FnMut(std::ops::Range<usize>, Option<u64>)) {
        const PAGE: usize = 4096;
        let mut done = 0;
        while done < len {
            let a = addr.wrapping_add(done as u64);
            let n = (PAGE - (a as usize & (PAGE - 1))).min(len - done);
            f(done..done + n, self.ram.dma_offset(a as u32).map(|p| p as u64));
            done += n;
        }
    }

    /// Memory the ring and buffers name, as a DMA master reads it; what is
    /// not memory reads as zero.
    fn ram_read(&self, addr: u64, out: &mut [u8]) {
        let bus = &**self.ram.phys();
        self.each_page(addr, out.len(), |r, at| match at {
            Some(p) => {
                crate::physical::read_bytes(bus, p, &mut out[r]);
            }
            None => out[r].fill(0),
        });
    }

    fn ram_write(&self, addr: u64, data: &[u8]) {
        let bus = &**self.ram.phys();
        self.each_page(addr, data.len(), |r, at| {
            if let Some(p) = at {
                crate::physical::write_bytes(bus, p, &data[r]);
            }
        });
    }

    fn ram_read64(&self, addr: u64) -> u64 {
        let mut b = [0u8; 8];
        self.ram_read(addr, &mut b);
        u64::from_be_bytes(b)
    }

    pub fn read64(&self, off: u32) -> u64 {
        let s = self.st.lock().unwrap();
        match off & !7 {
            reg::MAC_CONTROL => s.mac_control,
            // Transmit-empty shows only while its interrupt is enabled:
            // IRIX answers it by disabling the interrupt, never clearing
            // the bit, and loops in its handler while any status bit reads
            // set.
            reg::INT_STATUS => {
                let mut low = s.int_status & 0xff;
                if s.dma_control & bits::DMA_TX_INT_ENABLE == 0 {
                    low &= !bits::INT_TX_EMPTY;
                }
                low | (u64::from(s.rx_mcl_read & 0x1f) << 8) | (u64::from(s.tx_read & 0x1ff) << 16)
            }
            reg::DMA_CONTROL => s.dma_control,
            reg::TX_RING_PTR => u64::from(s.tx_write & 0x1ff) | (u64::from(s.tx_read & 0x1ff) << 16),
            // The MCL FIFO: write pointer, read pointer, depth, a byte
            // each (IRIX reads the read pointer alone, as a byte).
            reg::RX_FIFO => {
                let depth = s.mcl_fifo.len() as u64 & 0x1f;
                let rptr = u64::from(s.rx_mcl_read & 0x1f);
                (((rptr + depth) & 0x1f) << 16) | (rptr << 8) | depth
            }
            reg::PHY_DATA => s.phy_data & !bits::PHY_BUSY,
            reg::PHY_ADDRESS => s.phy_address,
            reg::STATION => s.station,
            reg::MULTICAST => s.multicast,
            reg::TX_RING_BASE => s.tx_ring_base,
            o => s.other.get(&o).copied().unwrap_or(0),
        }
    }

    /// A 1-, 2- or 4-byte read of the 64-bit register holding `off`
    /// (big-endian): IRIX reads the transmit read pointer as a halfword.
    pub fn read_part(&self, off: u32, size: u32) -> u64 {
        let shift = 8 * (8 - (off & 7) - size);
        (self.read64(off & !7) >> shift) & ((1u64 << (8 * size)) - 1)
    }

    /// A 1-, 2- or 4-byte write into the 64-bit register holding `off`,
    /// merged with the rest of it; IRIX starts a transmit by writing the
    /// ring's write pointer alone, a halfword at 0x36. The interrupt
    /// status is write-one-to-clear, so there the other bytes write zero.
    pub fn write_part(&self, off: u32, size: u32, val: u64) {
        let base = off & !7;
        let shift = 8 * (8 - (off & 7) - size);
        let mask = ((1u64 << (8 * size)) - 1) << shift;
        let old = if base == reg::INT_STATUS { 0 } else { self.read64(base) };
        self.write64(base, (old & !mask) | ((val << shift) & mask));
    }

    pub fn write64(&self, off: u32, val: u64) {
        let mut send = false;
        {
            let mut s = self.st.lock().unwrap();
            match off & !7 {
                reg::MAC_CONTROL => {
                    if val & bits::MAC_CORE_RESET != 0 {
                        s.reset_rings();
                    }
                    s.mac_control = val & !bits::MAC_CORE_RESET;
                }
                reg::INT_STATUS => s.int_status &= !(val & 0xff),
                reg::DMA_CONTROL => {
                    if val & bits::DMA_TX_DMA_ENABLE == 0 {
                        s.tx_read = 0;
                        s.tx_write = 0;
                    }
                    if val & bits::DMA_RX_DMA_ENABLE == 0 && s.dma_control & bits::DMA_RX_DMA_ENABLE != 0 {
                        s.mcl_fifo.clear();
                    }
                    s.dma_control = val;
                }
                // TX_ALIAS reaches DMA_CONTROL's transmit interrupt enable
                // alone; RX_ALIAS the receive interrupt bits.
                reg::TX_ALIAS => {
                    s.dma_control = (s.dma_control & !bits::DMA_TX_INT_ENABLE) | (val & bits::DMA_TX_INT_ENABLE);
                }
                reg::RX_ALIAS => {
                    s.dma_control = (s.dma_control & !0x3f0) | (val & 0x3f0);
                }
                reg::TX_RING_PTR => {
                    s.tx_write = (val as u32) % s.tx_ring_len();
                    send = true;
                }
                reg::PHY_ADDRESS => s.phy_address = val,
                reg::PHY_READ_INITIATE => {
                    let dev = ((s.phy_address >> 5) & 0x1f) as u32;
                    let r = (s.phy_address & 0x1f) as usize;
                    s.phy_data = if dev == PHY_ADDR { u64::from(s.phy.read(r)) } else { 0xffff };
                }
                reg::PHY_DATA => {
                    // Writing the data is what starts an MII write.
                    let dev = ((s.phy_address >> 5) & 0x1f) as u32;
                    let r = (s.phy_address & 0x1f) as usize;
                    if dev == PHY_ADDR {
                        s.phy.write(r, val as u16);
                    }
                    s.phy_data = val & 0xffff;
                }
                reg::STATION => s.station = val & 0xffff_ffff_ffff,
                reg::MULTICAST => s.multicast = val,
                reg::TX_RING_BASE => s.tx_ring_base = val & 0xffff_ffff,
                reg::MCL_RX_FIFO => {
                    if s.mcl_fifo.len() < RX_FIFO_DEPTH {
                        s.mcl_fifo.push_back(val as u32);
                    }
                    // The receive request is the FIFO running below the
                    // threshold: handing buffers back up to it ends the
                    // request. IRIX never clears it any other way after a
                    // good frame.
                    let threshold = ((s.dma_control >> bits::DMA_RX_INT_THRESHOLD_SHIFT) & 0x1f) as usize;
                    if s.mcl_fifo.len() >= threshold {
                        s.int_status &= !bits::INT_RX_THRESHOLD;
                    }
                }
                reg::TIMER | reg::STATION_ALT => {
                    s.other.insert(off & !7, val);
                }
                o => {
                    s.other.insert(o, val);
                }
            }
        }
        if send {
            self.transmit();
        }
        self.update_irq();
    }

    /// Send everything from the read pointer up to the write pointer.
    fn transmit(&self) {
        let mut frames = Vec::new();
        {
            let mut s = self.st.lock().unwrap();
            if s.dma_control & bits::DMA_TX_DMA_ENABLE == 0 {
                return;
            }
            let mut any = false;
            while s.tx_read != s.tx_write {
                let desc = s.tx_ring_base + u64::from(s.tx_read * TX_DESC_SIZE);
                let cmd = self.ram_read64(desc);
                let len = ((cmd & bits::TXCMD_LEN_MASK) + 1) as usize;
                let start = ((cmd >> bits::TXCMD_BUFSTART_SHIFT) & 0x7f) as u64;
                let mut frame = Vec::with_capacity(len);
                if start < u64::from(TX_DESC_SIZE) {
                    let mut inline = vec![0u8; (u64::from(TX_DESC_SIZE) - start) as usize];
                    self.ram_read(desc + start, &mut inline);
                    frame.extend_from_slice(&inline);
                }
                for p in 0..3u64 {
                    if cmd & (bits::TXCMD_PTR1 << p) == 0 {
                        continue;
                    }
                    let ptr = self.ram_read64(desc + 8 + 8 * p);
                    let addr = ptr & 0xffff_fff8;
                    let plen = (((ptr >> 32) & 0xffff) + 1) as usize;
                    let mut data = vec![0u8; plen];
                    self.ram_read(addr, &mut data);
                    frame.extend_from_slice(&data);
                }
                frame.truncate(len);
                // The status: sent, without error, and how long.
                let stat = bits::TXSTAT_SENT | bits::TXSTAT_SUCCESS | (len as u64 & 0xffff);
                self.ram_write(desc, &stat.to_be_bytes());
                if cmd & bits::TXCMD_TXINT != 0 {
                    s.int_status |= bits::INT_TX_PACKET_SENT;
                }
                s.tx_read = (s.tx_read + 1) % s.tx_ring_len();
                s.tx_frames += 1;
                frames.push(frame);
                any = true;
            }
            if any {
                // Latched when the ring runs dry, so an interrupt enabled
                // after the ring pointer was written still sees it.
                s.int_status |= bits::INT_TX_EMPTY;
            }
        }
        if frames.is_empty() {
            return;
        }
        if let Some(link) = self.link.lock().unwrap().as_mut() {
            for f in frames {
                Self::trace("tx", &f);
                let _ = link.to_net.push(f);
            }
            if let Some(w) = &link.tx_wake {
                w.notify();
            }
        }
    }

    /// `IRIS_IP32_MEC_TRACE`: report frames sent and received, and why a
    /// received one was dropped, the first 40 of each.
    fn trace(what: &str, frame: &[u8]) {
        use std::sync::atomic::{AtomicU32, Ordering};
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static N: AtomicU32 = AtomicU32::new(0);
        if !*ON.get_or_init(|| std::env::var_os("IRIS_IP32_MEC_TRACE").is_some()) {
            return;
        }
        if N.fetch_add(1, Ordering::Relaxed) >= 120 {
            return;
        }
        let hx = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(":");
        let et = if frame.len() >= 14 { u16::from_be_bytes([frame[12], frame[13]]) } else { 0 };
        eprintln!("mec: {what} len {} dst {} src {} type {et:#06x}", frame.len(),
                  hx(&frame[..6.min(frame.len())]), hx(&frame[6.min(frame.len())..12.min(frame.len())]));
    }

    /// Whether the receive filter takes a frame to `dst`.
    fn accepts(s: &State, dst: &[u8]) -> bool {
        if dst.len() < 6 {
            return false;
        }
        let filter = s.mac_control & bits::MAC_FILTER_MASK;
        if filter == bits::MAC_FILTER_PROMISC {
            return true;
        }
        if dst.iter().all(|&b| b == 0xff) {
            return true;
        }
        if dst[0] & 1 != 0 {
            // A multicast: all of them when asked to match any.
            return filter == bits::MAC_FILTER_MATCHMULTI || filter == bits::MAC_FILTER_ALLMULTI;
        }
        let station = s.station.to_be_bytes();
        dst[..6] == station[2..8]
    }

    /// Take frames from the backend into the buffers the driver has handed
    /// over. Called from the machine's polling loop.
    pub fn poll(&self) {
        self.receive();
        self.update_irq();
    }

    fn receive(&self) {
        let mut link = self.link.lock().unwrap();
        let Some(link) = link.as_mut() else { return };
        loop {
            let mut s = self.st.lock().unwrap();
            if s.dma_control & bits::DMA_RX_DMA_ENABLE == 0 {
                // Not receiving: frames are not kept for later.
                while let Ok(f) = link.from_net.pop() {
                    Self::trace("rx DROPPED: receive DMA off", &f);
                }
                return;
            }
            let Ok(frame) = link.from_net.peek() else { return };
            if !Self::accepts(&s, frame) {
                Self::trace(&format!("rx DROPPED by filter (mac_control {:#x}, station {:012x})", s.mac_control, s.station), frame);
                let _ = link.from_net.pop();
                continue;
            }
            let Some(buf) = s.mcl_fifo.pop_front() else {
                // No buffer: it waits for one.
                return;
            };
            let frame = link.from_net.pop().unwrap_or_default();
            Self::trace(&format!("rx into {buf:#010x}"), &frame);
            let mut data = frame;
            if data.len() < MIN_FRAME {
                data.resize(MIN_FRAME, 0);
            }
            let fcs = crc32(&data);
            data.extend_from_slice(&fcs.to_le_bytes());
            let offset = (((s.dma_control >> bits::DMA_RX_DMA_OFFSET_SHIFT) & 7) * 8) as u64;
            let start = u64::from(buf) + offset;
            if offset as usize + RX_PAD + data.len() > RX_BUF_SIZE {
                s.rx_dropped += 1;
                s.mcl_fifo.push_front(buf);
                continue;
            }
            self.ram_write(start, &[0u8; RX_PAD]);
            self.ram_write(start + RX_PAD as u64, &data);
            let dst = &data[..6];
            let mut stat = bits::RXSTAT_RECEIVED | (data.len() as u64 & 0xffff) | ((s.rx_seq & 0x1f) << bits::RXSTAT_SEQ_SHIFT);
            if dst.iter().all(|&b| b == 0xff) {
                stat |= bits::RXSTAT_BCAST;
            } else if dst[0] & 1 != 0 {
                stat |= bits::RXSTAT_MCAST;
            } else {
                stat |= bits::RXSTAT_MATCHMAC;
            }
            s.rx_seq = s.rx_seq.wrapping_add(1);
            s.rx_mcl_read = (s.rx_mcl_read + 1) & 0x1f;
            // The status last: the driver takes the buffer once it is set.
            self.ram_write(u64::from(buf), &stat.to_be_bytes());
            s.int_status |= bits::INT_RX_THRESHOLD;
            s.rx_frames += 1;
        }
    }
}

/// The Ethernet CRC-32 of `data`, as its FCS is computed.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// The bus side, addressed from MACE's map: the registers are 64-bit, so a
/// 32-bit read is the half it names and a 32-bit store the low word; bytes
/// and halfwords are parts of a register (IRIX writes the transmit ring's
/// write pointer as a halfword).
impl crate::traits::BusDevice for Mec {
    fn read8(&self, addr: u64) -> crate::traits::BusRead8 {
        crate::traits::BusRead8::ok(self.read_part((addr - crate::ip32::MACE_MEC as u64) as u32, 1) as u8)
    }
    fn read16(&self, addr: u64) -> crate::traits::BusRead16 {
        crate::traits::BusRead16::ok(self.read_part((addr - crate::ip32::MACE_MEC as u64) as u32, 2) as u16)
    }
    fn read32(&self, addr: u64) -> crate::traits::BusRead32 {
        let v = self.read64(((addr - crate::ip32::MACE_MEC as u64) as u32) & !7);
        crate::traits::BusRead32::ok(if addr & 4 == 0 { (v >> 32) as u32 } else { v as u32 })
    }
    fn read64(&self, addr: u64) -> crate::traits::BusRead64 {
        crate::traits::BusRead64::ok(Mec::read64(self, (addr - crate::ip32::MACE_MEC as u64) as u32))
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.write_part((addr - crate::ip32::MACE_MEC as u64) as u32, 1, val as u64);
        crate::traits::BUS_OK
    }
    fn write16(&self, addr: u64, val: u16) -> u32 {
        self.write_part((addr - crate::ip32::MACE_MEC as u64) as u32, 2, val as u64);
        crate::traits::BUS_OK
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        Mec::write64(self, ((addr - crate::ip32::MACE_MEC as u64) as u32) & !7, val as u64);
        crate::traits::BUS_OK
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        Mec::write64(self, (addr - crate::ip32::MACE_MEC as u64) as u32, val);
        crate::traits::BUS_OK
    }
}

/// The registers, the rings' pointers, the receive buffers handed over, the
/// PHY and the counters; loading drives the interrupt again. The link to
/// the host's network is the machine's, not the chip's.
impl crate::traits::Saveable for Mec {
    fn save_state(&self) -> toml::Value {
        let st = self.st.lock().unwrap();
        let mut t = toml::map::Map::new();
        for (k, v) in [
            ("mac_control", st.mac_control), ("int_status", st.int_status),
            ("dma_control", st.dma_control), ("tx_ring_base", st.tx_ring_base),
            ("station", st.station), ("multicast", st.multicast), ("rx_seq", st.rx_seq),
            ("phy_address", st.phy_address), ("phy_data", st.phy_data),
            ("tx_frames", st.tx_frames), ("rx_frames", st.rx_frames), ("rx_dropped", st.rx_dropped),
        ] {
            t.insert(k.into(), hex_u64(v));
        }
        t.insert("tx_read".into(), hex_u32(st.tx_read));
        t.insert("tx_write".into(), hex_u32(st.tx_write));
        t.insert("rx_mcl_read".into(), hex_u32(st.rx_mcl_read));
        t.insert("mcl_fifo".into(), toml::Value::Array(st.mcl_fifo.iter().map(|&a| hex_u32(a)).collect()));
        t.insert("phy".into(), u16_slice_to_toml(&st.phy.regs));
        let mut other = toml::map::Map::new();
        let mut keys: Vec<_> = st.other.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            other.insert(format!("{k:#x}"), hex_u64(st.other[&k]));
        }
        t.insert("other".into(), toml::Value::Table(other));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        {
            let u = |k: &str| get_field(v, k).and_then(toml_u64).unwrap_or(0);
            let w = |k: &str| get_field(v, k).and_then(toml_u32).unwrap_or(0);
            let mut st = self.st.lock().unwrap();
            *st = State::new();
            st.mac_control = u("mac_control");
            st.int_status = u("int_status");
            st.dma_control = u("dma_control");
            st.tx_ring_base = u("tx_ring_base");
            st.station = u("station");
            st.multicast = u("multicast");
            st.rx_seq = u("rx_seq");
            st.phy_address = u("phy_address");
            st.phy_data = u("phy_data");
            st.tx_frames = u("tx_frames");
            st.rx_frames = u("rx_frames");
            st.rx_dropped = u("rx_dropped");
            st.tx_read = w("tx_read");
            st.tx_write = w("tx_write");
            st.rx_mcl_read = w("rx_mcl_read");
            if let Some(toml::Value::Array(a)) = get_field(v, "mcl_fifo") {
                st.mcl_fifo = a.iter().map(|x| toml_u32(x).unwrap_or(0)).collect();
            }
            if let Some(p) = get_field(v, "phy") {
                load_u16_slice(p, &mut st.phy.regs);
            }
            if let Some(toml::Value::Table(o)) = get_field(v, "other") {
                for (k, x) in o {
                    let off = u32::from_str_radix(k.trim_start_matches("0x"), 16)
                        .map_err(|_| format!("mec: bad offset {k:?}"))?;
                    st.other.insert(off, toml_u64(x).unwrap_or(0));
                }
            }
        }
        self.update_irq();
        Ok(())
    }
}

impl crate::traits::Device for Mec {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("mec".to_string(), "MACE Ethernet: mec status".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write;
        if cmd != "mec" {
            return Err("Command not found".to_string());
        }
        match args.first().copied().unwrap_or("status") {
            "status" => {
                let s = self.st.lock().unwrap();
                writeln!(w, "MEC: station {:012x}, MAC control {:#x}, DMA control {:#x}, interrupt status {:#x}{}",
                    s.station, s.mac_control, s.dma_control, s.int_status,
                    if self.link.lock().unwrap().is_some() { ", network attached" } else { ", no network" }).ok();
                writeln!(w, "  tx ring {:#x} read {} write {}; rx fifo {} entries, sequence {}",
                    s.tx_ring_base, s.tx_read, s.tx_write, s.mcl_fifo.len(), s.rx_seq).ok();
                writeln!(w, "  frames: {} sent, {} received, {} dropped", s.tx_frames, s.rx_frames, s.rx_dropped).ok();
                Ok(())
            }
            _ => Err("Usage: mec status".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ip32::RAM_BASE;

    fn mec() -> (Mec, Arc<crate::dev::crime_mem::O2Memory>) {
        let ram = Arc::new(crate::dev::crime_mem::O2Memory::single(1 << 20));
        (Mec::new(ram.clone()), ram)
    }

    fn put(ram: &Arc<crate::dev::crime_mem::O2Memory>, off: usize, b: &[u8]) {
        ram.write(off, b);
    }

    fn get(ram: &Arc<crate::dev::crime_mem::O2Memory>, off: usize, n: usize) -> Vec<u8> {
        ram.bytes(off, (off + n) - (off)).to_vec()
    }

    fn mii_read(m: &Mec, phy: u32, r: u32) -> u64 {
        m.write64(reg::PHY_ADDRESS, u64::from((phy << 5) | r));
        m.write64(reg::PHY_READ_INITIATE, 1);
        m.read64(reg::PHY_DATA)
    }

    #[test]
    fn save_load_round_trip() {
        use crate::traits::Saveable;
        let (src, _) = mec();
        {
            let mut st = src.st.lock().unwrap();
            st.mac_control = 0x0000_0002;
            st.int_status = 0x41;
            st.dma_control = 0x0000_8001;
            st.tx_ring_base = 0x0123_4000;
            st.tx_read = 3;
            st.tx_write = 9;
            st.station = 0x0800_6912_3456;
            st.multicast = 0xffff;
            st.mcl_fifo = [0x0010_0000u32, 0x0010_1000].into_iter().collect();
            st.rx_seq = 5;
            st.rx_mcl_read = 7;
            st.phy.regs[4] = 0x0061;
            st.phy_address = 0x101;
            st.phy_data = 0x45e1;
            st.other.insert(0x80, 0xabcd);
            st.tx_frames = 10;
            st.rx_frames = 20;
            st.rx_dropped = 1;
        }
        let v1 = src.save_state();
        let (dst, _) = mec();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "Mec save_state mismatch after load_state round-trip");
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    /// The driver probes every MII address: one PHY answers, a QS6612 that
    /// can autonegotiate and has a link; the rest read all ones.
    #[test]
    fn one_phy_answers_and_it_has_a_link() {
        let (m, _) = mec();
        assert_eq!(mii_read(&m, 0, 1), 0xffff);
        let bmsr = mii_read(&m, PHY_ADDR, 1);
        assert_eq!(bmsr & 0x0008, 0x0008, "able to autonegotiate");
        assert_eq!(bmsr & 0x0004, 0x0004, "link up");
        assert_eq!(bmsr & 0x0100, 0, "no extended status");
        assert_eq!((mii_read(&m, PHY_ADDR, 2), mii_read(&m, PHY_ADDR, 3)), (0x0181, 0x4401), "QS6612");
        assert_eq!(m.read64(reg::PHY_DATA) & bits::PHY_BUSY, 0);
        // A reset: isolation and the reset bit itself do not stay set.
        m.write64(reg::PHY_ADDRESS, u64::from(PHY_ADDR << 5));
        m.write64(reg::PHY_DATA, 0x8400);
        assert_eq!(mii_read(&m, PHY_ADDR, 0) & 0x8400, 0);
        assert_eq!(mii_read(&m, PHY_ADDR, 31) & 0x1c, 0x18, "100TX full duplex");
    }

    /// A frame in two parts -- 6 bytes inline, the rest through a pointer --
    /// goes out whole; the descriptor gets its status, INT_STATUS shows the
    /// read pointer past it and TX_EMPTY, which interrupts once enabled.
    #[test]
    fn a_descriptor_is_sent_and_reports_back() {
        let (m, ram) = mec();
        let (to_net, mut tx_cons) = rtrb::RingBuffer::new(8);
        let (_rx_prod, from_net) = rtrb::RingBuffer::new(8);
        m.attach(to_net, from_net, None);
        let (irq, line) = crate::irq::LevelProbe::line();
        m.connect_irq(line);
        let ring = 0x2000usize;
        let data = 0x8008usize;
        let frame: Vec<u8> = (0..70u8).collect();
        put(&ram, ring + 128 - 6, &frame[..6]);
        put(&ram, data, &frame[6..]);
        let cmd: u64 = (70 - 1) | ((128 - 6) << 16) | bits::TXCMD_PTR1;
        put(&ram, ring, &cmd.to_be_bytes());
        let ptr: u64 = (RAM_BASE as u64 + data as u64) | ((64 - 1) << 32);
        put(&ram, ring + 8, &ptr.to_be_bytes());
        m.write64(reg::TX_RING_BASE, RAM_BASE as u64 + ring as u64);
        m.write64(reg::DMA_CONTROL, bits::DMA_TX_DMA_ENABLE);
        m.write64(reg::TX_RING_PTR, 1);
        assert_eq!(tx_cons.pop().unwrap(), frame);
        let stat = u64::from_be_bytes(get(&ram, ring, 8).try_into().unwrap());
        assert_eq!(stat & (bits::TXSTAT_SENT | bits::TXSTAT_SUCCESS), bits::TXSTAT_SENT | bits::TXSTAT_SUCCESS);
        let is = m.read64(reg::INT_STATUS);
        assert_eq!((is >> 16) & 0x1ff, 1, "read pointer past the descriptor");
        assert_eq!(is & bits::INT_TX_EMPTY, 0, "hidden while its interrupt is off (IRIX's handler loops on it)");
        assert!(!m.interrupting() && !irq.level(), "not before the transmit interrupt is enabled");
        m.write64(reg::TX_ALIAS, 1);
        assert!(m.interrupting() && irq.level());
        assert_eq!(m.read64(reg::INT_STATUS) & bits::INT_TX_EMPTY, bits::INT_TX_EMPTY, "latched, so enabling later still sees it");
        assert_eq!(m.read64(reg::DMA_CONTROL) & bits::DMA_TX_DMA_ENABLE, bits::DMA_TX_DMA_ENABLE, "TX_ALIAS leaves DMA alone");
        m.write64(reg::INT_STATUS, bits::INT_TX_EMPTY);
        assert!(!m.interrupting() && !irq.level(), "write one to clear");
    }

    /// A received frame lands in the oldest buffer handed over: padding, the
    /// frame, its CRC; then the status word, and RX_THRESHOLD.
    #[test]
    fn a_frame_is_received_into_the_oldest_buffer() {
        let (m, ram) = mec();
        let (to_net, _tx_cons) = rtrb::RingBuffer::new(8);
        let (mut rx_prod, from_net) = rtrb::RingBuffer::new(8);
        m.attach(to_net, from_net, None);
        let (irq, line) = crate::irq::LevelProbe::line();
        m.connect_irq(line);
        m.write64(reg::STATION, 0x0800_6912_3456);
        let (b0, b1) = (0x4000usize, 0x5000usize);
        m.write64(reg::MCL_RX_FIFO, RAM_BASE as u64 + b0 as u64);
        m.write64(reg::MCL_RX_FIFO, RAM_BASE as u64 + b1 as u64);
        m.write64(reg::DMA_CONTROL, bits::DMA_RX_DMA_ENABLE | bits::DMA_RX_INT_ENABLE | (4 << bits::DMA_RX_DMA_OFFSET_SHIFT));
        let mut frame = vec![0x08, 0x00, 0x69, 0x12, 0x34, 0x56, 2, 0, 0xde, 0xad, 0xbe, 0xef, 0x08, 0x00];
        frame.extend_from_slice(&[0xaa; 20]);
        rx_prod.push(frame.clone()).unwrap();
        rx_prod.push(vec![0x02, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 2]).unwrap(); // not for us
        m.poll();
        let stat = u64::from_be_bytes(get(&ram, b0, 8).try_into().unwrap());
        assert_ne!(stat & bits::RXSTAT_RECEIVED, 0);
        assert_eq!(stat & 0xffff, (MIN_FRAME + 4) as u64, "padded to 60, plus the CRC");
        assert_eq!(get(&ram, b0 + 32 + 2, frame.len()), frame, "the frame at offset 34");
        assert_eq!(get(&ram, b1, 8), vec![0u8; 8], "the other frame was not for this station");
        assert!(m.interrupting() && irq.level());
        assert_eq!((m.read64(reg::INT_STATUS) >> 8) & 0x1f, 1,
                   "the MCL FIFO read pointer, which OpenBSD reads to find filled buffers");
        m.write64(reg::INT_STATUS, bits::INT_RX_THRESHOLD);
        assert!(!m.interrupting() && !irq.level());
        assert_eq!(m.counts().1, 1);
    }
}
