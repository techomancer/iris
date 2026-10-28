//! MACE's serial DMA: the ring buffers between the UARTs and memory.

use std::sync::atomic::Ordering;
use std::sync::Mutex;

use crate::dev::crime_mem::O2Memory;
use crate::snapshot::{get_field, hex_u32, hex_u64, toml_u32, toml_u64};
use crate::ip32::*;

/// The serial DMA control registers, rings and status bits, as MACE's
/// datasheet lays them out.
pub mod sdma {
    /// Control bit 10: the channel is held in reset, pointers at zero.
    pub const CTRL_RESET: u64 = 0x400;
    /// Control bit 9: the engine runs. Clear, it is frozen with its state.
    pub const CTRL_ENABLE: u64 = 0x200;
    /// Control bits 7:5: which depth condition raises the threshold bit.
    pub const CTRL_COND: u64 = 0xe0;
    /// The bits a control register keeps; the rest read as zero.
    pub const CTRL_MASK: u64 = 0x6e0;
    /// Pointers and depths count 32-byte blocks in bits 11:5.
    pub const PTR_MASK: u32 = 0xfe0;
    /// One block short of the ring: the engine never fills the last slot.
    pub const RING_FULL: u32 = 0xfe0;
    pub const BLOCK: u32 = 32;
    /// Where the eight 4 KB rings start: bits 31:15 of the ring base register.
    pub const RING_BASE_MASK: u64 = 0xffff_8000;
    /// Transmit pair commands, in the top two bits of a control byte.
    pub const TX_CMD_THR: u8 = 1;
    pub const TX_CMD_MCR: u8 = 2;
    /// Transmit control bit 5: tell software once this pair is done.
    pub const TX_PREQ: u8 = 0x20;
    /// Receive control bit 7: this slot holds a real character.
    pub const RX_VALID: u8 = 0x80;

    /// Port `p`'s six ISA status bits, from bit 20 for port 0.
    pub const fn group(p: usize) -> u64 { 0x3f << (20 + 6 * p) }
    /// The 16550's own interrupt request.
    pub const fn dir(p: usize) -> u64 { 1 << (20 + 6 * p) }
    pub const fn tx_thir(p: usize) -> u64 { 1 << (21 + 6 * p) }
    pub const fn tx_preq(p: usize) -> u64 { 1 << (22 + 6 * p) }
    pub const fn tx_memerr(p: usize) -> u64 { 1 << (23 + 6 * p) }
    pub const fn rx_thir(p: usize) -> u64 { 1 << (24 + 6 * p) }
    pub const fn rx_overrun(p: usize) -> u64 { 1 << (25 + 6 * p) }
}

/// The serial DMA channels' registers: port 0 at the start, port 1 16 KB on.
pub const MACE_SDMA: u32 = MACE_BASE + 0x0031_8000;
pub const MACE_SDMA_SIZE: u32 = 0x0000_8000;

/// One direction of a serial port's DMA: its control register and the two
/// pointers into its ring, as byte offsets with only bits 11:5 kept.
#[derive(Clone, Copy)]
struct SdmaChan {
    ctrl: u64,
    rptr: u32,
    wptr: u32,
    /// Blocks and characters moved, so a quiet console can be told apart
    /// from an engine that never ran.
    blocks: u64,
    bytes: u64,
}

impl SdmaChan {
    /// Out of reset a channel is held in reset, which is what software that
    /// reads the control register before ever writing it gets back.
    const RESET: Self = Self { ctrl: sdma::CTRL_RESET, rptr: 0, wptr: 0, blocks: 0, bytes: 0 };

    /// Bytes of ring in use, always a whole number of blocks.
    fn depth(&self) -> u32 {
        self.wptr.wrapping_sub(self.rptr) & sdma::PTR_MASK
    }

    fn in_reset(&self) -> bool {
        self.ctrl & sdma::CTRL_RESET != 0
    }

    fn running(&self) -> bool {
        self.ctrl & (sdma::CTRL_RESET | sdma::CTRL_ENABLE) == sdma::CTRL_ENABLE
    }

    /// The threshold bit, as a level: true exactly while the depth meets the
    /// condition in the control register. A pure level is what lets the
    /// driver's own control writes take it down again; anything latched here
    /// would be an interrupt nobody can clear. It does not wait for ENABLE:
    /// a disabled channel is frozen, not switched off.
    fn threshold(&self, output: bool) -> bool {
        if self.in_reset() {
            return false;
        }
        let n = self.depth() >> 5;
        // The fill-level codes read the same comparator both ways round: an
        // input channel wants attention once it is that full, an output
        // channel once it has drained below it.
        let level = |at: u32| if output { n < at } else { n >= at };
        match (self.ctrl & sdma::CTRL_COND) >> 5 {
            1 => level(32),
            2 => level(64),
            3 => level(96),
            4 => n == 0,
            5 => n != 0,
            6 => n == 127,
            7 => n != 127,
            _ => false,
        }
    }
}

/// Both directions of one serial port, and whether the guest is using them.
#[derive(Clone, Copy)]
struct SdmaPort {
    tx: SdmaChan,
    rx: SdmaChan,
    /// The guest has taken a channel out of reset, so the port's interrupt
    /// bits follow the DMA model rather than the 16550 alone. See
    /// [`MaceSerialDma`].
    owned: bool,
    /// Latches. Only `preq` can ever be set: the model has no memory errors
    /// to report on receive, and applies back-pressure rather than overrun.
    preq: bool,
    tx_memerr: bool,
    rx_overrun: bool,
}

impl SdmaPort {
    const RESET: Self =
        Self { tx: SdmaChan::RESET, rx: SdmaChan::RESET, owned: false, preq: false, tx_memerr: false, rx_overrun: false };
}

/// What the engines reach outside themselves: memory, the two UARTs, and
/// where the rings are.
pub struct SdmaIo<'a> {
    pub ram: &'a O2Memory,
    pub com: [&'a Com16550; 2],
    /// The ring block's physical address, from the ring base register.
    pub ring_base: u32,
}

impl SdmaIo<'_> {
    /// Where block `ptr` of ring `ring` is on the bus, if it is memory. The
    /// rings are a DMA master's view of memory, as the Ethernet's are.
    fn block(&self, ram: &O2Memory, ring: u32, ptr: u32) -> Option<u64> {
        ram.dma_offset(self.ring_base | (ring << 12) | ptr).map(|p| p as u64)
    }
}

/// MACE's serial DMA: the rings that carry IRIX's console once its tty
/// driver is attached.
///
/// Each port has a transmit and a receive ring of 32-byte blocks in a 32 KB
/// area the ring base register names. A block is four 64-bit pairs, each four
/// control bytes then the four data bytes they describe. Software fills
/// transmit blocks and moves the write pointer; the engine sends them and
/// moves the read pointer. On receive the engine packs characters into
/// blocks and software follows behind.
///
/// Transmit is drained at once, inside the write of the write pointer: the
/// depth is back to zero by the time the driver reads it, and a polled
/// console write between two batches cannot overtake output still queued.
/// Receive fills from the UART's input queue on the service tick and
/// whenever the read pointer moves, never past one block short of full;
/// characters that do not fit wait in the queue.
///
/// Only a port whose guest has taken a channel out of reset is "owned" by
/// the model. The PROM, NetBSD and OpenBSD never touch these registers, and
/// for them the 16550's interrupt keeps driving all six of the port's status
/// bits exactly as before. An owned port's bits are the datasheet's: the two
/// thresholds and the latches, read through the mask, with the UART's own
/// request held low because in DMA mode the engine consumes it.
pub struct MaceSerialDma {
    /// MACE's ISA interrupt controller, which reads the ports' status bits
    /// through its mask.
    isa: std::sync::OnceLock<std::sync::Arc<crate::dev::mace_isa::MaceIsa>>,
    ports: Mutex<[SdmaPort; 2]>,
    /// Either port owned, readable without the lock, so the per-access hooks
    /// cost nothing on a machine that never uses DMA.
    any_owned: std::sync::atomic::AtomicBool,
}

impl Default for MaceSerialDma {
    fn default() -> Self { Self::new() }
}

impl MaceSerialDma {
    pub fn new() -> Self {
        Self { isa: std::sync::OnceLock::new(), ports: Mutex::new([SdmaPort::RESET; 2]), any_owned: Default::default() }
    }

    /// Every channel back in reset and neither port owned, as at power-on.
    /// A warm restart that left them running would leave the next kernel's
    /// pointers out of step with the engine's, and steal the PROM's input.
    pub fn reset(&self) {
        *self.ports.lock().unwrap() = [SdmaPort::RESET; 2];
        self.any_owned.store(false, Ordering::Release);
    }

    pub fn any_owned(&self) -> bool {
        self.any_owned.load(Ordering::Acquire)
    }

    /// Port and register number. Gaps within a port's 16 KB page alias to
    /// its eight registers.
    fn decode(addr: u32) -> (usize, u32) {
        (((addr >> 14) & 1) as usize, (addr >> 3) & 7)
    }

    pub fn read(&self, addr: u32) -> u64 {
        let (p, reg) = Self::decode(addr);
        let st = self.ports.lock().unwrap();
        let port = &st[p];
        (match reg {
            0 => return port.tx.ctrl,
            1 => port.tx.rptr,
            2 => port.tx.wptr,
            3 => port.tx.depth(),
            4 => return port.rx.ctrl,
            5 => port.rx.rptr,
            6 => port.rx.wptr,
            _ => port.rx.depth(),
        }) as u64
    }

    pub fn write(&self, addr: u32, val: u64, io: &SdmaIo) {
        let (p, reg) = Self::decode(addr);
        let mut st = self.ports.lock().unwrap();
        let port = &mut st[p];
        let ptr = val as u32 & sdma::PTR_MASK;
        match reg {
            0 | 4 => {
                let tx = reg == 0;
                let ch = if tx { &mut port.tx } else { &mut port.rx };
                ch.ctrl = val & sdma::CTRL_MASK;
                if ch.in_reset() {
                    ch.rptr = 0;
                    ch.wptr = 0;
                    if tx {
                        port.preq = false;
                        port.tx_memerr = false;
                    } else {
                        port.rx_overrun = false;
                    }
                }
                port.owned = !(port.tx.in_reset() && port.rx.in_reset());
                if tx {
                    Self::tx_run(p, port, io);
                } else {
                    Self::rx_run(p, port, io);
                }
                let owned = st.iter().any(|q| q.owned);
                self.any_owned.store(owned, Ordering::Release);
            }
            1 => port.tx.rptr = ptr,
            2 => {
                port.tx.wptr = ptr;
                Self::tx_run(p, port, io);
                // In loopback what was just sent is already waiting to come
                // back in.
                Self::rx_run(p, port, io);
            }
            5 => {
                port.rx.rptr = ptr;
                Self::rx_run(p, port, io);
            }
            // The receive write pointer is the engine's, and depths are
            // computed.
            _ => {}
        }
    }

    /// Move whatever can be moved: characters that arrived between register
    /// accesses (the TCP console, a script's `feed`), and any transmit work
    /// left over.
    pub fn poll(&self, io: &SdmaIo) {
        {
            let mut st = self.ports.lock().unwrap();
            for (p, port) in st.iter_mut().enumerate() {
                if port.owned {
                    Self::tx_run(p, port, io);
                    Self::rx_run(p, port, io);
                }
            }
        }
        // Blocks moved: thresholds and overruns may have changed.
        self.update_irq();
    }

    /// Send every block between the read and write pointers.
    fn tx_run(p: usize, port: &mut SdmaPort, io: &SdmaIo) {
        let ring = 4 + 2 * p as u32;
        while port.tx.running() && port.tx.rptr != port.tx.wptr {
            let mut b = [0u8; sdma::BLOCK as usize];
            {
                let ram = &*io.ram;
                let Some(i) = io.block(&ram, ring, port.tx.rptr) else {
                    // Nowhere to read from: the hardware stops the channel and
                    // says why.
                    port.tx_memerr = true;
                    port.tx.ctrl &= !sdma::CTRL_ENABLE;
                    return;
                };
                crate::physical::read_bytes(&**ram.phys(), i, &mut b);
            }
            for pair in b.chunks_exact(8) {
                for slot in 0..4 {
                    let (c, d) = (pair[slot], pair[4 + slot]);
                    match c >> 6 {
                        sdma::TX_CMD_THR => {
                            io.com[p].transmit(d);
                            port.tx.bytes += 1;
                        }
                        sdma::TX_CMD_MCR => io.com[p].write_mcr(d),
                        // Skip, or a delay: there is no wire to be idle on.
                        _ => {}
                    }
                    if c & sdma::TX_PREQ != 0 {
                        port.preq = true;
                    }
                }
            }
            port.tx.blocks += 1;
            port.tx.rptr = (port.tx.rptr + sdma::BLOCK) & sdma::PTR_MASK;
        }
    }

    /// Pack received characters into blocks while there is room.
    ///
    /// Each block is written whole, characters from the first slot on with
    /// the valid bit and every unused slot as padding (control 0, data 0xff),
    /// because IRIX's reader stops at the first all-zero control word and
    /// walks off the start of its buffer if a block it is handed has none in
    /// its first pair. The block is in memory before the write pointer says
    /// so: the service thread fills while the CPU reads.
    fn rx_run(p: usize, port: &mut SdmaPort, io: &SdmaIo) {
        let ring = 5 + 2 * p as u32;
        while port.rx.running() && port.rx.depth() < sdma::RING_FULL && !io.com[p].input_empty() {
            let ram = &*io.ram;
            let Some(i) = io.block(&ram, ring, port.rx.wptr) else { return };
            let mut chars = [0u8; 16];
            let n = io.com[p].take_input(&mut chars);
            if n == 0 {
                return;
            }
            let mut b = [0u8; sdma::BLOCK as usize];
            for (j, &c) in chars.iter().enumerate() {
                let (ctrl, data) = if j < n { (sdma::RX_VALID, c) } else { (0, 0xff) };
                b[8 * (j / 4) + j % 4] = ctrl;
                b[8 * (j / 4) + 4 + j % 4] = data;
            }
            crate::physical::write_bytes(&**ram.phys(), i, &b);
            port.rx.blocks += 1;
            port.rx.bytes += n as u64;
            port.rx.wptr = (port.rx.wptr + sdma::BLOCK) & sdma::PTR_MASK;
        }
    }

    /// Wire the ports' status into MACE's ISA interrupt controller.
    pub fn connect_irq(&self, isa: std::sync::Arc<crate::dev::mace_isa::MaceIsa>) {
        let _ = self.isa.set(isa);
    }

    /// Hand the ports' ownership and status bits to MACE's ISA controller.
    pub fn update_irq(&self) {
        if let Some(isa) = self.isa.get() {
            let (owned, bits) = if self.any_owned() { self.status() } else { ([false; 2], 0) };
            isa.set_serial_dma(owned, bits);
        }
    }

    /// Which ports are owned, and their status bits before the mask.
    pub fn status(&self) -> ([bool; 2], u64) {
        let st = self.ports.lock().unwrap();
        let mut bits = 0;
        for (p, port) in st.iter().enumerate() {
            for (on, bit) in [
                (port.tx.threshold(true), sdma::tx_thir(p)),
                (port.preq, sdma::tx_preq(p)),
                (port.tx_memerr, sdma::tx_memerr(p)),
                (port.rx.threshold(false), sdma::rx_thir(p)),
                (port.rx_overrun, sdma::rx_overrun(p)),
            ] {
                if on {
                    bits |= bit;
                }
            }
        }
        ([st[0].owned, st[1].owned], bits)
    }

    /// Software wrote the ISA status register: a pair-request latch whose
    /// bit it wrote as zero is cleared. Every other serial bit is a level or
    /// fatal, and is not the guest's to change.
    pub fn status_written(&self, val: u64) {
        let mut st = self.ports.lock().unwrap();
        for (p, port) in st.iter_mut().enumerate() {
            if val & sdma::tx_preq(p) == 0 {
                port.preq = false;
            }
        }
    }

    /// One line per port: who owns it, its control registers, pointers and
    /// what has moved.
    pub fn summary(&self) -> String {
        let st = self.ports.lock().unwrap();
        st.iter()
            .enumerate()
            .map(|(p, s)| {
                format!(
                    "port {p}{}: tx ctrl {:03x} r {:03x} w {:03x} ({} blocks, {} bytes); \
                     rx ctrl {:03x} r {:03x} w {:03x} ({} blocks, {} bytes)",
                    if s.owned { " (DMA)" } else { "" },
                    s.tx.ctrl, s.tx.rptr, s.tx.wptr, s.tx.blocks, s.tx.bytes,
                    s.rx.ctrl, s.rx.rptr, s.rx.wptr, s.rx.blocks, s.rx.bytes,
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl crate::traits::Device for MaceSerialDma {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("sdma".to_string(), "MACE serial DMA: sdma status".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write;
        if cmd != "sdma" {
            return Err("Command not found".to_string());
        }
        match args.first().copied().unwrap_or("status") {
            "status" => {
                let ports = self.ports.lock().unwrap();
                for (n, p) in ports.iter().enumerate() {
                    writeln!(w, "serial {n}: {}{}{}{}", if p.owned { "DMA-driven" } else { "programmed I/O" },
                        if p.preq { ", request" } else { "" }, if p.tx_memerr { ", tx memory error" } else { "" },
                        if p.rx_overrun { ", rx overrun" } else { "" }).ok();
                    for (dir, c) in [("tx", &p.tx), ("rx", &p.rx)] {
                        writeln!(w, "  {dir}: control {:#x} read {:#05x} write {:#05x}, {} blocks / {} bytes moved",
                            c.ctrl, c.rptr, c.wptr, c.blocks, c.bytes).ok();
                    }
                }
                Ok(())
            }
            _ => Err("Usage: sdma status".to_string()),
        }
    }
}

impl SdmaChan {
    fn save(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("ctrl".into(), hex_u64(self.ctrl));
        t.insert("rptr".into(), hex_u32(self.rptr));
        t.insert("wptr".into(), hex_u32(self.wptr));
        t.insert("blocks".into(), hex_u64(self.blocks));
        t.insert("bytes".into(), hex_u64(self.bytes));
        toml::Value::Table(t)
    }

    fn load(v: &toml::Value) -> Self {
        let u = |k: &str| get_field(v, k).and_then(toml_u64).unwrap_or(0);
        let p = |k: &str| get_field(v, k).and_then(toml_u32).unwrap_or(0);
        Self { ctrl: u("ctrl"), rptr: p("rptr"), wptr: p("wptr"), blocks: u("blocks"), bytes: u("bytes") }
    }
}

/// Both ports' channels and latches; loading hands the ports' status to
/// MACE ISA again.
impl crate::traits::Saveable for MaceSerialDma {
    fn save_state(&self) -> toml::Value {
        let ports = self.ports.lock().unwrap().iter().map(|p| {
            let mut t = toml::map::Map::new();
            t.insert("tx".into(), p.tx.save());
            t.insert("rx".into(), p.rx.save());
            t.insert("owned".into(), toml::Value::Boolean(p.owned));
            t.insert("preq".into(), toml::Value::Boolean(p.preq));
            t.insert("tx_memerr".into(), toml::Value::Boolean(p.tx_memerr));
            t.insert("rx_overrun".into(), toml::Value::Boolean(p.rx_overrun));
            toml::Value::Table(t)
        }).collect();
        let mut t = toml::map::Map::new();
        t.insert("ports".into(), toml::Value::Array(ports));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut any = false;
        if let Some(toml::Value::Array(a)) = get_field(v, "ports") {
            let mut ports = self.ports.lock().unwrap();
            for (p, e) in ports.iter_mut().zip(a) {
                let b = |k: &str| get_field(e, k).and_then(|x| x.as_bool()).unwrap_or(false);
                *p = SdmaPort {
                    tx: get_field(e, "tx").map_or(SdmaChan::RESET, SdmaChan::load),
                    rx: get_field(e, "rx").map_or(SdmaChan::RESET, SdmaChan::load),
                    owned: b("owned"),
                    preq: b("preq"),
                    tx_memerr: b("tx_memerr"),
                    rx_overrun: b("rx_overrun"),
                };
                any |= p.owned;
            }
        }
        self.any_owned.store(any, Ordering::Release);
        self.update_irq();
        Ok(())
    }
}

impl crate::traits::Resettable for MaceSerialDma {
    fn power_on(&self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_threshold_conditions_follow_the_datasheet_table() {
        let blocks = [0u32, 31, 32, 64, 96, 127];
        // Input channel, per code 1..7, for each depth above.
        let input: [[bool; 6]; 7] = [
            [false, false, true, true, true, true],
            [false, false, false, true, true, true],
            [false, false, false, false, true, true],
            [true, false, false, false, false, false],
            [false, true, true, true, true, true],
            [false, false, false, false, false, true],
            [true, true, true, true, true, false],
        ];
        for code in 0..8u64 {
            for (i, &n) in blocks.iter().enumerate() {
                let ch = SdmaChan { ctrl: code << 5, rptr: 0, wptr: n * 32, blocks: 0, bytes: 0 };
                let (inp, out) = if code == 0 {
                    (false, false)
                } else {
                    let v = input[code as usize - 1][i];
                    // The fill levels invert for an output channel.
                    (v, if code <= 3 { !v } else { v })
                };
                assert_eq!(ch.threshold(false), inp, "input, code {code}, {n} blocks");
                assert_eq!(ch.threshold(true), out, "output, code {code}, {n} blocks");
                let held = SdmaChan { ctrl: ch.ctrl | sdma::CTRL_RESET, ..ch };
                assert!(!held.threshold(false) && !held.threshold(true), "never in reset");
            }
        }
    }

    #[test]
    fn save_load_round_trip() {
        use crate::traits::Saveable;
        let src = MaceSerialDma::new();
        {
            let mut p = src.ports.lock().unwrap();
            p[0].tx = SdmaChan { ctrl: sdma::CTRL_ENABLE | 4 << 5, rptr: 0x40, wptr: 0x80, blocks: 3, bytes: 77 };
            p[0].rx = SdmaChan { ctrl: sdma::CTRL_ENABLE | 5 << 5, rptr: 0x20, wptr: 0x60, blocks: 1, bytes: 9 };
            p[0].owned = true;
            p[0].preq = true;
            p[1].rx_overrun = true;
        }
        src.any_owned.store(true, Ordering::Release);
        let v1 = src.save_state();

        let dst = MaceSerialDma::new();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "MaceSerialDma save_state mismatch after load_state round-trip");
        assert!(dst.any_owned(), "ownership is derived again on load");
    }
}
