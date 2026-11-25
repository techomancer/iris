// Bus status constants.
//
// These MUST match the corresponding ExecStatus values in mips_exec.rs.
// Enforced by compile-time asserts at the bottom of this file.
//
//   BUS_OK   == EXEC_COMPLETE  == 0x0000_0000
//   BUS_BUSY == EXEC_RETRY     == 0x0000_0100
//   BUS_ERR  == EXEC_BUS_ERR   == exec_exception(EXC_DBE)  == 0x0800_001C
//   BUS_VCE  == EXEC_BUS_VCE   == exec_exception(EXC_VCED) == 0x0800_007C
//
// Write operations return a plain u32 equal to one of these constants.
// The hot path in mips_exec.rs can pass write results through as ExecStatus
// with zero conversion.
//
// BUS_VCE is only ever returned by MipsCache (src/cpu/mips_cache_v2.rs).
// BusDevice implementations must never return BUS_VCE.
pub const BUS_OK:   u32 = 0x0000_0000;
pub const BUS_BUSY: u32 = 0x0000_0100;
pub const BUS_ERR:  u32 = 0x0800_001C; // EXEC_IS_EXCEPTION | (EXC_DBE  << CAUSE_EXCCODE_SHIFT)
pub const BUS_VCE:  u32 = 0x0800_007C; // EXEC_IS_EXCEPTION | (EXC_VCED << CAUSE_EXCCODE_SHIFT)

/// Result of an 8-bit bus read.
/// `status == BUS_OK` means the read succeeded; `data` is valid only then.
#[derive(Clone, Copy, Debug)]
pub struct BusRead8  { pub status: u32, pub data: u8  }
/// Result of a 16-bit bus read.
#[derive(Clone, Copy, Debug)]
pub struct BusRead16 { pub status: u32, pub data: u16 }
/// Result of a 32-bit bus read.
#[derive(Clone, Copy, Debug)]
pub struct BusRead32 { pub status: u32, pub data: u32 }
/// Result of a 64-bit bus read.
#[derive(Clone, Copy, Debug)]
pub struct BusRead64 { pub status: u32, pub data: u64 }

macro_rules! impl_bus_read {
    ($t:ident, $d:ty) => {
        impl $t {
            #[inline(always)] pub fn ok(data: $d) -> Self  { Self { status: BUS_OK,   data } }
            #[inline(always)] pub fn busy() -> Self         { Self { status: BUS_BUSY, data: 0 } }
            #[inline(always)] pub fn err() -> Self          { Self { status: BUS_ERR,  data: 0 } }
            #[inline(always)] pub fn vce() -> Self          { Self { status: BUS_VCE,  data: 0 } }
            #[inline(always)] pub fn is_ok(self) -> bool    { self.status == BUS_OK }
        }
    }
}
impl_bus_read!(BusRead8,  u8);
impl_bus_read!(BusRead16, u16);
impl_bus_read!(BusRead32, u32);
impl_bus_read!(BusRead64, u64);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    Reset(bool),
    Interrupt(u32, bool),
}

/// What 4 KiB of runnable memory is, whatever address a processor reaches it
/// at: the device that holds the bytes ([`CodeBank`]) and the page within it.
/// The JIT keys compiled code by it, so the code follows the bytes when a
/// memory controller moves a bank, and two processors that see the same bytes
/// at different addresses share one compiled page.
#[cfg(feature = "jitv2")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CodeKey {
    pub bank: u32,
    /// Page number within the bank: byte offset / 4096.
    pub page: u32,
}

#[cfg(feature = "jitv2")]
impl CodeKey {
    /// The key of no page.
    pub const NONE: CodeKey = CodeKey { bank: u32::MAX, page: u32::MAX };

    /// The first of the bank numbers kept for pages no bank names
    /// ([`Self::at_address`]); no [`CodeBank`] counts that far.
    const ADDRESS_BANKS: u32 = u32::MAX - 0x100;

    /// The key of a page that no bank on the bus names (a device behind a
    /// bridge that does not forward the question): its address, in address
    /// space `space` of the JIT pool. Private to that space, since only its
    /// CPUs are known to see the same bytes there.
    pub const fn at_address(space: usize, frame: u32, uncached: bool) -> CodeKey {
        CodeKey { bank: Self::ADDRESS_BANKS + space as u32, page: frame << 1 | uncached as u32 }
    }

    /// Whether this is an [`Self::at_address`] key.
    pub const fn is_address(self) -> bool {
        self.bank >= Self::ADDRESS_BANKS && self.bank != u32::MAX
    }

    /// A bus that is one flat memory (a test bus, a tool) names a page by its
    /// address, in bank 0, which no [`CodeBank`] is ever given.
    pub const fn flat(addr: u64) -> CodeKey {
        CodeKey { bank: 0, page: (addr >> 12) as u32 }
    }

    /// [`Self::flat`] as a bus answers it, counted by `gen`.
    pub const fn flat_page(addr: u64, gen: *const std::sync::atomic::AtomicU64) -> CodePage {
        CodePage { key: Self::flat(addr), gen, words: std::ptr::null() }
    }
}

/// What a bus answers for a page of runnable memory: its key, and the bank's
/// own write counter for the page. The counter belongs to the bank, not to an
/// address: it stays the page's counter wherever a memory controller puts the
/// bank (a ppmem window's counter for an address does not).
#[cfg(feature = "jitv2")]
#[derive(Clone, Copy, Debug)]
pub struct CodePage {
    pub key: CodeKey,
    pub gen: *const std::sync::atomic::AtomicU64,
    /// The page's 1024 words in the bank's own storage, for a bank that can
    /// hand them out: the JIT's compile workers read the page there, with no
    /// bus and no address. Like `gen` it belongs to the bank, stays put when
    /// the bank moves on the bus, and is valid while the device lives. Null
    /// for a device that cannot (a flash whose reads depend on its mode, a
    /// test bus); such a page is read through the bus at a CPU's address.
    pub words: *const u32,
}

/// The identity of one device that holds runnable memory: a RAM bank, a PROM,
/// a flash part. Each takes the next number from one counter when it is built,
/// so no two in the process share one.
#[cfg(feature = "jitv2")]
#[derive(Debug)]
pub struct CodeBank(u32);

#[cfg(feature = "jitv2")]
impl CodeBank {
    pub fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
        CodeBank(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }

    pub fn id(&self) -> u32 {
        self.0
    }

    /// The key of the page holding byte `offset` of this bank.
    #[inline]
    pub fn key(&self, offset: u64) -> CodeKey {
        CodeKey { bank: self.0, page: (offset >> 12) as u32 }
    }

    /// The page holding byte `offset` of this bank, counted by `gen`.
    #[inline]
    pub fn page(&self, offset: u64, gen: *const std::sync::atomic::AtomicU64) -> CodePage {
        CodePage { key: self.key(offset), gen, words: std::ptr::null() }
    }
}

#[cfg(feature = "jitv2")]
impl Default for CodeBank {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified bus device interface supporting 8/16/32/64-bit accesses.
///
/// Read methods return `BusRead{N}` with `status == BUS_OK` on success.
/// Write methods return a plain `u32` equal to `BUS_OK`, `BUS_BUSY`, or `BUS_ERR`.
/// The write return value is layout-compatible with `ExecStatus` (see mips_exec.rs).
///
/// Devices implement only the widths they natively support.
/// Unimplemented widths return `BusReadN::err()` / `BUS_ERR` by default.
pub trait BusDevice: Send + Sync {
    // 8-bit access
    fn read8  (&self, _addr: u64) -> BusRead8  { BusRead8::err()  }
    fn write8 (&self, _addr: u64, _val: u8)  -> u32 { BUS_ERR }

    // 16-bit access
    fn read16 (&self, _addr: u64) -> BusRead16 { BusRead16::err() }
    fn write16(&self, _addr: u64, _val: u16) -> u32 { BUS_ERR }

    // 32-bit access
    fn read32 (&self, _addr: u64) -> BusRead32 { BusRead32::err() }
    fn write32(&self, _addr: u64, _val: u32) -> u32 { BUS_ERR }

    // 64-bit access
    fn read64 (&self, _addr: u64) -> BusRead64 { BusRead64::err() }
    fn write64(&self, _addr: u64, _val: u64) -> u32 { BUS_ERR }

    /// Why a failed `size`-byte access at `addr` failed, in a few words, for
    /// the CPU's bus-error line. Asked only after the access has already
    /// failed; it never changes the status, the exception or any device
    /// state. A device overrides it for a failure it can explain, such as a
    /// probe of a card that is not there.
    fn describe_fault(&self, _addr: u64, _size: usize) -> Option<String> { None }

    /// 64-bit access issued by a DMA engine (e.g. MC's VDMA), as opposed to
    /// CPU-driven PIO. Defaults to plain `read64`/`write64` — most devices don't
    /// distinguish. A device whose PIO protocol has GO-triggered side effects
    /// (REX3's HOSTRW: a GO-space access both returns/consumes the current word
    /// AND arms the next one) can override these to implement the DMA-specific
    /// sequencing real hardware uses instead, without forcing every other
    /// BusDevice to care about the distinction.
    fn dma_read64 (&self, addr: u64) -> BusRead64 { self.read64(addr) }
    fn dma_write64(&self, addr: u64, val: u64) -> u32 { self.write64(addr, val) }

    /// Bulk form of [`dma_write64`](Self::dma_write64): hand the device a whole
    /// run of qwords destined for `addr` in one call.
    ///
    /// This exists to collapse the per-qword producer/consumer round trip. For
    /// REX3's HOSTRW port the scalar path costs a GFIFO push per 8 bytes; the
    /// batch form pushes **one** token carrying the whole slice, which the
    /// register processor streams into the host buffer before a single draw.
    ///
    /// Returns `BUS_OK` when the whole slice was accepted, or `BUS_BUSY` when
    /// none of it was. **It is all-or-nothing**: a device must not consume a
    /// prefix and report busy, because the DMA worker has no EXEC_RETRY and
    /// will re-issue the identical slice — a partial commit duplicates data
    /// (the bug class `try_push2` exists to prevent). `BUS_ERR` means the
    /// address is not batchable and the caller should fall back to scalar.
    ///
    /// The default implementation is the obvious scalar loop, which is correct
    /// for every device but wins nothing.
    ///
    /// It spins on BUS_BUSY rather than returning it. That is not laziness: a
    /// scalar loop cannot un-consume the words it already pushed, so returning
    /// busy mid-slice would break the all-or-nothing contract and the caller's
    /// retry would duplicate the prefix. Spinning is what the VDMA worker did
    /// per word before batching existed, and it keeps the default honest. A
    /// device that wants real back-pressure must override this and check
    /// capacity for the whole slice up front, the way REX3 does.
    fn dma_write64_bulk(&self, addr: u64, vals: &[u64]) -> u32 {
        for &v in vals.iter() {
            loop {
                let st = self.dma_write64(addr, v);
                if st == BUS_OK { break; }
                if st != BUS_BUSY { return st; }
                std::hint::spin_loop();
            }
        }
        BUS_OK
    }

    /// Bulk form of [`dma_read64`](Self::dma_read64): fill `out` with a run of
    /// qwords read from `addr`.
    ///
    /// Mirrors [`dma_write64_bulk`](Self::dma_write64_bulk). For REX3 this
    /// replaces one `wait_idle()` full-pipeline drain *per qword* with one per
    /// batch: the shader fills its host buffer, then the whole thing is copied
    /// out here.
    ///
    /// Returns `BUS_OK` on success. On any other status the contents of `out`
    /// are unspecified and the caller must not use them.
    fn dma_read64_bulk(&self, addr: u64, out: &mut [u64]) -> u32 {
        for slot in out.iter_mut() {
            let r = self.dma_read64(addr);
            if !r.is_ok() {
                return r.status;
            }
            *slot = r.data;
        }
        BUS_OK
    }

    /// Return a pointer directly into the device's backing store at `addr`.
    /// `addr` must be 8-byte aligned. The pointer is valid for the lifetime of `&self`.
    /// Returns `None` for devices that do not support direct pointer access (default).
    /// Data at the pointer is in the same layout as `read64` returns (host u64 chunks
    /// with the two u32 halves swapped relative to native order — `rotate_left(32)`).
    fn mem_ptr(&self, _addr: u64) -> Option<*const u64> {
        None
    }

    /// Read `buf.len()` consecutive u64 chunks starting at `addr` into `buf`.
    /// `addr` must be 8-byte aligned. Each u64 in `buf` is in the same byte
    /// order as two consecutive `read32()` calls in big-endian (MIPS) word order:
    /// `buf[i] = (read32(addr + i*8) as u64) << 32 | read32(addr + i*8 + 4) as u64`.
    /// Returns BUS_OK on success or the first error status encountered.
    /// Default: pairs of `read32` calls so devices that only implement 32-bit access work.
    fn read_block(&self, addr: u64, buf: &mut [u64]) -> u32 {
        let mut a = addr;
        for slot in buf.iter_mut() {
            let hi = self.read32(a);
            if hi.status != BUS_OK { return hi.status; }
            let lo = self.read32(a.wrapping_add(4));
            if lo.status != BUS_OK { return lo.status; }
            *slot = ((hi.data as u64) << 32) | lo.data as u64;
            a = a.wrapping_add(8);
        }
        BUS_OK
    }

    /// Write `buf.len()` consecutive u64 chunks starting at `addr` from `buf`.
    /// `addr` must be 8-byte aligned. Each u64 is split as two `write32` calls
    /// in big-endian (MIPS) word order: high word first, then low word.
    /// Returns BUS_OK on success or the first error status encountered.
    /// Default: pairs of `write32` calls so devices that only implement 32-bit access work.
    fn write_block(&self, addr: u64, buf: &[u64]) -> u32 {
        let mut a = addr;
        for &val in buf.iter() {
            let s = self.write32(a, (val >> 32) as u32);
            if s != BUS_OK { return s; }
            let s = self.write32(a.wrapping_add(4), val as u32);
            if s != BUS_OK { return s; }
            a = a.wrapping_add(8);
        }
        BUS_OK
    }

    /// Atomically replace the aligned word at `addr` if it holds `old`: whether
    /// it did, or `None` when this is not RAM and the caller stores as usual.
    /// What a store-conditional does on a machine with several CPUs.
    fn cas32(&self, _addr: u64, _old: u32, _new: u32) -> Option<bool> { None }
    /// The doubleword form of [`Self::cas32`].
    fn cas64(&self, _addr: u64, _old: u64, _new: u64) -> Option<bool> { None }

    /// Masked 64-bit write: only bytes where the corresponding mask byte is 0xFF are written.
    /// `addr` must be 8-byte aligned; `val` and `mask` are in big-endian (MIPS) byte order.
    /// Default: decompose into individual write8 calls for set bytes.
    fn write64_masked(&self, addr: u64, val: u64, mask: u64) -> u32 {
        for i in 0..8usize {
            let bit = 7 - i;
            let byte_mask = (mask >> (bit * 8)) & 0xFF;
            if byte_mask != 0 {
                let byte_val = (val >> (bit * 8)) as u8;
                let ws = self.write8(addr + i as u64, byte_val);
                if ws != BUS_OK { return ws; }
            }
        }
        BUS_OK
    }

    /// An uncached SB (`size` 1) or SH (`size` 2) to `addr`, with the whole
    /// source register `gpr` for a bus that can see the lanes the store does
    /// not select. The CPU calls this instead of `write8`/`write16` for an
    /// uncached SB or SH.
    ///
    /// The architectural byte or halfword is the low bits of `gpr`, which is
    /// all the default delivers. A partial-word write drives all of
    /// SysAD\[63:0\] in its data cycle, but the R10000 User's Manual (v2.0,
    /// p. 104, "SysAD\[63:0\] Data Cycle Encoding") defines only the lanes the
    /// address and size select; the other lanes are not specified. So only a
    /// device that models a path observing them (a memory ECC view) overrides
    /// this, and none in this tree does. A bus that forwards accesses must
    /// forward this call too.
    fn write_narrow_from(&self, addr: u64, size: usize, gpr: u64) -> u32 {
        if size == 1 { self.write8(addr, gpr as u8) } else { self.write16(addr, gpr as u16) }
    }

    /// JIT v2 (clanker/jitv2/jit-v2-design.md §2.4): pointer to the generation counter
    /// for the 4KB physical page containing `addr`, or null if this device doesn't
    /// back JIT-compilable memory (MMIO, etc). RAM devices return one counter per
    /// page, bumped on every mutation to that page; ROM devices may point every
    /// page at a single counter that is never bumped (content is immutable).
    /// The pointer is valid for the lifetime of the device.
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, _addr: u64) -> *const std::sync::atomic::AtomicU64 {
        std::ptr::null()
    }

    /// JIT v2: which bank holds the 4 KiB page at `addr` and where in it
    /// (the page's [`CodeKey`]), with the bank's counter for it, or `None`
    /// where no bank answers (MMIO, a hole, a device that does not say). The
    /// JIT keys a page fetched from there by its address and counts it with
    /// [`Self::gen_ptr`], as it would a ROM. The answer changes when a memory controller
    /// moves a bank; the bytes and the counter a key names do not. A bus that
    /// forwards accesses forwards this too. `addr` is the start of the page.
    #[cfg(feature = "jitv2")]
    fn code_page(&self, _addr: u64) -> Option<CodePage> {
        None
    }
}

pub trait FifoDevice: Send + Sync {
    fn read_fifo(&self) -> u8;
    fn write_fifo(&self, val: u8, notify: bool);
}

/// Status bits returned by DMA read/write/advance operations.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct DmaStatus(pub u32);

impl DmaStatus {
    pub const OK:         u32 = 0x00; // no flags — normal transfer
    pub const EOP:        u32 = 0x01; // end-of-packet descriptor boundary reached
    pub const EOX:        u32 = 0x02; // end-of-chain: descriptor chain exhausted, channel deactivated
    pub const IRQ:        u32 = 0x04; // DMA interrupt raised (xie was set on EOX)
    pub const NOT_ACTIVE: u32 = 0x08; // transfer refused — channel not active
    pub const OVERFLOW:   u32 = 0x20; // byte count exhausted mid-transfer

    pub fn ok()          -> Self { Self(Self::OK) }
    pub fn is_ok(self)   -> bool { self.0 == Self::OK }
    pub fn eop(self)     -> bool { self.0 & Self::EOP        != 0 }
    pub fn eox(self)     -> bool { self.0 & Self::EOX        != 0 }
    pub fn irq(self)     -> bool { self.0 & Self::IRQ        != 0 }
    pub fn not_active(self) -> bool { self.0 & Self::NOT_ACTIVE != 0 }
    pub fn refused(self) -> bool { self.0 & (Self::NOT_ACTIVE | Self::OVERFLOW) != 0 }
}

impl std::ops::BitOr for DmaStatus {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
}
impl std::ops::BitOrAssign for DmaStatus {
    fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
}

pub trait DmaClient: Send + Sync {
    /// Read the next value from the DMA channel, with the channel's status.
    fn read(&self) -> Option<(u32, DmaStatus)>;
    /// Write a value to the DMA channel. Returns the channel's status.
    fn write(&self, val: u32, eop: bool) -> DmaStatus;
    /// How far the device may read ahead of the guest: `Some(n)` when the
    /// guest is polling this channel's position and the device has read `n`
    /// words beyond the position the guest last saw (see HAL2's codec A).
    /// `None` when it is not being polled.
    fn read_ahead_of_poll(&self) -> Option<u64> { None }
    /// The peripheral overran its side of the transfer (the SEEQ's receive
    /// FIFO filled before the channel drained it). A channel that stops on
    /// that condition stops here.
    fn overflow(&self) {}
    /// Whether a receive channel can take a frame now; see [`RxRoom`].
    fn rx_room(&self) -> RxRoom { RxRoom::Room }
    /// Wait up to `timeout` for the guest to make the channel active, for a
    /// device that reached its data phase before the driver armed the
    /// channel. Returns whether the channel is active. A channel that can
    /// signal its arming wakes the caller at once (HPC3); the default sleeps
    /// a short slice and returns false, so a caller retrying in a loop polls.
    fn wait_active(&self, timeout: std::time::Duration) -> bool {
        std::thread::sleep(timeout.min(std::time::Duration::from_micros(100)));
        false
    }
}

/// What a receive channel can take now. The emulator's host side uses it to
/// hold frames off the virtual wire rather than lose them, which is a choice
/// of the model, not hardware behaviour (see the SEEQ's receive hold).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RxRoom {
    /// A frame now lands in a descriptor and the channel carries on.
    Room,
    /// The channel is stopped: a frame now is lost, as on the wire. The hold
    /// does not wait for it.
    Stopped,
    /// A frame now fills the descriptor that ends the chain, stopping the
    /// channel, while the driver still has filled descriptors to reap.
    Last,
}

/// A terminal device failure that stops the machine and prevents restart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineFatal {
    pub source: &'static str,
    pub message: String,
}

impl MachineFatal {
    pub fn new(source: &'static str, message: impl Into<String>) -> Self {
        Self { source, message: message.into() }
    }
}

impl std::fmt::Display for MachineFatal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.source, self.message)
    }
}

/// Asynchronous system-level events sent from devices to the machine event loop.
#[derive(Debug)]
pub enum MachineEvent {
    /// Full system reset (SIN bit in CPUCTRL0).
    HardReset,
    /// Soft power-off (front panel power state = 0).
    PowerOff,
}

/// Where devices send [`MachineEvent`]s, stamped with the machine's power-on
/// count, which every reset and restore bumps (`powered_on`): the event
/// thread drops a hard reset sent to a machine since replaced. Once that
/// thread listens, a hard reset also holds every CPU until the next power-on.
#[derive(Clone)]
pub struct MachineEvents {
    tx: std::sync::mpsc::SyncSender<(u64, MachineEvent)>,
    power_ons: std::sync::Arc<std::sync::atomic::AtomicU64>,
    holds: std::sync::Arc<ResetHolds>,
}

#[derive(Default)]
struct ResetHolds {
    cpus: std::sync::OnceLock<std::sync::Arc<crate::cpu::smp::World>>,
    listening: std::sync::atomic::AtomicBool,
}

impl ResetHolds {
    fn set(&self, on: bool) {
        if let Some(world) = self.cpus.get() {
            if on {
                world.hold_for_reset();
            } else {
                world.reset_done();
            }
        }
    }
}

pub struct MachineEventQueue {
    rx: std::sync::mpsc::Receiver<(u64, MachineEvent)>,
    power_ons: std::sync::Arc<std::sync::atomic::AtomicU64>,
    holds: std::sync::Arc<ResetHolds>,
}

impl MachineEvents {
    /// A queue of `capacity` events; a send to a full one is dropped.
    pub fn channel(capacity: usize) -> (Self, MachineEventQueue) {
        let (tx, rx) = std::sync::mpsc::sync_channel(capacity);
        let power_ons = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let holds = std::sync::Arc::new(ResetHolds::default());
        (Self { tx, power_ons: power_ons.clone(), holds: holds.clone() }, MachineEventQueue { rx, power_ons, holds })
    }

    /// The CPUs a hard reset stops.
    pub fn set_cpus(&self, world: std::sync::Arc<crate::cpu::smp::World>) {
        let _ = self.holds.cpus.set(world);
    }

    pub fn try_send(&self, event: MachineEvent) -> Result<(), std::sync::mpsc::TrySendError<(u64, MachineEvent)>> {
        use std::sync::mpsc::TrySendError;
        // Held before it is queued: the reset may run before this returns.
        let hold = matches!(event, MachineEvent::HardReset) && self.holds.listening.load(std::sync::atomic::Ordering::SeqCst);
        if hold {
            self.holds.set(true);
        }
        let sent = self.tx.try_send((self.power_ons.load(std::sync::atomic::Ordering::SeqCst), event));
        // Full: a waiting reset needs the hold. Disconnected: no reset will come.
        if hold && matches!(sent, Err(TrySendError::Disconnected(_))) {
            self.holds.set(false);
        }
        sent
    }

    /// [`Self::try_send`]: a device never waits on the queue.
    pub fn send(&self, event: MachineEvent) -> Result<(), std::sync::mpsc::TrySendError<(u64, MachineEvent)>> {
        self.try_send(event)
    }

    pub fn powered_on(&self) {
        self.power_ons.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.holds.set(false);
    }
}

impl MachineEventQueue {
    /// The event thread runs: from now on a hard reset holds the CPUs.
    pub fn listen(&self) {
        self.holds.listening.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// No reset will follow: let the CPUs run on.
    pub fn release_holds(&self) {
        self.holds.set(false);
    }

    pub fn recv(&self) -> Option<(u64, MachineEvent)> {
        self.rx.recv().ok()
    }

    pub fn try_recv(&self) -> Result<MachineEvent, std::sync::mpsc::TryRecvError> {
        self.rx.try_recv().map(|(_, e)| e)
    }

    /// Ask with the machine locked, so no restore runs in between.
    pub fn is_current(&self, sent: u64) -> bool {
        sent == self.power_ons.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Restore hardware to power-on state.
/// Called with all device threads stopped.
pub trait Resettable {
    fn power_on(&self);
}

/// Serialize / deserialize device register state to/from TOML.
/// Memory bulk data (RAM) is handled separately as raw binary.
pub trait Saveable {
    fn save_state(&self) -> toml::Value;
    fn load_state(&self, v: &toml::Value) -> Result<(), String>;
}

pub trait Device: Send + Sync {
    fn step(&self, cycles: u64);
    fn stop(&self);
    fn start(&self);
    fn is_running(&self) -> bool;

    fn signal(&self, _signal: Signal) {}

    fn register_commands(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    fn execute_command(&self, _cmd: &str, _args: &[&str], _writer: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        Err("Command not found".to_string())
    }

    /// A sound card's report for the monitor's `audio status`
    /// (see `host_audio::Meter::report`); other devices have none.
    fn audio_report(&self) -> Option<String> {
        None
    }

    /// Frames a sound card has moved between the guest and the host, inputs
    /// and outputs together, since it was built; other devices have none.
    /// It only goes up while sound plays or records, so a front-end compares
    /// two samples for an activity light.
    fn audio_frames(&self) -> Option<u64> {
        None
    }
}

// Compile-time asserts: BUS_* constants must equal the corresponding ExecStatus values.
// If these fail, update BOTH sets of constants to agree.
// See also: mips_exec.rs EXEC_BUS_ERR / EXEC_BUS_VCE / EXEC_COMPLETE / EXEC_RETRY.
const _: () = assert!(BUS_OK   == crate::cpu::mips_exec::EXEC_COMPLETE);
const _: () = assert!(BUS_BUSY == crate::cpu::mips_exec::EXEC_RETRY);
const _: () = assert!(BUS_ERR  == crate::cpu::mips_exec::EXEC_BUS_ERR);
const _: () = assert!(BUS_VCE  == crate::cpu::mips_exec::EXEC_BUS_VCE);
