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
// BUS_VCE is only ever returned by MipsCache (src/mips_cache_v2.rs).
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
    fn read8  (&self, _addr: u32) -> BusRead8  { BusRead8::err()  }
    fn write8 (&self, _addr: u32, _val: u8)  -> u32 { BUS_ERR }

    // 16-bit access
    fn read16 (&self, _addr: u32) -> BusRead16 { BusRead16::err() }
    fn write16(&self, _addr: u32, _val: u16) -> u32 { BUS_ERR }

    // 32-bit access
    fn read32 (&self, _addr: u32) -> BusRead32 { BusRead32::err() }
    fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_ERR }

    // 64-bit access
    fn read64 (&self, _addr: u32) -> BusRead64 { BusRead64::err() }
    fn write64(&self, _addr: u32, _val: u64) -> u32 { BUS_ERR }

    /// Masked 64-bit write: only bytes where the corresponding mask byte is 0xFF are written.
    /// `addr` must be 8-byte aligned; `val` and `mask` are in big-endian (MIPS) byte order.
    /// Default: decompose into individual write8 calls for set bytes.
    fn write64_masked(&self, addr: u32, val: u64, mask: u64) -> u32 {
        for i in 0..8usize {
            let bit = 7 - i;
            let byte_mask = (mask >> (bit * 8)) & 0xFF;
            if byte_mask != 0 {
                let byte_val = (val >> (bit * 8)) as u8;
                let ws = self.write8(addr + i as u32, byte_val);
                if ws != BUS_OK { return ws; }
            }
        }
        BUS_OK
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
    pub const ROWN:       u32 = 0x10; // write refused — ROWN=0, host owns descriptor
    pub const OVERFLOW:   u32 = 0x20; // byte count exhausted mid-transfer

    pub fn ok()          -> Self { Self(Self::OK) }
    pub fn is_ok(self)   -> bool { self.0 == Self::OK }
    pub fn eop(self)     -> bool { self.0 & Self::EOP        != 0 }
    pub fn eox(self)     -> bool { self.0 & Self::EOX        != 0 }
    pub fn irq(self)     -> bool { self.0 & Self::IRQ        != 0 }
    pub fn refused(self) -> bool { self.0 & (Self::NOT_ACTIVE | Self::ROWN | Self::OVERFLOW) != 0 }
}

impl std::ops::BitOr for DmaStatus {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
}
impl std::ops::BitOrAssign for DmaStatus {
    fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
}

pub trait DmaClient: Send + Sync {
    /// Returns (value, status, writeback).
    /// writeback is an optional (addr, val16) memory write to be executed by the caller
    /// under its own lock (e.g. SeeqState) for atomicity with state updates.
    fn read(&self) -> Option<(u32, DmaStatus, Option<(u32, u16)>)>;
    /// Write a value to the DMA channel.
    /// Returns (status, writeback) where writeback is an optional (addr, val16) memory write
    /// to be executed by the caller under its own lock (e.g. SeeqState) for atomicity.
    fn write(&self, val: u32, eop: bool) -> (DmaStatus, Option<(u32, u16)>);
}

/// Asynchronous system-level events sent from devices to the machine event loop.
#[derive(Debug)]
pub enum MachineEvent {
    /// Full system reset (SIN bit in CPUCTRL0).
    HardReset,
    /// Soft power-off (front panel power state = 0).
    PowerOff,
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
    fn get_clock(&self) -> u64;

    fn signal(&self, _signal: Signal) {}

    fn register_commands(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    fn execute_command(&self, _cmd: &str, _args: &[&str], _writer: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        Err("Command not found".to_string())
    }
}

// Compile-time asserts: BUS_* constants must equal the corresponding ExecStatus values.
// If these fail, update BOTH sets of constants to agree.
// See also: mips_exec.rs EXEC_BUS_ERR / EXEC_BUS_VCE / EXEC_COMPLETE / EXEC_RETRY.
const _: () = assert!(BUS_OK   == crate::mips_exec::EXEC_COMPLETE);
const _: () = assert!(BUS_BUSY == crate::mips_exec::EXEC_RETRY);
const _: () = assert!(BUS_ERR  == crate::mips_exec::EXEC_BUS_ERR);
const _: () = assert!(BUS_VCE  == crate::mips_exec::EXEC_BUS_VCE);
