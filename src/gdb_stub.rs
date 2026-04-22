/// GDB Remote Serial Protocol stub for Iris.
///
/// Exposes the MIPS R4400 CPU to a GDB client via TCP.
/// Uses the `gdbstub` crate for protocol handling and `gdbstub_arch` for
/// the MIPS64 register layout.
///
/// Entry point: `start_gdb_server(port, cpu_debug)` — spawns the listener thread.
///
/// ## Execution model
///
/// The GDB stub uses `run_debug_loop` (via `CpuDebug::run_blocking`) for both
/// continue and single-step. This gives us correct breakpoint handling and avoids
/// races with the normal CPU execution thread.
///
/// The event loop is poll-based: `wait_for_stop_reason` polls `is_running()` every
/// millisecond, and returns `IncomingData` when the connection has a pending byte
/// (which the gdbstub machinery then reads to handle Ctrl-C or other packets).

use std::collections::HashMap;
use std::convert::TryInto;
use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use gdbstub::common::Signal;
use gdbstub::conn::{Connection, ConnectionExt};
use gdbstub::stub::run_blocking::{self, BlockingEventLoop};
use gdbstub::stub::{DisconnectReason, GdbStub, GdbStubError, SingleThreadStopReason};
use gdbstub::target::ext::base::single_register_access::SingleRegisterAccess;
use gdbstub::target::ext::base::singlethread::{
    SingleThreadBase, SingleThreadResume, SingleThreadResumeOps, SingleThreadSingleStep,
    SingleThreadSingleStepOps,
};
use gdbstub::target::ext::base::BaseOps;
use gdbstub::target::ext::breakpoints::{
    Breakpoints, BreakpointsOps, HwWatchpoint, HwWatchpointOps, SwBreakpoint, SwBreakpointOps,
    WatchKind,
};
use gdbstub::target::ext::target_description_xml_override::{
    TargetDescriptionXmlOverride, TargetDescriptionXmlOverrideOps,
};
use gdbstub::target::{Target, TargetError, TargetResult};
use gdbstub_arch::mips::reg::id::MipsRegId;
use gdbstub_arch::mips::reg::{MipsCoreRegs, MipsCp0Regs, MipsFpuRegs};
use gdbstub_arch::mips::MipsBreakpointKind;

use crate::mips_exec::BpType;

// ── Custom MIPS64 arch — 72-register g packet ────────────────────────────────
//
// With mips:isa64 and org.gnu.gdb.mips.linux (reg 72 "fp"), GDB computes the
// g-packet size from regs 0-71 only = 72 × 8 = 576 bytes. reg 72 (fp) is not
// counted. gdbstub_arch::mips::Mips64 uses MipsCoreRegs<u64> which is 72 regs
// by default, so we match it exactly here.

/// Register set that matches mips64-linux-gdb's g packet for mips:isa64: 72 × 8 = 576 bytes.
/// gdb_serialize emits 72 registers (0-71); reg 72 (fp) is not in the g-packet.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IrisMipsRegs(pub MipsCoreRegs<u64>);

impl gdbstub::arch::Registers for IrisMipsRegs {
    type ProgramCounter = u64;

    fn pc(&self) -> u64 {
        self.0.pc
    }

    fn gdb_serialize(&self, mut write_byte: impl FnMut(Option<u8>)) {
        // Serialize all fields of MipsCoreRegs<u64> except fir.
        // Order must match gdbstub_arch's MipsCoreRegs gdb_serialize exactly,
        // minus the final fir field.
        let write_u64 = |v: u64, w: &mut dyn FnMut(Option<u8>)| {
            for b in v.to_be_bytes() { w(Some(b)); }
        };
        // GDB g-packet (with set mips abi n64): 72 × 8 = 576 bytes.
        //   0-31: GPRs
        //   32: status, 33: lo, 34: hi, 35: badvaddr, 36: cause, 37: pc
        //   38-69: FPRs, 70: fcsr, 71: fir
        // reg 72 (fp, org.gnu.gdb.mips.linux) is NOT in the g-packet.
        let r = &self.0;
        for i in 0..32 { write_u64(r.r[i], &mut write_byte); }
        write_u64(r.cp0.status,   &mut write_byte); // 32
        write_u64(r.lo,           &mut write_byte); // 33
        write_u64(r.hi,           &mut write_byte); // 34
        write_u64(r.cp0.badvaddr, &mut write_byte); // 35
        write_u64(r.cp0.cause,    &mut write_byte); // 36
        write_u64(r.pc,           &mut write_byte); // 37
        for i in 0..32 { write_u64(r.fpu.r[i], &mut write_byte); }
        write_u64(r.fpu.fcsr,     &mut write_byte); // 70
        write_u64(r.fpu.fir,      &mut write_byte); // 71
        // reg 72 (fp/restart) is declared in org.gnu.gdb.mips.linux but GDB does NOT
        // include it in the g-packet size calculation — so we stop at 72 regs = 576 bytes.
    }

    fn gdb_deserialize(&mut self, bytes: &[u8]) -> Result<(), ()> {
        if bytes.len() < 38 * 8 { return Err(()); } // at minimum need GPRs + special regs
        let mut chunks = bytes.chunks_exact(8);
        let mut next = || -> Result<u64, ()> {
            Ok(u64::from_be_bytes(chunks.next().ok_or(())?.try_into().unwrap()))
        };
        let r = &mut self.0;
        for i in 0..32 { r.r[i] = next()?; }
        r.cp0.status   = next()?; // 32
        r.lo           = next()?; // 33
        r.hi           = next()?; // 34
        r.cp0.badvaddr = next()?; // 35
        r.cp0.cause    = next()?; // 36
        r.pc           = next()?; // 37
        for i in 0..32 { r.fpu.r[i] = next()?; }
        r.fpu.fcsr     = next()?; // 70
        r.fpu.fir      = next()?; // 71
        Ok(())
    }
}

/// Custom RegId that stops at regnum 70 (fcsr) so gdbstub computes g-packet
/// size as 71 × 8 = 568 bytes, matching mips64-linux-gdb's expectation.
#[derive(Debug, Clone, Copy)]
pub struct IrisMipsRegId(pub MipsRegId<u64>);

impl gdbstub::arch::RegId for IrisMipsRegId {
    fn from_raw_id(id: usize) -> Option<(Self, Option<core::num::NonZeroUsize>)> {
        let sz8 = Some(core::num::NonZeroUsize::new(8)?);
        // QEMU MIPS64 g packet numbering (gdb_num_core_regs=73):
        //   0-31: GPRs, 32: status, 33: lo, 34: hi, 35: badvaddr, 36: cause, 37: pc
        //   38-69: FPRs, 70: fcsr, 71: fir, 72: fp (unused)
        let reg = match id {
            0..=31  => MipsRegId::Gpr(id as u8),
            32      => MipsRegId::Status,
            33      => MipsRegId::Lo,
            34      => MipsRegId::Hi,
            35      => MipsRegId::Badvaddr,
            36      => MipsRegId::Cause,
            37      => MipsRegId::Pc,
            38..=69 => MipsRegId::Fpr((id as u8) - 38),
            70      => MipsRegId::Fcsr,
            71      => MipsRegId::Fir,
            // 72 = fp (org.gnu.gdb.mips.linux) — not in g-packet, accessible via p/P only
            72      => MipsRegId::Gpr(0), // alias to r0 (zero) for p/P reads
            _       => return None,
        };
        Some((IrisMipsRegId(reg), sz8))
    }
}

/// Custom Arch: Mips64 with 71-register g packet (fir excluded).
pub enum IrisMips64 {}

impl gdbstub::arch::Arch for IrisMips64 {
    type Usize = u64;
    type Registers = IrisMipsRegs;
    type RegId = IrisMipsRegId;
    type BreakpointKind = MipsBreakpointKind;

    fn target_description_xml() -> Option<&'static str> {
        None // We provide it via TargetDescriptionXmlOverride
    }
}

// ── Stop reason ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    SwBreakpoint,
    HwBreakpoint,
    Watchpoint { addr: u64, kind: WatchKind },
    DoneStep,
    Interrupted,
}

impl StopReason {
    fn to_gdb(&self) -> SingleThreadStopReason<u64> {
        match self {
            StopReason::SwBreakpoint | StopReason::HwBreakpoint => {
                SingleThreadStopReason::SwBreak(())
            }
            StopReason::Watchpoint { addr, kind } => SingleThreadStopReason::Watch {
                tid: (),
                kind: *kind,
                addr: *addr,
            },
            StopReason::DoneStep => SingleThreadStopReason::DoneStep,
            StopReason::Interrupted => SingleThreadStopReason::Signal(Signal::SIGINT),
        }
    }
}

// ── CpuDebug trait — erases MipsCpu<T,C> generics ────────────────────────────

/// Type-erased interface to MipsCpu for the GDB stub.
pub trait CpuDebug: Send + Sync {
    /// Stop the CPU execution thread and wait for it to exit.
    fn stop(&self);
    /// Start the normal (fast) CPU execution thread.
    fn start(&self);
    /// True if a CPU thread (debug or normal) is running.
    fn is_running(&self) -> bool;

    /// Execute exactly one logical MIPS instruction (branch + delay slot as a unit),
    /// blocking until done. Used for GDB single-step (si).
    fn step_one(&self) -> StopReason;

    /// Execute up to `count` instructions (None = forever), blocking until stopped.
    /// This uses run_debug_loop internally so breakpoints work correctly.
    fn run_blocking(&self, count: Option<usize>) -> StopReason;

    /// Read all core registers.
    fn read_regs(&self) -> MipsCoreRegs<u64>;
    /// Write all core registers.
    fn write_regs(&self, regs: &MipsCoreRegs<u64>);
    /// Read one register by GDB register ID. Returns None if unknown.
    fn read_reg(&self, id: MipsRegId<u64>) -> Option<u64>;
    /// Write one register by GDB register ID.
    fn write_reg(&self, id: MipsRegId<u64>, val: u64);

    /// Read `buf.len()` bytes starting at virtual address `addr`.
    /// Handles unaligned accesses by reading aligned words and extracting bytes.
    fn read_mem(&self, addr: u64, buf: &mut [u8]) -> Result<(), ()>;
    /// Write `data.len()` bytes to virtual address `addr`.
    fn write_mem(&self, addr: u64, data: &[u8]) -> Result<(), ()>;

    /// Add a breakpoint of `kind` at `addr`. Returns the internal BP id.
    fn add_bp(&self, addr: u64, kind: BpType) -> usize;
    /// Remove a breakpoint by internal id.
    fn remove_bp(&self, id: usize);

    /// Return the stop reason from the last `run_blocking` call.
    fn last_stop_reason(&self) -> StopReason;
}

// ── CpuDebug adapter impl lives in mips_exec.rs (see MipsCpuDebugAdapter) ───
// The adapter is implemented in mips_exec.rs to avoid importing generic bounds here.

// ── Breakpoint map ────────────────────────────────────────────────────────────

/// Tracks GDB-owned breakpoints (keyed by addr + BpType) → internal bp id.
/// Avoids conflicts with monitor-side breakpoints.
struct BpMap {
    map: HashMap<(u64, u8), usize>,
}

impl BpMap {
    fn new() -> Self {
        Self { map: HashMap::new() }
    }

    fn add(&mut self, addr: u64, kind: BpType, cpu: &dyn CpuDebug) -> bool {
        let key = (addr, kind as u8);
        if self.map.contains_key(&key) {
            return true; // already present
        }
        let id = cpu.add_bp(addr, kind);
        self.map.insert(key, id);
        true
    }

    fn remove(&mut self, addr: u64, kind: BpType, cpu: &dyn CpuDebug) -> bool {
        let key = (addr, kind as u8);
        if let Some(id) = self.map.remove(&key) {
            cpu.remove_bp(id);
            true
        } else {
            false
        }
    }

    fn remove_all(&mut self, cpu: &dyn CpuDebug) {
        for (_, id) in self.map.drain() {
            cpu.remove_bp(id);
        }
    }
}

// ── IrisTarget ────────────────────────────────────────────────────────────────

/// Execution state from the GDB perspective.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExecState {
    Stopped,
    /// CPU is running; event loop should poll is_running().
    Running,
    /// Single-step completed synchronously; stop reason ready.
    StepDone,
}

pub struct IrisTarget {
    pub cpu: Arc<dyn CpuDebug>,
    bp_map: BpMap,
    exec_state: ExecState,
    last_stop: StopReason,
}

impl IrisTarget {
    pub fn new(cpu: Arc<dyn CpuDebug>) -> Self {
        Self {
            cpu,
            bp_map: BpMap::new(),
            exec_state: ExecState::Stopped,
            last_stop: StopReason::Interrupted,
        }
    }
}

impl Drop for IrisTarget {
    fn drop(&mut self) {
        // Remove all GDB-owned breakpoints on disconnect.
        self.bp_map.remove_all(self.cpu.as_ref());
    }
}

// ── Target impl ──────────────────────────────────────────────────────────────

impl Target for IrisTarget {
    type Arch = IrisMips64;
    type Error = &'static str;

    #[inline(always)]
    fn base_ops(&mut self) -> BaseOps<Self::Arch, Self::Error> {
        BaseOps::SingleThread(self)
    }

    #[inline(always)]
    fn support_breakpoints(&mut self) -> Option<BreakpointsOps<Self>> {
        Some(self)
    }

    #[inline(always)]
    fn support_target_description_xml_override(
        &mut self,
    ) -> Option<TargetDescriptionXmlOverrideOps<'_, Self>> {
        Some(self)
    }
}

// ── SingleThreadBase ──────────────────────────────────────────────────────────

impl SingleThreadBase for IrisTarget {
    fn read_registers(&mut self, regs: &mut IrisMipsRegs) -> TargetResult<(), Self> {
        regs.0 = self.cpu.read_regs();
        //eprintln!("GDB: read_registers PC={:#018x}", regs.0.pc);
        Ok(())
    }

    fn write_registers(&mut self, regs: &IrisMipsRegs) -> TargetResult<(), Self> {
        self.cpu.write_regs(&regs.0);
        Ok(())
    }

    #[inline(always)]
    fn support_single_register_access(
        &mut self,
    ) -> Option<gdbstub::target::ext::base::single_register_access::SingleRegisterAccessOps<'_, (), Self>>
    {
        Some(self)
    }

    fn read_addrs(&mut self, start_addr: u64, data: &mut [u8]) -> TargetResult<usize, Self> {
        if self.cpu.read_mem(mips_sign_extend(start_addr), data).is_err() {
            data.fill(0);
        }
        Ok(data.len())
    }

    fn write_addrs(&mut self, start_addr: u64, data: &[u8]) -> TargetResult<(), Self> {
        self.cpu
            .write_mem(mips_sign_extend(start_addr), data)
            .map_err(|_| TargetError::NonFatal)
    }

    #[inline(always)]
    fn support_resume(&mut self) -> Option<SingleThreadResumeOps<'_, Self>> {
        Some(self)
    }
}

// ── SingleRegisterAccess ──────────────────────────────────────────────────────

impl SingleRegisterAccess<()> for IrisTarget {
    fn read_register(
        &mut self,
        _tid: (),
        reg_id: IrisMipsRegId,
        buf: &mut [u8],
    ) -> TargetResult<usize, Self> {
        let val = self.cpu.read_reg(reg_id.0).ok_or(TargetError::NonFatal)?;
        let bytes = val.to_be_bytes();
        let n = buf.len().min(8);
        buf[..n].copy_from_slice(&bytes[..n]);
        Ok(n)
    }

    fn write_register(
        &mut self,
        _tid: (),
        reg_id: IrisMipsRegId,
        val: &[u8],
    ) -> TargetResult<(), Self> {
        let mut bytes = [0u8; 8];
        let n = val.len().min(8);
        // val is big-endian from GDB; right-align into 8-byte buffer
        let offset = 8 - n;
        bytes[offset..].copy_from_slice(&val[..n]);
        self.cpu.write_reg(reg_id.0, u64::from_be_bytes(bytes));
        Ok(())
    }
}

// ── SingleThreadResume ────────────────────────────────────────────────────────

impl SingleThreadResume for IrisTarget {
    fn resume(&mut self, _signal: Option<Signal>) -> Result<(), Self::Error> {
        //eprintln!("GDB: resume() called (continue)");
        // Mark as running; the event loop (IrisEventLoop::wait_for_stop_reason)
        // will spawn the debug thread and poll for completion.
        self.exec_state = ExecState::Running;
        Ok(())
    }

    #[inline(always)]
    fn support_single_step(&mut self) -> Option<SingleThreadSingleStepOps<'_, Self>> {
        Some(self)
    }
}

impl SingleThreadSingleStep for IrisTarget {
    fn step(&mut self, _signal: Option<Signal>) -> Result<(), Self::Error> {
        //eprintln!("GDB: step() called");
        // Execute one logical MIPS instruction (branch + delay slot as a unit).
        let reason = self.cpu.step_one();
        //eprintln!("GDB: step() done, reason={:?}, exec_state->StepDone", reason);
        self.last_stop = reason;
        self.exec_state = ExecState::StepDone;
        Ok(())
    }
}

// ── Breakpoints ───────────────────────────────────────────────────────────────

impl Breakpoints for IrisTarget {
    #[inline(always)]
    fn support_sw_breakpoint(&mut self) -> Option<SwBreakpointOps<'_, Self>> {
        Some(self)
    }

    #[inline(always)]
    fn support_hw_watchpoint(&mut self) -> Option<HwWatchpointOps<'_, Self>> {
        Some(self)
    }
}

/// Sign-extend a MIPS virtual address: if bits[63:32] are zero and bit 31 is set,
/// extend to 0xffffffff_xxxxxxxx. GDB sends 32-bit MIPS addresses without the upper half.
fn mips_sign_extend(addr: u64) -> u64 {
    if addr >> 32 == 0 && addr & 0x8000_0000 != 0 {
        addr | 0xffff_ffff_0000_0000
    } else {
        addr
    }
}

impl SwBreakpoint for IrisTarget {
    fn add_sw_breakpoint(
        &mut self,
        addr: u64,
        _kind: MipsBreakpointKind,
    ) -> TargetResult<bool, Self> {
        Ok(self.bp_map.add(mips_sign_extend(addr), BpType::Pc, self.cpu.as_ref()))
    }

    fn remove_sw_breakpoint(
        &mut self,
        addr: u64,
        _kind: MipsBreakpointKind,
    ) -> TargetResult<bool, Self> {
        Ok(self.bp_map.remove(mips_sign_extend(addr), BpType::Pc, self.cpu.as_ref()))
    }
}

impl HwWatchpoint for IrisTarget {
    fn add_hw_watchpoint(
        &mut self,
        addr: u64,
        _len: u64,
        kind: WatchKind,
    ) -> TargetResult<bool, Self> {
        let bp_kind = match kind {
            WatchKind::Write => BpType::VirtWrite,
            WatchKind::Read => BpType::VirtRead,
            WatchKind::ReadWrite => BpType::VirtWrite, // approximate
        };
        Ok(self.bp_map.add(mips_sign_extend(addr), bp_kind, self.cpu.as_ref()))
    }

    fn remove_hw_watchpoint(
        &mut self,
        addr: u64,
        _len: u64,
        kind: WatchKind,
    ) -> TargetResult<bool, Self> {
        let bp_kind = match kind {
            WatchKind::Write => BpType::VirtWrite,
            WatchKind::Read => BpType::VirtRead,
            WatchKind::ReadWrite => BpType::VirtWrite,
        };
        Ok(self.bp_map.remove(mips_sign_extend(addr), bp_kind, self.cpu.as_ref()))
    }
}

// ── TargetDescriptionXmlOverride ──────────────────────────────────────────────

/// MIPS64 target description XML.
///
/// Declares 64-bit GPRs, CP0 regs, and FPU regs matching the gdbstub_arch
/// MipsCoreRegs<u64> layout (72 registers total).
/// NOTE: GDB still requires `set mips abi n64` to display 64-bit GPRs correctly.
/// The XML alone is not sufficient — mips:isa64 in <architecture> does NOT set ABI.
const MIPS64_TARGET_XML: &str = concat!(
    r#"<?xml version="1.0"?>"#,
    r#"<!DOCTYPE target SYSTEM "gdb-target.dtd">"#,
    r#"<target version="1.0">"#,
    // No <architecture> tag — GDB must be told the ABI externally via:
    //   set mips abi n64
    // The org.gnu.gdb.mips.linux feature below suppresses GDB's stack-frame
    // heuristic (which otherwise sends a spurious 0x03 before single-step packets).
    r#"<feature name="org.gnu.gdb.mips.cpu">"#,
    r#"<reg name="r0" bitsize="64" regnum="0"/>"#,
    r#"<reg name="r1" bitsize="64"/>"#,
    r#"<reg name="r2" bitsize="64"/>"#,
    r#"<reg name="r3" bitsize="64"/>"#,
    r#"<reg name="r4" bitsize="64"/>"#,
    r#"<reg name="r5" bitsize="64"/>"#,
    r#"<reg name="r6" bitsize="64"/>"#,
    r#"<reg name="r7" bitsize="64"/>"#,
    r#"<reg name="r8" bitsize="64"/>"#,
    r#"<reg name="r9" bitsize="64"/>"#,
    r#"<reg name="r10" bitsize="64"/>"#,
    r#"<reg name="r11" bitsize="64"/>"#,
    r#"<reg name="r12" bitsize="64"/>"#,
    r#"<reg name="r13" bitsize="64"/>"#,
    r#"<reg name="r14" bitsize="64"/>"#,
    r#"<reg name="r15" bitsize="64"/>"#,
    r#"<reg name="r16" bitsize="64"/>"#,
    r#"<reg name="r17" bitsize="64"/>"#,
    r#"<reg name="r18" bitsize="64"/>"#,
    r#"<reg name="r19" bitsize="64"/>"#,
    r#"<reg name="r20" bitsize="64"/>"#,
    r#"<reg name="r21" bitsize="64"/>"#,
    r#"<reg name="r22" bitsize="64"/>"#,
    r#"<reg name="r23" bitsize="64"/>"#,
    r#"<reg name="r24" bitsize="64"/>"#,
    r#"<reg name="r25" bitsize="64"/>"#,
    r#"<reg name="r26" bitsize="64"/>"#,
    r#"<reg name="r27" bitsize="64"/>"#,
    r#"<reg name="r28" bitsize="64"/>"#,
    r#"<reg name="r29" bitsize="64"/>"#,
    r#"<reg name="r30" bitsize="64"/>"#,
    r#"<reg name="r31" bitsize="64"/>"#,
    r#"<reg name="lo" bitsize="64" regnum="33"/>"#,
    r#"<reg name="hi" bitsize="64" regnum="34"/>"#,
    r#"<reg name="pc" bitsize="64" regnum="37" type="code_ptr"/>"#,
    r#"</feature>"#,
    r#"<feature name="org.gnu.gdb.mips.cp0">"#,
    r#"<reg name="status" bitsize="64" regnum="32"/>"#,
    r#"<reg name="badvaddr" bitsize="64" regnum="35" type="data_ptr"/>"#,
    r#"<reg name="cause" bitsize="64" regnum="36"/>"#,
    r#"</feature>"#,
    r#"<feature name="org.gnu.gdb.mips.fpu">"#,
    r#"<reg name="f0" bitsize="64" type="ieee_double" regnum="38"/>"#,
    r#"<reg name="f1" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f2" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f3" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f4" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f5" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f6" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f7" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f8" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f9" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f10" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f11" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f12" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f13" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f14" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f15" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f16" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f17" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f18" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f19" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f20" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f21" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f22" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f23" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f24" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f25" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f26" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f27" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f28" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f29" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f30" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="f31" bitsize="64" type="ieee_double"/>"#,
    r#"<reg name="fcsr" bitsize="64" regnum="70" group="float"/>"#,
    r#"<reg name="fir" bitsize="64" regnum="71" group="float"/>"#,
    r#"</feature>"#,
    // org.gnu.gdb.mips.linux: suppresses the MIPS stack-frame heuristic backward
    // scan that otherwise sends a spurious 0x03 before single-step packets.
    // reg 72 "fp" is the last reg in QEMU's gdb_num_core_regs=73 layout.
    r#"<feature name="org.gnu.gdb.mips.linux">"#,
    r#"<reg name="fp" bitsize="64" regnum="72"/>"#,
    r#"</feature>"#,
    r#"</target>"#,
);

impl TargetDescriptionXmlOverride for IrisTarget {
    fn target_description_xml(
        &self,
        annex: &[u8],
        offset: u64,
        length: usize,
        buf: &mut [u8],
    ) -> TargetResult<usize, Self> {
        if annex != b"target.xml" {
            return Ok(0);
        }
        let xml = MIPS64_TARGET_XML.as_bytes();
        let start = offset as usize;
        if start >= xml.len() {
            return Ok(0);
        }
        let end = (start + length).min(xml.len());
        let n = end - start;
        buf[..n].copy_from_slice(&xml[start..end]);
        Ok(n)
    }
}

// ── Event loop ────────────────────────────────────────────────────────────────

struct IrisEventLoop;

impl BlockingEventLoop for IrisEventLoop {
    type Target = IrisTarget;
    type Connection = LoggingStream;
    type StopReason = SingleThreadStopReason<u64>;

    fn wait_for_stop_reason(
        target: &mut IrisTarget,
        conn: &mut Self::Connection,
    ) -> Result<
        run_blocking::Event<Self::StopReason>,
        run_blocking::WaitForStopReasonError<
            <Self::Target as Target>::Error,
            <Self::Connection as Connection>::Error,
        >,
    > {
        use run_blocking::Event;
        //eprintln!("GDB: wait_for_stop_reason entered, exec_state={}", target.exec_state as u8);

        // If a single-step just completed synchronously, report it immediately.
        if target.exec_state == ExecState::StepDone {
            //eprintln!("GDB: wait_for_stop_reason: StepDone -> TargetStopped({:?})", target.last_stop);
            target.exec_state = ExecState::Stopped;
            return Ok(Event::TargetStopped(target.last_stop.to_gdb()));
        }

        // Launch the CPU in a background thread so we can poll for Ctrl-C.
        if target.exec_state == ExecState::Running {
            //eprintln!("GDB: launching gdb-continue thread");
            let cpu = target.cpu.clone();
            thread::Builder::new()
                .name("gdb-continue".to_string())
                .spawn(move || { cpu.run_blocking(None); })
                .map_err(|_| run_blocking::WaitForStopReasonError::Target("spawn failed"))?;

            // Wait until run_debug_loop has set running=true before we start polling.
            // Without this we'd see is_running()==false immediately (thread not started yet).
            for _ in 0..200 {
                if target.cpu.is_running() { break; }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        // Poll: wait for CPU to stop or Ctrl-C from GDB.
        loop {
            if !target.cpu.is_running() {
                target.exec_state = ExecState::Stopped;
                let reason = target.cpu.last_stop_reason();
                target.last_stop = reason;
                //eprintln!("GDB: continue stopped, reason={:?}", reason);
                return Ok(Event::TargetStopped(reason.to_gdb()));
            }

            match conn.peek() {
                Ok(Some(byte)) => {
                    //eprintln!("GDB: incoming 0x{:02x} during run", byte);
                    // Only 0x03 (Ctrl-C) is valid while running — pass it to gdbstub
                    // which will call on_interrupt. Any other byte (e.g. stray '+' ACK)
                    // is ignored here; gdbstub shouldn't send packets mid-execution.
                    if byte == 0x03 {
                        return Ok(Event::IncomingData(byte));
                    }
                    // Consume and discard non-interrupt bytes.
                    let _ = conn.read();
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(1)),
                Err(e) => return Err(run_blocking::WaitForStopReasonError::Connection(e)),
            }
        }
    }

    fn on_interrupt(
        target: &mut IrisTarget,
    ) -> Result<Option<Self::StopReason>, <Self::Target as Target>::Error> {
        //eprintln!("GDB: on_interrupt called, exec_state={:?}, is_running={}",
        //    target.exec_state as u8, target.cpu.is_running());
        // Spurious 0x03 from GDB's stack-frame heuristic. It fires before the step
        // packet is sent, while we are either Stopped (idle, from_idle=true) or
        // Polling (inside wait_for_stop_reason, from_idle=false but CPU not running).
        // In both cases the CPU is stopped and the interrupt is noise.
        // Return DoneStep so gdbstub sends a stop reply and GDB can then send si again.
        if !target.cpu.is_running() {
            //eprintln!("GDB: on_interrupt: CPU not running (spurious), returning DoneStep");
            target.exec_state = ExecState::Stopped;
            return Ok(Some(SingleThreadStopReason::DoneStep));
        }
        //eprintln!("GDB: on_interrupt: real interrupt, stopping CPU");
        target.cpu.stop();
        target.exec_state = ExecState::Stopped;
        target.last_stop = StopReason::Interrupted;
        Ok(Some(SingleThreadStopReason::Signal(Signal::SIGINT)))
    }
}

// ── Logging stream wrapper ────────────────────────────────────────────────────

struct LoggingStream(TcpStream);

impl gdbstub::conn::Connection for LoggingStream {
    type Error = io::Error;
    fn write(&mut self, byte: u8) -> Result<(), Self::Error> {
        use io::Write;
        Write::write_all(&mut self.0, &[byte])
    }
    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        use io::Write;
        Write::write_all(&mut self.0, buf)
    }
    fn flush(&mut self) -> Result<(), Self::Error> {
        use io::Write;
        Write::flush(&mut self.0)
    }
    fn on_session_start(&mut self) -> Result<(), Self::Error> {
        self.0.set_nodelay(true)
    }
}

impl gdbstub::conn::ConnectionExt for LoggingStream {
    fn read(&mut self) -> Result<u8, Self::Error> {
        use io::Read;
        self.0.set_nonblocking(false)?;
        let mut buf = [0u8; 1];
        Read::read_exact(&mut self.0, &mut buf)?;
        //eprintln!("GDB <<< 0x{:02x} {:?}", buf[0], char::from(buf[0]));
        Ok(buf[0])
    }
    fn peek(&mut self) -> Result<Option<u8>, Self::Error> {
        self.0.set_nonblocking(true)?;
        let mut buf = [0u8; 1];
        match TcpStream::peek(&mut self.0, &mut buf) {
            Ok(_) => Ok(Some(buf[0])),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Spawn the GDB stub listener thread.
///
/// Accepts one connection at a time. Breakpoints set by GDB are automatically
/// removed when the client disconnects.
pub fn start_gdb_server(port: u16, cpu: Arc<dyn CpuDebug>) {
    thread::Builder::new()
        .name("gdb-listener".to_string())
        .spawn(move || {
            let addr = format!("127.0.0.1:{}", port);
            let listener = match TcpListener::bind(&addr) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("GDB stub: failed to bind {}: {}", addr, e);
                    return;
                }
            };
            eprintln!(
                "GDB stub: listening on {} (GDB: target remote {})",
                addr, addr
            );

            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let peer = stream
                            .peer_addr()
                            .map(|a| a.to_string())
                            .unwrap_or_default();
                        eprintln!("GDB stub: connected from {}", peer);

                        // Stop the CPU before starting a GDB session so we start
                        // from a known-halted state. (In developer mode it's already stopped.)
                        cpu.stop();

                        let mut target = IrisTarget::new(cpu.clone());
                        let stream = LoggingStream(stream);
                        let gdb = GdbStub::new(stream);

                        match gdb.run_blocking::<IrisEventLoop>(&mut target) {
                            Ok(DisconnectReason::Disconnect) => {
                                eprintln!("GDB stub: client disconnected");
                            }
                            Ok(DisconnectReason::TargetExited(code)) => {
                                eprintln!("GDB stub: target exited ({})", code);
                            }
                            Ok(DisconnectReason::TargetTerminated(sig)) => {
                                eprintln!("GDB stub: target terminated ({:?})", sig);
                            }
                            Ok(DisconnectReason::Kill) => {
                                eprintln!("GDB stub: killed by GDB");
                            }
                            Err(e) if e.is_target_error() => {
                                eprintln!("GDB stub: target error: {}", e.into_target_error().unwrap());
                            }
                            Err(e) => {
                                eprintln!("GDB stub: error: {:?}", e);
                            }
                        }
                        // IrisTarget::drop() removes all GDB breakpoints.
                    }
                    Err(e) => {
                        eprintln!("GDB stub: accept error: {}", e);
                    }
                }
            }
        })
        .expect("failed to spawn gdb-listener thread");
}
