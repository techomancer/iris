//! The GDB stub's view of a CPU (`crate::gdb_stub::CpuDebug`).

use super::*;

// ── CpuDebug adapter ─────────────────────────────────────────────────────────

use std::sync::atomic::AtomicI32;
use crate::gdb_stub::{CpuDebug, StopReason};
use gdbstub_arch::mips::reg::{MipsCoreRegs, MipsCp0Regs, MipsFpuRegs};
use gdbstub_arch::mips::reg::id::MipsRegId;
use gdbstub::target::ext::breakpoints::WatchKind;

/// Tracks the stop reason from the last `run_blocking` call.
/// Stored in an Arc<Mutex<>> so the spawned continue thread can write it.
pub(super) struct StopState {
    reason: parking_lot::Mutex<StopReason>,
}

impl StopState {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self { reason: parking_lot::Mutex::new(StopReason::Interrupted) })
    }
    fn set(&self, r: StopReason) { *self.reason.lock() = r; }
    fn get(&self) -> StopReason { *self.reason.lock() }
}

/// Wraps `Arc<MipsCpu<T,C>>` to implement `CpuDebug`.
pub struct MipsCpuDebugAdapter<T: Tlb + Send + 'static, C: CpuModel + Send + 'static> {
    cpu: Arc<MipsCpu<T, C>>,
    stop_state: Arc<StopState>,
    // Allocator for GDB-owned breakpoint IDs (starts at 10000).
    next_gdb_bp_id: parking_lot::Mutex<usize>,
}

impl<T: Tlb + Send + 'static, C: CpuModel + Send + 'static> MipsCpuDebugAdapter<T, C> {
    pub fn new(cpu: Arc<MipsCpu<T, C>>) -> Arc<Self> {
        Arc::new(Self {
            cpu,
            stop_state: StopState::new(),
            next_gdb_bp_id: parking_lot::Mutex::new(10000),
        })
    }
}

impl<T: Tlb + Send + 'static, C: CpuModel + Send + 'static> CpuDebug
    for MipsCpuDebugAdapter<T, C>
{
    fn stop(&self) {
        self.cpu.stop();
    }

    fn start(&self) {
        self.cpu.start();
    }

    fn is_running(&self) -> bool {
        self.cpu.is_running()
    }

    fn step_one(&self) -> StopReason {
        use crate::cpu::mips_exec::{EXEC_BREAKPOINT, EXEC_RETRY};
        self.cpu.stop(); // ensure no thread is running
        let mut exec = self.cpu.executor.lock();
        exec.last_bp_hit = None;

        // EXEC_RETRY means the instruction did not retire and PC is unmoved, so
        // the same PC must be stepped again — bounded like validate.rs's
        // MAX_RETRIES_PER_INSTRUCTION so a stuck device can't spin here.
        const MAX_STEP_RETRIES: u32 = 100_000;

        // Execute one instruction. If it's a branch, also execute the delay
        // slot, so a single GDB-visible step lands past it rather than
        // stopping mid-delay-slot. `core.in_delay_slot` (set by
        // branch_delay, cleared by handle_exec_complete) is the real signal
        // that a branch/jump was just taken and its delay slot hasn't run
        // yet — status codes carry no PC-related information of their own
        // (every handler sets core.pc itself before returning).
        //eprintln!("GDB: step_one: PC={:#018x}", exec.core.pc);
        let mut status = exec.step_int();
        let mut retries = 0u32;
        while status == EXEC_RETRY && retries < MAX_STEP_RETRIES {
            retries += 1;
            std::hint::spin_loop();
            status = exec.step_int();
        }
        if status == EXEC_RETRY {
            // Still busy after the cap: a device is stuck, not transiently
            // busy. Report a stop rather than a phantom "step completed".
            drop(exec);
            self.stop_state.set(StopReason::Interrupted);
            return StopReason::Interrupted;
        }
        if status == EXEC_BREAKPOINT {
            // Sitting on an active breakpoint: step_int() matches before it
            // dispatches, so without a one-shot skip every later single-step
            // re-detects it (as run_debug_loop's first_step does for `cont`).
            exec.skip_breakpoints = true;
            status = exec.step_int();
        }
        //eprintln!("GDB: step_one: after step status={:#010x} PC={:#018x}", status, exec.core.pc);
        let reason = if status == EXEC_BREAKPOINT {
            drop(exec);
            return StopReason::SwBreakpoint;
        } else if exec.core.in_delay_slot {
            // Branch taken — execute delay slot instruction too.
            let ds_status = exec.step_int();
            if ds_status == EXEC_BREAKPOINT {
                StopReason::SwBreakpoint
            } else {
                StopReason::DoneStep
            }
        } else {
            StopReason::DoneStep
        };
        drop(exec);
        self.stop_state.set(reason);
        reason
    }

    fn run_blocking(&self, count: Option<usize>) -> StopReason {
        // run_debug_loop with wait=true blocks until the CPU stops.
        // We discard the text output (sink writer).
        //eprintln!("GDB: run_blocking: calling run_debug_loop");
        let sink = Box::new(std::io::sink());
        self.cpu.run_debug_loop(count, true, sink);
        //eprintln!("GDB: run_blocking: run_debug_loop returned, is_running={}", self.cpu.is_running());

        // Inspect executor state to determine stop reason.
        let reason = if let Some(exec) = self.cpu.executor.try_lock() {
            if let Some(bp_id) = exec.last_bp_hit {
                let bp = exec.breakpoints.iter().find(|b| b.id == bp_id);
                match bp.map(|b| b.kind) {
                    Some(BpType::Pc) | Some(BpType::VirtFetch) | Some(BpType::PhysFetch) => {
                        StopReason::SwBreakpoint
                    }
                    Some(BpType::VirtRead) | Some(BpType::PhysRead) => {
                        StopReason::Watchpoint { addr: exec.core.pc, kind: WatchKind::Read }
                    }
                    Some(BpType::VirtWrite) | Some(BpType::PhysWrite) => {
                        StopReason::Watchpoint { addr: exec.core.pc, kind: WatchKind::Write }
                    }
                    _ => StopReason::SwBreakpoint,
                }
            } else {
                StopReason::DoneStep
            }
        } else {
            StopReason::DoneStep
        };

        self.stop_state.set(reason);
        reason
    }

    fn read_regs(&self) -> MipsCoreRegs<u64> {
        let exec = self.cpu.executor.lock();
        let core = &exec.core;
        let mut r = [0u64; 32];
        for i in 0..32 { r[i] = core.read_gpr(i as u32); }
        MipsCoreRegs {
            r,
            lo: core.lo,
            hi: core.hi,
            pc: core.pc,
            cp0: MipsCp0Regs {
                status: core.cp0_status as u64,
                badvaddr: core.cp0_badvaddr,
                cause: core.cp0_cause as u64,
            },
            fpu: MipsFpuRegs {
                r: core.fpr,
                fcsr: core.fpu_fcsr as u64,
                fir: core.fpu_fir as u64,
            },
        }
    }

    fn write_regs(&self, regs: &MipsCoreRegs<u64>) {
        let mut exec = self.cpu.executor.lock();
        let core = &mut exec.core;
        for i in 1..32 { core.gpr[i] = regs.r[i]; } // r[0] always 0
        core.lo = regs.lo;
        core.hi = regs.hi;
        core.pc = regs.pc;
        core.cp0_status = regs.cp0.status as u32;
        core.cp0_badvaddr = regs.cp0.badvaddr;
        core.cp0_cause = regs.cp0.cause as u32;
        core.fpr = regs.fpu.r;
        core.fpu_fcsr = regs.fpu.fcsr as u32;
        core.fpu_fir  = regs.fpu.fir as u32;
        // The debugger just wrote cp0_status by direct field write; re-derive.
        exec.resync_privilege_state();
    }

    fn read_reg(&self, id: MipsRegId<u64>) -> Option<u64> {
        let exec = self.cpu.executor.lock();
        let core = &exec.core;
        Some(match id {
            MipsRegId::Gpr(i) if i < 32 => core.read_gpr(i as u32),
            MipsRegId::Lo => core.lo,
            MipsRegId::Hi => core.hi,
            MipsRegId::Pc => core.pc,
            MipsRegId::Status => core.cp0_status as u64,
            MipsRegId::Badvaddr => core.cp0_badvaddr,
            MipsRegId::Cause => core.cp0_cause as u64,
            MipsRegId::Fpr(i) if i < 32 => core.fpr[i as usize],
            MipsRegId::Fcsr => core.fpu_fcsr as u64,
            MipsRegId::Fir => core.fpu_fir as u64,
            _ => return None,
        })
    }

    fn write_reg(&self, id: MipsRegId<u64>, val: u64) {
        let mut exec = self.cpu.executor.lock();
        let core = &mut exec.core;
        match id {
            MipsRegId::Gpr(i) if i > 0 && i < 32 => { core.gpr[i as usize] = val; }
            MipsRegId::Lo => { core.lo = val; }
            MipsRegId::Hi => { core.hi = val; }
            MipsRegId::Pc => { core.pc = val; }
            MipsRegId::Status => { core.cp0_status = val as u32; }
            MipsRegId::Cause => { core.cp0_cause = val as u32; }
            MipsRegId::Fpr(i) if i < 32 => { core.fpr[i as usize] = val; }
            MipsRegId::Fcsr => { core.fpu_fcsr = val as u32; }
            _ => {}
        }
        // `core`'s borrow ends with the match. Unconditional rather than gated on
        // MipsRegId::Status — this is a debugger path, and a resync after a GPR write
        // costs nothing next to never getting one after a Status write.
        exec.resync_privilege_state();
    }

    fn read_mem(&self, addr: u64, buf: &mut [u8]) -> Result<(), ()> {
        if buf.is_empty() { return Ok(()); }
        let mut exec = self.cpu.executor.lock();

        let mut i = 0usize;
        while i < buf.len() {
            // Read a word at the aligned address, then extract the needed bytes.
            let cur_addr = addr.wrapping_add(i as u64);
            let aligned = cur_addr & !3u64;
            let offset = (cur_addr & 3) as usize;
            let remaining = buf.len() - i;
            let avail = 4 - offset;
            let to_copy = remaining.min(avail);

            match exec.debug_read(aligned, 4) {
                Ok(word) => {
                    let word_bytes = (word as u32).to_be_bytes(); // MIPS is big-endian
                    for j in 0..to_copy {
                        buf[i + j] = word_bytes[offset + j];
                    }
                }
                Err(_) => {
                    // Fill with 0 on error rather than aborting the entire read.
                    for j in 0..to_copy { buf[i + j] = 0; }
                }
            }
            i += to_copy;
        }
        Ok(())
    }

    fn write_mem(&self, addr: u64, data: &[u8]) -> Result<(), ()> {
        if data.is_empty() { return Ok(()); }
        let mut exec = self.cpu.executor.lock();

        let mut i = 0usize;
        while i < data.len() {
            let cur_addr = addr.wrapping_add(i as u64);
            let aligned = cur_addr & !3u64;
            let offset = (cur_addr & 3) as usize;
            let remaining = data.len() - i;
            let avail = 4 - offset;
            let to_write = remaining.min(avail);

            if to_write == 4 {
                // Aligned 4-byte write.
                let val = u32::from_be_bytes([data[i], data[i+1], data[i+2], data[i+3]]);
                exec.debug_write(aligned, val as u64, 4, u64::MAX);
            } else {
                // Partial word: read-modify-write.
                let old = match exec.debug_read(aligned, 4) {
                    Ok(v) => v as u32,
                    Err(_) => 0,
                };
                let mut word_bytes = old.to_be_bytes();
                for j in 0..to_write { word_bytes[offset + j] = data[i + j]; }
                let new_val = u32::from_be_bytes(word_bytes);
                exec.debug_write(aligned, new_val as u64, 4, u64::MAX);
            }
            i += to_write;
        }
        Ok(())
    }

    fn add_bp(&self, addr: u64, kind: BpType) -> usize {
        let id = {
            let mut next = self.next_gdb_bp_id.lock();
            let id = *next;
            *next += 1;
            id
        };
        let mut exec = self.cpu.executor.lock();
        exec.add_breakpoint(id, addr, kind);
        //eprintln!("GDB: add_bp id={} addr={:#018x} kind={:?} pc_bp_count={}", id, addr, kind, exec.pc_bp_count);
        id
    }

    fn remove_bp(&self, id: usize) {
        self.cpu.executor.lock().remove_breakpoint(id);
    }

    fn last_stop_reason(&self) -> StopReason {
        self.stop_state.get()
    }
}
