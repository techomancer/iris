//! The CPU's monitor commands and what they lean on: formatting helpers, the
//! debug run loop, image loading, the backtrace walker. Runs on the monitor
//! thread, against a stopped core (`MipsCpu::pause`).

use super::*;

// Helper to format PC with symbol
pub(super) fn format_pc_symbol(pc: u64, symbols: &SymbolTable) -> String {
    let mut lookup = symbols.lookup(pc);
    let mut effective_pc = pc;
    
    // If not found and address is KSEG1 (0xFFFFFFFF_A...), try KSEG0 (0xFFFFFFFF_8...)
    if lookup.is_none() && (pc >> 32) == 0xFFFFFFFF && ((pc >> 29) & 0x7) == 5 {
        let kseg0_pc = (pc & 0x1FFFFFFF) | 0xFFFF_FFFF_8000_0000;
        if let Some(res) = symbols.lookup(kseg0_pc) {
            lookup = Some(res);
            effective_pc = kseg0_pc;
        }
    }

    if let Some((sym_addr, name)) = lookup {
        let offset = effective_pc - sym_addr;
        if offset > 256 {
            return String::new();
        }
        if offset == 0 {
            return format!(" <{}>", name);
        } else {
            return format!(" <{}+0x{:x}>", name, offset);
        }
    }
    String::new()
}

// Helper to parse command arguments (registers or values)
pub(super) fn parse_reg_name(arg: &str) -> Option<usize> {
    match exp::parse_reg_target(arg) {
        Some(RegTarget::Gpr(n)) => Some(n as usize),
        _ => None,
    }
}

/// Copy `data` to `vaddr` and zero-fill out to `memsz`. Writes go through the
/// CPU's virtual-address path, so KSEG0/KSEG1 mapping, the MC address mask and
/// bank remapping all apply — never poke `Memory` behind the bus.
pub(super) fn load_range<T: Tlb, C: CpuModel>(
    exec: &mut MipsExecutor<T, C>,
    vaddr: u64,
    data: &[u8],
    memsz: u64,
) -> Result<(), String> {
    if memsz == 0 {
        return Ok(());
    }
    probe_mapped(exec, vaddr)?;
    probe_mapped(exec, vaddr.wrapping_add(memsz - 1))?;

    let mut off = 0u64;
    while off < memsz {
        let addr = vaddr.wrapping_add(off);
        let byte = |k: u64| *data.get((off + k) as usize).unwrap_or(&0) as u64;
        let (size, val) = if memsz - off >= 4 && addr % 4 == 0 {
            (4usize, (byte(0) << 24) | (byte(1) << 16) | (byte(2) << 8) | byte(3))
        } else {
            (1usize, byte(0))
        };
        let status = exec.debug_write(addr, val, size, full_mask_for_usize(size));
        if status != EXEC_COMPLETE {
            return Err(format!(
                "write failed at {:#018x} (status {:#010x}) — is RAM mapped and addressable there?",
                addr, status
            ));
        }
        off += size as u64;
    }
    invalidate_loaded_range(exec, vaddr, memsz);
    Ok(())
}

/// A physical address typed at the monitor. `parse_cpu_arg` sign-extends a
/// 32-bit value, as it should for a virtual address, which would put a physical
/// address from 2 GB to 4 GB out of reach.
#[cfg(feature = "jitv2")]
fn monitor_phys(arg: u64) -> u64 {
    if arg >> 32 == 0xFFFF_FFFF { arg & 0xFFFF_FFFF } else { arg }
}

/// Unmapped physical addresses are backed by `UnmappedRam`, which accepts every
/// write and reads back zero — so a write into the void is completely silent.
/// Probe with a non-zero pattern before committing an image.
///
/// The probe goes to the bus, not through `debug_write`: KSEG0 is cacheable, so
/// a cached write is absorbed by L1D and reads back fine with nothing behind it.
/// The original word is put back, so this leaves no trace.
pub(super) fn probe_mapped<T: Tlb, C: CpuModel>(exec: &mut MipsExecutor<T, C>, vaddr: u64) -> Result<(), String> {
    let tr = exec.debug_translate(vaddr);
    if tr.is_exception() {
        return Err(format!("{:#018x} does not translate (status {:#010x})", vaddr, tr.status));
    }
    let phys = tr.phys;
    let unmapped = || {
        Err(format!(
            "nothing is mapped at {:#018x} (physical {:#010x}) — writes there go nowhere. \
             The PROM maps RAM during POST: boot to the PROM prompt first, or use --load-elf, \
             which maps the banks itself",
            vaddr, phys
        ))
    };

    const PATTERN: u32 = 0x5A5A_A5A5;
    if phys % 4 == 0 {
        let saved = exec.sysad.read32(phys);
        if !saved.is_ok() || exec.sysad.write32(phys, PATTERN) != BUS_OK {
            return unmapped();
        }
        let got = exec.sysad.read32(phys);
        exec.sysad.write32(phys, saved.data);
        if !got.is_ok() || got.data != PATTERN {
            return unmapped();
        }
    } else {
        let saved = exec.sysad.read8(phys);
        if !saved.is_ok() || exec.sysad.write8(phys, PATTERN as u8) != BUS_OK {
            return unmapped();
        }
        let got = exec.sysad.read8(phys);
        exec.sysad.write8(phys, saved.data);
        if !got.is_ok() || got.data != PATTERN as u8 {
            return unmapped();
        }
    }
    Ok(())
}

/// Drop stale cache copies of a just-loaded range: writeback+invalidate L1D and
/// L2 so the new bytes reach memory, then invalidate L1I so a stale decoded
/// instruction can never execute. The v1 JIT needs no explicit flush (its
/// CodeCache dies with the CPU thread, and loading requires a stopped CPU);
/// jitv2 self-invalidates from the per-page generation counter the writes bump.
pub(super) fn invalidate_loaded_range<T: Tlb, C: CpuModel>(exec: &mut MipsExecutor<T, C>, vaddr: u64, len: u64) {
    let (_, iline) = exec.cache.get_config(CACH_PI);
    let (_, dline) = exec.cache.get_config(CACH_PD);
    let step = (iline.min(dline).max(4)) as u64;
    let mut addr = vaddr & !(step - 1);
    let end = vaddr.wrapping_add(len);
    while addr < end {
        let tr = exec.debug_translate(addr);
        if !tr.is_exception() {
            let phys = tr.phys;
            exec.cache.cache_op(C_HWBINV | CACH_PD, addr, phys);
            exec.cache.cache_op(C_HWBINV | CACH_SD, addr, phys);
            exec.cache.cache_op(C_HINV | CACH_PI, addr, phys);
        }
        addr = addr.wrapping_add(step);
    }
}

pub(super) fn parse_cpu_arg(arg: &str, core: &MipsCore, symbols: Option<&SymbolTable>) -> Result<u64, String> {
    exp::parse_and_eval(arg, core, symbols)
}


pub(super) fn decode_status(val: u32) -> String {
    let mut s = String::new();
    s.push_str("CU:");
    for i in (0..4).rev() {
        if (val & (1 << (28 + i))) != 0 { s.push_str(&format!("{}", i)); } else { s.push('_'); }
    }
    
    if (val & STATUS_RP) != 0 { s.push_str(" RP"); }
    if (val & STATUS_FR) != 0 { s.push_str(" FR"); }
    if (val & STATUS_RE) != 0 { s.push_str(" RE"); }
    if (val & STATUS_BEV) != 0 { s.push_str(" BEV"); }
    if (val & STATUS_TS) != 0 { s.push_str(" TS"); }
    if (val & STATUS_SR) != 0 { s.push_str(" SR"); }
    if (val & STATUS_CH) != 0 { s.push_str(" CH"); }
    if (val & STATUS_CE) != 0 { s.push_str(" CE"); }
    if (val & STATUS_DE) != 0 { s.push_str(" DE"); }

    s.push_str(" IM:");
    for i in (0..8).rev() {
        if (val & (1 << (8 + i))) != 0 { s.push_str(&format!("{}", i)); } else { s.push('_'); }
    }

    if (val & STATUS_KX) != 0 { s.push_str(" KX"); }
    if (val & STATUS_SX) != 0 { s.push_str(" SX"); }
    if (val & STATUS_UX) != 0 { s.push_str(" UX"); }

    let ksu = (val >> STATUS_KSU_SHIFT) & 3;
    match ksu {
        0 => s.push_str(" K:K"),
        1 => s.push_str(" K:S"),
        2 => s.push_str(" K:U"),
        _ => s.push_str(" K:?"),
    }

    if (val & STATUS_ERL) != 0 { s.push_str(" ERL"); }
    if (val & STATUS_EXL) != 0 { s.push_str(" EXL"); }
    if (val & STATUS_IE) != 0 { s.push_str(" IE"); }

    s
}

pub(super) fn decode_cause(val: u32) -> String {
    let mut s = String::new();
    if (val & CAUSE_BD) != 0 { s.push_str("BD "); }
    
    let ce = (val >> CAUSE_CE_SHIFT) & 3;
    if ce != 0 { s.push_str(&format!("CE:{} ", ce)); }

    s.push_str("IP:");
    for i in (0..8).rev() {
        if (val & (1 << (8 + i))) != 0 { s.push_str(&format!("{}", i)); } else { s.push('_'); }
    }

    let exc = (val >> CAUSE_EXCCODE_SHIFT) & 0x1F;
    let exc_name = match exc {
        EXC_INT => "INT",
        EXC_MOD => "MOD",
        EXC_TLBL => "TLBL",
        EXC_TLBS => "TLBS",
        EXC_ADEL => "ADEL",
        EXC_ADES => "ADES",
        EXC_IBE => "IBE",
        EXC_DBE => "DBE",
        EXC_SYS => "SYS",
        EXC_BP => "BP",
        EXC_RI => "RI",
        EXC_CPU => "CPU",
        EXC_OV => "OV",
        EXC_TR => "TR",
        EXC_FPE => "FPE",
        EXC_WATCH => "WATCH",
        _ => "?",
    };
    s.push_str(&format!(" Exc:{:02x}({})", exc, exc_name));

    s
}

pub(super) fn is_call_instruction(instr: u32) -> bool {
    let op = (instr >> 26) & 0x3F;
    match op {
        OP_JAL => true,
        OP_SPECIAL => {
            let funct = instr & 0x3F;
            funct == FUNCT_JALR
        }
        OP_REGIMM => {
            let rt = (instr >> 16) & 0x1F;
            rt == RT_BGEZAL || rt == RT_BLTZAL || rt == RT_BGEZALL || rt == RT_BLTZALL
        }
        _ => false,
    }
}

impl<T: Tlb + Send + 'static, C: CpuModel + Send + 'static> MipsCpu<T, C> {
    pub fn run_debug_loop(&self, mut count: Option<usize>, wait: bool, mut writer: Box<dyn Write + Send>) {
        self.halt(); // Ensure stopped before running

        self.running.store(true, Ordering::SeqCst);

        let executor = self.executor.clone();
        let running = self.running.clone();
        let debug = self.debug.clone();
        let exception_mask = self.exception_mask.clone();
        let trace_file = self.trace_file.clone();
        let stop_note = self.stop_note.clone();
        let cpu_id = self.executor.lock().core.cpu_id;

        let task = move || {
            crate::cpu::smp::set_current_cpu(cpu_id);
            let mut exec = executor.lock();
            // Why this run ends, for the stop note (`status`, `wait`).
            let mut reason: Option<String> = None;

            // Clear any breakpoint id left from a previous run/step, or a stop
            // caused by something else (retry, exception_mask) is reported by
            // run_blocking() — and to GDB — as a phantom breakpoint hit.
            exec.last_bp_hit = None;

            // Same reasoning as MipsCpu::start(): this closure runs on its own
            // fresh OS thread ("MIPS-Debug"), so the host FPU rounding mode
            // needs to be re-synced from the guest's tracked FCSR.RM rather
            // than assuming the platform default matches.
            crate::platform::set_fpu_mode((exec.core.fpu_fcsr & 0x3) as u8);

            // The CPU counts as running for the duration of this loop: resume
            // the virtual CP0 Count and compare timer (latched by the stop()
            // above), and re-latch on the way out. For a long `run` this
            // gives the guest live timer ticks; for a single `step` the
            // running window is microseconds, so Count stays effectively
            // frozen between steps and IP7 never fires mid-inspection (use
            // the `ip7` command to inject one deliberately).
            exec.core.on_cpu_start();

            if !wait {
                let _ = writeln!(writer, "Running...");
            }

            let mut first_step = true;
            let mut steps_since_yield = 0;
            // Consecutive EXEC_RETRY count for the instruction currently being
            // attempted (see the EXEC_RETRY arm); reset once anything retires.
            let mut retry_count: u32 = 0;
            const MAX_RUN_RETRIES: u32 = 100_000;

            loop {
                if !running.load(Ordering::Relaxed) {
                    writeln!(writer, "Interrupted").unwrap();
                    reason = Some("interrupted by `stop`".to_string());
                    break;
                }

                // Check step count
                if let Some(c) = count {
                    if c == 0 { reason = Some("step finished".to_string()); break; }
                    count = Some(c - 1);
                }

                // Capture snapshot BEFORE executing — this is the state to restore on undo.
                // memory_writes are collected during step() into pending_memory_writes, then
                // patched into the snapshot afterwards so a single undo entry is self-contained.
                #[cfg(feature = "developer")]
                let undo_snap_idx = if exec.undo_buffer.is_enabled() {
                    exec.pending_memory_writes.clear();
                    let snapshot = exec.create_snapshot();
                    let idx = exec.undo_buffer.push_get_idx(snapshot);
                    Some(idx)
                } else {
                    None
                };

                // Try step with breakpoints enabled.
                // step_int() now pushes (pc, instr) into traceback on successful fetch.
                let mut status = exec.step_int();

                if status == EXEC_BREAKPOINT {
                    // If we hit the temporary breakpoint (ID 0), stop immediately
                    if exec.last_bp_hit == Some(0) {
                        // Don't step over, just let the match below handle the break
                    } else if first_step {
                        // If we hit a user breakpoint right at the start of a command,
                        // step over it to resume execution.
                        exec.skip_breakpoints = true;
                        status = exec.step_int();
                    }
                }

                if first_step {
                    first_step = false;
                }

                // Patch the memory writes collected during step() into the pre-step snapshot.
                #[cfg(feature = "developer")]
                if let Some(idx) = undo_snap_idx {
                    let writes = std::mem::take(&mut exec.pending_memory_writes);
                    if let Some(ref mut snap) = exec.undo_buffer.snapshots[idx] {
                        snap.memory_writes = writes;
                    }
                }

                // Display executed instruction from traceback (already captured by step())
                let insn_trace = debug.load(Ordering::Relaxed) || mips_log(MIPS_LOG_INSN);
                if insn_trace || count.is_some() || trace_file.lock().is_some() {
                    if let Some(entry) = exec.traceback.get_last(1).into_iter().next() {
                        let symbols = exec.symbols.lock();
                        let sym_str = format_pc_symbol(entry.pc, &symbols);
                        let dis = mips_dis::disassemble(entry.instr, entry.pc, Some(&symbols));
                        let info = format!("{:#018x}{}: {:#010x} {}", entry.pc, sym_str, entry.instr, dis);
                        if insn_trace {
                            dlog_dev!(LogModule::Mips, "{}", info);
                            writeln!(writer, "{}", info).unwrap();
                        }
                        if count.is_some() {
                            // Route through devlog writers so Exec: lines serialize with
                            // cache/device dlog output on the same TCP socket.
                            crate::dlog_unconditional!("Exec: {}", info);
                        }
                        if let Some(ref mut f) = *trace_file.lock() {
                            let _ = writeln!(f, "{}", info);
                        }
                    } else if insn_trace {
                        writeln!(writer, "PC: {:016x} (Fetch failed)", exec.core.pc).unwrap();
                    }
                }

                let pc = exec.core.pc;
                match status {
                    EXEC_RETRY => {
                        // Nothing retired, so the same PC must be attempted
                        // again (validate.rs documents the same contract);
                        // breaking out made a transient bus-busy a real stop.
                        retry_count += 1;
                        if retry_count > MAX_RUN_RETRIES {
                            writeln!(writer, "PC={:016x}: Retry (Bus Busy) — still busy after {} attempts, giving up", pc, MAX_RUN_RETRIES).unwrap();
                            reason = Some("bus busy, gave up retrying".to_string());
                            break;
                        }
                        // Refund the step budget: a retry retires no
                        // instruction, so `step N` must still retire N.
                        if let Some(c) = count { count = Some(c + 1); }
                        std::hint::spin_loop();
                        continue;
                    }
                    s if s & EXEC_IS_EXCEPTION != 0 && s & EXEC_IS_TLB_REFILL == 0 => {
                        let code = (s >> crate::cpu::mips_core::CAUSE_EXCCODE_SHIFT) & 0x1F;
                        let mask = exception_mask.load(Ordering::Relaxed);
                        if (mask & (1 << code)) != 0 {
                            writeln!(writer, "PC={:016x}: Exception code={}", pc, code).unwrap();
                            reason = Some(format!("caught exception code {} (`exception` catch)", code));
                            break;
                        }
                    }
                    s if s & EXEC_IS_EXCEPTION != 0 && s & EXEC_IS_TLB_REFILL != 0 => {
                        let code = (s >> crate::cpu::mips_core::CAUSE_EXCCODE_SHIFT) & 0x1F;
                        let mask = exception_mask.load(Ordering::Relaxed);
                        if (mask & (1 << code)) != 0 {
                            writeln!(writer, "PC={:016x}: TLB Miss code={}", pc, code).unwrap();
                            reason = Some(format!("caught TLB refill, code {} (`exception` catch)", code));
                            break;
                        }
                    }
                    EXEC_BREAKPOINT => {
                        let why = exec.breakpoint_hit_reason();
                        if exec.last_bp_hit != Some(0) {
                            writeln!(writer, "PC={:016x}: {}", pc, why).unwrap();
                        }
                        reason = Some(why);
                        break;
                    }
                    _ => {}
                }

                // Something retired: the retry budget is per instruction
                // attempt, not cumulative over a long run.
                retry_count = 0;

                steps_since_yield += 1;
                if steps_since_yield >= 500000 {
                    steps_since_yield = 0;
                    drop(exec);
                    thread::sleep(Duration::from_millis(1));
                    exec = executor.lock();

                    if !running.load(Ordering::Relaxed) {
                        writeln!(writer, "Interrupted").unwrap();
                        break;
                    }
                }
            }
            #[cfg(feature = "developer")]
            { exec.skip_interrupts = false; }
            if let Some(ref mut f) = *trace_file.lock() { let _ = f.flush(); }

            // Print next instruction
            let next_pc = exec.core.pc;
            match exec.debug_fetch_instr(next_pc) {
                 Ok(instr) => {
                     let symbols = exec.symbols.lock();
                     let sym_str = format_pc_symbol(next_pc, &symbols);
                     let dis = mips_dis::disassemble(instr, next_pc, Some(&symbols));
                     writeln!(writer, "Next: {:016x}{}: {:08x} {}", next_pc, sym_str, instr, dis).unwrap();
                 }
                 Err(_) => {
                     writeln!(writer, "Next: {:016x} (Fetch failed)", next_pc).unwrap();
                 }
            }

            let stop_pc = exec.core.pc;
            stop_note.lock().record(stop_pc, reason.unwrap_or_else(|| "stopped".to_string()));

            // Clear temporary breakpoint (used by run/finish)
            exec.clear_temp_breakpoint();

            exec.core.on_cpu_stop();
            drop(exec);
            running.store(false, Ordering::SeqCst);
        };

        let handle = thread::Builder::new().name("MIPS-Debug".to_string()).spawn(task).unwrap();

        if wait {
            let _ = handle.join();
        } else {
            *self.thread.lock() = Some(handle);
        }
    }

    /// Load a static ELF32 MSB image: copy each `PT_LOAD` to its `p_vaddr`,
    /// zero-fill to `p_memsz`, set PC to `e_entry`. Returns a segment summary.
    pub fn load_elf(&self, path: &str) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        self.load_elf_bytes(&bytes, path)
    }

    /// `load_elf` on an image already in memory. `name` only labels errors —
    /// nothing here touches the filesystem, which is what lets a host embedding
    /// IRIS as a library run a bare-metal image it carries in its own binary
    /// (`crate::benchsuite`) without writing a temporary file. A sandboxed app
    /// has nowhere to write one.
    pub fn load_elf_bytes(&self, bytes: &[u8], name: &str) -> Result<String, String> {
        let elf = crate::elf::parse(bytes).map_err(|e| format!("{}: {}", name, e))?;
        self.check_stopped()?;
        let mut exec = self.try_lock_executor()?;
        let mut out = String::new();
        for s in &elf.segments {
            load_range(&mut exec, s.vaddr, &bytes[s.offset..s.offset + s.filesz], s.memsz)?;
            out.push_str(&format!(
                "  {:#018x}  filesz {:#x}  memsz {:#x}  {}\n",
                s.vaddr, s.filesz, s.memsz, crate::elf::flag_str(s.flags)
            ));
        }
        exec.core.pc = elf.entry;
        out.push_str(&format!("  entry {:#018x}\n", elf.entry));
        Ok(out)
    }

    /// Load raw bytes at a virtual address. PC is not touched.
    pub fn load_bin(&self, path: &str, vaddr: u64) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        self.check_stopped()?;
        let mut exec = self.try_lock_executor()?;
        load_range(&mut exec, vaddr, &bytes, bytes.len() as u64)?;
        Ok(format!("  {:#018x}  {} bytes\n", vaddr, bytes.len()))
    }

    pub(super) fn monitor_commands(&self) -> Vec<(String, String)> {
        vec![
            ("cpu".to_string(), "CPU commands: start, stop, run, step, regs, cop0, cop1, mem, dis, jump, translate, trace, undo, sym, loadsym".to_string()),
            ("bp".to_string(), "Breakpoint commands: bp add <addr> [type] [if <expr>], bp list, bp del <id>, bp enable/disable <id>, bp ignore <id> <n>, bp clear, bp save/load <file>".to_string()),
            ("b".to_string(), "Alias for bp add".to_string()),
            ("bl".to_string(), "Alias for bp list".to_string()),
            ("bd".to_string(), "Alias for bp disable".to_string()),
            ("be".to_string(), "Alias for bp enable".to_string()),
            ("bb".to_string(), "Alias for bp delete".to_string()),
            ("tlb".to_string(), "TLB commands: tlb dump | tlb trans <vaddr> [asid] | tlb debug <on|off> [DEV]".to_string()),
            ("start".to_string(), "Start CPU execution thread".to_string()),
            ("stop".to_string(), "Stop CPU execution thread".to_string()),
            ("status".to_string(), "CPU running/stopped, PC, and why it last stopped (never blocks)".to_string()),
            ("wait".to_string(), "wait [ms]: block until the CPU stops (breakpoint, exception) or ms pass; says why".to_string()),
            ("ip7".to_string(), "Set pending IP7 (CP0 Compare timer) interrupt — simulate a timer fire while stopped".to_string()),
            ("exception".to_string(), "Control exception breaks: exception <class|code|all> <on|off>".to_string()),
            ("run".to_string(), "Run instructions until exception or breakpoint: run [addr]".to_string()),
            ("step".to_string(), "Step n instructions or until address: step [count|addr]".to_string()),
            ("next".to_string(), "Step over function calls: next [count]".to_string()),
            ("finish".to_string(), "Run until function return (jr ra)".to_string()),
            ("fin".to_string(), "Alias for finish".to_string()),
            ("s".to_string(), "Alias for step".to_string()),
            #[cfg(feature = "developer")]
            ("si".to_string(), "Step suppressing interrupt delivery (alias for step, no interrupts taken) [DEV]".to_string()),
            ("n".to_string(), "Alias for next".to_string()),
            ("regs".to_string(), "Dump registers".to_string()),
            ("r".to_string(), "Alias for regs".to_string()),
            ("c".to_string(), "Alias for run".to_string()),
            ("cont".to_string(), "Alias for run".to_string()),
            ("cop0".to_string(), "Dump COP0 registers".to_string()),
            ("cop1".to_string(), "Dump COP1 registers".to_string()),
            ("eframe".to_string(), "IRIX exception frame: eframe <addr> [o32|64] — registers, decoded cause/sr, then a backtrace of the interrupted context".to_string()),
            ("putbuf".to_string(), "IRIX kernel console log (putbuf ring): putbuf [size]  (alias dmesg; needs the kernel's symbols)".to_string()),
            ("mem".to_string(), "Dump virtual memory: mem <addr> [count] [b|h|w|d]".to_string()),
            ("m".to_string(), "Alias for mem".to_string()),
            ("mw".to_string(), "Write virtual memory: mw <addr> <val> [size: b|h|w|d]".to_string()),
            ("stack".to_string(), "Dump stack memory: stack [addr] [count]".to_string()),
            ("bt".to_string(), "Print backtrace: bt [frames]".to_string()),
            ("ms".to_string(), "Read string from virtual memory: ms <addr> [max_len]".to_string()),
            ("dis".to_string(), "Disassemble virtual memory: dis [addr] [count]".to_string()),
            ("d".to_string(), "Alias for dis".to_string()),
            ("jump".to_string(), "Set PC to address: jump <addr>".to_string()),
            ("setreg".to_string(), "Set register value: setreg <reg> <value>".to_string()),
            ("translate".to_string(), "Translate virtual address: translate <addr>".to_string()),
            ("t".to_string(), "Alias for translate".to_string()),
            ("debug".to_string(), "CPU instruction trace: debug <on|off|file <path>> [DEV]".to_string()),
            ("ex".to_string(), "Alias for exception".to_string()),
            ("undo".to_string(), "Undo N instructions or control undo buffer: undo [count] | undo <on|off|clear|resize <n>> [DEV]".to_string()),
            ("dt".to_string(), "Disassemble traceback: dt [count] | dt file <path> [count]".to_string()),
            ("idleprof".to_string(), "Locate idle/spin loops: idleprof <on|off|report [count]|stats>".to_string()),
            #[cfg(feature = "idle-pause")]
            ("idle".to_string(), "Idle parking: idle <on|off|status>".to_string()),
            #[cfg(feature = "instr_stats")]
            ("instrstats".to_string(), "Per-instruction decode/execution counters: instrstats [report|clear|dump] [DEV]".to_string()),
            ("u".to_string(), "Alias for undo [DEV]".to_string()),
            ("sym".to_string(), "Lookup symbol: sym <addr> | sym <name> (lists every copy; pick one with name:N)".to_string()),
            ("loadsym".to_string(), "Load symbols from file: loadsym <file>".to_string()),
            ("loadelf".to_string(), "Load a static ELF32 MSB binary and set PC to its entry: loadelf <file>".to_string()),
            ("loadbin".to_string(), "Load raw bytes at a virtual address: loadbin <file> <addr>".to_string()),
            ("proc".to_string(), "IRIX kernel introspection: proc info  (requires `loadsym` first)".to_string()),
            ("l1i".to_string(), "L1 Instruction Cache commands: l1i <check|dump> <addr|index>".to_string()),
            ("l1d".to_string(), "L1 Data Cache commands: l1d <check|dump> <addr|index> | l1d wb <vaddr> <size> | l1d pwb <paddr> <size>".to_string()),
            ("l2".to_string(), "L2 Cache commands: l2 <check|dump> <addr|index>".to_string()),
            ("ll".to_string(), "LL/SC state: ll (llbit/lladdr) | ll stats | ll clear (histogram needs --features llstats)".to_string()),
            #[cfg(feature = "jitv2")]
            ("j2".to_string(), "JIT v2 introspection: j2 pcp | j2 dumppcp [addr] [path] (capture page+memory for the jitv2-pcp-dump offline analyzer) | j2 corpus [dir] (dump every cached page to a corpus dir for offline codegen measurement) | j2 intrun [N] (instructions sharing one pending-interrupt check; 1 = per-instruction) | j2 status (alias: stats) | j2 inline [on|off] | j2 dispatch [on|off] | j2 fallback [on|off] | j2 inline_mem [on|off] | j2 memhelpers [on|off] | j2 pagewb [on|off] | j2 threads (read-only) | j2 <alu|fpu|branch|loadstore|cop0> [on|off] | j2 instrs [category] | j2 flush | j2 clear <paddr> | j2 deny <paddr> | j2 html [path] | j2 lockstep (status only; always on when built) | j2 lstate [full] [N] (recent lockstep step history, state entering each instr) (see also: jitcheck <n> for JIT-vs-interpreter determinism checking)".to_string()),
            #[cfg(feature = "developer")]
            ("trace".to_string(), "Execution trace capture: trace start <path> | trace stop | trace status".to_string()),
        ]
    }

    pub(super) fn monitor_execute(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        // Handle "cpu" prefix by shifting args
        let (actual_cmd, actual_args) = if cmd == "cpu" {
            if args.is_empty() {
                return Err("Usage: cpu <command> [args...]".to_string());
            }
            (args[0], &args[1..])
        } else {
            (cmd, args)
        };

        // One rule for the running CPU, the same for every command:
        //  - run control (run/step/next/finish) starts from a stopped CPU,
        //    and says so if it isn't;
        //  - start/stop/status/wait/help never touch the core lock;
        //  - everything else holds the core still while it works (`pause`):
        //    a running CPU is stopped for the command and restarted after.
        const RUN_CONTROL: &[&str] = &["run", "c", "cont", "finish", "fin", "step", "s", "si", "next", "n"];
        const LOCK_FREE: &[&str] = &["start", "stop", "status", "wait", "help"];
        if RUN_CONTROL.contains(&actual_cmd) && self.is_running() {
            return Err(format!("CPU is running: `stop` it before `{}` (or `wait` for a breakpoint)", actual_cmd));
        }
        let _paused = if RUN_CONTROL.contains(&actual_cmd) || LOCK_FREE.contains(&actual_cmd) {
            None
        } else {
            Some(self.pause())
        };

        if actual_cmd == "bp" || actual_cmd == "b" || actual_cmd == "bd" || actual_cmd == "be" || actual_cmd == "bb" || actual_cmd == "bl" {
            let (subcmd, args) = if actual_cmd == "bp" {
                if actual_args.is_empty() {
                    return Err("Usage: bp <add|list|del|enable|disable|ignore|clear|save|load> ...".to_string());
                }
                (actual_args[0], &actual_args[1..])
            } else {
                let s = match actual_cmd {
                    "b" => "add",
                    "bd" => "disable",
                    "be" => "enable",
                    "bb" => "delete",
                    "bl" => "list",
                    _ => unreachable!(),
                };
                (s, actual_args)
            };

            let mut exec = self.executor.lock();
            match subcmd {
                "add" => {
                    if args.is_empty() { return Err("Usage: bp add <addr> [type]".to_string()); }

                    // Check for "if <expr>" at the end
                    let (args_before_if, cond_expr, cond_src) = {
                        if let Some(pos) = args.iter().position(|&a| a == "if") {
                            let expr_str = args[pos+1..].join(" ");
                            let symbols = exec.symbols.lock();
                            let expr = exp::parse_and_fold_checked(&expr_str, Some(&symbols))
                                .map_err(|e| format!("bad condition: {}", e))?;
                            (&args[..pos], Some(expr), Some(expr_str))
                        } else {
                            (args, None, None)
                        }
                    };

                    if args_before_if.is_empty() { return Err("Usage: bp add <addr> [type] [if <expr>]".to_string()); }

                    let addr = {
                        let symbols = exec.symbols.lock();
                        parse_cpu_arg(args_before_if[0], &exec.core, Some(&symbols))?
                    };
                    
                    let kind = if args_before_if.len() > 1 {
                        match args_before_if[1] {
                            "pc" => BpType::Pc,
                            "r" | "read" => BpType::VirtRead,
                            "w" | "write" => BpType::VirtWrite,
                            "f" | "fetch" => BpType::VirtFetch,
                            "pr" | "pread" => BpType::PhysRead,
                            "pw" | "pwrite" => BpType::PhysWrite,
                            "pf" | "pfetch" => BpType::PhysFetch,
                            _ => return Err("Invalid breakpoint type. Options: pc, r, w, f, pr, pw, pf".to_string()),
                        }
                    } else {
                        BpType::Pc
                    };

                    let id = exec.next_bp_id;
                    exec.next_bp_id += 1;
                    exec.add_breakpoint(id, addr, kind);

                    let line = {
                        let symbols = exec.symbols.lock();
                        format_bp_line(id, addr, kind, true, cond_src.as_deref(), &symbols)
                    };
                    if let Some(bp) = exec.breakpoints.iter_mut().find(|bp| bp.id == id) {
                        bp.condition = cond_expr;
                        bp.condition_src = cond_src;
                    }
                    writeln!(writer, "Breakpoint added: {}", line).unwrap();
                    return Ok(());
                }
                "list" => {
                    writeln!(writer, "Breakpoints:").unwrap();
                    let symbols = exec.symbols.lock();
                    for bp in &exec.breakpoints {
                        if bp.id == 0 { continue; } // Skip internal breakpoint
                        let line = format_bp_line(bp.id, bp.addr, bp.kind, bp.enabled,
                                                  bp.condition_src.as_deref(), &symbols);
                        let mut extra = format!("  [hits {}", bp.hits);
                        if bp.ignore > 0 {
                            extra.push_str(&format!(", ignoring next {}", bp.ignore));
                        }
                        extra.push(']');
                        writeln!(writer, "  {}{}", line, extra).unwrap();
                    }
                    return Ok(());
                }
                "del" | "delete" => {
                    if args.is_empty() { return Err("Usage: bp del <id>".to_string()); }
                    let id = args[0].parse::<usize>().map_err(|_| "Invalid ID")?;
                    if exec.remove_breakpoint(id) {
                        writeln!(writer, "Breakpoint {} deleted", id).unwrap();
                        return Ok(());
                    } else {
                        return Err(format!("Breakpoint {} not found", id));
                    }
                }
                "enable" => {
                    if args.is_empty() { return Err("Usage: bp enable <id>".to_string()); }
                    let id = args[0].parse::<usize>().map_err(|_| "Invalid ID")?;
                    if exec.set_breakpoint_enabled(id, true) {
                        writeln!(writer, "Breakpoint {} enabled", id).unwrap();
                        return Ok(());
                    } else {
                        return Err(format!("Breakpoint {} not found", id));
                    }
                }
                "ignore" => {
                    if args.len() < 2 { return Err("Usage: bp ignore <id> <count>".to_string()); }
                    let id = args[0].parse::<usize>().map_err(|_| "Invalid ID")?;
                    let n = args[1].parse::<u64>().map_err(|_| "Invalid count")?;
                    let bp = exec.breakpoints.iter_mut().find(|b| b.id == id && id != 0)
                        .ok_or_else(|| format!("Breakpoint {} not found", id))?;
                    bp.ignore = n;
                    writeln!(writer, "Breakpoint {} will let the next {} hit{} pass", id, n, if n == 1 { "" } else { "s" }).unwrap();
                    return Ok(());
                }
                "clear" => {
                    let ids: Vec<usize> = exec.breakpoints.iter().map(|b| b.id).filter(|&i| i != 0).collect();
                    for id in &ids {
                        exec.remove_breakpoint(*id);
                    }
                    writeln!(writer, "Deleted {} breakpoint{}", ids.len(), if ids.len() == 1 { "" } else { "s" }).unwrap();
                    return Ok(());
                }
                "save" => {
                    // One `bp add` line per breakpoint: the file is what you
                    // would type, so it can be edited and `bp load`ed after a
                    // restart. Addresses are saved numerically (symbols may not
                    // be loaded yet when it is read back).
                    let path = args.first().ok_or("Usage: bp save <file>")?;
                    let mut out = String::new();
                    let mut n = 0;
                    for bp in exec.breakpoints.iter().filter(|b| b.id != 0) {
                        if matches!(bp.kind, BpType::WriteValue | BpType::RegValue) { continue; }
                        if !bp.enabled { out.push_str("# disabled\n"); }
                        out.push_str(&format!("bp add {:#x} {}", bp.addr, bp.kind.name()));
                        if let Some(c) = &bp.condition_src {
                            out.push_str(" if ");
                            out.push_str(c);
                        }
                        out.push('\n');
                        n += 1;
                    }
                    std::fs::write(path, out).map_err(|e| format!("{}: {}", path, e))?;
                    writeln!(writer, "Saved {} breakpoint{} to {}", n, if n == 1 { "" } else { "s" }, path).unwrap();
                    return Ok(());
                }
                "load" => {
                    let path = args.first().ok_or("Usage: bp load <file>")?;
                    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
                    let mut disable_next = false;
                    let mut added = 0;
                    for (lineno, line) in text.lines().enumerate() {
                        let line = line.trim();
                        if line == "# disabled" { disable_next = true; continue; }
                        if line.is_empty() || line.starts_with('#') { continue; }
                        let words: Vec<&str> = line.split_whitespace().collect();
                        let rest = match words.as_slice() {
                            ["bp", "add", rest @ ..] | ["b", rest @ ..] => rest,
                            _ => return Err(format!("{}:{}: not a `bp add` line", path, lineno + 1)),
                        };
                        let (before_if, cond) = match rest.iter().position(|&w| w == "if") {
                            Some(p) => (&rest[..p], Some(rest[p + 1..].join(" "))),
                            None => (rest, None),
                        };
                        let symbols_arc = exec.symbols.clone();
                        let symbols = symbols_arc.lock();
                        let expr = match &cond {
                            Some(c) => Some(exp::parse_and_fold_checked(c, Some(&symbols))
                                .map_err(|e| format!("{}:{}: bad condition: {}", path, lineno + 1, e))?),
                            None => None,
                        };
                        let addr = before_if.first()
                            .ok_or_else(|| format!("{}:{}: missing address", path, lineno + 1))
                            .and_then(|a| parse_cpu_arg(a, &exec.core, Some(&symbols)))?;
                        drop(symbols);
                        let kind = match before_if.get(1).copied().unwrap_or("pc") {
                            "pc" => BpType::Pc, "r" | "read" => BpType::VirtRead, "w" | "write" => BpType::VirtWrite,
                            "f" | "fetch" => BpType::VirtFetch, "pr" | "pread" => BpType::PhysRead,
                            "pw" | "pwrite" => BpType::PhysWrite, "pf" | "pfetch" => BpType::PhysFetch,
                            k => return Err(format!("{}:{}: unknown breakpoint type {}", path, lineno + 1, k)),
                        };
                        let id = exec.next_bp_id;
                        exec.next_bp_id += 1;
                        exec.add_breakpoint(id, addr, kind);
                        if let Some(bp) = exec.breakpoints.iter_mut().find(|bp| bp.id == id) {
                            bp.condition = expr;
                            bp.condition_src = cond;
                        }
                        if disable_next {
                            exec.set_breakpoint_enabled(id, false);
                            disable_next = false;
                        }
                        added += 1;
                    }
                    writeln!(writer, "Loaded {} breakpoint{} from {}", added, if added == 1 { "" } else { "s" }, path).unwrap();
                    return Ok(());
                }
                "disable" => {
                    if args.is_empty() { return Err("Usage: bp disable <id>".to_string()); }
                    let id = args[0].parse::<usize>().map_err(|_| "Invalid ID")?;
                    if exec.set_breakpoint_enabled(id, false) {
                        writeln!(writer, "Breakpoint {} disabled", id).unwrap();
                        return Ok(());
                    } else {
                        return Err(format!("Breakpoint {} not found", id));
                    }
                }
                _ => return Err("Unknown bp subcommand".to_string()),
            }
        }

        if actual_cmd == "loadsym" {
            if actual_args.is_empty() {
                return Err("Usage: loadsym <file>".to_string());
            }
            let exec = self.try_lock_executor()?;
            let mut symbols = exec.symbols.lock();
            match symbols.load(actual_args[0]) {
                Ok(count) => {
                    writeln!(writer, "Loaded {} symbols from {}", count, actual_args[0]).unwrap();
                    return Ok(());
                },
                Err(e) => return Err(format!("Failed to load symbols: {}", e)),
            }
        }

        if actual_cmd == "sym" {
            if actual_args.is_empty() {
                return Err("Usage: sym <addr|name>".to_string());
            }
            let exec = self.executor.lock();
            let symbols = exec.symbols.lock();
            // A name: every copy, with the map file it came from.
            let copies = symbols.copies(actual_args[0]);
            if !copies.is_empty() {
                let base = actual_args[0].rsplit_once(':').map(|(b, _)| b).unwrap_or(actual_args[0]);
                for (i, addr, file) in &copies {
                    writeln!(writer, "{}:{} = {:016x}  ({})", base, i, addr, file).unwrap();
                }
                if copies.len() > 1 {
                    writeln!(writer, "ambiguous: use {}:N to pick one", base).unwrap();
                }
                return Ok(());
            }
            let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
            
            if let Some((sym_addr, name)) = symbols.lookup(addr) {
                let offset = addr - sym_addr;
                if offset == 0 {
                    writeln!(writer, "{:016x} = {}", addr, name).unwrap();
                } else {
                    writeln!(writer, "{:016x} = {} + 0x{:x}", addr, name, offset).unwrap();
                }
            } else {
                writeln!(writer, "{:016x} = ???", addr).unwrap();
            }
            return Ok(());
        }

        if actual_cmd == "proc" {
            let mut exec = self.try_lock_executor()?;
            // Helper closures around debug_read so the body stays readable.
            // All reads here are big-endian (IRIX kernel).
            let read_u32 = |exec: &mut MipsExecutor<T, C>, vaddr: u64| -> Result<u32, String> {
                exec.debug_read(vaddr, 4)
                    .map(|v| v as u32)
                    .map_err(|e| format!("debug_read({:#x}): {:?}", vaddr, e))
            };
            let read_bytes = |exec: &mut MipsExecutor<T, C>, vaddr: u64, n: usize| -> Result<Vec<u8>, String> {
                let mut out = Vec::with_capacity(n);
                // debug_read returns up to 8 bytes per call; chunk it.
                let mut a = vaddr;
                let mut remaining = n;
                while remaining > 0 {
                    let step = remaining.min(4);
                    let v = exec.debug_read(a, step)
                        .map_err(|e| format!("debug_read({:#x}): {:?}", a, e))?;
                    // big-endian: high byte first within `step` bytes
                    for i in (0..step).rev() {
                        out.push(((v >> (i * 8)) & 0xff) as u8);
                    }
                    a += step as u64;
                    remaining -= step;
                }
                Ok(out)
            };
            // Probe `utsname` symbol and read sysname/release.
            let utsname_va = {
                let symbols = exec.symbols.lock();
                match symbols.get_addr("utsname") {
                    Some(a) => a,
                    None => return Err("symbol 'utsname' not found — run `loadsym <unix.nm>` first".to_string()),
                }
            };
            // IRIX utsname has 5 fields of SYS_NMLN bytes each. SYS_NMLN is
            // 257 on IRIX 5.3 (large for POSIX), but we only read the front
            // null-terminated strings. Read 256 bytes per field to be safe.
            const NMLN: usize = 257;
            let buf = read_bytes(&mut exec, utsname_va, NMLN * 5)?;
            let read_str = |off: usize| -> String {
                let end = buf[off..off + NMLN].iter().position(|&b| b == 0).unwrap_or(NMLN);
                String::from_utf8_lossy(&buf[off..off + end]).into_owned()
            };
            let sysname  = read_str(0 * NMLN);
            let nodename = read_str(1 * NMLN);
            let release  = read_str(2 * NMLN);
            let version  = read_str(3 * NMLN);
            let machine  = read_str(4 * NMLN);

            let sub = actual_args.first().copied().unwrap_or("info");
            if sub == "info" {
                writeln!(writer, "utsname @ {:#x}:", utsname_va).unwrap();
                writeln!(writer, "  sysname  = {:?}", sysname).unwrap();
                writeln!(writer, "  nodename = {:?}", nodename).unwrap();
                writeln!(writer, "  release  = {:?}", release).unwrap();
                writeln!(writer, "  version  = {:?}", version).unwrap();
                writeln!(writer, "  machine  = {:?}", machine).unwrap();
                let nproc_va = exec.symbols.lock().get_addr("nproc").ok_or_else(|| "symbol 'nproc' not found".to_string())?;
                let proc_va  = exec.symbols.lock().get_addr("proc").ok_or_else(|| "symbol 'proc' not found".to_string())?;
                let nproc = read_u32(&mut exec, nproc_va)?;
                let proc_ptr = read_u32(&mut exec, proc_va)?;
                writeln!(writer, "nproc    @ {:#x} = {}", nproc_va, nproc).unwrap();
                writeln!(writer, "proc[]   @ {:#x} -> {:#x}", proc_va, proc_ptr).unwrap();
                return Ok(());
            }
            return Err("Usage: proc info".to_string());
        }

        // Bare `ll` reports live reservation state. `ll stats` / `ll clear`
        // fall through to the histogram arm below (feature `llstats`) — this
        // early return used to swallow every `ll ...` form, which silently made
        // that arm dead code even in a build that had it compiled in.
        if actual_cmd == "ll" && actual_args.is_empty() {
            let exec = self.try_lock_executor()?;
            let llbit  = exec.cache.get_llbit();
            let lladdr = exec.cache.get_lladdr();
            let phys   = (lladdr as u64) << 4;
            writeln!(writer, "llbit:  {}", if llbit { "SET" } else { "clear" }).unwrap();
            writeln!(writer, "lladdr: {:08x}  (phys {:010x})", lladdr, phys).unwrap();
            #[cfg(feature = "llstats")]
            writeln!(writer, "(`ll stats` for the per-address retry histogram)").unwrap();
            return Ok(());
        }

        if actual_cmd == "l1d" && (actual_args.first().copied() == Some("wb") || actual_args.first().copied() == Some("pwb")) {
            let is_phys = actual_args[0] == "pwb";
            if actual_args.len() < 3 {
                return Err(format!("Usage: l1d {} <{}> <size>", actual_args[0], if is_phys { "paddr" } else { "vaddr" }));
            }
            let mut exec = self.try_lock_executor()?;
            let symbols_arc = exec.symbols.clone();
            let symbols = symbols_arc.lock();
            let addr = parse_cpu_arg(actual_args[1], &exec.core, Some(&symbols))?;
            let size = parse_cpu_arg(actual_args[2], &exec.core, Some(&symbols))?;
            drop(symbols);

            let (virt_addr, phys_addr) = if is_phys {
                (None, addr)
            } else {
                let tr = exec.debug_translate(addr);
                if tr.is_exception() {
                    return Err(format!("Virtual {:016x} -> Translation Failed", addr));
                }
                (Some(addr), tr.phys)
            };

            let n = exec.cache.writeback(virt_addr, phys_addr, size);
            writeln!(writer, "l1d {}: wrote back {} dirty line(s) covering phys [{:016x}, {:016x})",
                actual_args[0], n, phys_addr, phys_addr + size).unwrap();
            return Ok(());
        }

        if actual_cmd == "l1i" || actual_cmd == "l1d" || actual_cmd == "l2" {
            if actual_args.is_empty() {
                return Err(format!("Usage: {} <check|dump> <addr|index>", actual_cmd));
            }
            let cache_name = actual_cmd;
            let op = actual_args[0];
            let val_str = if actual_args.len() > 1 { actual_args[1] } else { "0" };

            let mut exec = self.try_lock_executor()?;
            let symbols = exec.symbols.lock();
            let val = parse_cpu_arg(val_str, &exec.core, Some(&symbols))?;
            drop(symbols);

            // Perform translation first if needed, as it requires mutable access to exec
            let (virt_addr, phys_addr) = if (op == "check" || op == "probe") && val >= 0x8000_0000 && !exec.memory.contains(val) {
                let tr = exec.debug_translate(val);
                if !tr.is_exception() {
                    let pa = tr.phys;
                    writeln!(writer, "Virtual {:016x} -> Physical {:016x}", val, pa).unwrap();
                    (val, pa)
                } else {
                    writeln!(writer, "Virtual {:016x} -> Translation Failed", val).unwrap();
                    (val, val)
                }
            } else {
                (val, val)
            };

            // Call debug methods through the MipsCache trait
            match op {
                "check" | "probe" => {
                    writeln!(writer, "{}", exec.cache.debug_probe(cache_name, virt_addr, phys_addr)).unwrap();
                }
                "dump" => {
                    writeln!(writer, "{}", exec.cache.debug_dump_line(cache_name, val as usize)).unwrap();
                }
                _ => return Err("Unknown operation. Use check or dump".to_string()),
            }
            return Ok(());
        }

        // Helper to dump registers
        let dump_regs = |exec: &mut MipsExecutor<T, C>, out: &mut dyn Write| {
            writeln!(out, "PC: {:#018x}", exec.core.pc).unwrap();
            writeln!(out, "HI: {:#018x} LO: {:#018x}", exec.core.hi, exec.core.lo).unwrap();
            for i in 0..32 {
                let val = exec.core.gpr[i];
                write!(out, "{:4}(${:02}): {:#018x}  ", mips_dis::reg_name(i as u32), i, val).unwrap();
                if (i + 1) % 4 == 0 { writeln!(out).unwrap(); }
            }
            writeln!(out, "CP0 Status: {:#010x} ({})", exec.core.cp0_status, decode_status(exec.core.cp0_status)).unwrap();
            writeln!(out, "CP0 Cause:  {:#010x} ({})", exec.core.cp0_cause, decode_cause(exec.core.cp0_cause)).unwrap();
            writeln!(out, "CP0 EPC: {:#018x} BadVAddr: {:#018x}", exec.core.cp0_epc, exec.core.cp0_badvaddr).unwrap();
        };

        match actual_cmd {
            "help" => {
                writeln!(writer, "CPU Commands:").unwrap();
                for (c, h) in self.register_commands() {
                    writeln!(writer, "  {:12} - {}", c, h).unwrap();
                }
                Ok(())
            }
            "start" => {
                self.start();
                writeln!(writer, "CPU started").unwrap();
                Ok(())
            }
            "stop" => {
                let was_running = self.is_running();
                // Note first, as the breakpoint path does: `stop` clears
                // `running`, and a client that sees that must find the note.
                // The pc is only readable once the core is stopped, so it is
                // filled in after.
                if was_running {
                    self.stop_note.lock().record(0, "stopped by `stop`".to_string());
                }
                self.stop();
                if was_running {
                    let pc = self.executor.lock().core.pc;
                    self.stop_note.lock().pc = pc;
                }
                writeln!(writer, "CPU stopped").unwrap();
                Ok(())
            }
            "status" => {
                // Never blocks: while running, the core is not touched.
                // Read the flag before the note: a stop records its note
                // first, so a CPU seen stopped has its note.
                let running = self.is_running();
                let note = self.stop_note.lock().clone();
                if running {
                    writeln!(writer, "running").unwrap();
                    if note.seq > 0 {
                        writeln!(writer, "  last stop #{}: {} at {:016x}", note.seq, note.reason, note.pc).unwrap();
                    }
                    return Ok(());
                }
                let exec = self.executor.lock();
                let pc = exec.core.pc;
                let symbols = exec.symbols.lock();
                let sym_str = format_pc_symbol(pc, &symbols);
                writeln!(writer, "stopped pc={:016x}{}", pc, sym_str).unwrap();
                if note.seq > 0 {
                    writeln!(writer, "  stop #{}: {}", note.seq, note.reason).unwrap();
                }
                let hz = exec.core.count_hz;
                writeln!(writer, "  count_hz={} ({:.3} MHz)  count={:#010x} compare={:#010x}",
                    hz, hz as f64 / 1e6, exec.core.count_peek(), exec.core.cp0_compare as u32).unwrap();
                Ok(())
            }
            "wait" => {
                // wait [ms]: block until the CPU stops (a breakpoint, an
                // exception catch) or ms pass (default 10000), then say why.
                let ms: u64 = actual_args.first().and_then(|a| a.parse().ok()).unwrap_or(10_000);
                let deadline = std::time::Instant::now() + Duration::from_millis(ms);
                while self.is_running() && std::time::Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(5));
                }
                if self.is_running() {
                    writeln!(writer, "still running after {} ms", ms).unwrap();
                    return Ok(());
                }
                let note = self.stop_note.lock().clone();
                let exec = self.executor.lock();
                let symbols = exec.symbols.lock();
                let pc = exec.core.pc;
                writeln!(writer, "stopped pc={:016x}{}", pc, format_pc_symbol(pc, &symbols)).unwrap();
                if note.seq > 0 {
                    writeln!(writer, "  stop #{}: {}", note.seq, note.reason).unwrap();
                }
                Ok(())
            }
            "ip7" => {
                // The compare timer is silenced while the CPU is stopped
                // (on_cpu_stop) — this injects the interrupt it would have
                // raised, through the same pending word, for debugging
                // interrupt delivery under manual stepping.
                let exec = self.try_lock_executor()?;
                exec.core.set_interrupt(7);
                writeln!(writer, "IP7 set pending (delivered on next executed instruction, mask permitting)").unwrap();
                Ok(())
            }
            "run" | "c" | "cont" => {
                let block = actual_args.first() == Some(&"block");
                let args_rest = if block { &actual_args[1..] } else { actual_args };
                let until_pc = if !args_rest.is_empty() {
                    let exec = self.executor.lock();
                    let symbols = exec.symbols.lock();
                    let pc = parse_cpu_arg(args_rest[0], &exec.core, Some(&symbols))?;
                    println!("Running until PC = {:016x}", pc);
                    Some(pc)
                } else {
                    None
                };

                if let Some(pc) = until_pc {
                    self.executor.lock().set_temp_breakpoint(pc);
                }
                self.run_debug_loop(None, block, writer);
                Ok(())
            }
            "finish" | "fin" => {
                let actual_args = if actual_args.first() == Some(&"block") { &actual_args[1..] } else { actual_args };
                let mut exec = self.executor.lock();
                let ret_addr = exec.get_return_address();
                if let Some(addr) = ret_addr {
                    exec.set_temp_breakpoint(addr);
                    drop(exec);
                    self.run_debug_loop(None, true, writer);
                    Ok(())
                } else {
                    Err("Could not determine return address".to_string())
                }
            }
            "step" | "s" | "si" => {
                let actual_args = if actual_args.first() == Some(&"block") { &actual_args[1..] } else { actual_args };
                let mut count = Some(1);
                let mut until_pc = None;

                if !actual_args.is_empty() {
                    let arg = actual_args[0];
                    // If it parses as a number and doesn't start with 0x, treat as count.
                    // Otherwise treat as address/register.
                    if let Ok(c) = arg.parse::<usize>() {
                        if !arg.starts_with("0x") {
                            count = Some(c);
                        } else {
                            let exec = self.executor.lock();
                            let symbols = exec.symbols.lock();
                            until_pc = Some(parse_cpu_arg(arg, &exec.core, Some(&symbols))?);
                            count = None;
                        }
                    } else {
                        let exec = self.executor.lock();
                        let symbols = exec.symbols.lock();
                        until_pc = Some(parse_cpu_arg(arg, &exec.core, Some(&symbols))?);
                        count = None;
                    }
                }

                if let Some(pc) = until_pc {
                    self.executor.lock().set_temp_breakpoint(pc);
                }
                #[cfg(feature = "developer")]
                if cmd == "si" { self.executor.lock().skip_interrupts = true; }
                self.run_debug_loop(count, true, writer);
                Ok(())
            }
            "next" | "n" => {
                let actual_args = if actual_args.first() == Some(&"block") { &actual_args[1..] } else { actual_args };
                let count = if !actual_args.is_empty() {
                    actual_args[0].parse().unwrap_or(1)
                } else {
                    1
                };

                if count > 1 {
                    writeln!(writer, "Warning: 'next' with count > 1 is not fully supported, executing once.").unwrap();
                }

                let mut exec = self.executor.lock();
                let pc = exec.core.pc;
                let is_call = if let Ok(instr) = exec.debug_fetch_instr(pc) {
                    is_call_instruction(instr)
                } else {
                    false
                };

                if is_call {
                    if is_call {
                         if let Ok(instr) = exec.debug_fetch_instr(pc) {
                             let symbols = exec.symbols.lock();
                             let sym_str = format_pc_symbol(pc, &symbols);
                             let dis = mips_dis::disassemble(instr, pc, Some(&symbols));
                             writeln!(writer, "Exec: {:016x}{}: {:08x} {}", pc, sym_str, instr, dis).unwrap();
                         }
                    }
                }

                drop(exec); // Release lock before running loop

                if is_call {
                    self.executor.lock().set_temp_breakpoint(pc + 8);
                    self.run_debug_loop(None, true, writer);
                } else {
                    self.run_debug_loop(Some(1), true, writer);
                }
                Ok(())
            }
            "regs" | "r" => {
                let mut exec = self.try_lock_executor()?;
                if actual_args.is_empty() {
                    dump_regs(&mut exec, &mut writer);
                } else {
                    let symbols = exec.symbols.lock();
                    for arg in actual_args {
                        match parse_cpu_arg(arg, &exec.core, Some(&symbols)) {
                            Ok(val) => {
                                let sym_str = format_pc_symbol(val, &symbols);
                                writeln!(writer, "{}: {:016x} ({}){}", arg, val, val, sym_str).unwrap();
                            }
                            Err(e) => writeln!(writer, "{}", e).unwrap(),
                        }
                    }
                }
                Ok(())
            }
            "cop0" => {
                let mut exec = self.try_lock_executor()?;
                writeln!(writer, "COP0 Registers:").unwrap();
                for i in 0..32 {
                    let val = exec.core.read_cp0(i);
                    let name = mips_dis::cp0_reg_name(i);
                    if name != "?" {
                        write!(writer, "  {:2} {:8}: {:#018x}", i, name, val).unwrap();
                        if i == 12 { // Status
                            write!(writer, " {}", decode_status(val as u32)).unwrap();
                        } else if i == 13 { // Cause
                            write!(writer, " {}", decode_cause(val as u32)).unwrap();
                        }
                        writeln!(writer).unwrap();
                    }
                }
                let c = &exec.core;
                // count_hz is fixed for the life of the core (DEFAULT_COUNT_HZ
                // unless overridden by `[clock] fixed_mhz` / the CLI).
                writeln!(writer, "  IP7 timer: count_hz={} ({:.3} MHz)  fired={}",
                    c.count_hz, c.count_hz as f64 / 1e6,
                    c.fasttick_count.load(Ordering::Relaxed)).unwrap();
                Ok(())
            }
            "cop1" => {
                let exec = self.try_lock_executor()?;
                let status = exec.core.cp0_status;
                let fr1 = (status & crate::cpu::mips_core::STATUS_FR) != 0;
                let cu1 = (status & crate::cpu::mips_core::STATUS_CU1) != 0;
                writeln!(writer, "COP1 Registers (FPU):").unwrap();
                // The FR bit decides register-file interpretation: FR=1 is 32
                // independent 64-bit registers; FR=0 packs even/odd pairs into
                // one 64-bit slot (odd = upper 32 bits of the even slot), and
                // double/long ops must use the even register. `f{NN}` below is
                // always the raw physical slot; the "FR=0 view" line spells out
                // what the guest actually sees for a paired slot.
                writeln!(writer, "  cp0_status={:#010x}  CU1={}  FR={}  ({}-bit FPU register mode)",
                    status, cu1 as u8, fr1 as u8, if fr1 { 64 } else { 32 }).unwrap();
                if !cu1 {
                    writeln!(writer, "  NOTE: CU1 clear — any FP instruction here raises Coprocessor Unusable").unwrap();
                }
                for i in 0..32 {
                    let val = exec.core.fpr[i];
                    let f32_val = f32::from_bits(val as u32);
                    let f64_val = f64::from_bits(val);
                    if fr1 {
                        writeln!(writer, "  f{:02}: {:#018x}  (s: {:e}  d: {:e})", i, val, f32_val, f64_val).unwrap();
                    } else if i % 2 == 0 {
                        // FR=0: even slot holds FPR(i) in low32, FPR(i+1) in high32; full 64 bits are FPR(i) as d/l.
                        let lo = f32::from_bits(val as u32);
                        let hi = f32::from_bits((val >> 32) as u32);
                        writeln!(writer, "  f{:02}/f{:02}: {:#018x}  (f{:02} s: {:e}  f{:02} s: {:e}  f{:02} d: {:e})",
                            i, i + 1, val, i, lo, i + 1, hi, i, f64_val).unwrap();
                    }
                }
                // FCSR decode. Layout (R4000/MIPS III): [1:0] RM, [6:2] sticky
                // Flags, [11:7] Enables, [17:12] Cause (+ CE at 17), bit 23 =
                // FCC0, bits [31:25] = FCC[7:1], bit 24 = FS (flush-to-zero).
                let fcsr = exec.core.fpu_fcsr;
                let rm = fcsr & 0x3;
                let rm_name = ["RN (nearest-even)", "RZ (toward zero)", "RP (toward +inf)", "RM (toward -inf)"][rm as usize];
                let bits5 = |v: u32, base: u32| -> String {
                    // I U O Z V, low bit first
                    let names = ["I", "U", "O", "Z", "V"];
                    let set: Vec<&str> = names.iter().enumerate()
                        .filter(|&(k, _)| v & (1 << (base + k as u32)) != 0)
                        .map(|(_, n)| *n).collect();
                    if set.is_empty() { "-".to_string() } else { set.join(",") }
                };
                let fcc = {
                    let b0 = (fcsr >> 23) & 1;
                    let hi = (fcsr >> 25) & 0x7F;
                    (hi << 1) | b0
                };
                writeln!(writer, "Control Registers:").unwrap();
                writeln!(writer, "  FIR:  {:#010x}", exec.core.fpu_fir).unwrap();
                writeln!(writer, "  FCCR: {:#010x}", exec.core.fpu_fccr).unwrap();
                writeln!(writer, "  FEXR: {:#010x}", exec.core.fpu_fexr).unwrap();
                writeln!(writer, "  FENR: {:#010x}", exec.core.fpu_fenr).unwrap();
                writeln!(writer, "  FCSR: {:#010x}", fcsr).unwrap();
                writeln!(writer, "    RM      = {} ({})", rm, rm_name).unwrap();
                writeln!(writer, "    Cause   = {}  (CE={})", bits5(fcsr, 12), (fcsr >> 17) & 1).unwrap();
                writeln!(writer, "    Enables = {}", bits5(fcsr, 7)).unwrap();
                writeln!(writer, "    Flags   = {}  (sticky)", bits5(fcsr, 2)).unwrap();
                writeln!(writer, "    FS      = {}  (flush denormals to zero)", (fcsr >> 24) & 1).unwrap();
                writeln!(writer, "    FCC     = {:#04x}  (bits 7..0)", fcc).unwrap();
                Ok(())
            }
            "mem" | "m" | "memory" => {
                if actual_args.is_empty() { return Err("Usage: mem <addr> [count] [b|h|w|d]".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                let count: u64 = if actual_args.len() > 1 { actual_args[1].parse().unwrap_or(1) } else { 1 };
                let size: usize = match actual_args.get(2).copied() {
                    None | Some("w") | Some("word") => 4,
                    Some("b") | Some("byte") => 1,
                    Some("h") | Some("half") => 2,
                    Some("d") | Some("double") => 8,
                    Some(o) => return Err(format!("Invalid size {}. Use b, h, w, or d", o)),
                };
                for i in 0..count {
                    let curr_addr = addr.wrapping_add(i * size as u64);
                    match exec.debug_read(curr_addr, size) {
                        Ok(val) => writeln!(writer, "{:#018x}: {:#0w$x}", curr_addr, val, w = 2 + 2 * size).unwrap(),
                        Err(e) => writeln!(writer, "{:#018x}: Error {:?}", curr_addr, e).unwrap(),
                    }
                }
                Ok(())
            }
            "eframe" | "ef" => {
                // eframe <addr> [o32|64]: an IRIX exception frame (struct
                // eframe_s, sys/reg.h): 8-byte register slots; o32 kernels
                // (5.3, 6.5 on IP22) put 4 argument-save slots first. Then a
                // backtrace of the interrupted context from its EPC/sp/ra.
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let base = {
                    let symbols = symbols_arc.lock();
                    parse_cpu_arg(actual_args.first().ok_or("Usage: eframe <addr> [o32|64]")?, &exec.core, Some(&symbols))?
                };
                let wide = match actual_args.get(1).copied() {
                    Some("64") | Some("n64") => true,
                    Some("o32") | Some("32") => false,
                    None => exec.core.cp0_status & STATUS_KX != 0,
                    Some(o) => return Err(format!("unknown layout {} (o32 or 64)", o)),
                };
                let at = if wide { 0 } else { 4 };
                let names: [&str; 31] = if wide {
                    ["at", "v0", "v1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3",
                     "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"]
                } else {
                    ["at", "v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
                     "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"]
                };
                let slot = |exec: &mut MipsExecutor<T, C>, i: u64| -> Result<u64, String> {
                    let a = base.wrapping_add((at + i) * 8);
                    exec.debug_read(a, 8).map_err(|e| format!("cannot read frame slot at {:#x}: {:?}", a, e))
                };
                let mut regs = [0u64; 31];
                for (i, r) in regs.iter_mut().enumerate() {
                    *r = slot(&mut exec, i as u64)?;
                }
                let sr = slot(&mut exec, 31)?;
                let lo = slot(&mut exec, 32)?;
                let hi = slot(&mut exec, 33)?;
                let badvaddr = slot(&mut exec, 34)?;
                let cause = slot(&mut exec, 35)?;
                let epc = slot(&mut exec, 36)?;
                writeln!(writer, "eframe at {:#x} ({} layout)", base, if wide { "64-bit" } else { "o32" }).unwrap();
                for (i, name) in names.iter().enumerate() {
                    write!(writer, "{:>3}: {:#018x}  ", name, regs[i]).unwrap();
                    if i % 4 == 3 { writeln!(writer).unwrap(); }
                }
                writeln!(writer).unwrap();
                let symbols = symbols_arc.lock();
                writeln!(writer, "  sr:       {:#010x} ({})", sr as u32, decode_status(sr as u32)).unwrap();
                writeln!(writer, "  cause:    {:#010x} ({})", cause as u32, decode_cause(cause as u32)).unwrap();
                writeln!(writer, "  badvaddr: {:#018x}", badvaddr).unwrap();
                writeln!(writer, "  epc:      {:#018x}{}", epc, format_pc_symbol(epc, &symbols)).unwrap();
                writeln!(writer, "  ra:       {:#018x}{}", regs[30], format_pc_symbol(regs[30], &symbols)).unwrap();
                writeln!(writer, "  lo/hi:    {:#x} / {:#x}", lo, hi).unwrap();
                drop(symbols);
                writeln!(writer, "Interrupted context:").unwrap();
                write!(writer, "{}", exec.backtrace_from(epc, regs[28], regs[30], 20)).unwrap();
                Ok(())
            }
            "putbuf" | "dmesg" => {
                // putbuf [size]: the IRIX kernel's console message ring (what
                // savecore puts in the crash log): panic text you otherwise
                // only see on the screen. 6.5 has `char *putbuf` sized by
                // `putbufsz`; an older kernel with a plain array needs [size].
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let (buf_sym, ndx_sym, sz_sym) = {
                    let symbols = symbols_arc.lock();
                    (symbols.get_addr("putbuf"), symbols.get_addr("putbufndx"), symbols.get_addr("putbufsz"))
                };
                let buf_sym = buf_sym.ok_or("no `putbuf` symbol: loadsym the kernel's map first")?;
                let ndx_sym = ndx_sym.ok_or("no `putbufndx` symbol")?;
                let read32 = |exec: &mut MipsExecutor<T, C>, a: u64| -> Result<u64, String> {
                    exec.debug_read(a, 4).map_err(|e| format!("cannot read {:#x}: {:?}", a, e))
                };
                let ndx = read32(&mut exec, ndx_sym)?;
                let (buf, size) = match (actual_args.first().and_then(|a| a.parse::<u64>().ok()), sz_sym) {
                    (Some(n), _) => (buf_sym, n),
                    (None, Some(sz)) => {
                        let p = read32(&mut exec, buf_sym)?;
                        let p = if p & 0x8000_0000 != 0 { p | 0xFFFF_FFFF_0000_0000 } else { p };
                        (p, read32(&mut exec, sz)?)
                    }
                    (None, None) => return Err("no `putbufsz`: give the buffer size, `putbuf <size>`".to_string()),
                };
                if size == 0 || size > (1 << 20) {
                    return Err(format!("implausible putbuf size {}", size));
                }
                let start = ndx.saturating_sub(size);
                let mut text = Vec::with_capacity(size as usize);
                for i in start..ndx {
                    let a = buf.wrapping_add(i % size);
                    let b = exec.debug_read(a, 1).map_err(|e| format!("cannot read {:#x}: {:?}", a, e))? as u8;
                    if b != 0 { text.push(b); }
                }
                writeln!(writer, "{}", String::from_utf8_lossy(&text).replace('\r', "")).unwrap();
                Ok(())
            }
            "stack" => {
                let mut exec = self.try_lock_executor()?;
                let sp = exec.core.read_gpr(29);
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                
                let addr = if !actual_args.is_empty() {
                    parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?
                } else {
                    sp
                };
                
                let count = if actual_args.len() > 1 { actual_args[1].parse().unwrap_or(16) } else { 16 };
                
                for i in 0..count {
                    let curr_addr = addr.wrapping_add(i * 8); // 64-bit stack slots usually
                    match exec.debug_read(curr_addr, 8) {
                        Ok(val) => writeln!(writer, "{:016x}: {:016x}", curr_addr, val).unwrap(),
                        Err(_) => writeln!(writer, "{:016x}: ????????????????", curr_addr).unwrap(),
                    }
                }
                Ok(())
            }
            "bt" | "backtrace" => {
                writeln!(writer, "{}", self.try_lock_executor()?.backtrace(20)).unwrap();
                Ok(())
            }
            "mw" => {
                if actual_args.len() < 2 { return Err("Usage: mw <addr> <val> [size: b|h|w|d]".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                let val = u64::from_str_radix(actual_args[1].trim_start_matches("0x"), 16)
                    .or_else(|_| actual_args[1].parse::<u64>())
                    .map_err(|_| "Invalid value".to_string())?;
                
                let size: usize = if actual_args.len() > 2 {
                    match actual_args[2] {
                        "b" | "byte" => 1,
                        "h" | "half" => 2,
                        "w" | "word" => 4,
                        "d" | "double" => 8,
                        _ => return Err("Invalid size. Use b, h, w, or d".to_string()),
                    }
                } else {
                    4
                };

                let mask = full_mask_for_usize(size);

                match exec.debug_write(addr, val, size, mask) {
                    EXEC_COMPLETE => writeln!(writer, "Wrote {:x} to {:016x}", val, addr).unwrap(),
                    e => writeln!(writer, "Error writing to {:016x}: {:?}", addr, e).unwrap(),
                }
                Ok(())
            }
            "ms" => {
                if actual_args.is_empty() { return Err("Usage: ms <addr> [max_len]".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                let max_len = if actual_args.len() > 1 {
                    actual_args[1].parse::<usize>().unwrap_or(256)
                } else {
                    256
                };

                let mut bytes = Vec::new();
                let mut curr = addr;
                for _ in 0..max_len {
                    match exec.debug_read(curr, 1) {
                        Ok(val) => {
                            let b = val as u8;
                            if b == 0 { break; }
                            bytes.push(b);
                            curr = curr.wrapping_add(1);
                        }
                        Err(_) => break,
                    }
                }
                
                let s = String::from_utf8_lossy(&bytes);
                writeln!(writer, "{:016x}: \"{}\"", addr, s).unwrap();
                Ok(())
            }
            "dis" | "d" | "disasm" | "disassemble" => {
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let addr = if !actual_args.is_empty() {
                    parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?
                } else {
                    exec.core.pc
                };
                let count = if actual_args.len() > 1 { actual_args[1].parse().unwrap_or(1) } else { 1 };
                
                for i in 0..count {
                    let curr_addr = addr.wrapping_add(i * 4);
                    match exec.debug_fetch_instr(curr_addr) {
                        Ok(instr) => {
                            let sym_str = format_pc_symbol(curr_addr, &symbols);
                            writeln!(writer, "{:#018x}{}: {:#010x} {}", curr_addr, sym_str, instr, mips_dis::disassemble(instr, curr_addr, Some(&symbols))).unwrap()
                        },
                        Err(_) => writeln!(writer, "{:#018x}: Could not fetch", curr_addr).unwrap(),
                    }
                }
                Ok(())
            }
            "loadelf" => {
                if actual_args.is_empty() { return Err("Usage: loadelf <file>".to_string()); }
                write!(writer, "{}", self.load_elf(actual_args[0])?).unwrap();
                Ok(())
            }
            "loadbin" => {
                if actual_args.len() < 2 { return Err("Usage: loadbin <file> <addr>".to_string()); }
                let addr = {
                    let exec = self.try_lock_executor()?;
                    let symbols_arc = exec.symbols.clone();
                    let symbols = symbols_arc.lock();
                    parse_cpu_arg(actual_args[1], &exec.core, Some(&symbols))?
                };
                write!(writer, "{}", self.load_bin(actual_args[0], addr)?).unwrap();
                Ok(())
            }
            "jump" => {
                if actual_args.is_empty() { return Err("Usage: jump <addr>".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                exec.core.pc = addr;
                writeln!(writer, "PC set to {:016x}", addr).unwrap();
                Ok(())
            }
            "setreg" => {
                if actual_args.len() < 2 { return Err("Usage: setreg <reg> <value>".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let target = exp::parse_reg_target(actual_args[0])
                    .ok_or_else(|| format!("Unknown register: {}", actual_args[0]))?;
                let val = parse_cpu_arg(actual_args[1], &exec.core, Some(&symbols))?;
                exp::write_reg_target(&target, &mut exec.core, val);
                writeln!(writer, "{} = {:016x}", actual_args[0], val).unwrap();
                Ok(())
            }
            "translate" | "t" | "trans" => {
                if actual_args.is_empty() { return Err("Usage: translate <addr>".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                let tr = exec.debug_translate(addr);
                if tr.is_exception() {
                    writeln!(writer, "Exception(0x{:08x})", tr.status).unwrap();
                } else {
                    let c = match tr.status & 0x7 {
                        3 => "Cacheable", 5 => "CacheableCoherent", _ => "Uncached",
                    };
                    writeln!(writer, "Translated {{ phys_addr: 0x{:08x}, cache_attr: {} }}", tr.phys, c).unwrap();
                }
                Ok(())
            }
            "debug" => {
                if actual_args.is_empty() {
                    return Err("Usage: debug <on|off|file> [filename]".to_string());
                }
                match actual_args[0] {
                    "on" | "1" => {
                        self.debug.store(true, Ordering::Relaxed);
                        *self.trace_file.lock() = None;
                        writeln!(writer, "CPU debug enabled (console output)").unwrap();
                    }
                    "off" | "0" => {
                        self.debug.store(false, Ordering::Relaxed);
                        *self.trace_file.lock() = None;
                        writeln!(writer, "CPU debug disabled").unwrap();
                    }
                    "file" => {
                        // file <path>: trace to file only, no socket spam
                        let path = actual_args.get(1).ok_or("Usage: debug file <filename>")?;
                        match std::fs::File::create(path) {
                            Ok(f) => {
                                self.debug.store(false, Ordering::Relaxed);
                                *self.trace_file.lock() = Some(std::io::BufWriter::new(f));
                                writeln!(writer, "CPU trace -> {}", path).unwrap();
                            }
                            Err(e) => return Err(format!("Cannot open {}: {}", path, e)),
                        }
                    }
                    _ => return Err("Usage: debug <on|off|file> [filename]".to_string()),
                }
                Ok(())
            }
            "ex" | "exception" | "exc" => {
                if actual_args.len() < 2 {
                    return Err("Usage: exception <class|code|all> <on|off>".to_string());
                }
                let target = actual_args[0];
                let enable = match actual_args[1] {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("Usage: exception <class|code|all> <on|off>".to_string()),
                };

                let mut mask = self.exception_mask.load(Ordering::Relaxed);
                let set_bit = |m: &mut u32, bit: u32, val: bool| {
                    if val { *m |= 1 << bit; } else { *m &= !(1 << bit); }
                };

                match target {
                    "all" => mask = if enable { 0xFFFFFFFF } else { 0 },
                    "int" => set_bit(&mut mask, EXC_INT, enable),
                    "tlb" => {
                        set_bit(&mut mask, EXC_MOD, enable);
                        set_bit(&mut mask, EXC_TLBL, enable);
                        set_bit(&mut mask, EXC_TLBS, enable);
                    },
                    "addr" => {
                        set_bit(&mut mask, EXC_ADEL, enable);
                        set_bit(&mut mask, EXC_ADES, enable);
                    },
                    "bus" => {
                        set_bit(&mut mask, EXC_IBE, enable);
                        set_bit(&mut mask, EXC_DBE, enable);
                    },
                    "sys" => {
                        set_bit(&mut mask, EXC_SYS, enable);
                        set_bit(&mut mask, EXC_BP, enable);
                    },
                    "ri" => {
                        set_bit(&mut mask, EXC_RI, enable);
                        set_bit(&mut mask, EXC_CPU, enable);
                    },
                    "arith" => {
                        set_bit(&mut mask, EXC_OV, enable);
                        set_bit(&mut mask, EXC_TR, enable);
                        set_bit(&mut mask, EXC_FPE, enable);
                    },
                    "watch" => set_bit(&mut mask, EXC_WATCH, enable),
                    "vce" => {
                        set_bit(&mut mask, EXC_VCEI, enable);
                        set_bit(&mut mask, EXC_VCED, enable);
                    },
                    s => {
                        if let Ok(code) = s.parse::<u32>() {
                            if code < 32 {
                                set_bit(&mut mask, code, enable);
                            } else {
                                return Err("Invalid exception code".to_string());
                            }
                        } else {
                            return Err("Unknown exception class or code".to_string());
                        }
                    }
                }
                self.exception_mask.store(mask, Ordering::Relaxed);
                writeln!(writer, "Exception mask set to {:08x}", mask).unwrap();
                Ok(())
            }
            "undo" | "u" => {
                #[cfg(feature = "developer")]
                {
                // Handle both "cpu undo on|off|clear" and step-back "undo [count]"
                if !actual_args.is_empty() {
                    match actual_args[0] {
                        "on" | "1" if actual_args[0] == "on" || actual_args[0] == "1" => {
                            let mut exec = self.try_lock_executor()?;
                            exec.undo_buffer.enable();
                            writeln!(writer, "CPU undo buffer enabled").unwrap();
                            return Ok(());
                        }
                        "off" | "0" if actual_args[0] == "off" || actual_args[0] == "0" => {
                            let mut exec = self.try_lock_executor()?;
                            exec.undo_buffer.disable();
                            writeln!(writer, "CPU undo buffer disabled").unwrap();
                            return Ok(());
                        }
                        "clear" => {
                            let mut exec = self.try_lock_executor()?;
                            exec.undo_buffer.clear();
                            writeln!(writer, "CPU undo buffer cleared").unwrap();
                            return Ok(());
                        }
                        "resize" => {
                            let new_capacity: usize = actual_args.get(1)
                                .and_then(|s| s.parse().ok())
                                .ok_or("Usage: undo resize <capacity>")?;
                            let mut exec = self.try_lock_executor()?;
                            let old_capacity = exec.undo_buffer.capacity();
                            exec.undo_buffer.resize(new_capacity);
                            writeln!(writer, "CPU undo buffer capacity: {} -> {}", old_capacity, exec.undo_buffer.capacity()).unwrap();
                            return Ok(());
                        }
                        _ => {}
                    }
                }

                // If not a special command, treat as step-back count
                let count = if !actual_args.is_empty() {
                    actual_args[0].parse().unwrap_or(1)
                } else {
                    1
                };

                let mut exec = self.try_lock_executor()?;

                if !exec.undo_buffer.can_undo(count) {
                    return Err(format!("Cannot undo {} steps (only {} available)", count, exec.undo_buffer.count));
                }

                // Clone the snapshot from 'count' steps back to avoid borrow issues
                if let Some(snapshot) = exec.undo_buffer.get(count).cloned() {
                    // Restore CPU state
                    exec.restore_snapshot(&snapshot);

                    // Restore memory writes in reverse order
                    for mem_write in snapshot.memory_writes.iter().rev() {
                        let _ = exec.debug_write(mem_write.virt_addr, mem_write.old_value, mem_write.size, 0);
                    }

                    // Pop the consumed snapshots so subsequent undos go further back.
                    for _ in 0..count {
                        exec.undo_buffer.pop();
                    }

                    writeln!(writer, "Undid {} instruction(s), PC now at {:016x}", count, exec.core.pc).unwrap();
                    Ok(())
                } else {
                    Err(format!("Failed to retrieve undo snapshot"))
                }
                }
                #[cfg(not(feature = "developer"))]
                Err("undo requires a developer build".to_string())
            }
            "ll" => {
                #[cfg(feature = "llstats")]
                {
                    let mut exec = self.try_lock_executor()?;
                    match actual_args.first().copied() {
                        Some("clear") => {
                            for e in exec.ll_stats.iter_mut() {
                                *e = LlSiteStats { key: u64::MAX, ..Default::default() };
                            }
                            writeln!(writer, "ll stats cleared").unwrap();
                            return Ok(());
                        }
                        Some("stats") => {
                            if exec.ll_stats.iter().all(|e| e.key == u64::MAX) {
                                writeln!(writer, "ll stats: no LL/SC activity recorded").unwrap();
                                return Ok(());
                            }
                            // Worst max_run first: a kernel LL/SC loop that
                            // cannot make progress is the thing being hunted,
                            // and it sorts straight to the top.
                            let mut rows: Vec<(u64, LlSiteStats)> = exec.ll_stats.iter()
                                .filter(|e| e.key != u64::MAX)
                                .map(|e| (e.key, *e))
                                .collect();
                            // Lead with reservations that were never resolved:
                            // `ll - (ok + fail)` counts LLs whose SC never ran
                            // at all. That is the abandoned-reservation
                            // signature — a thread that took a reservation and
                            // then went to sleep without completing its
                            // LL/SC loop — and it is what a stuck kernel lock
                            // looks like from here. A row with ll == ok and no
                            // failures is a loop working perfectly and is the
                            // least interesting thing in the table, so sorting
                            // on max_run alone buried the signal.
                            let unresolved = |st: &LlSiteStats| st.ll.saturating_sub(st.ok + st.fail);
                            rows.sort_by(|a, b| unresolved(&b.1).cmp(&unresolved(&a.1))
                                .then(b.1.max_run.cmp(&a.1.max_run))
                                .then(b.1.fail.cmp(&a.1.fail))
                                .then(b.1.ll.cmp(&a.1.ll)));
                            let mut out = String::new();
                            out.push_str("ll stats — per-LL-address reservation histogram\n");
                            out.push_str("  (unresolved = LLs whose SC never ran: a reservation taken\n");
                            out.push_str("   and abandoned, i.e. the thread slept mid-loop. Sorted first.\n");
                            out.push_str("   max_run = longest run of consecutive SC failures; a couple is\n");
                            out.push_str("   normal, hundreds means a stuck lock.)\n\n");
                            out.push_str(&format!("{:>12}  {:>10} {:>10} {:>10} {:>8} {:>8} {:>10}\n",
                                "phys", "ll", "sc_ok", "sc_fail", "cur_run", "max_run", "unresolved"));
                            // Totals over EVERY row, not just the 40 printed —
                            // with ~1000 live addresses a truncated sum is
                            // actively misleading.
                            let mut tot_ll = 0u64; let mut tot_ok = 0u64; let mut tot_fail = 0u64;
                            let mut tot_unres = 0u64;
                            for (_, st) in rows.iter() {
                                tot_ll += st.ll; tot_ok += st.ok; tot_fail += st.fail;
                                tot_unres += unresolved(st);
                            }
                            for (lladdr, st) in rows.iter().take(40) {
                                // LLAddr holds phys >> 4.
                                out.push_str(&format!("{:>12}  {:>10} {:>10} {:>10} {:>8} {:>8} {:>10}\n",
                                    format!("{:#010x}", (*lladdr as u64) << 4),
                                    st.ll, st.ok, st.fail, st.cur_run, st.max_run,
                                    unresolved(st)));
                            }
                            if rows.len() > 40 {
                                out.push_str(&format!("  ... {} more addresses\n", rows.len() - 40));
                            }
                            out.push_str(&format!("\n  {} addresses (all rows); ll={} sc_ok={} sc_fail={} unresolved={}\n",
                                rows.len(), tot_ll, tot_ok, tot_fail, tot_unres));
                            write!(writer, "{}", out).unwrap();
                            return Ok(());
                        }
                        _ => return Err("Usage: ll [stats|clear]".to_string()),
                    }
                }
                #[cfg(not(feature = "llstats"))]
                Err("ll stats requires --features llstats".to_string())
            }
            "tlb" => {
                if actual_args.is_empty() { return Err("Usage: tlb <dump|trans|debug> ...".to_string()); }
                let exec = self.try_lock_executor()?;
                let tlb = &exec.tlb;

                match actual_args[0] {
                    "dump" => {
                        for i in 0..tlb.num_entries() {
                            let entry_str = tlb.format_entry(i);
                            if !entry_str.is_empty() {
                                writeln!(writer, "{}", entry_str).unwrap();
                            }
                        }
                    }
                    "trans" | "translate" => {
                        if actual_args.len() < 2 { return Err("Usage: tlb trans <vaddr> [asid]".to_string()); }
                        let vaddr = u64::from_str_radix(actual_args[1].trim_start_matches("0x"), 16)
                            .map_err(|_| "Invalid address".to_string())?;
                        let asid = if actual_args.len() > 2 {
                            u8::from_str_radix(actual_args[2].trim_start_matches("0x"), 16)
                                .map_err(|_| "Invalid ASID".to_string())?
                        } else {
                            0
                        };
                        writeln!(writer, "{}", tlb.debug_translate(vaddr, asid)).unwrap();
                    }
                    _ => return Err("Unknown TLB subcommand".to_string()),
                }
                Ok(())
            }
            #[cfg(feature = "jitv2")]
            "j2" => {
                // Kept in sync with the `("j2", ...)` entry in the command
                // help table above — this is the bare-`j2` usage line and is
                // a separate, hand-maintained string, so a new subcommand has
                // to be added in BOTH places or it stays undiscoverable from
                // one of them (`intrun`/`flushkeep`/`corpus`/`dumppcp` were
                // all missing here after being added only to the help table).
                if actual_args.is_empty() {
                    return Err("Usage: j2 <analyze <addr>|pcp [addr]|dumppcp [addr] [path]|corpus [dir]|status|inline|dispatch|fallback|inline_mem|memhelpers [on|off]|pagewb|instrs|threads|opt [none|speed]|intrun [N]|min-instrs|max-instrs|min-calls|lockstep|lstate [full] [N]|hugepages|flush|clear <paddr>|deny <paddr>|html [path]>".to_string());
                }
                // "flush" needs the CPU genuinely stopped, not just this
                // lock momentarily free — try_lock_executor() succeeding
                // only proves no one holds the lock *right now* (MipsCpu::step
                // releases it between batches), which isn't enough: a
                // manual flush must be certain nothing is mid-dispatch
                // through JIT-compiled code referencing the pool it's about
                // to clear. Rather than stop/start the CPU ourselves here
                // (this runs on the monitor console thread, not the CPU
                // thread — flush_from_cpu_thread's "as good as stopped"
                // reasoning doesn't apply), just require the developer to
                // have already run `stop` first, same as they'd need to for
                // `j2 pcp` to see anything meaningful anyway.
                if actual_args[0] == "flush" {
                    if self.is_running() {
                        return Err("j2 flush: stop the CPU first (`stop`) — flushing while it's running could race a concurrently-executing compiled function".to_string());
                    }
                    let mut exec = self.try_lock_executor()?;
                    exec.jitv2_flush_from_here(false);
                    // Mirrors jitv2_track_pcp's own post-flush recovery —
                    // self.pcp would otherwise dangle into the just-cleared
                    // pool.
                    exec.pcp = std::ptr::null_mut();
                    writeln!(writer, "jitv2: manual flush complete").unwrap();
                    return Ok(());
                }
                let mut exec = self.try_lock_executor()?;

                match actual_args[0] {
                    "inline" if self.executor.lock().core.smp => {
                        return Err("j2 inline: several CPUs share the compile pool, which needs its threads".to_string());
                    }
                    "inline" => {
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 inline compile: {}", if exec.jitv2_inline_compile { "on" } else { "off" }).unwrap();
                            }
                            Some("on") => {
                                #[cfg(feature = "lightning")]
                                return Err("j2 inline on: unavailable under `lightning` — compiles always go to the async queue, never inline".to_string());
                                // Switching TO inline: stop the async
                                // compile queue (if running) and build a
                                // fresh Codegen for Jitv2::codegen's idle
                                // slot, reusing the queue's own still-live
                                // shared arena rather than reserving a new
                                // one — no leaked reservation, and inline
                                // dispatch picks up right where the queue
                                // left off, address-space-wise. The queue's
                                // own (now-discarded) Codegen wrapper is
                                // simply dropped; its arena lives on via the
                                // new Codegen built on top of it below.
                                #[cfg(not(feature = "lightning"))]
                                {
                                    // Lock `jitv2` ONCE for both stop() and the
                                    // codegen hand-back. The previous form
                                    // `if let Some(cg) = exec.jitv2.lock().compile_queue.stop() { *exec.jitv2.lock()... }`
                                    // deadlocked: the first `.lock()` temporary
                                    // in the `if let` scrutinee lives until the
                                    // whole `if let` completes, so the inner
                                    // `exec.jitv2.lock()` re-locked the same
                                    // (non-reentrant) mutex — hanging the
                                    // console thread the instant `j2 inline on`
                                    // ran, before the CPU even started.
                                    {
                                        let mut jit = exec.jitv2.lock();
                                        let stopped = jit.compile_queue.stop();
                                        if let Some(old) = stopped.into_iter().next() {
                                            let (shared, state) = old.shared_arena();
                                            drop(old);
                                            *jit.codegen.lock() = Some(crate::cpu::jitv2::codegen::Codegen::new_with_shared_arena(shared, state));
                                        }
                                    }
                                    exec.jitv2_inline_compile = true;
                                    writeln!(writer, "j2 inline compile: on").unwrap();
                                }
                            }
                            Some("off") => {
                                #[cfg(feature = "jitv2_lockstep")]
                                return Err("j2 inline off: unavailable under `jitv2_lockstep` — every region must compile+run synchronously on this thread for per-instruction verification to fire; the async queue is deliberately left stopped (see MipsExecutor::jitv2_inline_compile's doc comment)".to_string());
                                // Switching TO threaded: reuse the inline
                                // slot's own still-live shared arena for the
                                // queue's new worker Codegen (built inside
                                // start() — see that method's own doc
                                // comment) rather than reserving a fresh
                                // one. The old Jitv2::codegen wrapper is
                                // simply dropped once its arena handle is
                                // extracted; the arena itself lives on.
                                #[cfg(not(feature = "jitv2_lockstep"))]
                                {
                                exec.jitv2_inline_compile = false;
                                let bus = exec.sysad.clone();
                                let old = exec.jitv2.lock().codegen.lock().take();
                                match old {
                                    Some(old) => {
                                        let (shared, state) = old.shared_arena();
                                        drop(old);
                                        let mut jit = exec.jitv2.lock();
                                        let stats = jit.stats.clone();
                                        jit.compile_queue.start_with_shared_arena(bus, stats, shared, state);
                                    }
                                    None => {
                                        return Err("j2 inline off: no Codegen available (already owned by a running compile queue?)".to_string());
                                    }
                                }
                                writeln!(writer, "j2 inline compile: off").unwrap();
                                }
                            }
                            Some(_) => return Err("Usage: j2 inline [on|off]".to_string()),
                        }
                    }
                    "dispatch" => {
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 dispatch: {}", if exec.jitv2_dispatch_enabled { "on" } else { "off" }).unwrap();
                            }
                            Some("on") => {
                                exec.jitv2_dispatch_enabled = true;
                                writeln!(writer, "j2 dispatch: on").unwrap();
                            }
                            Some("off") => {
                                #[cfg(feature = "lightning")]
                                return Err("j2 dispatch off: unavailable under `lightning` — JIT dispatch is always on".to_string());
                                #[cfg(not(feature = "lightning"))]
                                {
                                    exec.jitv2_dispatch_enabled = false;
                                    writeln!(writer, "j2 dispatch: off (real JIT dispatch gate skipped entirely, interpreter-only)").unwrap();
                                }
                            }
                            Some(_) => return Err("Usage: j2 dispatch [on|off]".to_string()),
                        }
                    }
                    "fallback" => {
                        // Interpreter-fallback region admission (analyzer's
                        // FALLBACK_ENABLED global): when on, an unsupported
                        // (Excluded) instruction is kept in a compiled region
                        // as a fallback head run through the interpreter,
                        // instead of ending the region. A process-global read
                        // by the analyzer at compile time — already-compiled
                        // regions are unaffected until re-compiled, so a `j2
                        // flush` (CPU stopped) is required for the change to
                        // take effect. Defaults off while the path is being
                        // validated against live boots.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 fallback: {}", if crate::cpu::jitv2::analyzer::fallback_enabled() { "on" } else { "off" }).unwrap();
                            }
                            Some(on @ ("on" | "off")) => {
                                crate::cpu::jitv2::analyzer::set_fallback_enabled(on == "on");
                                writeln!(writer, "j2 fallback: {} — run `j2 flush` (CPU stopped) for it to take effect on already-compiled regions", on).unwrap();
                            }
                            Some(_) => return Err("Usage: j2 fallback [on|off]".to_string()),
                        }
                    }
                    "entrypre" => {
                        // Entry-word interrupt preamble. Normally an
                        // externally-dispatched entry word bypasses its own
                        // preamble (`skip_entry_preamble`), so a pending
                        // interrupt is sampled one head later. Turning this on
                        // emits the check at the entry word too.
                        //
                        // Read by codegen at compile time, so a `j2 flush` (CPU
                        // stopped) is required for it to affect regions that are
                        // already compiled.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 entrypre: {}", if crate::cpu::jitv2::entry_preamble_forced() { "on" } else { "off" }).unwrap();
                            }
                            Some(on @ ("on" | "off")) => {
                                crate::cpu::jitv2::set_entry_preamble_forced(on == "on");
                                writeln!(writer, "j2 entrypre: {} — run `j2 flush` (CPU stopped) for it to take effect on already-compiled regions", on).unwrap();
                            }
                            Some(_) => return Err("Usage: j2 entrypre [on|off]".to_string()),
                        }
                    }
                    "inline_mem" => {
                        // Inline load/store fast path: compiled code performs
                        // the nutlb probe + L1-D tag check itself and reads or
                        // writes the line directly, instead of calling out to
                        // the Rust `read*_fn`/`write*_fn` wrappers on every
                        // access (docs/jit-inline-memory.md).
                        //
                        // Read by codegen at *emit* time, so — like `j2
                        // fallback` — already-compiled regions keep whatever
                        // they were built with until a `j2 flush` (CPU
                        // stopped). That is exactly what makes it usable for
                        // A/B benchmarking on a live guest: toggle, flush,
                        // measure, repeat, without rebuilding.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 inline_mem: {}",
                                    if crate::cpu::jitv2::codegen::inline_mem_enabled() { "on" } else { "off" }).unwrap();
                            }
                            Some(on @ ("on" | "off")) => {
                                // Under jitv2_lockstep the inline path is
                                // forced off for correctness (it bypasses the
                                // callout wrappers lockstep's load/store
                                // verification lives in — see
                                // `codegen::inline_mem_enabled`). Say so
                                // rather than accepting the toggle and
                                // silently ignoring it.
                                if cfg!(feature = "jitv2_lockstep") {
                                    writeln!(writer, "j2 inline_mem: forced off by jitv2_lockstep (the inline path bypasses the callout hooks lockstep verifies loads/stores through) — cannot be enabled in this build").unwrap();
                                } else {
                                    crate::cpu::jitv2::codegen::set_inline_mem_enabled(on == "on");
                                    writeln!(writer, "j2 inline_mem: {} — run `j2 flush` (CPU stopped) for it to take effect on already-compiled regions", on).unwrap();
                                }
                            }
                            Some(_) => return Err("Usage: j2 inline_mem [on|off]".to_string()),
                        }
                    }
                    "memhelpers" => {
                        // Shared memory-helper calls vs inline TLB/cache guards:
                        // when enabled, load/store instructions emit a call to
                        // a shared helper function instead of inlining the ~50
                        // IR instructions for the probe.
                        //
                        // Read at compile time, so a `j2 flush` (CPU stopped)
                        // is required for it to take effect on already-compiled
                        // regions.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 memhelpers: {}",
                                    if crate::cpu::jitv2::codegen::mem_helpers_enabled() { "on" } else { "off" }).unwrap();
                            }
                            Some(on @ ("on" | "off")) => {
                                crate::cpu::jitv2::codegen::set_mem_helpers_enabled(on == "on");
                                writeln!(writer, "j2 memhelpers: {} — run `j2 flush` (CPU stopped) for it to take effect on already-compiled regions", on).unwrap();
                            }
                            Some(_) => return Err("Usage: j2 memhelpers [on|off]".to_string()),
                        }
                    }
                    "pagewb" => {
                        // Diagnostic: write back dirty L1-D/L2 lines covering
                        // a physical code page the instant a fetch crosses
                        // onto it (jitv2_track_pcp), before the JIT can
                        // compile against it — chases stale-RAM-vs-cache
                        // divergence for relocated/self-modified code. Takes
                        // effect immediately (unlike `j2 fallback`/category
                        // gates, this isn't read at compile time — no `j2
                        // flush` needed). Defaults off: real per-page-crossing
                        // cost on every fetch miss while the CPU is running.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 pagewb: {}", if jitv2_pagewb_enabled() { "on" } else { "off" }).unwrap();
                            }
                            Some(on @ ("on" | "off")) => {
                                set_jitv2_pagewb_enabled(on == "on");
                                writeln!(writer, "j2 pagewb: {}", on).unwrap();
                            }
                            Some(_) => return Err("Usage: j2 pagewb [on|off]".to_string()),
                        }
                    }
                    cat @ ("alu" | "fpu" | "branch" | "loadstore" | "cop0") => {
                        // Category gate: bulk on/off over opcode_support's per-InstrKind
                        // runtime enable table (ENABLED) — every InstrKind whose category()
                        // intersects this one gets set together. Same underlying table a
                        // future `j2 instr <name> [on|off]` will flip one bit of; disabling
                        // a whole category and then re-enabling one instruction under test
                        // composes naturally since there's only ever one table. Same
                        // process-global-read-at-compile-time contract as `j2 fallback`:
                        // already-compiled regions are unaffected until a `j2 flush`. Useful
                        // for bisecting a live-boot divergence into one functional unit
                        // before narrowing further.
                        let bit = match cat {
                            "alu" => crate::cpu::mips_instr_stats::InstrCategory::ALU,
                            "fpu" => crate::cpu::mips_instr_stats::InstrCategory::FPU,
                            "branch" => crate::cpu::mips_instr_stats::InstrCategory::BRANCH,
                            "loadstore" => crate::cpu::mips_instr_stats::InstrCategory::LOADSTORE,
                            "cop0" => crate::cpu::mips_instr_stats::InstrCategory::COP0,
                            _ => unreachable!(),
                        };
                        match actual_args.get(1).copied() {
                            None => {
                                let on = crate::cpu::jitv2::opcode_support::category_enabled(bit);
                                writeln!(writer, "j2 {}: {}", cat, if on { "on" } else { "off" }).unwrap();
                            }
                            Some(on @ ("on" | "off")) => {
                                crate::cpu::jitv2::opcode_support::set_category_enabled(bit, on == "on");
                                writeln!(writer, "j2 {}: {} — run `j2 flush` (CPU stopped) for it to take effect on already-compiled regions", cat, on).unwrap();
                            }
                            Some(_) => return Err(format!("Usage: j2 {} [on|off]", cat)),
                        }
                    }
                    "instrs" => {
                        // Inspection dump for the per-instruction toggle table above —
                        // without this there's no way to see which instructions are
                        // currently enabled/disabled short of reading the source, which
                        // defeats the point of a runtime bisection tool.
                        let filter = match actual_args.get(1).copied() {
                            None => None,
                            Some("alu") => Some(crate::cpu::mips_instr_stats::InstrCategory::ALU),
                            Some("fpu") => Some(crate::cpu::mips_instr_stats::InstrCategory::FPU),
                            Some("branch") => Some(crate::cpu::mips_instr_stats::InstrCategory::BRANCH),
                            Some("loadstore") => Some(crate::cpu::mips_instr_stats::InstrCategory::LOADSTORE),
                            Some("cop0") => Some(crate::cpu::mips_instr_stats::InstrCategory::COP0),
                            Some(_) => return Err("Usage: j2 instrs [alu|fpu|branch|loadstore|cop0]".to_string()),
                        };
                        let _ = crate::cpu::jitv2::opcode_support::write_status(&mut writer, filter);
                    }
                    "threads" => {
                        // Read-only: thread_count is fixed at process
                        // startup (iris.toml's jitv2.threads or
                        // --jitv2-threads), never mutable at runtime — see
                        // CompileQueue::set_thread_count's own doc comment.
                        if actual_args.len() > 1 {
                            return Err("j2 threads is read-only — set via iris.toml [jitv2] threads or --jitv2-threads at startup".to_string());
                        }
                        let jit = exec.jitv2.lock();
                        writeln!(writer, "j2 threads: {}", jit.compile_queue.thread_count()).unwrap();
                    }
                    "opt" => {
                        // Process-wide, not per-Codegen (opt_level is baked
                        // into the ISA Flags at JITModule-construction time)
                        // — only takes effect on the next Codegen::new()/
                        // reset() (a flush), not retroactively for functions
                        // already compiled. See Codegen::set_opt_level_speed's
                        // own doc comment.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 opt: {}", if crate::cpu::jitv2::codegen::Codegen::opt_level_speed() { "speed" } else { "none" }).unwrap();
                            }
                            Some("speed") => {
                                crate::cpu::jitv2::codegen::Codegen::set_opt_level_speed(true);
                                writeln!(writer, "j2 opt: speed (takes effect on the next flush/reset)").unwrap();
                            }
                            Some("none") => {
                                crate::cpu::jitv2::codegen::Codegen::set_opt_level_speed(false);
                                writeln!(writer, "j2 opt: none (takes effect on the next flush/reset)").unwrap();
                            }
                            Some(_) => return Err("Usage: j2 opt [none|speed]".to_string()),
                        }
                    }
                    "min-instrs" => {
                        // Applies to future compile decisions only — an
                        // offset already denylisted for being too short
                        // under the old threshold stays denylisted until its
                        // page's next gen bump (same as every other sticky
                        // rejection, §6.4); lowering the threshold doesn't
                        // retroactively un-denylist anything.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 min-instrs: {}", crate::cpu::jitv2::comp::min_instrs_to_compile()).unwrap();
                            }
                            Some(n) => match n.parse::<usize>() {
                                Ok(n) => {
                                    crate::cpu::jitv2::comp::set_min_instrs_to_compile(n);
                                    writeln!(writer, "j2 min-instrs: {}", crate::cpu::jitv2::comp::min_instrs_to_compile()).unwrap();
                                }
                                Err(_) => return Err("Usage: j2 min-instrs [N]".to_string()),
                            },
                        }
                    }
                    "max-instrs" => {
                        // Same sticky-denylist caveat as min-instrs: applies
                        // to future compile decisions only, doesn't
                        // retroactively affect anything already compiled or
                        // denylisted under the old budget.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 max-instrs: {}", crate::cpu::jitv2::comp::max_instrs_per_compile()).unwrap();
                            }
                            Some(n) => match n.parse::<usize>() {
                                Ok(n) => {
                                    crate::cpu::jitv2::comp::set_max_instrs_per_compile(n);
                                    writeln!(writer, "j2 max-instrs: {}", crate::cpu::jitv2::comp::max_instrs_per_compile()).unwrap();
                                }
                                Err(_) => return Err("Usage: j2 max-instrs [N]".to_string()),
                            },
                        }
                    }

                    "intrun" => {
                        // How many consecutive instructions share one
                        // pending-interrupt check. See
                        // `CODEGEN_INTERRUPT_RUN` for why this is the knob
                        // that controls GPR store->load forwarding in emitted
                        // code, and what raising it costs (interrupt-sampling
                        // latency, bounded by this value).
                        //
                        // Honoured under `developer` too (that build exists
                        // to debug and test things). Forced to 1 only under
                        // jitv2_lockstep, which verifies the unmodified
                        // per-instruction emission — see
                        // `Codegen::interrupt_run`.
                        // Report `opt_level` alongside the value, always:
                        // this knob only removes an alias-analysis barrier,
                        // and at `opt_level=none` Cranelift runs no
                        // store-to-load forwarding for it to unblock, so
                        // raising intrun legitimately changes almost nothing
                        // in emitted code. That looks exactly like a broken
                        // knob unless the opt level is visible next to it.
                        // `developer` defaults to `none`
                        // (CODEGEN_OPT_LEVEL_SPEED) but can run `speed` like
                        // any other build — hence a note pointing at how,
                        // not a claim that the combination is unavailable.
                        let opt = if crate::cpu::jitv2::codegen::Codegen::opt_level_speed() { "speed" } else { "none" };
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 intrun: {} (max {}, opt_level={})",
                                    crate::cpu::jitv2::codegen::Codegen::interrupt_run(),
                                    crate::cpu::jitv2::codegen::MAX_INTERRUPT_RUN, opt).unwrap();
                                if !crate::cpu::jitv2::codegen::Codegen::opt_level_speed() {
                                    writeln!(writer, "  note: opt_level=none — Cranelift runs no store-to-load forwarding, so this knob has little effect on emitted code. Set IRIS_OPT_SPEED=1 at launch, or `j2 opt speed` then stop + `j2 flush` to rebuild.").unwrap();
                                }
                            }
                            Some(n) => match n.parse::<u32>() {
                                Ok(n) if n >= 1 => {
                                    crate::cpu::jitv2::codegen::Codegen::set_interrupt_run(n);
                                    writeln!(writer, "j2 intrun: {} (max {}, opt_level={})",
                                        crate::cpu::jitv2::codegen::Codegen::interrupt_run(),
                                        crate::cpu::jitv2::codegen::MAX_INTERRUPT_RUN, opt).unwrap();
                                    if !crate::cpu::jitv2::codegen::Codegen::opt_level_speed() {
                                        writeln!(writer, "  note: opt_level=none — Cranelift runs no store-to-load forwarding, so this knob has little effect on emitted code. Set IRIS_OPT_SPEED=1 at launch, or `j2 opt speed` then stop + `j2 flush` to rebuild.").unwrap();
                                    }
                                    if cfg!(feature = "jitv2_lockstep") {
                                        writeln!(writer, "  (forced to 1: jitv2_lockstep verifies the unmodified per-instruction emission)").unwrap();
                                    }
                                    // Already-compiled regions baked the old
                                    // policy into their emitted code. Not
                                    // flushed here: `j2 flush` requires the
                                    // CPU stopped (it would otherwise race a
                                    // concurrently-executing compiled
                                    // function), and this command does not.
                                    // Same "takes effect on the next relevant
                                    // event" contract as `j2 opt`.
                                    writeln!(writer, "  applies to code compiled from now on; run `stop` then `j2 flush` to rebuild what already exists").unwrap();
                                }
                                _ => return Err(format!("Usage: j2 intrun [1..{}]", crate::cpu::jitv2::codegen::MAX_INTERRUPT_RUN)),
                            },
                        }
                    }
                    "min-calls" => {
                        // Only applies to exec_decoded's real dispatch gate,
                        // on the async (non-inline) compile path — see
                        // count_dispatch_and_check_threshold's own doc
                        // comment for why jitv2_inline_compile/tests/lockstep
                        // are exempt by construction, not by this value.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 min-calls: {}", crate::cpu::jitv2::min_calls_before_compile()).unwrap();
                            }
                            Some(n) => match n.parse::<u64>() {
                                Ok(n) => {
                                    crate::cpu::jitv2::set_min_calls_before_compile(n);
                                    writeln!(writer, "j2 min-calls: {}", crate::cpu::jitv2::min_calls_before_compile()).unwrap();
                                }
                                Err(_) => return Err("Usage: j2 min-calls [N]".to_string()),
                            },
                        }
                    }
                    #[cfg(feature = "jitv2_lockstep")]
                    "lockstep" => {
                        // Lockstep is not a runtime toggle anymore: if the build
                        // has the jitv2_lockstep feature, every verifiable
                        // (straight-line ALU/load-store/CP1) JIT instruction
                        // self-checks against the interpreter, always — it's
                        // driven from the JIT block itself. Kept as an
                        // informational command.
                        #[cfg(feature = "jitv2_lockstep")]
                        writeln!(writer, "j2 lockstep: always on (built with jitv2_lockstep). Straight-line ALU/load-store/CP1 instructions self-check every dispatch; branch/jump not yet verified.").unwrap();
                        #[cfg(not(feature = "jitv2_lockstep"))]
                        writeln!(writer, "j2 lockstep: not built (rebuild with the jitv2_lockstep feature to enable).").unwrap();
                    }
                    #[cfg(feature = "jitv2_lockstep")]
                    "lstate" => {
                        // j2 lstate [full] [N]: dump the rolling lockstep step
                        // history (MipsExecutor::ls_history). The state shown
                        // for each instruction is what ENTERED it — i.e. the
                        // previous step's final state — which is exactly the
                        // context the DIVERGENCE report drops (lockstep_compare
                        // keeps only ls_before's pc/slot fields, not the
                        // register file). After a divergence bail the last
                        // entry is the inputs to the instruction that diverged.
                        let hist = &exec.ls_history;
                        if hist.is_empty() {
                            writeln!(writer, "j2 lstate: lockstep history is empty (no bracketed instruction has stepped yet)").unwrap();
                        } else {
                            let (full, count) = match (actual_args.get(1).copied(), actual_args.get(2).copied()) {
                                (None, _) => (false, 8usize),
                                (Some("full"), None) => (true, 1usize),
                                (Some("full"), Some(n)) => (true, n.parse().map_err(|_| "Usage: j2 lstate [full] [N]".to_string())?),
                                (Some(n), _) => (false, n.parse().map_err(|_| "Usage: j2 lstate [full] [N]".to_string())?),
                            };
                            let count = count.clamp(1, hist.len());
                            let symbols_arc = exec.symbols.clone();
                            let symbols = symbols_arc.lock();
                            writeln!(writer, "j2 lstate: {} of {} step(s), oldest first; register state is what ENTERED each instruction", count, hist.len()).unwrap();
                            let start = hist.len() - count;
                            for (i, e) in hist.iter().skip(start).enumerate() {
                                let bd = if e.bd == LOCKSTEP_BD_LIVE { "  [bd:live]" } else if e.bd != 0 { "  [bd]" } else { "" };
                                writeln!(writer, "  #{:<3} {:#018x}: {:08x}  {}{}", start + i, e.pc, e.raw,
                                    crate::cpu::mips_dis::disassemble(e.raw, e.pc, Some(&symbols)), bd).unwrap();
                                if full {
                                    let s = &e.before;
                                    for row in 0..8 {
                                        let mut line = String::new();
                                        for col in 0..4 {
                                            let r = row * 4 + col;
                                            line.push_str(&format!("  {:>4}={:016x}", crate::cpu::mips_dis::reg_name(r as u32), s.gpr[r]));
                                        }
                                        writeln!(writer, "  {}", line).unwrap();
                                    }
                                    writeln!(writer, "      hi={:016x}  lo={:016x}  pc={:016x}", s.hi, s.lo, s.pc).unwrap();
                                    writeln!(writer, "      status={:08x} (FR={})  cause={:08x}  epc={:016x}  fcsr={:08x}",
                                        s.status, (s.status & crate::cpu::mips_core::STATUS_FR != 0) as u8, s.cause, s.epc, s.fcsr).unwrap();
                                    for row in 0..8 {
                                        let mut line = String::new();
                                        for col in 0..4 {
                                            let r = row * 4 + col;
                                            line.push_str(&format!("  f{:<2}={:016x}", r, s.fpr[r]));
                                        }
                                        writeln!(writer, "  {}", line).unwrap();
                                    }
                                }
                            }
                        }
                    }
                    "analyze" => {
                        // Runs the REAL analyzer walk (Analyzer::walk_bounded,
                        // the same one handle_request uses to decide what a
                        // region actually contains) against live memory, and
                        // prints every visited word with its disassembly and
                        // classification — the static-analysis counterpart to
                        // `j2 pcp` (which only shows what's already been
                        // compiled/published). Built for exactly the kind of
                        // question this session kept needing answered ad hoc
                        // via temporary eprintln instrumentation: "starting
                        // from this exact word, what does the walk include,
                        // and why does it stop where it does" — now a
                        // permanent, no-rebuild-required tool.
                        //
                        // Takes a PHYSICAL address, not virtual: translation
                        // isn't always available (CPU stopped mid-exception,
                        // page not currently TLB-mapped, or the address came
                        // from a log/j2 pcp dump that already gives phys) —
                        // reading via `sysad.read32` directly (the same
                        // primitive `handle_request` itself uses) sidesteps
                        // needing a live translation at all.
                        if actual_args.len() < 2 {
                            return Err("Usage: j2 analyze <phys_addr>".to_string());
                        }
                        let symbols_arc = exec.symbols.clone();
                        let symbols = symbols_arc.lock();
                        let phys_addr = monitor_phys(parse_cpu_arg(actual_args[1], &exec.core, Some(&symbols))?);
                        let entry_word = ((phys_addr & 0xFFF) >> 2) as u16;
                        let page_base_phys = phys_addr & !0xFFF;

                        let mut words = [0u32; crate::cpu::jitv2::ENTRIES_PER_PAGE];
                        for (i, w) in words.iter_mut().enumerate() {
                            let a = page_base_phys + (i as u64) * 4;
                            let r = exec.sysad.read32(a);
                            if !r.is_ok() {
                                return Err(format!("failed to read word at phys {:#010x} (offset {:#x} into page)", a, i * 4));
                            }
                            *w = r.data;
                        }

                        let (walked, non_empty) = exec.jitv2_inline_analyzer.walk_bounded(&words, entry_word, page_base_phys as u32, usize::MAX);
                        writeln!(writer, "analyze phys={:#010x}: page_base_phys={:#010x} entry_page_off={:#05x}", phys_addr, page_base_phys, entry_word * 4).unwrap();
                        if !non_empty {
                            writeln!(writer, "  region is EMPTY — entry word itself is Excluded with no emitter and fallback is off, or otherwise undispatchable").unwrap();
                        } else {
                            let visited: Vec<_> = crate::cpu::jitv2::analyzer::instrs_linear(walked).collect();
                            writeln!(writer, "  {} instructions visited:", visited.len()).unwrap();
                            for instr in &visited {
                                let paddr = page_base_phys + (instr.word as u64) * 4;
                                let dis = mips_dis::disassemble(instr.raw, paddr, Some(&symbols));
                                let mut tags = Vec::new();
                                if instr.word == entry_word { tags.push("ENTRY"); }
                                if instr.is_fallback { tags.push("fallback"); }
                                if instr.is_branch_fallback_successor { tags.push("branch-fallback-successor"); }
                                if instr.is_slot_only { tags.push("slot-only"); }
                                let tag_str = if tags.is_empty() { String::new() } else { format!(" [{}]", tags.join(",")) };
                                let exits = match (instr.fallthrough_exit, instr.taken_exit) {
                                    (None, None) => String::new(),
                                    (ft, tk) => format!(" (fallthrough_exit={:?} taken_exit={:?})", ft, tk),
                                };
                                writeln!(writer, "    page_off={:#05x} phys={:#010x}: {:08x} {}{}{}", instr.word * 4, paddr, instr.raw, dis, tag_str, exits).unwrap();
                            }
                        }
                    }
                    // Available under both `comp.rs` implementations: the dump
                    // format is implementation-agnostic (the default impl
                    // reconstructs the bitmaps from its per-`JitEntry` state
                    // behind `PhysicalCodePage::dump_*`).
                    "dumppcp" => {
                        // Capture a PhysicalCodePage + its raw 4KB of memory to
                        // a file for offline analysis with the `jitv2-pcp-dump`
                        // binary (which re-runs the real reachability walker and
                        // disassembles every requested/compiled/denylisted
                        // offset). `PcpDump` already had capture/to_bytes/
                        // from_bytes and a whole offline analyzer built on it —
                        // there was simply no command to trigger a capture, so
                        // the analyzer had nothing to read.
                        //
                        // Usage: j2 dumppcp [addr] [path]
                        //   addr — page to dump; defaults to the tracked page
                        //          (or the page PC translates to).
                        //   path — output file; defaults to
                        //          `jitv2_pcp_<pfn>.bin` in the cwd.
                        let explicit_addr = if let Some(&a) = actual_args.get(1) {
                            if a.starts_with("./") || a.contains('/') || a.ends_with(".bin") {
                                None // it's a path, not an address
                            } else {
                                let symbols_arc = exec.symbols.clone();
                                let symbols = symbols_arc.lock();
                                Some(parse_cpu_arg(a, &exec.core, Some(&symbols))?)
                            }
                        } else { None };

                        let page_ptr = if explicit_addr.is_none() && !exec.pcp.is_null() {
                            exec.pcp
                        } else {
                            let pc = explicit_addr.unwrap_or(exec.core.pc);
                            let result = exec.debug_translate(pc);
                            if result.is_exception() {
                                return Err(format!("{:#018x} doesn't translate", pc));
                            }
                            let Some(pfn) = exec.jitv2_frame(result.phys) else {
                                return Err(format!("phys {:#x} is above the JIT's physical range", result.phys));
                            };
                            let page_base = pfn as u64 * crate::cpu::jitv2::PAGE_SIZE as u64;
                            let sysad = exec.sysad.clone();
                            let mut jit = exec.jitv2.lock();
                            let fr1 = (exec.core.cp0_status & crate::cpu::mips_core::STATUS_FR) != 0;
                            let slot = jit.page_for(pfn, page_base, sysad.as_ref(), fr1);
                            match slot {
                                Some(slot) => jit.page_ptr(slot),
                                None => return Err("page pool full".to_string()),
                            }
                        };
                        let page = unsafe { &*page_ptr };
                        let sysad = exec.sysad.clone();
                        let dump = crate::cpu::jitv2::pcp_dump::PcpDump::capture(page, sysad.as_ref())
                            .map_err(|st| format!("capture failed: bus status {:#x}", st))?;
                        // A path argument is whichever arg is not the address.
                        let path = actual_args.iter().skip(1)
                            .find(|a| a.contains('/') || a.ends_with(".bin"))
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| crate::cpu::jitv2::pcp_dump::default_dump_path(dump.pfn));
                        std::fs::write(&path, dump.to_bytes())
                            .map_err(|e| format!("write {}: {}", path, e))?;
                        writeln!(writer, "wrote {} (pfn {:#x}, gen {} entry_gen {}, fr1={})",
                            path, dump.pfn, dump.current_gen, dump.entry_gen, dump.fr1).unwrap();
                        writeln!(writer, "analyse with: jitv2-pcp-dump {} [--offset <hex>] [--compile]", path).unwrap();
                    }
                    // Bulk corpus capture, replacing the old
                    // `jitv2_corpus_dump` Cargo feature (see
                    // `jitv2/pcp_dump.rs`'s module doc for why that one went
                    // away). Everything a corpus needs is already sitting in
                    // the live pcp cache after a boot — every page the JIT
                    // touched, with its entry bitmaps — so this just walks the
                    // pool and writes each claimed page out in the same `.pcp`
                    // format `dumppcp` uses. No feature flag, so a corpus can
                    // be taken from the production-shaped build whose emitted
                    // code is the thing worth measuring.
                    "corpus" => {
                        // Usage: j2 corpus [dir]   (default: jitv2_corpus)
                        let dir = actual_args.get(1).copied().unwrap_or(crate::cpu::jitv2::pcp_dump::CORPUS_DIR);
                        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {}", dir, e))?;

                        // Capture under the lock into memory first, then write
                        // outside it: `capture` reads 1024 words per page off
                        // the bus, and holding the jitv2 lock across a few
                        // thousand of those plus the filesystem would stall
                        // every dispatching CPU thread for the whole dump.
                        let sysad = exec.sysad.clone();
                        let mut unreadable = 0usize;
                        let dumps: Vec<crate::cpu::jitv2::pcp_dump::PcpDump> = {
                            let jit = exec.jitv2.lock();
                            let mut v = Vec::new();
                            for page in jit.claimed_pages() {
                                // A claimed page whose memory won't read right
                                // now (device unmapped under it, say) is skipped
                                // rather than failing the whole dump — one bad
                                // page shouldn't cost the other few thousand.
                                match crate::cpu::jitv2::pcp_dump::PcpDump::capture(page, sysad.as_ref()) {
                                    Ok(d) => v.push(d),
                                    Err(_) => unreadable += 1,
                                }
                            }
                            v
                        };

                        let mut written = 0usize;
                        let mut failed = 0usize;
                        let mut total_calls = 0u64;
                        for dump in &dumps {
                            // pfn-named, not timestamped like `default_dump_path`:
                            // a corpus is a set keyed by page, and re-running
                            // `j2 corpus` into the same directory should refresh
                            // it rather than pile up a second copy of every page.
                            let path = std::path::Path::new(dir)
                                .join(format!("pcp_{:08x}.pcp", dump.pfn));
                            match std::fs::write(&path, dump.to_bytes()) {
                                Ok(()) => {
                                    written += 1;
                                    total_calls = total_calls.saturating_add(dump.call_count);
                                }
                                Err(e) => {
                                    if failed == 0 {
                                        writeln!(writer, "j2 corpus: write {}: {}", path.display(), e).unwrap();
                                    }
                                    failed += 1;
                                }
                            }
                        }
                        writeln!(writer, "j2 corpus: wrote {} page(s) to {}/ ({} unreadable, {} write failure(s))",
                            written, dir, unreadable, failed).unwrap();
                        if total_calls > 0 {
                            writeln!(writer, "  total recorded dispatches across corpus: {}", total_calls).unwrap();
                        }
                        writeln!(writer, "  measure with: IRIS_CORPUS_DIR={} IRIS_OPT_SPEED=1 cargo test --release --features jitv2 zz_corpus_sizes -- --nocapture", dir).unwrap();
                    }
                    "pcp" => {
                        // Optional address argument: inspect an arbitrary
                        // page's compile state (e.g. "why does this function
                        // never show up as JIT-tagged in dt") instead of only
                        // whatever page execution is currently stopped on —
                        // previously silently ignored (this whole command
                        // took no arguments at all), which made it impossible
                        // to check a page other than the live one.
                        // actual_args[0] is "pcp" itself here (this arm is
                        // already inside `match actual_args[0]`) — the
                        // address, if given, is actual_args[1]. Passing [0]
                        // here fed the literal string "pcp" to parse_cpu_arg,
                        // which tried (and failed) to resolve it as a symbol
                        // name — "Symbol not found: pcp" — found live.
                        let explicit_addr = if let Some(&addr_arg) = actual_args.get(1) {
                            let symbols_arc = exec.symbols.clone();
                            let symbols = symbols_arc.lock();
                            Some(parse_cpu_arg(addr_arg, &exec.core, Some(&symbols))?)
                        } else {
                            None
                        };
                        // `self.pcp` is null whenever the CPU is stopped
                        // (`MipsCpu::stop()`'s `nanotlb_invalidate()` — see
                        // its own doc comment for why that invariant exists)
                        // — which is exactly when a developer is most likely
                        // to run this command. Rather than just reporting
                        // "nothing tracked," re-derive the page the same way
                        // a real fetch would: translate the target PC with
                        // `debug_translate` (side-effect-free — no TLB/nanotlb
                        // state touched) and look it up via `page_for`, which
                        // finds the existing pool entry if this PFN was ever
                        // seen before (the overwhelmingly common case here —
                        // we're inspecting whatever page execution just
                        // stopped on, or an explicitly requested one) without
                        // disturbing `self.pcp` itself. An explicit address
                        // always re-derives (never trusts self.pcp, which
                        // tracks a different page whenever one was asked for).
                        let page_ptr = if explicit_addr.is_none() && !exec.pcp.is_null() {
                            exec.pcp
                        } else {
                            let pc = explicit_addr.unwrap_or(exec.core.pc);
                            let result = exec.debug_translate(pc);
                            if result.is_exception() {
                                writeln!(writer, "No PhysicalCodePage tracked (self.pcp is null) and PC {:#018x} doesn't translate.", pc).unwrap();
                                return Ok(());
                            }
                            let Some(pfn) = exec.jitv2_frame(result.phys) else {
                                writeln!(writer, "No PhysicalCodePage tracked (self.pcp is null) and PC {:#018x} is at phys {:#x}, above the JIT's physical range.", pc, result.phys).unwrap();
                                return Ok(());
                            };
                            let page_base = pfn as u64 * crate::cpu::jitv2::PAGE_SIZE as u64;
                            let sysad = exec.sysad.clone();
                            let fr1 = (exec.core.cp0_status & crate::cpu::mips_core::STATUS_FR) != 0;
                            let mut jit = exec.jitv2.lock();
                            match jit.page_for(pfn, page_base, sysad.as_ref(), fr1) {
                                Some(slot) => jit.page_ptr(slot),
                                None => {
                                    writeln!(writer, "No PhysicalCodePage tracked (self.pcp is null) and the page pool is full (can't even allocate a fresh lookup entry).").unwrap();
                                    return Ok(());
                                }
                            }
                        };
                        // Safety: try_lock_executor holds the same lock every
                        // mutator of jitv2/pcp takes, and pcp is only ever
                        // reassigned or nulled by the exec thread under that
                        // lock (jitv2_track_pcp, nanotlb_invalidate) — safe
                        // to dereference for the duration of this borrow.
                        // `page_ptr` (the re-derived case) points into the
                        // same `Jitv2::pages` pool under the same lock
                        // discipline, so the same reasoning applies to it.
                        let page = unsafe { &*page_ptr };
                        let pc = explicit_addr.unwrap_or(exec.core.pc);
                        let entry_offset = ((pc & 0xFFF) >> 2) as usize;

                        // §13: one function per page now, so "published"/
                        // "denylisted" are page-wide bitmap scans over
                        // `compiled`/`denied` rather than per-offset flags —
                        // no more per-entry func/gen/call_count/
                        // block_include_count (overlap between entries is
                        // structurally impossible under this design, so that
                        // diagnostic no longer applies).
                        writeln!(writer, "pfn={:#010x}  gen={}  entry_gen={}  scheduled(in-flight)={}  compiles_since_flush={}  rejected_compiles={}",
                            page.pfn, page.current_gen(), page.entry_gen(), page.is_scheduled(), page.compiles_since_flush(), page.rejected_compiles()).unwrap();
                        // §13: one function per page, so FR mode is a single
                        // whole-page value — pinned at page-claim time from the
                        // live STATUS_FR (PhysicalCodePage::fr1) and every
                        // FPR-access emitter in the one function bakes its
                        // register-packing scheme from it. emit_fr_mode_guard
                        // bails+kills an entry whose live STATUS_FR no longer
                        // matches this. " FR-MISMATCH" flags a standing
                        // disagreement with the live bit right now.
                        {
                            let page_fr1 = page.is_fr1();
                            let live_fr1 = (exec.core.cp0_status & crate::cpu::mips_core::STATUS_FR) != 0;
                            writeln!(writer, "  fr1 (pinned)={}  live STATUS_FR={}  (cp0_status={:#010x}){}",
                                page_fr1 as u8, live_fr1 as u8, exec.core.cp0_status,
                                if page_fr1 != live_fr1 { "  FR-MISMATCH" } else { "" }).unwrap();
                        }
                        #[cfg(feature = "developer")]
                        writeln!(writer, "  schedule_attempts={}  sends_attempted={}  sends_dropped_queue_full={}",
                            page.schedule_attempts(), page.sends_attempted(), page.sends_dropped_queue_full()).unwrap();
                        writeln!(writer, "  [prepare] bounced={}", page.prepare_bounced()).unwrap();
                        writeln!(writer, "  [analyze] rejected={}", page.analyze_rejected()).unwrap();
                        writeln!(writer, "  [codegen] rejected={} oom_bounced={}", page.codegen_rejected(), page.codegen_oom_bounced()).unwrap();
                        writeln!(writer, "  [finalize] failed={}", page.finalize_failed()).unwrap();
                        writeln!(writer, "  [seal] gap_blocked={}", page.seal_gap_blocked()).unwrap();
                        writeln!(writer, "  [publish] compiles_since_flush={}", page.compiles_since_flush()).unwrap();
                        // Churn avoidance: how often a compile request for
                        // this page was answered by re-validating the code
                        // already installed (its generation moved but every
                        // word the last compile decoded was unchanged), vs.
                        // how often it had a snapshot to compare against and
                        // genuinely had to recompile. A page with a large
                        // `skipped` is one whose gen churns for reasons
                        // unrelated to its code. Neither counts a request
                        // that had no snapshot to compare against at all.
                        // Both are since-last-flush, not lifetime.
                        writeln!(writer, "  [churn] redundant_skipped={} redundant_rejected={} (since last flush)",
                            page.redundant_skipped(), page.redundant_rejected()).unwrap();
                        writeln!(writer, "pc={:#018x}  page_off={:#05x}", pc, entry_offset * 4).unwrap();
                        writeln!(
                            writer,
                            "  entry_offset: requested={} published={} entry_valid={} denylisted={}",
                            page.is_requested(entry_offset),
                            page.is_published(entry_offset),
                            page.is_runnable(entry_offset),
                            page.is_denylisted(entry_offset),
                        ).unwrap();

                        let mut requested = 0usize;
                        let mut published = 0usize;
                        let mut denylisted = 0usize;
                        for off in 0..crate::cpu::jitv2::ENTRIES_PER_PAGE {
                            if page.is_requested(off) { requested += 1; }
                            if page.is_published(off) { published += 1; }
                            if page.is_denylisted(off) { denylisted += 1; }
                        }
                        writeln!(
                            writer,
                            "totals: {} / {} offsets requested, {} published, {} denylisted",
                            requested, crate::cpu::jitv2::ENTRIES_PER_PAGE, published, denylisted,
                        ).unwrap();

                        // One function serves the whole page — print its
                        // pointer unconditionally (even when nothing is
                        // published: a non-null `func` with 0 published
                        // entries is itself a diagnostic signal, e.g. a stale
                        // pointer left over from before a `kill()`-only
                        // decline), then list every published entry offset
                        // that dispatches into it (vaddr = vbase | offset*4).
                        let vbase = pc & !0xFFFu64;
                        let func = page.func();
                        #[cfg(feature = "developer")]
                        let dev_cols = format!(" instrs={} code_size={} calls={}",
                            page.instr_count.load(Ordering::Relaxed),
                            page.code_size.load(Ordering::Relaxed),
                            page.call_count.load(Ordering::Relaxed));
                        #[cfg(not(feature = "developer"))]
                        let dev_cols = String::new();
                        writeln!(writer, "  func={:#014x}{}", func as usize, dev_cols).unwrap();
                        // One row per offset that's flagged in ANY of the
                        // three bitmaps (requested/compiled/denylisted) —
                        // R/C/D shown together so a state that shouldn't
                        // coexist (e.g. compiled AND denylisted: a stale
                        // `compiled` bit denylist() alone didn't clear, the
                        // exact EXEC_FALLBACK-forever hazard `kill()` exists
                        // to prevent) is visible at a glance instead of
                        // needing three separate lists cross-referenced by
                        // hand. `page_off` is the byte offset (off*4),
                        // matching every other value this command prints —
                        // an earlier version of the denylisted-only list
                        // printed the bare word index instead, silently 4x
                        // too small (found live: reported 0x240 when the
                        // real byte offset, matching everything else this
                        // command and jitv2-pcp-dump print, was 0x900).
                        if requested > 0 || published > 0 || denylisted > 0 {
                            writeln!(writer, "  offsets (R=requested C=compiled D=denylisted):").unwrap();
                            for off in 0..crate::cpu::jitv2::ENTRIES_PER_PAGE {
                                let r = page.is_requested(off);
                                let c = page.is_published(off);
                                let d = page.is_denylisted(off);
                                if !r && !c && !d { continue; }
                                let vaddr = vbase | ((off as u64) * 4);
                                writeln!(writer, "    page_off={:#05x} vaddr={:#018x}  R={} C={} D={}",
                                    off * 4, vaddr, r, c, d).unwrap();
                            }
                        }
                    }
                    "clear" => {
                        // Reset the PhysicalCodePage covering `paddr` back to
                        // a freshly-claimed-but-never-compiled state —
                        // requested/compiled/denied bitmaps, the compiled
                        // func pointer, and every diagnostic counter all go
                        // back to default, in place. Keeps `pfn`/`gen`: this
                        // is NOT reset_to_unclaimed (that evicts the page
                        // from the pool entirely, changing its identity) —
                        // the page stays claimed under the same physical
                        // frame, it just looks like nothing was ever
                        // compiled for it. Doesn't require the CPU to be
                        // stopped first: reset_compiled_state only writes
                        // fields dispatch reads through
                        // is_runnable()/is_published(), which will simply see
                        // "unpublished" the instant this returns.
                        if exec.core.smp {
                            return Err("j2 clear: another CPU may be running this page; use j2 flush".to_string());
                        }
                        if actual_args.len() < 2 { return Err("Usage: j2 clear <paddr>".to_string()); }
                        let symbols_arc = exec.symbols.clone();
                        let symbols = symbols_arc.lock();
                        let paddr = monitor_phys(parse_cpu_arg(actual_args[1], &exec.core, Some(&symbols))?);
                        drop(symbols);
                        let pfn = exec.jitv2_monitor_pfn(paddr)?;
                        let page_base = pfn as u64 * crate::cpu::jitv2::PAGE_SIZE as u64;
                        let sysad = exec.sysad.clone();
                        let fr1 = (exec.core.cp0_status & crate::cpu::mips_core::STATUS_FR) != 0;
                        let mut jit = exec.jitv2.lock();
                        match jit.page_for(pfn, page_base, sysad.as_ref(), fr1) {
                            Some(slot) => {
                                let page_ptr = jit.page_ptr(slot);
                                unsafe { (*page_ptr).reset_compiled_state(); }
                                writeln!(writer, "j2 clear: page pfn={:#010x} (paddr {:#010x}) compiled state reset (pfn/gen kept)", pfn, page_base).unwrap();
                            }
                            None => return Err("j2 clear: page pool is full (can't even allocate a fresh lookup entry)".to_string()),
                        }
                    }
                    "deny" => {
                        // Sticky-denylist the single entry offset covering
                        // `paddr` — the page's other offsets are untouched.
                        // `denylist()` alone (clearing the `denied` bit) only
                        // blocks *future* compiles from covering this offset
                        // — it does NOT retroactively clear an already-set
                        // `compiled` bit, so an offset published before this
                        // call keeps dispatching through the page's existing
                        // `func` forever. `kill()` is the un-publish call
                        // (clears the `compiled` bit); call both so an
                        // already-compiled offset actually stops running
                        // right away instead of only being blocked from
                        // recompiling.
                        //
                        // Note this is still not permanent on a page that
                        // keeps mutating: `publish()`'s "real invalidation"
                        // path (snap_gen > prev entry_gen — the underlying
                        // bytes actually changed, §13.6) resets the whole
                        // page's `denied` bitmap back to all-eligible on its
                        // next successful compile, on the theory that a
                        // sticky rejection against old bytes shouldn't
                        // outlive them. A `j2 deny` on a page that is still
                        // actively recompiling (`compiles_since_flush`
                        // climbing in `j2 pcp`) can therefore be silently
                        // undone by the very next real recompile — `kill()`
                        // still stops the specific `func` this call saw, but
                        // a fresh compile can re-publish the same offset
                        // right after.
                        if actual_args.len() < 2 { return Err("Usage: j2 deny <paddr>".to_string()); }
                        let symbols_arc = exec.symbols.clone();
                        let symbols = symbols_arc.lock();
                        let paddr = monitor_phys(parse_cpu_arg(actual_args[1], &exec.core, Some(&symbols))?);
                        drop(symbols);
                        let pfn = exec.jitv2_monitor_pfn(paddr)?;
                        let page_base = pfn as u64 * crate::cpu::jitv2::PAGE_SIZE as u64;
                        let offset = (paddr & 0xFFF) as usize >> 2;
                        let sysad = exec.sysad.clone();
                        let fr1 = (exec.core.cp0_status & crate::cpu::mips_core::STATUS_FR) != 0;
                        let mut jit = exec.jitv2.lock();
                        match jit.page_for(pfn, page_base, sysad.as_ref(), fr1) {
                            Some(slot) => {
                                let page_ptr = jit.page_ptr(slot);
                                unsafe { (*page_ptr).denylist(offset); (*page_ptr).kill(offset); }
                                writeln!(writer, "j2 deny: page pfn={:#010x} offset={:#05x} (paddr {:#010x}) denylisted and killed", pfn, offset * 4, paddr).unwrap();
                            }
                            None => return Err("j2 deny: page pool is full (can't even allocate a fresh lookup entry)".to_string()),
                        }
                    }
                    "seal-queue" => {
                        // Read-only peek at the shared arena's seal-queue —
                        // answers "is there a permanent gap, and where"
                        // without needing per-worker-thread introspection
                        // (there is none: every jitv2-compile-N thread looks
                        // identical from the outside, and a
                        // finalize_batch_nonforced call that never happens
                        // for some FuncId leaves no trace on the thread that
                        // should have made it, only a stuck placeholder
                        // here). See SharedArena::seal_queue_snapshot's own
                        // doc comment for how to read this, and its own
                        // doc comment on why the front entry alone isn't
                        // always where the queue is actually stuck (a
                        // patched front with thousands still queued behind
                        // it is a real, confirmed-live shape).
                        //
                        // Usage: j2 seal-queue [N]  (default 2) — print the
                        // first N entries from the front, so the real gap
                        // (usually just past the front) is visible directly
                        // instead of only reported as an index.
                        let count: usize = actual_args.get(1)
                            .map(|s| s.parse().map_err(|_| format!("j2 seal-queue: bad count '{}'", s)))
                            .transpose()?
                            .unwrap_or(2);
                        let jit = exec.jitv2.lock();
                        match jit.compile_queue.seal_queue_snapshot() {
                            None => {
                                writeln!(writer, "seal queue: pool never started (no arena built yet)").unwrap();
                            }
                            Some(snap) => {
                                writeln!(writer, "position={:#x}  sealed_up_to={:#x}  queue_len={}",
                                    snap.position, snap.sealed_up_to, snap.queue_len).unwrap();
                                if let Some(gap_idx) = snap.first_gap_index {
                                    writeln!(writer, "first real gap: entry #{} (0 = the front entry itself)", gap_idx).unwrap();
                                } else {
                                    writeln!(writer, "no gap found (queue empty or fully contiguous — should be draining)").unwrap();
                                }
                                let describe_page = |writer: &mut String, page_ptr: *mut crate::cpu::jitv2::PhysicalCodePage| {
                                    if page_ptr.is_null() {
                                        writeln!(writer, "    page: none (test-only compile path, no real page)").unwrap();
                                    } else {
                                        // Safety: same reasoning as `j2 pcp`'s own page
                                        // dereference — try_lock_executor holds the lock
                                        // every mutator of the page pool takes, so nothing
                                        // frees/reclaims this slot out from under this
                                        // read. The pointer could still name a slot that's
                                        // since been reset to a different pfn if a flush
                                        // ran between the compile and this command — report
                                        // it as-is; a stale pfn here is itself a useful
                                        // signal, not a crash risk.
                                        let page = unsafe { &*page_ptr };
                                        writeln!(writer, "    page: pfn={:#010x} phys_base={:#010x}",
                                            page.pfn, page.pfn as u64 * crate::cpu::jitv2::PAGE_SIZE as u64).unwrap();
                                    }
                                };
                                match jit.compile_queue.seal_queue_entries() {
                                    None => {}
                                    Some(entries) => {
                                        for (i, (start, end, is_placeholder, tid, page_ptr)) in entries.iter().take(count).enumerate() {
                                            let marker = if Some(i) == snap.first_gap_index { "  <-- first gap" } else { "" };
                                            writeln!(writer, "  #{} [{:#x}, {:#x})  unpatched_placeholder={}{}",
                                                i, start, end, is_placeholder, marker).unwrap();
                                            writeln!(writer, "    created by thread {:?}", tid).unwrap();
                                            let mut buf = String::new();
                                            describe_page(&mut buf, *page_ptr);
                                            write!(writer, "{}", buf).unwrap();
                                        }
                                        if entries.len() > count {
                                            writeln!(writer, "  ... {} more entries not shown (j2 seal-queue {} to see more)",
                                                entries.len() - count, count + 8).unwrap();
                                        }
                                    }
                                }
                                if snap.first_gap_index.is_some_and(|i| snap.first_gap_is_unpatched_placeholder == Some(true) && i < usize::MAX) {
                                    writeln!(writer, "^ the marked entry's compile reserved this range (push_placeholder) and its \
                                        matching finalize_batch_nonforced call never ran (or is still running) — every entry \
                                        behind it is permanently blocked until this one is patched. Re-run this command after a \
                                        pause: if its range/thread/page haven't changed, that compile is genuinely stuck (not \
                                        just slow).").unwrap();
                                }
                            }
                        }
                    }
                    "html" => {
                        // See src/cpu/jitv2_html_j2wp.rs for the full collect+render
                        // implementation.
                        let path = actual_args.get(1).copied().unwrap_or("jitv2.html");
                        let jitv2_arc = exec.jitv2.clone();
                        let bus = exec.sysad.clone();
                        crate::cpu::jitv2_html_j2wp::write_jitv2_html(
                            &jitv2_arc, &bus, &exec.memory.clone(), &mut exec.jitv2_inline_analyzer, path, &mut *writer,
                        )?;
                    }
                    "stats" | "status" => {
                        let jit = exec.jitv2.lock();
                        writeln!(writer, "pages: {} / {} used", jit.pages_used(), jit.capacity()).unwrap();
                        // Churn avoidance: "skipped" is compile work that did not
                        // happen because the page's generation moved without
                        // any word the last compile decoded changing;
                        // "rejected" had a snapshot to check and still had
                        // to compile. A high skipped:rejected ratio is the
                        // mechanism paying for itself; near-zero skipped
                        // means the RAM is buying nothing on this workload.
                        {
                            let (skipped, rejected) = jit.redundant_compile_totals();
                            let considered = skipped + rejected;
                            let pct = if considered > 0 { (skipped as f64 / considered as f64) * 100.0 } else { 0.0 };
                            writeln!(writer, "churn (since last flush): {skipped} compiles skipped as redundant, {rejected} had to recompile ({pct:.1}% avoided of {considered} checked)").unwrap();
                        }
                        // Report the EFFECTIVE inline state: jitv2_lockstep forces
                        // it on regardless of the runtime field (see exec_decoded's
                        // `inline_compile`), so printing the raw field would
                        // misleadingly say "off" on a lockstep build that's
                        // actually inline-compiling every region.
                        #[cfg(feature = "jitv2_lockstep")]
                        {
                            writeln!(writer, "inline compile: on (forced by jitv2_lockstep)").unwrap();
                            // Positive proof lockstep is actually verifying: this
                            // counts real JIT-vs-interpreter compares. Stuck at 0
                            // means the JIT never ran (nothing compiled) and
                            // verification is silently OFF.
                            let verified = crate::cpu::mips_exec::LOCKSTEP_VERIFICATIONS.load(Ordering::Relaxed);
                            writeln!(writer, "lockstep: {} instructions verified (0 => NOT verifying — JIT not running)", verified).unwrap();
                        }
                        #[cfg(not(feature = "jitv2_lockstep"))]
                        writeln!(writer, "inline compile: {}", if exec.jitv2_inline_compile { "on" } else { "off" }).unwrap();
                        #[cfg(feature = "developer")]
                        {
                            let dt_calls = crate::cpu::mips_exec::DEV_TRACE_BP_CALLS.load(Ordering::Relaxed);
                            writeln!(writer, "dev trace hook: {} calls (0 => emit_dev_trace_bp never reached from compiled code)", dt_calls).unwrap();
                        }
                        // Functions compiled into the active Codegen's
                        // Cranelift arena since the last mega_flush — a
                        // diagnostic count only now; the actual flush trigger
                        // is CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES (real bytes
                        // reserved), printed separately below. Branch on
                        // whether the pool is actually spawned, not on
                        // `jit.codegen`'s Option — that slot stays `Some`
                        // the entire time the pool runs (reserved for
                        // `j2 inline on` to take later, see `Machine::new`'s
                        // own comment), so it's never a reliable "who owns
                        // compilation right now" signal on its own.
                        if jit.compile_queue.is_running() {
                            writeln!(writer, "codegen: {} functions compiled (owned by async compile thread pool, {} threads)",
                                jit.compile_queue.function_count(), jit.compile_queue.thread_count()).unwrap();
                        } else if let Some(n) = jit.codegen.lock().as_ref().map(|c| c.function_count()) {
                            writeln!(writer, "codegen: {} functions compiled", n).unwrap();
                        }
                        {
                            // Real per-worker arena byte counts have no
                            // cross-thread mirror today (only function_count
                            // does — see CompileQueue::function_count's own
                            // doc comment) — only readable here while the
                            // pool isn't running (the idle inline slot).
                            let stats = if jit.compile_queue.is_running() {
                                None
                            } else {
                                jit.codegen.lock().as_ref().map(|c| c.packing_stats())
                            };
                            if let Some((used, reserved)) = stats {
                                writeln!(writer, "arena reserved: {} / {} bytes ({:.1}%), used {} bytes ({:.1}% packing) — real flush trigger",
                                    reserved, crate::cpu::jitv2::CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES,
                                    reserved as f64 * 100.0 / crate::cpu::jitv2::CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES as f64,
                                    used, if reserved == 0 { 0.0 } else { used as f64 * 100.0 / reserved as f64 }).unwrap();
                            } else if jit.compile_queue.is_running() {
                                writeln!(writer, "arena reserved: n/a while pooled (no cross-thread byte mirror yet — see function_count for a live signal instead)").unwrap();
                            }
                            // Mega-flush count: each flush wipes ALL published
                            // entries, forcing everything to recompile. A count
                            // that keeps climbing during steady-state execution
                            // is the arena-fill/flush/recompile death spiral —
                            // the working set exceeds the arena threshold.
                            let flushes = crate::cpu::jit_feedback::JIT_FEEDBACK.flush_events.load(Ordering::Relaxed);
                            writeln!(writer, "mega-flushes (arena/pool wipes, each recompiles everything): {}", flushes).unwrap();
                        }
                        // Arena bytes: this block reports emitted code size (arena
                        // bytes + the bytes/instruction histogram), and `developer`
                        // distorts exactly that number — it forces `opt_level=none`
                        // and emits a 4-argument `call_indirect` per instruction
                        // (`emit_dev_trace_bp`), measured live at ~380 code
                        // bytes/guest instruction where the same pages compile
                        // to ~92 without it. Gating the metric on the feature
                        // that invalidates it meant it could never be read for
                        // the build that ships. The underlying counters are two
                        // u32 per pooled page (32 KiB at full capacity), so
                        // keeping them always-on costs nothing.
                        {
                            let code_bytes = jit.code_bytes_used();
                            writeln!(writer, "arena bytes (sum of published code_size): {} ({:.1} KiB) — proxy for the Cranelift arena, for when `packing_stats` cannot be read (pool busy); the arena packs, so this is close but excludes alignment padding",
                                code_bytes, code_bytes as f64 / 1024.0).unwrap();
                        }
                        // Everything below is genuinely developer-only
                        // (rejection reasons, fallback counters, stats fields
                        // that do not exist without the feature).
                        #[cfg(feature = "developer")]
                        {
                            let compiles = jit.stats.compiles.load(Ordering::Relaxed);
                            let failed = jit.stats.failed_compiles.load(Ordering::Relaxed);
                            let kills = jit.stats.kill_entry_calls.load(Ordering::Relaxed);
                            writeln!(writer, "compiles: {} ok, {} failed (codegen gap or analyzer-excluded), {} kill_entry_fn bailouts (FR-mismatch)",
                                compiles, failed, kills).unwrap();
                            let fb_words = jit.stats.fallback_words.load(Ordering::Relaxed);
                            let fb_regions = jit.stats.fallback_regions.load(Ordering::Relaxed);
                            writeln!(writer, "fallback: {} interp-fallback words compiled across {} regions ({})",
                                fb_words, fb_regions,
                                if crate::cpu::jitv2::analyzer::fallback_enabled() { "j2 fallback: on" } else { "j2 fallback: off" }).unwrap();
                            // Gate on the buckets, not on `failed`: a
                            // dirty-page deferral is deliberately not counted
                            // as a failed compile, so a run whose only
                            // rejections are deferrals would print nothing at
                            // all if this asked `failed > 0`.
                            let any_reject = crate::cpu::jitv2::RejectReason::ALL.iter()
                                .any(|r| jit.stats.reject_reasons[r.index()].load(Ordering::Relaxed) > 0);
                            if any_reject {
                                writeln!(writer, "  rejections by reason:").unwrap();
                                // Driven by RejectReason::ALL, not a list
                                // repeated here: a variant added to the enum
                                // but forgotten here is invisible in `j2
                                // stats` no matter how often it fires, and
                                // the only symptom is `failed` not matching
                                // the sum of the printed buckets.
                                for reason in crate::cpu::jitv2::RejectReason::ALL {
                                    let n = jit.stats.reject_reasons[reason.index()].load(Ordering::Relaxed);
                                    if n == 0 { continue; }
                                    writeln!(writer, "    {:>7}  {}", n, reason.label()).unwrap();
                                }
                            }

                            let dispatches = jit.stats.compile_queue_dispatches.load(Ordering::Relaxed);
                            let full = jit.stats.compile_queue_full.load(Ordering::Relaxed);
                            let depth_sum = jit.stats.compile_queue_depth_sum.load(Ordering::Relaxed);
                            if dispatches > 0 {
                                let avg_depth = depth_sum as f64 / dispatches as f64;
                                let full_pct = full as f64 * 100.0 / dispatches as f64;
                                writeln!(writer, "compile queue: {} / {} now, {} dispatches, {} full ({:.1}%), avg depth at dispatch {:.1}",
                                    jit.compile_queue.queue_occupancy(), crate::cpu::jitv2::COMPILE_QUEUE_CAPACITY,
                                    dispatches, full, full_pct, avg_depth).unwrap();
                            }

                            let pending_threshold = jit.stats.batch_flushes_pending_threshold.load(Ordering::Relaxed);
                            let queue_drain = jit.stats.batch_flushes_queue_drain.load(Ordering::Relaxed);
                            if pending_threshold > 0 || queue_drain > 0 {
                                let max_pending = jit.stats.batch_max_pending.load(Ordering::Relaxed);
                                writeln!(writer, "force-seal sweeps: {} pending-threshold, {} queue-drain, largest backlog {} entries",
                                    pending_threshold, queue_drain, max_pending).unwrap();
                            }
                            // Packing quality (bytes actually used by compiled
                            // code vs. bytes reserved for it, per the arena's
                            // own host-page segments) — only readable while
                            // this thread holds the real Codegen directly
                            // (inline mode, or the worker not yet started);
                            // when the async worker owns it, there's no lock
                            // a status command should contend for just to
                            // print a diagnostic (same reasoning as
                            // `function_count`'s own mirror-vs-direct split
                            // above, but packing_stats isn't mirrored since
                            // it's purely informational, not something the
                            // hot compile path needs a lock-free read of).
                            // Bind the guard to a local rather than using the
                            // `lock()` temporary directly in the `if let`
                            // scrutinee: the temporary otherwise lives until
                            // the end of the enclosing block, which outlives
                            // `jit` now that the stats output is split into
                            // several cfg-gated blocks.
                            let codegen_guard = jit.codegen.lock();
                            if let Some(codegen) = codegen_guard.as_ref() {
                                let (used, reserved) = codegen.packing_stats();
                                if reserved > 0 {
                                    let pct = used as f64 * 100.0 / reserved as f64;
                                    writeln!(writer, "  packing: {} / {} bytes used ({:.1}%) across this Codegen's host-page segments",
                                        used, reserved, pct).unwrap();
                                }
                            }

                        }
                        // Size histogram: real distribution of published regions'
                        // instruction counts, scanned across every pooled page — ground
                        // truth against MAX_INSTRS_PER_COMPILE (comp.rs),
                        // since a region can land shorter than the budget
                        // (branch/exclusion/page boundary) or, for a branch's
                        // own head instruction, effectively longer once its
                        // mandatory delay slot is counted in. Paired with each
                        // bucket's own code-size distribution (avg/min/max) —
                        // shows whether code size actually scales with
                        // instruction count or is dominated by fixed
                        // per-region overhead (preamble, CU1/FR guard).
                        {
                            // Real distribution of published regions' instruction
                            // counts, scanned across every pooled page — ground
                            // truth against MAX_INSTRS_PER_COMPILE (comp.rs),
                            // since a region can land shorter than the budget
                            // (branch/exclusion/page boundary) or, for a branch's
                            // own head instruction, effectively longer once its
                            // mandatory delay slot is counted in. Paired with each
                            // bucket's own code-size distribution (avg/min/max) —
                            // shows whether code size actually scales with
                            // instruction count or is dominated by fixed
                            // per-region overhead (preamble, CU1/FR guard).
                            let hist = jit.code_size_by_instr_count();
                            let total: u32 = hist.iter().flatten().map(|b| b.count).sum();
                            let total_bytes: u64 = hist.iter().flatten().map(|b| b.sum_bytes).sum();
                            let total_instrs: u64 = hist.iter().enumerate()
                                .filter_map(|(n, b)| b.map(|b| n as u64 * b.count as u64))
                                .sum();
                            if total > 0 {
                                writeln!(writer, "instruction-count histogram ({} published entries):", total).unwrap();
                                for (n, bucket) in hist.iter().enumerate() {
                                    let Some(b) = bucket else { continue };
                                    let pct = b.count as f64 * 100.0 / total as f64;
                                    let avg = b.sum_bytes as f64 / b.count as f64;
                                    writeln!(writer, "  {:>3} instr: {:>7}  ({:5.1}%)  code bytes avg={:>6.1} min={:>5} max={:>5}",
                                        n, b.count, pct, avg, b.min_bytes, b.max_bytes).unwrap();
                                }
                                if total_instrs > 0 {
                                    writeln!(writer, "  overall: {:.1} code bytes/instruction ({} bytes across {} instructions)",
                                        total_bytes as f64 / total_instrs as f64, total_bytes, total_instrs).unwrap();
                                }
                            }
                        }
                    }
                    #[cfg(all(target_os = "linux", feature = "developer"))]
                    "hugepages" => {
                        let jit = exec.jitv2.lock();
                        let (ptr, len) = jit.codegen.lock().as_ref()
                            .map(|c| c.arena_range())
                            .unwrap_or((0, 0));
                        if len == 0 {
                            writeln!(writer, "jitv2 arena not yet allocated (no compile has run) or owned by the async compile thread right now — try again after `stop`/a compile.").unwrap();
                            return Ok(());
                        }
                        writeln!(writer, "arena: {:#x}..{:#x} ({} bytes, {} MiB)", ptr, ptr + len, len, len / (1024 * 1024)).unwrap();
                        match crate::cpu::jitv2::paged_memory::anon_hugepages_in_range(ptr, len) {
                            Some(hugepage_bytes) => {
                                let pct = hugepage_bytes as f64 * 100.0 / len as f64;
                                writeln!(writer, "AnonHugePages within arena: {} bytes ({} MiB, {:.1}% of reservation) — from /proc/self/smaps",
                                    hugepage_bytes, hugepage_bytes / (1024 * 1024), pct).unwrap();
                                if hugepage_bytes == 0 {
                                    writeln!(writer, "  0% is suspicious once real code has compiled and finalized — check `cat /sys/kernel/mm/transparent_hugepage/enabled` (want `madvise` or `always`, not `never`).").unwrap();
                                }
                            }
                            None => {
                                writeln!(writer, "/proc/self/smaps unavailable (sandboxed/restricted environment?) — can't verify hugepage status this way.").unwrap();
                            }
                        }
                    }
                    _ => return Err("Usage: j2 <analyze <addr>|pcp [addr]|status|inline [on|off]|dispatch [on|off]|fallback [on|off|<category>]|memhelpers [on|off]|pagewb [on|off]|instrs|threads|opt [none|speed]|min-instrs [N]|max-instrs [N]|min-calls [N]|lockstep|lstate [full] [N]|hugepages|flush|clear <paddr>|deny <paddr>>".to_string()),
                }
                Ok(())
            }
            #[cfg(feature = "developer")]
            "trace" => {
                if actual_args.is_empty() { return Err("Usage: trace <start|stop|status>".to_string()); }
                let mut exec = self.try_lock_executor()?;

                match actual_args[0] {
                    "start" => {
                        if actual_args.len() < 2 { return Err("Usage: trace start <path>".to_string()); }
                        let path = std::path::Path::new(actual_args[1]);
                        exec.trace_start(path).map_err(|e| format!("trace start failed: {}", e))?;
                        writeln!(writer, "Recording execution trace to {}", actual_args[1]).unwrap();
                    }
                    "stop" => {
                        let count = exec.trace_stop().map_err(|e| format!("trace stop failed: {}", e))?;
                        writeln!(writer, "Trace stopped: {} records written", count).unwrap();
                    }
                    "status" => {
                        writeln!(writer, "recording: {}", exec.trace_active()).unwrap();
                    }
                    _ => return Err("Usage: trace <start|stop|status>".to_string()),
                }
                Ok(())
            }
            "dt" | "traceback" => {
                // dt [N]              — dump last N instructions to console (default 10)
                // dt file <path> [N] — dump to file (default: entire buffer)
                let (file_path, count) = if actual_args.first().copied() == Some("file") {
                    let path = actual_args.get(1).copied().unwrap_or("/tmp/iris_trace.txt");
                    let n = actual_args.get(2).and_then(|s| s.parse().ok()).unwrap_or(TRACEBACK_SIZE);
                    (Some(path.to_string()), n)
                } else {
                    let n = actual_args.first().and_then(|s| s.parse().ok()).unwrap_or(10);
                    (None, n)
                };
                let requested = count;
                let count = count.min(TRACEBACK_SIZE);
                let exec = self.executor.lock();
                let symbols = exec.symbols.lock();
                let entries = exec.traceback.get_last(count);
                if requested > TRACEBACK_SIZE {
                    writeln!(writer, "note: asked for {} instructions; the history holds {}", requested, TRACEBACK_SIZE).unwrap();
                }
                let n = entries.len();
                let do_write = |w: &mut dyn Write| {
                    writeln!(w, "Execution Traceback (last {} instructions):", n).unwrap();
                    for entry in &entries {
                        let sym_str = format_pc_symbol(entry.pc, &symbols);
                        let dis = mips_dis::disassemble(entry.instr, entry.pc, Some(&symbols));
                        // Origin tag: plain interpreter entries are unmarked (the
                        // common case), every other arrival class gets its own
                        // distinct tag (see InstrOrigin::tag) so exactly how an
                        // instruction was reached — plain JIT body, entry word via
                        // back-edge, inlined delay slot, fallback, fallback
                        // successor (with or without back-edge) — stands out in
                        // the trace instead of a single collapsed [jit] bit.
                        writeln!(w, "{:#018x}{}: {:#010x} {}{}", entry.pc, sym_str, entry.instr, dis, entry.origin.tag()).unwrap();
                    }
                };
                if let Some(ref path) = file_path {
                    match std::fs::File::create(path) {
                        Ok(f) => {
                            let mut bw = std::io::BufWriter::new(f);
                            do_write(&mut bw);
                            let _ = std::io::Write::flush(&mut bw);
                            writeln!(writer, "Wrote {} instructions to {}", n, path).unwrap();
                        }
                        Err(e) => return Err(format!("Cannot open {}: {}", path, e)),
                    }
                } else {
                    do_write(&mut writer);
                }
                Ok(())
            }
            #[cfg(feature = "idle-pause")]
            "idle" => {
                let sub = actual_args.first().copied().unwrap_or("status");
                match sub {
                    "on" => {
                        crate::cpu::idle_park::set_idle_park_enabled(true);
                        writeln!(writer, "idle: parking enabled").unwrap();
                    }
                    "off" => {
                        crate::cpu::idle_park::set_idle_park_enabled(false);
                        writeln!(writer, "idle: parking disabled").unwrap();
                    }
                    "status" => {
                        use std::sync::atomic::Ordering::Relaxed as R;
                        let n = crate::cpu::idle_park::PARK_COUNT.load(R);
                        let ns = crate::cpu::idle_park::PARK_SLEEP_NS.load(R);
                        let irq = crate::cpu::idle_park::PARK_WAKE_IRQ.load(R);
                        let other = crate::cpu::idle_park::PARK_WAKE_OTHER.load(R);
                        writeln!(writer, "idle: {}  parks={} slept={:.3}s wake_irq={} wake_other={}",
                            if crate::cpu::idle_park::idle_park_enabled() { "on" } else { "off" },
                            n, ns as f64 / 1e9, irq, other).unwrap();
                        writeln!(writer, "  updates={} full={} largest={} distinct={}",
                            crate::cpu::idle_park::UPDATES.load(R),
                            crate::cpu::idle_park::WINDOW_FULL.load(R),
                            crate::cpu::idle_park::WINDOW_SAMPLES.load(R),
                            crate::cpu::idle_park::WINDOW_DISTINCT.load(R)).unwrap();
                        writeln!(writer, "  resets: irq={} ip={} im0={} ie0={} user={}",
                            crate::cpu::idle_park::RESET_IRQ.load(R),
                            crate::cpu::idle_park::RESET_IP.load(R),
                            crate::cpu::idle_park::RESET_IM0.load(R),
                            crate::cpu::idle_park::RESET_IE0.load(R),
                            crate::cpu::idle_park::RESET_USER.load(R)).unwrap();
                    }
                    _ => return Err("Usage: idle <on|off|status>".to_string()),
                }
                Ok(())
            }
            #[cfg(feature = "idle-pause")]
            "idleprof" => {
                let sub = actual_args.first().copied().unwrap_or("report");
                match sub {
                    "on" => {
                        self.idle_profile_arm();
                        writeln!(writer, "idleprof: armed (lock-free, histogram reset). Let the guest idle, then `stop; idleprof report`.").unwrap();
                    }
                    "off" => {
                        self.idle_profile_disarm();
                        writeln!(writer, "idleprof: disarmed.").unwrap();
                    }
                    "report" => {
                        let count = actual_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(32);
                        let mut exec = self.try_lock_executor()?;
                        exec.idle_profile_report(count, &mut writer);
                    }
                    "stats" => {
                        use std::sync::atomic::Ordering;
                        let n = crate::cpu::idle_park::PARK_COUNT.load(Ordering::Relaxed);
                        let ns = crate::cpu::idle_park::PARK_SLEEP_NS.load(Ordering::Relaxed);
                        let irq = crate::cpu::idle_park::PARK_WAKE_IRQ.load(Ordering::Relaxed);
                        let other = crate::cpu::idle_park::PARK_WAKE_OTHER.load(Ordering::Relaxed);
                        writeln!(writer, "idleprof: parks={} slept={:.3}s wake_irq={} wake_other={}",
                            n, ns as f64 / 1e9, irq, other).unwrap();
                    }
                    _ => return Err("Usage: idleprof <on | off | report [count] | stats>".to_string()),
                }
                Ok(())
            }
            #[cfg(feature = "instr_stats")]
            "instrstats" => {
                let sub = actual_args.first().copied().unwrap_or("report");
                let mut exec = self.try_lock_executor()?;
                match sub {
                    "report" => {
                        let _ = exec.instr_stats.write_report(&mut writer);
                    }
                    "clear" => {
                        exec.instr_stats.clear();
                        writeln!(writer, "instrstats: counters cleared").unwrap();
                    }
                    "dump" => {
                        match std::fs::File::create("instr_used.txt") {
                            Ok(f) => {
                                let mut w = std::io::BufWriter::new(f);
                                let _ = exec.instr_stats.write_used_list(&mut w);
                            }
                            Err(e) => return Err(format!("Cannot open instr_used.txt: {}", e)),
                        }
                        match std::fs::File::create("instr_counts.txt") {
                            Ok(f) => {
                                let mut w = std::io::BufWriter::new(f);
                                let _ = exec.instr_stats.write_counts(&mut w);
                            }
                            Err(e) => return Err(format!("Cannot open instr_counts.txt: {}", e)),
                        }
                        writeln!(writer, "instrstats: wrote instr_used.txt, instr_counts.txt").unwrap();
                    }
                    _ => return Err("Usage: instrstats [report | clear | dump]".to_string()),
                }
                Ok(())
            }
            _ => Err(format!("Unknown CPU command: {}", actual_cmd)),
        }
    }
}

impl<T: Tlb, C: CpuModel> MipsExecutor<T, C> {
    /// Dump the hottest sampled PCs and flag a likely idle-loop region: the
    /// smallest contiguous PC window (<=256 bytes) of always-interrupts-enabled
    /// samples that together account for the bulk of execution.
    #[cfg(feature = "idle-pause")]
    pub fn idle_profile_report(&mut self, top: usize, writer: &mut dyn Write) {
        let total = self.idle_profiler.total;
        if total == 0 {
            let _ = writeln!(writer, "idleprof: no samples (run `idleprof on`, let the guest idle, then `cpu stop`)");
            return;
        }

        let mut rows: Vec<(u64, IdleSample)> =
            self.idle_profiler.hist.iter().map(|(&pc, &s)| (pc, s)).collect();
        rows.sort_by(|a, b| b.1.count.cmp(&a.1.count));

        // Pre-fetch instruction words for the displayed PCs while we still hold
        // `&mut self`; disassembly below borrows `self.symbols`, which would
        // otherwise conflict with debug_fetch_instr's `&mut self`.
        let instrs: std::collections::HashMap<u64, Option<u32>> = rows
            .iter()
            .take(top)
            .map(|(pc, _)| (*pc, self.debug_fetch_instr(*pc).ok()))
            .collect();

        let symbols = self.symbols.lock();
        let _ = writeln!(
            writer,
            "idleprof: {} samples (stride {}), {} distinct PCs — top {}:",
            total, self.idle_profiler.stride, rows.len(), top.min(rows.len())
        );
        let _ = writeln!(writer, "  {:>16}  {:>7}  {:>5}  {:>4}  symbol / disasm", "pc", "count", "pct", "ie%");
        for (pc, s) in rows.iter().take(top) {
            let pct = s.count as f64 * 100.0 / total as f64;
            let iepct = s.ie_count as f64 * 100.0 / s.count as f64;
            let sym = format_pc_symbol(*pc, &symbols);
            let dis = match instrs.get(pc).copied().flatten() {
                Some(instr) => mips_dis::disassemble(instr, *pc, Some(&symbols)),
                None => "<fetch failed>".to_string(),
            };
            let _ = writeln!(
                writer,
                "  {:016x}  {:>7}  {:4.1}%  {:3.0}%  {}{}",
                pc, s.count, pct, iepct, sym, format!("  {}", dis)
            );
        }

        // Idle-loop heuristic: among the hottest PCs that are interrupt-enabled
        // ~always, find the tightest contiguous window covering >=50% of samples.
        let mut hot: Vec<(u64, IdleSample)> = rows
            .iter()
            .filter(|(_, s)| s.ie_count * 100 >= s.count * 99) // IE >= 99%
            .cloned()
            .collect();
        hot.sort_by_key(|(pc, _)| *pc);
        let mut best: Option<(u64, u64, u64)> = None; // (lo, hi, covered)
        for i in 0..hot.len() {
            let lo = hot[i].0;
            let mut covered = 0u64;
            let mut hi = lo;
            for &(pc, s) in &hot[i..] {
                if pc.wrapping_sub(lo) > 256 {
                    break;
                }
                hi = pc;
                covered += s.count;
            }
            if covered * 2 >= total && best.map_or(true, |(_, _, c)| covered > c) {
                best = Some((lo, hi, covered));
            }
        }
        match best {
            Some((lo, hi, covered)) => {
                let pct = covered as f64 * 100.0 / total as f64;
                let sym = format_pc_symbol(lo, &symbols);
                let _ = writeln!(
                    writer,
                    "\nidle-loop candidate: {:016x}..={:016x} ({} bytes){}  — {:.1}% of samples, interrupts enabled",
                    lo, hi, hi - lo + 4, sym, pct
                );
            }
            None => {
                let _ = writeln!(
                    writer,
                    "\nno clear idle-loop candidate (no interrupts-enabled PC window covers >=50% of samples)"
                );
            }
        }
    }

    /// Analyze function prologue to determine frame size and RA save location
    fn analyze_prologue(&mut self, start_pc: u64, current_pc: u64) -> (u64, Option<(i64, usize)>) {
        let mut frame_size = 0u64;
        let mut ra_info = None;
        
        let mut pc = start_pc;
        // Limit scanning to avoid infinite loops or huge scans
        while pc < current_pc && (pc - start_pc) < 1024 {
            let instr = match self.debug_fetch_instr(pc) {
                Ok(i) => i,
                Err(_) => break,
            };

            let op = (instr >> 26) & 0x3F;
            let rs = (instr >> 21) & 0x1F;
            let rt = (instr >> 16) & 0x1F;
            let imm = (instr & 0xFFFF) as i16;

            // ADDIU sp, sp, imm (0x09) or DADDIU sp, sp, imm (0x19)
            if (op == 0x09 || op == 0x19) && rs == 29 && rt == 29 {
                // Stack adjustment: usually negative to allocate space
                let adj = imm as i64;
                if adj < 0 {
                    frame_size = frame_size.wrapping_add((-adj) as u64);
                }
            }
            // SW ra, offset(sp) (0x2B)
            else if op == 0x2B && rs == 29 && rt == 31 {
                ra_info = Some((imm as i64, 4usize));
            }
            // SD ra, offset(sp) (0x3F)
            else if op == 0x3F && rs == 29 && rt == 31 {
                ra_info = Some((imm as i64, 8usize));
            }

            pc += 4;
        }
        (frame_size, ra_info)
    }

    pub fn get_return_address(&mut self) -> Option<u64> {
        let pc = self.core.pc;
        let sp = self.core.read_gpr(29);
        let ra = self.core.read_gpr(31);

        let sym_info = {
            let symbols = self.symbols.lock();
            symbols.lookup(pc).map(|(addr, _)| addr)
        };

        if let Some(start_addr) = sym_info {
            let (_, ra_info) = self.analyze_prologue(start_addr, pc);
            if let Some((offset, size)) = ra_info {
                let save_addr = sp.wrapping_add(offset as u64);
                match self.debug_read(save_addr, size) {
                    Ok(val) => Some(val),
                    Err(_) => Some(ra),
                }
            } else {
                Some(ra)
            }
        } else {
            Some(ra)
        }
    }

    pub fn backtrace(&mut self, max_frames: usize) -> String {
        let (pc, sp, ra) = (self.core.pc, self.core.read_gpr(29), self.core.read_gpr(31));
        self.backtrace_from(pc, sp, ra, max_frames)
    }

    /// Walk the stack from an arbitrary context (an exception frame's EPC,
    /// sp and ra, say) by prologue analysis.
    pub fn backtrace_from(&mut self, pc: u64, sp: u64, ra: u64, max_frames: usize) -> String {
        let mut output = String::new();
        let (mut pc, mut sp) = (pc, sp);

        writeln!(output, "Backtrace:").unwrap();

        for i in 0..max_frames {
            let symbols = self.symbols.lock();
            let sym_info = symbols.lookup(pc);
            let sym_str = if let Some((_, name)) = sym_info {
                let offset = pc.wrapping_sub(sym_info.unwrap().0);
                format!("{} + 0x{:x}", name, offset)
            } else {
                format!("0x{:016x}", pc)
            };

            writeln!(output, "#{:02} pc=0x{:016x} sp=0x{:016x} {}", i, pc, sp, sym_str).unwrap();

            if let Some((start_addr, _)) = sym_info {
                drop(symbols); // Release lock before calling analyze_prologue

                let (frame_size, ra_info) = self.analyze_prologue(start_addr, pc);

                if frame_size > 0 {
                    let prev_sp = sp.wrapping_add(frame_size);
                    let prev_pc = if let Some((offset, size)) = ra_info {
                        let save_addr = sp.wrapping_add(offset as u64);
                        match self.debug_read(save_addr, size) {
                            Ok(val) => val,
                            Err(_) => break, // Cannot read return address
                        }
                    } else {
                        ra // Leaf function, use current RA
                    };

                    sp = prev_sp;
                    pc = prev_pc;
                    if pc == 0 { break; }
                } else {
                    // No frame size found, assume leaf or end of chain
                    if ra == 0 || ra == pc { break; }
                    pc = ra;
                }
            } else {
                break; // No symbol info
            }
        }
        output
    }
}
