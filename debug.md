# Debugging and introspection

How to look inside a running IRIS machine: the monitor console and its
commands, the GDB stub, the MCP server for AI assistants, traces and
recordings, and the other diagnostic outputs. For how the emulator is built,
see [HACKING.md](HACKING.md); for configuration, [HELP.md](HELP.md) and
[FEATURES.md](FEATURES.md).

- [Build flavours](#build-flavours)
- [The monitor console](#the-monitor-console)
- [Command reference](#command-reference): [machine](#machine),
  [execution](#execution-control), [registers](#registers),
  [memory](#memory-and-loading), [symbols and IRIX](#symbols-and-irix-introspection),
  [breakpoints](#breakpoints), [exceptions](#exceptions),
  [history and tracing](#history-undo-and-tracing), [TLB and caches](#tlb-and-caches),
  [JIT v2](#jit-v2), [SCSI](#scsi-cd-rom-and-disks), [graphics](#graphics),
  [devices](#hardware-devices), [networking](#networking),
  [physical bus](#physical-bus), [logging](#logging),
  [performance](#performance-and-locks), [Ultra64](#n64-development-board)
- [Recipes](#recipes)
- [GDB stub](#gdb-stub)
- [MCP server](#mcp-server)
- [Other introspection](#other-introspection)

## Build flavours

What you can inspect depends on how iris was built:

| Build | Debugging |
|---|---|
| plain `--release` | Everything in this document except the **[DEV]** items. Breakpoints and the traceback work. |
| `lightning` (official releases) | Strips breakpoint checks and the traceback buffer for speed: breakpoints and `dt` do nothing. Inspection commands still work. |
| `developer` (`cargo run --profile developer --features developer`) | Adds the undo buffer, `si`, instruction tracing (`debug`, `trace`), `jitcheck`, per-module logging, graphics traces, and the `[DEV]` commands. The CPU starts **paused**: type `start`. Cannot be combined with `lightning`. |
| `--features gputrace` | Graphics traces and recordings (`rex/gr2/mgras trace`, `rex/mgras rec`) without the rest of `developer`; works with `lightning`. |
| `--features jitv2` | The `j2` commands. `jitv2_lockstep` checks every JIT instruction against the interpreter. |
| `--features perfetto` | Perfetto timeline traces ([docs/perfetto.md](docs/perfetto.md)). |

The `developer` *profile* alone only changes optimisation and debug-info
settings; the instrumentation comes from the `developer` *feature*.
Developer builds also show extra counters in the status bar (D: decode rate,
I$: L1I hit rate, UC: uncached fetches, cs: step count).

## The monitor console

The monitor listens on **127.0.0.1:8888** (`monitor_port` in the config or
`--monitor-port`; give each iris running at once its own port). It is also
available on the terminal iris was started from.

```sh
nc localhost 8888            # or: telnet 127.0.0.1 8888
tools/mon.py "regs" "cop0"   # one-shot: run commands and print their output
IRIS_MON_PORT=8898 tools/mon.py "regs"   # an iris with a non-default monitor_port
```

`help` lists every command the running build registered, with a one-line
usage string; `quit` or `exit` closes the connection. A command can be
prefixed with `cpu` (`cpu regs`) to address the CPU explicitly.

CPU command arguments are expressions: numbers are hexadecimal with `0x`,
octal with a leading `0`, decimal otherwise, and a 32-bit value with bit 31
set is sign-extended (so `0x88001000` means the KSEG0 address
`0xffffffff88001000`; write more than eight hex digits to avoid that).
Register names (`a0`, `sp`, `$5`, `r5`, `pc`, `hi`, `lo`, `c0_<name>`) and,
once a map is loaded, symbols can be used (`b bcopy`, `dis panic 20`,
`mem sp+16 4`).

**Commands and a running CPU.** `start`, `stop`, `status`, `wait` and `help`
never touch the CPU core. Run control (`run`, `step`, `next`, `finish`) needs
a stopped CPU and says so if it is running. Everything else pauses a running
CPU for the duration of the command and resumes it afterwards, so `bp add`,
`regs`, `mem` and friends work at any time and never hang.

A breakpoint hit while the CPU runs freely is reported by `status` and
`wait`, not printed.

**[DEV]** below marks commands that need a `developer` build. They are
accepted in every build but do nothing without the feature.

## Command reference

### Machine

| Command | Description |
|---|---|
| `machine-start` | Start the CPU and all peripherals. Idempotent: prints the transition it performed, e.g. `machine-start: halted -> starting -> running`, or `machine-start: running (no change)` |
| `machine-stop` | Stop the CPU and all peripherals. Idempotent: prints `machine-stop: running -> halted` |
| `machine-status [json]` | Guest state (`idle`, `starting`, `running`, `halted`) and whether the monitor is accepting commands; `json` emits one object |
| `reset` | Reset all hardware to power-on state |
| `save <name>` / `load <name>` | Save / load a snapshot under `saves/<name>/` (see [STORAGE.md](STORAGE.md#snapshots-and-rollback)) |
| `testdev` | Bare-metal test device status (`cpu-tests/`, `bench/`) |

The machine lifecycle is one value, and every surface reports it the same way:
`idle` (not started; the monitor is not listening), `starting` (the monitor
accepts commands but the CPU is parked), `running` (the CPU is executing), and
`halted` (stopped by `machine-stop`, `reset` or a guest power-off). `running` is
read from the CPU's own running flag, so the `start`/`stop` execution commands
and breakpoints are reflected without a separate bookkeeping step.

A fresh `--ci` machine is **ready for commands while still `starting`**: the
monitor is up and the peripherals are running, but the CPU stays parked until
`machine-start` (or `iris-ci start`). Do not sleep before sending it — poll
instead. Either retry the monitor TCP connect until it answers, or use the CI
socket, which iris creates only after the monitor has bound:

```sh
iris-ci --socket /tmp/iris.sock status --wait-ready 60
tools/mon.py "machine-start"
```

`machine-status` is available as soon as a monitor connection is accepted; the
`json` form is one object with `state`, `cpu_running`, `monitor_ready` and
`monitor_addr`. `iris-ci status` prints the same fields from the same source.
If the monitor could not bind its port (another iris is using it), the CI
socket still answers but reports `monitor_ready` false, so a harness can tell a
bind failure from a guest that never started.

### Execution control

| Command | Description |
|---|---|
| `start` | Start CPU execution (threaded mode) |
| `stop` | Stop the CPU; returns once it has fully stopped |
| `status` | Running or stopped, the PC, why it last stopped (which breakpoint, its condition and hit count), and the CP0 Count rate. Never blocks |
| `wait [ms]` | Block until the CPU stops (breakpoint, exception catch) or `ms` pass (default 10000), then say where and why |
| `run [addr]` / `c` / `cont` | Run until a breakpoint or exception, or until `addr` (temporary breakpoint) |
| `step [n\|addr]` / `s` | Step `n` instructions (default 1), or run until `addr` |
| `si` | Step without taking interrupts **[DEV]** |
| `next [n]` / `n` | Step over calls (`jal`/`bal` execute as one unit) |
| `finish` / `fin` | Run until the current function returns (`jr ra`) |
| `jump <addr>` | Set the PC |
| `ip7` | Make the CP0 Compare (IP7) interrupt pending, to step through timer delivery while stopped |

There are two execution modes. **Threaded mode** (`start`) runs the CPU on
its own thread at full speed, releasing the executor lock every 500,000
instructions so commands can get in; breakpoints still stop it (except in
`lightning` builds). **Debug mode** (`run`, `step`, `next`, `finish`) drives
the CPU through `run_debug_loop` with breakpoint checks and tracing; `run`
returns immediately, the stepping commands wait for completion. Starting a
debug command stops threaded mode first. The GDB stub uses the same debug
loop, so monitor and GDB breakpoints coexist.

### Several CPUs

The monitor talks to the machine's CPUs through one object, so with a single
CPU every command behaves as described elsewhere here. An Octane with
`[machine] cpus = 2` adds a choice of which CPU a command means.

| Command | Description |
|---|---|
| `cpus` | List the CPUs, whether each runs, and which is selected (`*`) |
| `cpu <n>` | Select CPU n for the commands that follow |
| `cpu <n> <command>` | Run one command on CPU n alone (`cpu 1 dis pc 8`, `cpu 1 bp add 0x80000180`) |
| `r<n>` | Registers of CPU n: `cpu <n> regs` |
| `s<n>` [count] | Step CPU n: `cpu <n> step` |

Without a number, `start`, `stop` and the breakpoint commands (`bp`, `b`, `bd`,
`be`, `bb`) apply to every CPU, `status` answers for each, and everything else
goes to the selected CPU. A number always means that CPU alone, even for `stop`.

To step one CPU while the other holds still:

```
stop
s1
s1 20
start
```

HEART's counter keeps running through a monitor stop, and the guest's
timeouts count on it, so a long stop can make IRIX report that a CPU missed its
clock. A breakpoint stops the CPU that hit it; the other keeps running until
`stop`. `heart status` shows each CPU's mask, interrupt outputs and why it is
held. `j2 inline` and `j2 clear` are refused with several CPUs. The GDB stub
sees CPU 0.

### Registers

| Command | Description |
|---|---|
| `regs` / `r` | GPRs, HI/LO, and the key CP0 registers (Status, Cause, EPC, BadVAddr) |
| `cop0` | All CP0 registers |
| `cop1` | FPU registers, FCSR and FIR |
| `setreg <reg> <val>` | Set a register (`a0`, `$5`, `r5`, `pc`, `hi`, `lo`, `c0_<n>` or `c0_<name>`) |

### Memory and loading

| Command | Description |
|---|---|
| `mem <addr> [n] [b\|h\|w\|d]` / `m` | Dump virtual memory in bytes, halves, words (default) or doublewords |
| `mw <addr> <val> [b\|h\|w\|d]` | Write virtual memory |
| `ms <addr> [max]` | Read a NUL-terminated string |
| `stack [addr] [n]` | Dump stack memory (default `$sp`) |
| `dis [addr] [n]` / `d` | Disassemble |
| `translate <addr>` / `t` | Translate a virtual address to physical with the current TLB and mode |
| `loadelf <file>` | Load a static big-endian ELF32 and set the PC to its entry (also `--load-elf`) |
| `loadbin <file> <addr>` | Load raw bytes at a virtual address |

Virtual accesses use kernel privilege, never change CP0 and never trigger
breakpoints. For physical addresses see [physical bus](#physical-bus).

### Symbols and IRIX introspection

| Command | Description |
|---|---|
| `loadsym <file>` | Load a symbol map in `nm` format (`prom.map`, `unix.map`) |
| `sym <addr>` | Nearest symbol to an address |
| `sym <name>` | Every copy of a name, with the map it came from |
| `bt [frames]` | Walk the stack frames for a call backtrace |
| `proc info` | IRIX kernel utsname and friends (needs the kernel's map) |
| `eframe <addr> [o32\|64]` / `ef` | Decode an IRIX exception frame (the `ep` of a panic): registers, decoded Cause and Status, EPC, then a backtrace of the interrupted context |
| `putbuf [size]` / `dmesg` | The IRIX kernel console log (what a panic printed); needs the kernel's map |

A name defined in more than one loaded map (the PROM and the kernel both have
`bcopy`) is ambiguous in expressions: the error lists the copies, and
`bcopy:1` picks one. Internal lookups (`proc`, `putbuf`) use the copy from
the map loaded last.

### Breakpoints

| Command | Description |
|---|---|
| `bp add <addr> [type] [if <expr>]` / `b` | Add a breakpoint. Types: `pc` (default); `r`, `w`, `f` (virtual read, write, fetch); `pr`, `pw`, `pf` (physical). The condition may use registers (`a0`, `c0_badvaddr`, ...) and symbols; a name that is neither is refused |
| `bp list` / `bl` | List breakpoints with type, symbol, condition and hit count |
| `bp del <id>` / `bb` | Delete |
| `bp enable <id>` / `be`, `bp disable <id>` / `bd` | Enable / disable |
| `bp ignore <id> <n>` | Let the next `n` hits pass |
| `bp clear` | Delete all |
| `bp save <file>` / `bp load <file>` | Write breakpoints as `bp add` lines / read them back. Breakpoints are not otherwise kept across restarts |

Monitor breakpoint IDs start at 1, GDB's at 10000; both kinds coexist.

### Exceptions

`exception <class|code|all> <on|off>` (alias `ex`) stops the CPU when an
exception of that class is taken. `ex tlb on` stops on every TLB exception.

| Class | Exceptions |
|---|---|
| `all` | everything |
| `int` | interrupts |
| `tlb` | TLB refill, invalid, modified |
| `addr` | address errors (load/store) |
| `bus` | bus errors (instruction/data) |
| `sys` | syscall, breakpoint |
| `ri` | reserved instruction, coprocessor unusable |
| `arith` | overflow, trap, FPE |
| `watch` | watchpoint |
| `vce` | virtual coherency exceptions |

### History, undo and tracing

| Command | Description |
|---|---|
| `dt [n]` / `dt file <path> [n]` | Disassemble the traceback: the last `n` instructions executed (the history holds 1,048,576) |
| `undo on` / `undo off` | Enable / disable the undo buffer **[DEV]** |
| `undo [n]` / `u` | Step back `n` instructions, reversing register and memory changes **[DEV]** |
| `undo clear` / `undo resize <n>` | Clear / resize the undo history **[DEV]** |
| `debug <on\|off\|file <path>>` | Print every executed instruction, or write them to a file **[DEV]** |
| `trace start <path>` / `trace stop` / `trace status` | Record full per-instruction architectural state to a file **[DEV]** |
| `idleprof <on\|off\|report [n]\|stats>` | Sample the PC to find idle and spin loops (`idle-pause` builds); `stats` prints park counters |
| `idle <on\|off\|status>` | Turn idle parking on or off at runtime, or show park counters and the detector's last window (`idle-pause` builds) |
| `instrstats [report\|clear\|dump]` | Per-instruction decode/execute counters (`--features instr_stats`) **[DEV]** |

The undo buffer tracks register changes and memory writes. It does not
restore peripheral state, so treat what it shows after undoing across device
accesses with suspicion.

### TLB and caches

| Command | Description |
|---|---|
| `tlb dump` | All TLB entries (48 on R4400/R5000, 64 on R10000) |
| `tlb trans <vaddr> [asid]` | Translate through the TLB, optionally with another ASID |
| `tlb debug <on\|off>` | Log TLB operations **[DEV]** |
| `l1i <check\|dump> <addr\|index>` | L1 instruction cache |
| `l1d <check\|dump> <addr\|index>` | L1 data cache |
| `l1d wb <vaddr> <size>` / `l1d pwb <paddr> <size>` | Write cached L1D data back to RAM (virtual / physical range) |
| `l2 <check\|dump> <addr\|index>` | Unified L2 (R4400) |
| `ll` / `ll stats` / `ll clear` | LL/SC state (LLbit, LLAddr); the histogram needs `--features llstats` |

The R5000 has only its 2-way L1s. The R10000 keeps shadow tag/data/ECC arrays
for CACHE instructions and diagnostics; its ordinary accesses go straight to
memory ([FEATURES.md](FEATURES.md#emulated-cpu)).

### JIT v2

With `--features jitv2`, `j2` controls and inspects the MIPS JIT (concepts in
[HACKING.md](HACKING.md#8-jitv2-the-mips-jit)). Settings read at compile time
take effect after `j2 flush`.

| Command | Effect | Default (release / `developer`) |
|---|---|---|
| `j2 status` (alias `j2 stats`) | Arena use, compile counts, skipped/rejected recompiles, reject reasons | — |
| `j2 dispatch [on\|off]` | Main switch for JIT dispatch; off = interpreter only | `on` |
| `j2 inline [on\|off]` | Compile synchronously on the CPU thread instead of the compile pool | `off` (`on` under lockstep) |
| `j2 threads` | Compile-pool thread count (read-only; set with `[jitv2] threads`) | — |
| `j2 opt [none\|speed]` | Cranelift optimisation level | `speed` / `none` |
| `j2 min-instrs [N]` / `j2 max-instrs [N]` | Minimum / maximum instructions walked per compile | `0` / `1024` |
| `j2 min-calls [N]` | Legacy setting; stored and reported but not used by whole-page dispatch | `0` |
| `j2 intrun [N]` | Instructions sharing one pending-interrupt check; 1 = every instruction (forced under lockstep) | `1` |
| `j2 fallback [on\|off]` | Keep an unsupported instruction inside a region as an interpreter call instead of ending the region | `off` |
| `j2 entrypre [on\|off]` | Check for interrupts at an externally dispatched entry word too | `off` |
| `j2 inline_mem [on\|off]` | Inline L1D loads and stores (forced off under lockstep) | `on` |
| `j2 memhelpers [on\|off]` | Call shared memory-helper functions instead of inlining the cache probe | — |
| `j2 pagewb [on\|off]` | Write back dirty L1D/L2 lines covering a code page when a fetch crosses onto it (diagnostic) | `off` |
| `j2 <alu\|fpu\|branch\|loadstore\|cop0> [on\|off]` | Enable or disable compiling one instruction category | `on` |
| `j2 instrs [category]` | List instructions and whether each has an emitter and is enabled | — |
| `j2 flush` | Drop all compiled code and reset the arena (stop the CPU first) | — |
| `j2 clear <paddr>` / `j2 deny <paddr>` | Reset one physical code page / deny one entry point | — |
| `j2 analyze <addr>` | Run the analyzer's walk on live memory and print each word with its classification | — |
| `j2 pcp [addr]` | Physical code page state: entries, generation, compiled code | — |
| `j2 dumppcp [addr] [path]` | Capture a page and its memory for the offline `jitv2-pcp-dump` | — |
| `j2 corpus [dir]` | Dump every cached page as `.pcp` files for offline codegen measurement | `jitv2_corpus/` |
| `j2 html [path]` | Render the physical code page visualiser | — |
| `j2 seal-queue` | Peek at the shared arena's seal queue | — |
| `j2 lockstep` / `j2 lstate [full] [N]` | Lockstep status / recent lockstep history (`jitv2_lockstep` builds) | — |

`jitcheck <n> [skip]` **[DEV]** captures the live state, runs `n`
instructions interpreter-only and through real JIT dispatch, steps past the
first `skip` divergences and stops at the next one.

Verification tools, cheapest first: the `jitv2/equiv_test.rs` unit tests, the
cpu-tests matrix (`cpu-tests/run/matrix.sh`), `jitcheck`,
`--features jitv2_lockstep` (every instruction checked against the
interpreter), `--features jitv2_smc_check` (writes onto executing pages), and
`--features fetchverify` (stale code both engines would agree on).

### SCSI, CD-ROM and disks

| Command | Description |
|---|---|
| `scsi status` | Attached devices, disc lists and queue positions |
| `scsi regs` | WD33C93A registers |
| `scsi eject <id>` | Cycle to the next disc on CD-ROM `id` |
| `scsi add <id> <path>` | Insert a disc image as next in the queue (checks the file exists) |
| `scsi list <id>` | Discs in the queue with ordinals |
| `scsi del <id> <ord>` | Remove a queued disc (not the active one) |
| `scsi next <id> <ord>` | Move a disc to position 1 (next on eject) |
| `scsi defer <on\|off>` | Deferred SCSI status interrupts |
| `scsi dayna` | DaynaPort MAC, addresses and counters |
| `scsi wdt [N]` / `scsi wdt file <path>` | WD33C93A transaction ring: last `N` / all to a file **[DEV]** |
| `scsi debug <on\|off>` | Per-command trace **[DEV]** |
| `scsi0 ...` / `scsi1 ...` | The same, for controller 0 / controller 1 (Indigo2 and O2) |
| `cow status` / `cow commit [id]` / `cow reset [id]` | Copy-on-write overlays (`cow1 ...` on the O2's second controller) |

On the O2 the controllers are Adaptec 7880s: `scsi regs` shows that chip's
interrupt status, queues and command totals, whether its worker thread is
running (commands run on it, off the CPU's thread), any completions held
for a paused sequencer and NetBSD's doorbell, and `wdt`, `defer`, `debug` and
`dayna` are not there. Ids are per controller, as in the `[scsi.N]` tables
(`[scsi.17]` is id 1 on controller 1, so `scsi1 eject 1`).

On the Octane the controllers are ISP1040s: `scsi notes` (`scsi1 notes`)
shows the command total and anything odd the driver did; `regs`, `wdt`,
`defer`, `debug` and `dayna` are not there. It also has `heart status`
(interrupt bits, mask, counter, bank descriptors), `heart unsupported`
(HEART registers touched but not modelled) and `ip30 memcfg|bridge|gfx|xbow|unmapped`
(the banks as HEART places them, the Bridge's interrupt routing, the graphics
widget's link, refused Crossbow accesses, and accesses to physical addresses
nothing is mapped at, with the PC; a device's own refusal is not listed). The
`phys` commands work on the Octane as well.

On the Fuel the one controller is a two-channel ISP12160: `scsi` (or
`scsi0`) and `cow` act on channel 0, `scsi1` and `cow1` on channel 1, and
ids run 0 to 15 on each (`[scsi.17]` is id 1 on channel 1). `scsi notes`
shows the whole chip's command total and anything odd the driver did, and
has no `scsi1` form; `regs`, `wdt`, `defer`, `debug` and `dayna` are not
there.

Queue position `[0]` is the mounted disc, `[1]` the next on eject. A disc
change reaches IRIX as a Unit Attention ("medium changed") on its next
TEST UNIT READY; IRIX's `mediad` may still need a manual remount
(`clanker/irix/cdrom-changer-eject-no-mediad-remount.md`).

### Graphics

Traces and recordings need `developer` or `--features gputrace`. The
graphics debugging workflow (test programs, pixel checks, offline replay) is
in the `irix-gfx-loop` skill (`clanker/skills/irix-gfx-loop/SKILL.md`).

**Newport (REX3)**

| Command | Description |
|---|---|
| `rex status` | All REX3 drawing registers |
| `rex jit <on\|off\|status\|list>` | REX3 shader JIT (`rex-jit`): switch, counters, compiled and failed shaders |
| `rex jit <disable\|enable> <dm0> <dm1>` | Take one draw mode out of (or back into) JIT dispatch |
| `rex trace <file> [gfifo,cpu\|all]` / `rex trace off\|flush` | Annotated register trace (also `IRIS_REX3_TRACE=<file>`) |
| `rex rec <file>[.txt]` / `rex rec mark\|off` | Replayable bus recording (also `IRIS_REX3_REC=<file>`); replay with `REX3_REPLAY=<file> cargo test --release rex3_replay_file` |
| `rex fbdump [DIR]` | Framebuffers as files in `DIR` (default `.`) |
| `rex cmap <on\|off>` | CMAP access trace |
| `rex debug <on\|off>` | REX3 register trace **[DEV]** |
| `vc2 status` / `vc2 ramdump` / `vc2 debug <on\|off>` | VC2 video timing and cursor state / RAM dump / trace **[DEV]** |
| `xmap status` / `xmap debug <on\|off>` | XMAP9 state / trace **[DEV]** |
| `cmap status` / `cmap debug <on\|off>` | CMAP state / trace **[DEV]** |
| `bt445 status` / `bt445 identity` / `bt445 debug <on\|off>` | RAMDAC state / reset the palette to a linear ramp / trace **[DEV]** |
| `dcb debug <on\|off>` | Display control bus trace **[DEV]** |
| `disp status` / `disp debug <on\|off>` | Display timing and framebuffer state |
| `disp compositor <gl\|sw>` | Switch between the GL and software compositor |
| `block debug <on\|off>` | Block/span draw logging to `block.log` **[DEV]** |
| `draw debug <on\|off>` | Draw a debug overlay on the framebuffer **[DEV]** |

**GR2 (XZ / Extreme)**

| Command | Description |
|---|---|
| `gr2 [status]` | Board summary |
| `gr2 hq` | HQ2 registers, FIFO state and the command interpreter's state |
| `gr2 ge <n> [start] [count]` | GE7 window words |
| `gr2 shram [start] [count]` | Shared RAM words |
| `gr2 ucode [start] [count]` | HQ2 microcode RAM |
| `gr2 vc1` / `gr2 vc1 sram <addr> [n]` | VC1 registers, cursor, DID table / SRAM (byte address) |
| `gr2 xmap [ch]` | XMAP5 mode table |
| `gr2 clut [start] [n] [ch]` | Colour lookup table entries |
| `gr2 dac` | Bt457 state |
| `gr2 pix <x> <y>` | One display pixel: VRAM word, DID, XMAP mode, composed RGB |
| `gr2 fbdump [dir]` | VRAM planes and the composed screen as PNG and raw files |
| `gr2 trace <file> [hq,re3,cpu\|all]` / `gr2 trace flush\|off` | Annotated command trace |
| `gr2 jit [on\|off\|sync\|list]` | RE3 span JIT (`gr2-jit`) mode, counters, shaders |
| `re3 [regs]` / `re3 pix <x> <y> [w] [h]` | RE3 raster registers / raw pixels |

**IMPACT (MGRAS)**

| Command | Description |
|---|---|
| `mgras [status]` | Board summary and inventory, with the microcode fingerprints (HQ, each GE11) and the IRIX release they came with when known |
| `mgras hq` | HQ3 flags, enables, DMA and raster-interface registers, FIFOs |
| `mgras gl` | The GL front end's current state |
| `mgras probe <lo> <hi> <value>` / `mgras probe off` | Make GE readbacks in `lo..=hi` answer `value` (to map what libGLcore decodes) |
| `mgras ge [n] [start] [count]` | GE11 microcode lines |
| `mgras ucode [start] [count]` | HQ3 microcode RAM |
| `mgras vc3` / `mgras vc3 sram <a> [n]` | VC3 registers, cursor, DID runs / SRAM words |
| `mgras xmap` | XMAP display modes per DID and register files |
| `mgras cmap [start] [n] [0\|1]` | Colormap entries |
| `mgras dac` | DAC registers and gamma |
| `mgras pix <x> <y>` | One pixel through the compositor (y = 0 is the top) |
| `mgras stats` | Command, primitive and DMA counters |
| `mgras jit` | Raster JIT (`gr4-jit`) state |
| `mgras fbdump [dir]` | Planes and the screen as PNG and raw files |
| `mgras shot <file.png>` / `mgras dump <file>` | Screenshot / the board state image that replay checkpoints hash (waits for the board to go idle) |
| `mgras trace <file> [hq,rss,cpu,tex\|all]` / `mgras trace flush\|off` | Annotated trace (also `IRIS_MGRAS_TRACE=<file>`) |
| `mgras rec <file>` / `mgras rec mark\|off` | Replayable recording; replay with `MGRAS_REPLAY=<file> cargo test --release --lib mgras_replay_file` |
| `rss [regs]` / `rss pix <x> <y> [w] [h]` | Raster registers (decoded IR, each engine's backlog) / raw pixels |

`iris-ci screenshot` captures the composed screen from any board, also
headless.

**Screen, pointer and keyboard (any board)**

These work the same on Newport, GR2, IMPACT and the O2, with a window, with
`--ci`, or alongside VNC viewers. Coordinates are screen pixels as a
screenshot shows them (0,0 top-left). The guest's mouse is relative and X
accelerates it, so `ptr` steers: it sends a move, reads back where the
board's hardware cursor went, and corrects until the pointer is on the
target. The first `ptr` move pins the pointer in the top-left corner once,
to learn the cursor's hotspot exactly.

| Command | Description |
|---|---|
| `screen info` | Size, hardware cursor, pointer position, whether the guest has enabled the mouse, buttons held |
| `screen shot <file.png> [x y w h]` | The composed screen as PNG (no host status bar), optionally cropped |
| `screen pix <x> <y>` | One pixel, `rrggbb` |
| `screen wait [quiet_ms] [timeout_ms]` | Wait until the screen has not changed for `quiet_ms` (default 500, at most 10000 ms) |
| `screen cursor` | The cursor glyph as text, with its hotspot |
| `ptr pos` | Where the pointer is |
| `ptr move <x> <y>` | Put the pointer on a pixel |
| `ptr click <x> <y> [left\|middle\|right] [count]` | Move, then click (`count` 2 for a double click) |
| `ptr down\|up [button]` | Press or release a button where the pointer is |
| `ptr drag <x1> <y1> <x2> <y2> [button]` | Press at one point, release at the other |
| `ptr rel <dx> <dy>` | Raw relative motion, as a mouse sends it |
| `ptr calibrate` | Pin the pointer in the top-left corner and learn the current cursor's hotspot |
| `key <name>[+<name>...] [down\|up]` | Press a key or chord (`Return`, `F1`, `ctrl+c`, `ctrl+alt+Delete`): X keysym names, short aliases (`ctrl`, `alt`, `shift`, `enter`, `esc`, `tab`, `bs`, `del`, `pgup`), single characters, or winit names (`KeyA`) |
| `key type <text>` | Type ASCII text on a US layout |
| `key names` | Every key name `key` accepts |

### Hardware devices

| Command | Description |
|---|---|
| `mc regs` | Memory Controller registers, including MEMCFG |
| `mc dma` | VDMA (graphics DMA) state |
| `mc dma xlate [vaddr] [w]` | Walk the VDMA address translation for `vaddr` (default MEMADR) without side effects; `w` = the transfer writes memory |
| `mc vdma <on\|off>` | VDMA trace to `vdma.log` |
| `eeprom <on\|off\|dump\|r <word>\|w <word> <val>>` | CPU/MC boot-configuration EEPROM (93C56; cache size etc.) |
| `nveeprom <on\|off\|dump\|r <word>\|w <word> <val>\|save [file]>` | Indigo2 NVRAM EEPROM (93CS56: PROM environment and MAC) |
| `hpc3 status` | HPC3 state |
| `pdma status` / `pdma chain <addr>` | PBUS DMA channels / decode a descriptor chain at a physical address |
| `pdma dump <on\|off\|hal\|scsi\|enet\|MASK>` | PDMA trace **[DEV]** |
| `ioc status` | IOC interrupt controller state |
| `ioc ticks` | 8254 timer service since the last call: ticks fired, acknowledged, merged; fire-to-acknowledge waits |
| `rtc status` / `rtc dump` | RTC registers (and any `[rtc_offset]`) / NVRAM bytes |
| `rtc save [file]` | Save the RTC NVRAM |
| `rtc r <offset>` / `rtc w <offset> <val>` | Read / write NVRAM bytes |
| `rtc debug <on\|off>` | RTC trace **[DEV]** |
| `pit status` / `pit debug <on\|off>` | 8254 timer state / trace **[DEV]** |
| `hal2 status` | HAL2 audio state |
| `ps2 status` / `ps2 debug <on\|off>` | PS/2 controller state / keyboard and mouse trace |
| `ps2 type <ascii>` / `ps2 enter` / `ps2 esc` | Type text / press Enter / press Esc (Esc stops PROM autoboot) |
| `ps2 key <shift\|ctrl\|alt> <down\|up>` | Hold or release a modifier |
| `ps2 mouse <dx> <dy> [buttons]` | Relative mouse motion with an optional button mask (1 left, 2 right, 4 middle) |
| `serial status` | SCC channel A/B registers and FIFOs |
| `vino status` / `vino debug <on\|off>` | VINO registers, channel state, descriptor cache / register and I2C trace |

`nveeprom save` (Indigo2) and `rtc save` (Indy) persist PROM changes before
a forced termination; a normal stop saves them automatically.

### Networking

| Command | Description |
|---|---|
| `seeq status` | SEEQ 8003 ethernet state |
| `net status [tcp\|udp\|icmp\|all]` | NAT connection table |
| `net interfaces` | Host interfaces available for PCAP |
| `net debug <tcp\|udp\|icmp> <on\|off>` | Per-packet trace **[DEV]** |

`IRIS_TFTP_TRACE` and `IRIS_RSH_TRACE` log the TFTP and rsh services
([FEATURES.md](FEATURES.md#environment-and-diagnostic-controls)).

### Physical bus

| Command | Description |
|---|---|
| `phys mem <addr> [n]` / `mm` | Dump physical memory |
| `phys dis <addr> [n]` / `md` | Disassemble physical memory |
| `phys map` | The decode table: pages by kind (untouched, filled but not yet mapped, shared, private), memory in use, slots faulted in, and the table as runs of devices. Every machine on `Physical` has the `phys` commands (Indy, Indigo2, O2, Octane, Fuel) |
| `phys trace <on\|off>` / `trace <on\|off>` | Log physical bus accesses |
| `phys error <on\|off>` | Stop on bus errors |
| `phys hole <on\|off>` | Stop on accesses to unmapped addresses |
| `phys bench` / `bench` | Memory write benchmark: fills the first fully mapped 64 MB of RAM (overwrites it) |

### Logging

Per-module logging; all of it needs a developer build to produce output.
**[DEV]**

| Command | Description |
|---|---|
| `log status` | Each module's state |
| `log <module\|all> <on\|off>` | Enable / disable |
| `log <module> mask <cat\|hex>` | Category mask |
| `log <module> file <path\|off>` | Send a module's log to a file |

Modules: `net hpc3 seeq hal2 mc rex3 mips ioc scsi pdma vino dcb vc2 cmap
xmap bt445 scc ps2 rtc eeprom l1i l1d l2c bedrock l1 usb`. Mask categories: `pdma` —
`hal enet scsi`; `mips` — `insn tlb mem fpu cp0`; `l1i`/`l1d`/`l2c` — `hit
miss op`; every module also takes `on`/`all`, `off`/`none` or a hex mask.
`bedrock` covers the Fuel's devices on its Bedrock hub, `l1` its system
controller and `usb` its USB host controller.
`IRIS_DEBUG_LOG` or `[debug] debug_log` set the same thing at startup.

### Performance and locks

| Command | Description |
|---|---|
| `perf snapshot` | Performance counters |
| `locks` | Every registered lock, whether it is held, and by which thread and call site (for `SpinLock`s) |
| `perfetto start FILE [cats=LIST] [clock=host\|guest]` | Start a Perfetto trace (`--features perfetto`) |
| `perfetto cats LIST` / `perfetto status` / `perfetto stop` | Change categories / show / stop |
| `idleprof ...` | See [history and tracing](#history-undo-and-tracing) |

See [docs/perfetto.md](docs/perfetto.md) for categories and analysis.

### N64 development board

With `[ultra64] enabled = true`: `ultra status | send <hex> | reset | r
<offset> | w <offset> <val> | dump <offset> <size> | disasm <offset> <count>
| load <path> [offset]`. See [HELP.md](HELP.md#n64-development-board-ultra64).

### SGI O2 (IP32)

The O2's devices register their own commands (see [the O2 guide](docs/o2-ip32.md));
its SCSI controllers answer to `scsi`/`cow` and `scsi1`/`cow1` as described under
[SCSI, CD-ROM and disks](#scsi-cd-rom-and-disks):

| Command | Description |
|---|---|
| `crime status` / `crime regs` / `crime irqlog [on\|off\|n]` / `crime touched` | CRIME interrupts, mask and memory banks / registers / the last *n* interrupt-register accesses (recorded only after `crime irqlog on`, which clears the old log; never in a `lightning` build) / registers the guest used (not kept in a `lightning` build) |
| `re status` / `re notes` | CRIME rendering engine state / notes on unusual commands |
| `gbe status` | GBE display engine: timing, frame buffer, tiles |
| `isa status` | MACE ISA interrupts |
| `pci status` / `pci config <slot>` | MACE PCI bridge / a function's configuration space |
| `mec status` | MACE Ethernet: rings, receive buffers, frame counts |
| `sdma status` | MACE serial DMA |
| `maceps2 status` / `maceps2 traffic` | PS/2 ports / recent bytes each way |
| `audio status` / `audio codec` | MACE audio rings and rates / AD1843 codec registers |
| `ust status` | MACE UST counter |
| `rtc status` / `rtc dump` / `rtc r <reg>` / `rtc w <reg> <val>` | DS1687 RTC |
| `onewire status` | MACE identity register and the ID chip holding the MAC address |

`o2 <command>` runs the same commands `[o2] control_port` serves, one per line:

| Command | Description |
|---|---|
| `o2 shot PATH` | Screen shot (PNG) |
| `o2 status` | GBE display state |
| `o2 pixel X Y` | One screen pixel |
| `o2 type TEXT` | Type on the PS/2 keyboard (`\r` is Enter) |
| `o2 key\|keydown\|keyup NAME` | Press, hold or release a key |
| `o2 mouse DX DY [BUTTONS]` / `o2 click [BUTTON]` | Move the mouse / click |
| `o2 pc` / `o2 regs` | CPU PC / registers as last published |
| `o2 peek ADDR LEN PATH` | Physical memory to a file (up to 1 MB) |
| `o2 retrace PATH` | Recorded rendering-engine primitives (`IRIS_IP32_RE_TRACE`) |
| `o2 scsi PATH` | The SCSI controllers' recent commands |
| `o2 unmapped` | Bus accesses that reached no device (the first 64 distinct), with the PC that made each (not kept in a `lightning` build) |
| `o2 ps2log PATH` / `o2 crimelog PATH` | PS/2 traffic / CRIME interrupt-register accesses recorded since `crime irqlog on` |

## Recipes

**Stop at a function and look around** (release or developer build):

```text
> loadsym unix.map
> b bcopy if a2 > 0x10000
> wait 30000
> regs
> bt
> dt 20
> c
```

`wait` returns when the breakpoint hits (or after 30 s) and says where the
CPU stopped.

**Read an IRIX panic.** Load the kernel's symbols, read what the kernel
printed, then decode the exception frame address (`ep`) the panic message
reported:

```text
> loadsym unix.map
> putbuf
> eframe <ep address from the panic message>
```

**Step backwards** (developer build):

```text
> undo on
> s 10
> u 3
> regs
```

For the IRIX side of kernel debugging (making boot verbose, walking device
discovery from probe to root mount), see the `irix-kernel-debug` skill in
`clanker/skills/`.

**Find out why a device looks stuck**: `status` (is the CPU running, why did
it stop), `locks` (who holds what), the device's own `status` command, then
`ioc status` / `hpc3 status` for its interrupt.

## GDB stub

`--gdb-port <port>` starts a GDB Remote Serial Protocol stub
(`src/gdb_stub.rs`) on `127.0.0.1:<port>`, one client at a time. Breakpoints
set by GDB are removed when the client disconnects.

```sh
cargo run --profile developer --features developer -- --gdb-port 1234   # CPU starts paused
cargo run --release -- --gdb-port 1234                                  # attach to a running system
```

Connect with `mips64-unknown-linux-gnu-gdb` or `gdb-multiarch`:

```
set architecture mips:isa64
set mips abi n64
set mips mask-address off
set heuristic-fence-post 0
set backtrace past-main on
set backtrace limit 0
target remote localhost:1234
```

- `set mips abi n64` is required; without it GDB shows the 64-bit registers
  as 32-bit.
- `set mips mask-address off` stops GDB from truncating 64-bit kernel
  addresses (`0xffffffff80010000`) in breakpoint packets.
- `set heuristic-fence-post 0` silences "can't find start of function"
  warnings when stepping without symbols.
- `set debug remote 1` logs every packet, useful for register layout or
  breakpoint address problems.

| GDB command | Effect |
|---|---|
| `info registers`, `p $pc`, `set $pc = ...` | Read / write registers |
| `x/Ni $pc`, `x/Nw 0x...`, `set {int}0x... = N` | Disassemble, read, write memory (byte-granular, through `debug_read`/`debug_write`) |
| `break *0x...`, `delete` | Software breakpoints on virtual addresses |
| `watch *0x...`, `rwatch *0x...` | Write / read watchpoints |
| `continue`, `stepi`, Ctrl-C | Run, single-step, interrupt |

`stepi` is synchronous. `continue` runs the CPU on a background thread and
reports when it stops. Register and memory access need a stopped CPU; a
developer build starts paused, so you can inspect before the first
instruction.

The g-packet holds 72 registers (576 bytes): r0–r31 (0–31), Status (32), LO
(33), HI (34), BadVAddr (35), Cause (36), PC (37), f0–f31 (38–69), FCSR (70),
FIR (71). A 73rd pseudo-register, `fp`, is declared in the target description
to keep GDB's frame heuristics quiet but is not in the g-packet.

## MCP server

`clanker/iris_mcp.py` exposes the monitor as Model Context Protocol tools,
so an AI assistant can debug a live machine. It needs the Python MCP SDK
(`pip install mcp`) and connects to the monitor on 127.0.0.1:8888, retrying
until iris is up.

```sh
python3 clanker/iris_mcp.py
```

Register it with your client, for example in `.mcp.json`:

```json
{ "mcpServers": { "iris": { "type": "stdio", "command": "python3",
    "args": ["/path/to/iris/clanker/iris_mcp.py"] } } }
```

| Tool | Monitor command |
|---|---|
| `run_command(command)` | anything |
| `get_status()`, `wait_for_stop(timeout_ms)` | `status`, `wait` |
| `start_cpu()`, `stop_cpu()` | `start`, `stop` |
| `step(count)`, `next_instruction(count)`, `continue_execution(until)`, `finish_function()` | `step`, `next`, `run`, `finish` |
| `get_registers()`, `read_cop0()`, `read_cop1()` | `regs`, `cop0`, `cop1` |
| `read_memory(address, count, size)`, `write_memory(address, value, size)`, `read_string(address, max_len)` | `mem`, `mw`, `ms` |
| `disassemble(address, count)` | `dis` |
| `add_breakpoint(address, kind, condition)`, `remove_breakpoint(id)`, `list_breakpoints()` | `bp add`, `bp del`, `bp list` |
| `backtrace(frames)`, `traceback(count)`, `undo(count)` | `bt`, `dt`, `undo` |
| `translate_address(address)`, `dump_tlb()` | `translate`, `tlb dump` |
| `lookup_symbol(name_or_addr)` | `sym` |
| `read_eframe(address, layout)`, `kernel_log(size)` | `eframe`, `putbuf` |
| `screenshot(x, y, width, height, max_width)` (returns the image) | `screen shot` |
| `screen_info()`, `screen_pixel(x, y)`, `wait_screen(quiet_ms, timeout_ms)` | `screen info`, `screen pix`, `screen wait` |
| `click(x, y, button, count)`, `move_pointer(x, y)`, `drag(x1, y1, x2, y2, button)` | `ptr click`, `ptr move`, `ptr drag` |
| `type_text(text)`, `press_key(keys, action)` | `key type`, `key` |

Commands without a dedicated tool (graphics, devices, `j2`) go through
`run_command`.

## Other introspection

- **VNC**: `--vnc` (or `[vnc] listen`) serves the screen, keyboard and mouse
  to any VNC viewer, from any board, with a window or under `--ci`. See
  [HELP.md](HELP.md#vnc).

- **`iris-ci`**: the CI control socket drives a running machine from scripts:
  boot and log in, run guest commands, copy files in and out, screenshots,
  snapshots, CD changes ([STORAGE.md](STORAGE.md#ci-control-socket-and-iris-ci)).
- **Serial console**: ports 8880 (channel A) and 8881 (channel B, the IRIX
  console); `serial_log` writes the console to a file ([HELP.md](HELP.md#serial-ports)).
- **Environment variables** for traces, JIT diagnostics and logging are
  listed in [FEATURES.md](FEATURES.md#environment-and-diagnostic-controls).
- **Crashes on Windows**: `crash_diag.rs` logs otherwise silent process
  deaths (`0xC000041D` and friends) with a symbolised stack to
  `iris-crash.log`.
- **Perfetto** timelines of interrupts, graphics, JIT and devices:
  [docs/perfetto.md](docs/perfetto.md).
