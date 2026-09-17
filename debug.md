# IRIS debugger

The IRIS emulator features a built-in monitor and debugger that allows for interactive inspection and control of the emulated machine. The monitor listens on TCP port 8888 by default, and is also available on the terminal `iris` was started from.

This is a guided tour of the debugger. [HELP.md](HELP.md#monitor-console) has the complete command reference, and [HACKING.md](HACKING.md#9-gdb-stub) covers the GDB stub.

> **Build flavour matters.** `lightning` builds (and the official releases) strip breakpoint checks and the traceback buffer, so breakpoints and `dt` do nothing there. The undo buffer, `si`, `debug` tracing and `trace` recording need a `developer` build (`cargo run --profile developer`), which also starts with the CPU paused.

## Connecting

You can connect to the debugger using `netcat` or `telnet`:

```bash
nc localhost 8888
```

## Execution control

| Command | Alias | Description |
| :--- | :--- | :--- |
| `start` | | Start the CPU execution thread. |
| `stop` | | Stop the CPU execution thread. |
| `run [addr]` | `c`, `cont` | Continue execution. If an address is provided, runs until that address is hit (temporary breakpoint). |
| `step [count\|addr]` | `s` | Step `count` instructions (default 1). If an address is provided (e.g., `step 0x88001000`), runs until that address. |
| `si` | | Step without taking interrupts (developer builds). |
| `next [count]` | `n` | Step over function calls (executes `jal`/`bal` as one unit). |
| `finish` | `fin` | Run until the current function returns (detects return address). |
| `jump <addr>` | | Force the PC to a specific address. |

## Breakpoints

Breakpoints can be set on execution (PC), memory reads, or memory writes.

| Command | Alias | Description |
| :--- | :--- | :--- |
| `bp list` | `bl` | List all defined breakpoints. |
| `bp add <addr> [type] [if <expr>]` | `b` | Add a breakpoint at `addr`. Type can be `pc` (default), `r` (read), `w` (write), `f` (fetch), `pr` (phys read), `pw` (phys write), `pf` (phys fetch). An optional `if <expr>` makes it conditional. |
| `bp del <id>` | `bb` | Delete breakpoint with the specified ID. |
| `bp enable <id>` | `be` | Enable a disabled breakpoint. |
| `bp disable <id>` | `bd` | Disable a breakpoint without deleting it. |

## Inspection

### Registers

| Command | Alias | Description |
| :--- | :--- | :--- |
| `regs` | `r` | Dump General Purpose Registers (GPRs), HI/LO, and key CP0 registers (Status, Cause, EPC, BadVAddr). |
| `cop0` | | Dump all Coprocessor 0 (System Control) registers. |
| `cop1` | | Dump Coprocessor 1 (FPU) registers and control/status registers. |

### Memory

| Command | Alias | Description |
| :--- | :--- | :--- |
| `mem <addr> [count]` | `m` | Dump virtual memory at `addr`. Default count is 1 word. |
| `mw <addr> <val> [size]` | | Write `val` to virtual memory at `addr`. Size can be `b` (byte), `h` (half), `w` (word), or `d` (double). Default is word. |
| `ms <addr> [max_len]` | | Read a null-terminated string from virtual memory. |
| `stack [addr] [count]` | | Dump stack memory. Defaults to current `$sp` if address is not provided. |
| `dis <addr> [count]` | `d` | Disassemble instructions at `addr`. |

### Translation and TLB

| Command | Alias | Description |
| :--- | :--- | :--- |
| `translate <addr>` | `t` | Translate a virtual address to a physical address using the current TLB and addressing mode. |
| `tlb dump` | | Dump all TLB entries. |
| `tlb trans <vaddr> [asid]` | | Debug translation of a virtual address with an optional ASID. |
| `tlb debug <on\|off>` | | Enable verbose logging of TLB operations. |

## Undo and time travel

The emulator maintains a circular buffer of previous CPU states, allowing you to step backwards in time.

> **⚠️ Warning:** The undo feature is powerful but can be **glitchy**. It tracks register changes and memory writes but may not perfectly restore peripheral state. It needs a `developer` build and must be explicitly enabled.

| Command | Alias | Description |
| :--- | :--- | :--- |
| `undo on` | `u on` | **Enable** the undo buffer. |
| `undo off` | `u off` | Disable the undo buffer. |
| `undo [count]` | `u` | Step back `count` instructions (default 1). Reverses register and memory changes. |
| `undo clear` | | Clear the undo history buffer. |
| `undo resize <n>` | | Change the undo buffer size. |

## Tracing and history

| Command | Alias | Description |
| :--- | :--- | :--- |
| `dt [count]` | | **Disassemble Traceback**: Show the last `count` instructions executed by the CPU. Useful for seeing how you got to the current PC. `dt file <path> [count]` writes it to a file. |
| `bt [frames]` | | **Backtrace**: Attempt to walk the stack frames to show the call stack. |
| `debug <on\|off\|file <path>>` | | Verbose CPU instruction tracing (prints every instruction executed). Developer builds. |
| `trace start <path>` / `trace stop` / `trace status` | | Record full per-instruction architectural state to a file (developer builds). |
| `phys trace <on\|off>` | | Log physical bus accesses (useful for debugging I/O). |
| `idleprof <on\|off\|report>` | | Sample the PC to find idle/spin loops (`idle-pause` builds). |
| `ip7` | | Make the timer interrupt pending, to step through interrupt delivery while stopped. |

## Exceptions

You can configure the debugger to stop execution when specific exceptions occur.

**Usage:** `exception <class|code|all> <on|off>`

| Class/Code | Description |
| :--- | :--- |
| `all` | All exceptions. |
| `int` | Interrupts. |
| `tlb` | TLB Refill / Invalid / Modified. |
| `addr` | Address Errors (Load/Store). |
| `bus` | Bus Errors (Instruction/Data). |
| `sys` | Syscall / Breakpoint. |
| `ri` | Reserved Instruction / Coprocessor Unusable. |
| `arith` | Arithmetic Overflow / Trap / FPE. |
| `watch` | Watchpoint. |
| `vce` | Virtual Coherency Exceptions. |

Example: `ex tlb on` will stop execution whenever a TLB exception occurs.

## Symbols

The debugger can load symbol maps (e.g., `prom.map`, `unix.map`) to display function names instead of raw addresses.

| Command | Description |
| :--- | :--- |
| `loadsym <file>` | Load a symbol map file (NM output format). |
| `sym <addr>` | Lookup the symbol nearest to `addr`. |
| `proc info` | IRIX kernel introspection (utsname) once kernel symbols are loaded. |

## Loading code

| Command | Description |
| :--- | :--- |
| `loadelf <file>` | Load a static big-endian ELF32 and set PC to its entry point (also `--load-elf` on the command line). |
| `loadbin <file> <addr>` | Load raw bytes at a virtual address. |

## Cache debugging

Commands to inspect the internal state of the emulated caches. The R4400 has L1I, L1D and an L2; the R5000 has only 2-way L1I/L1D.

| Command | Description |
| :--- | :--- |
| `l1i <check\|dump> <addr\|index>` | Inspect L1 Instruction Cache. |
| `l1d <check\|dump> <addr\|index>` | Inspect L1 Data Cache. |
| `l2 <check\|dump> <addr\|index>` | Inspect L2 Unified Cache (R4400). |
| `l1d wb <vaddr> <size>` / `l1d pwb <paddr> <size>` | Write cached L1D data back to RAM. |
| `ll` | LL/SC state (llbit, lladdr); `ll stats` with `--features llstats`. |

## JIT debugging

With `--features jitv2`, `j2` controls and inspects the MIPS JIT (see HACKING.md §7). `jitcheck <n> [skip]` (developer builds) runs `n` instructions both interpreter-only and through JIT dispatch from the same state and stops at the first divergence; `j2 dispatch off` runs interpreter-only.

## Example session

```text
> start
CPU started
> bp add 0x88002000
Breakpoint 1 added at 0000000088002000 (Pc)
> run
... execution ...
PC=0000000088002000: Breakpoint 1 hit
> u on
CPU undo buffer enabled
> s
Exec: 88002000 <func+0x0>: 27bdffd8 addiu sp, sp, -40
> regs
... registers ...
> u
Undid 1 instruction(s), PC now at 0000000088002000
> dt 5
Execution Traceback (last 5 instructions):
...
```
## Execution modes

The emulator runs the CPU in one of two ways:

1. **Threaded mode** (`start`): the CPU runs on its own thread at full speed, releasing the executor lock every 500,000 instructions so monitor commands can get in. Breakpoints still stop it (except in `lightning` builds). `stop` joins the thread and returns once the CPU has fully stopped.
2. **Debug mode** (`run`, `step`, `next`, `finish`): `run_debug_loop` drives the CPU with breakpoint checks and tracing. `run`/`c` return immediately and let it run; `step`/`next`/`finish` wait for it to finish. Starting a debug command stops threaded mode first. `stop` works on both.

The GDB stub uses the same debug loop, so monitor and GDB breakpoints coexist.

## MCP server integration

The emulator includes a Model Context Protocol (MCP) server that exposes monitor commands as tools for AI assistants.

### Start the server

1. Start IRIS with the monitor enabled (default port 8888).
2. Install the Python MCP SDK (`pip install mcp`) and run the MCP server script:
   ```bash
   python3 src/iris_mcp.py
   ```

### Available tools

The MCP server exposes the following tools to connected clients:

| Tool | Description | IRIS Command |
| :--- | :--- | :--- |
| `run_command(cmd)` | Run raw monitor command | (any) |
| `read_memory(addr, count)` | Read memory words | `mem` |
| `write_memory(addr, val, size)` | Write memory | `mw` |
| `read_string(addr, max_len)` | Read string | `ms` |
| `get_registers()` | Dump GPRs | `regs` |
| `read_cop0()` | Dump CP0 regs | `cop0` |
| `read_cop1()` | Dump FPU regs | `cop1` |
| `step(count)` | Step instruction(s) | `step` |
| `get_status()` | Running state and PC | `status` |
| `next_instruction(count)` | Step over call | `next` |
| `continue_execution(until)` | Run (optional breakpoint) | `run` |
| `finish_function()` | Run until return | `finish` |
| `add_breakpoint(addr, kind)` | Add breakpoint | `bp add` |
| `remove_breakpoint(id)` | Delete breakpoint | `bp del` |
| `list_breakpoints()` | List breakpoints | `bp list` |
| `backtrace(frames)` | Show stack trace | `bt` |
| `traceback(count)` | Show execution history | `dt` |
| `undo(count)` | Undo instructions | `undo` |
| `translate_address(addr)` | VA to PA translation | `translate` |
| `dump_tlb()` | Dump TLB entries | `tlb dump` |
| `lookup_symbol(name_or_addr)` | Symbol lookup | `sym` |
| `disassemble(addr, count)` | Disassemble instructions | `dis` |