# HACKING on IRIS

## How does this thing work?

IRIS is an SGI Indy (MIPS R4400) emulator written in Rust. It is not cycle-accurate
anywhere. IRIX doesn't expect it, and accuracy would just make things slower.

---

## 1. Data Path & Endianness

**Word-Transparent Architecture.** Host `u32`/`u64` values are bit-containers — no
internal byte-swapping. Endianness is handled only at "The Edge" (PROM/disk I/O) via
`swap_on_load`. The CPU thread handles byte/half-word packing internally via bit-shifts.
The bus and MC always see aligned word/double-word values.

Do not suggest `.to_be()` or `.to_le()` for memory or register logic.

---

## 2. Concurrency Model

Every device can run in its own thread. CPU, REX3, SCSI, and ethernet each have their
own thread. `hptimer.rs` provides a repeating-event timer for devices that don't need
a dedicated thread.

Synchronisation is per-device: each device locks its own internal state. Be careful
when calling back up to a parent device (e.g. from SCSI → HPC3) — that is where
deadlocks live. Ethernet had two of them; see memory notes for the gory details.

Memory access is YOLO: all devices can freely read/write DRAM. It works because 32-bit
and 64-bit transactions are halfway atomic (though unordered). Hardware sync points are
maintained where the real hardware requires them.

---

## 3. Device, Bus & Port Abstraction

The **MC (Memory Controller)** is the central crossbar. All traffic — CPU PIO and HPC3
DMA — passes through it.

```rust
pub enum BusStatus { Ready, Busy, Error, Data(u32), Data64(u64) }

pub trait BusDevice: Send + Sync {
    fn read8/16/32/64(&self, addr: u32) -> BusStatus;
    fn write8/16/32/64(&self, addr: u32, val: u32) -> BusStatus;
}

pub trait Device: Send + Sync {
    fn clock(&mut self);
    fn run(&mut self);
    fn stop(&mut self);
    fn start(&mut self);
    fn is_running(&self) -> bool;
    fn get_clock(&self) -> u64;
}
```

Devices usually implement both traits and expose getters for each. Devices connect to
each other via implementation-specific `connect()` / `set_phys()` functions.

**Physical address map** (`physical.rs`) uses a 64KB-granularity lookup table for O(1)
address decoding. There is a lot of unsafe in there — it's intentional and somewhat
cursed.

Address space hierarchy:
- Upstream: SysAd (64-bit, 8-byte enables)
- Downstream: MC-mapped 64KB pages (DRAM, GIO-Bus)
- GIO-Bus: sub-decoder for HPC3 (GIO32) and Newport (GIO64)

---

## 4. Hardware Notes

**HPC3** — self-programming DMA; fetches descriptors via the MC bus.

**REX3/Newport** — has a 16-word internal write FIFO in hardware. We enlarged it to
64K entries so it can absorb large DMA transfers and render pixels while the CPU does
other things.

**Memory** — emulated as `Vec<u32>`. Banks 2 and 3 can be enabled (up to 512MB). PROM
is fine with it; IRIX 6.5 uses 384MB, 5.3 uses up to 512MB.

**Cache** — fully emulated L2 was a mistake in hindsight but here we are.
- L1I: virtually indexed, physically tagged (VIPT)
- L1D: virtually indexed, physically tagged (VIPT)
- L2: physically indexed, physically tagged
- Decoded instructions are cached in L2; L1I entries point into L2 (inclusive).
- VCE fires when VA[14:12] doesn't match the L2's stored pidx for the same physical line.

**Count/Compare** — calibrates itself to real time to fire fasttick at 1KHz using a
15-bit fixed-point counter. It is cursed.

---

## 5. Interrupts

The CPU thread polls an `AtomicU64` interrupt bitmask every instruction cycle.
INT2/INT3 (Local0/Local1) logic maps to R4000 IP2/IP3.

---

## 6. CPU Execution Architecture

**MipsCore** (`mips_core.rs`) — pure register state: GPRs r0–r31 (64-bit), CP0, CP1
FPU registers, interrupt state (`AtomicU64`). No execution logic.

**MipsExecutor** (`mips_exec.rs`) — execution engine combining core + memory:

```rust
pub struct MipsExecutor<T: Tlb, C: MipsCache> {
    pub core: MipsCore,
    pub sysad: Arc<dyn BusDevice>,
    pub tlb: T,
    pub cache: C,
    in_delay_slot: bool,
    pub delay_slot_target: u64,
    // + breakpoints, traceback, symbol table, decode cache, hot-path fn ptrs, ...
}
```

Key methods:
- `exec(instr: u32) -> ExecStatus` — execute one already-fetched instruction
- `step() -> ExecStatus` — fetch from PC and execute

PC advancement and delay slots are managed internally by the executor.

### ExecStatus

`ExecStatus` is a `u32` bit-field, **not an enum**.

```rust
pub type ExecStatus = u32;

// Normal (non-exception) status — bits [15:8]
pub const EXEC_COMPLETE:           ExecStatus = 0x0000_0000; // advance PC by 4
pub const EXEC_COMPLETE_NO_INC:    ExecStatus = 0x0000_0080; // PC already set, no increment
pub const EXEC_RETRY:              ExecStatus = 0x0000_0100; // bus busy, retry same instr
pub const EXEC_BRANCH_DELAY:       ExecStatus = 0x0000_0200; // branch taken; target in delay_slot_target
pub const EXEC_BRANCH_LIKELY_SKIP: ExecStatus = 0x0000_0400; // branch-likely not taken, skip delay slot
pub const EXEC_BREAKPOINT:         ExecStatus = 0x0000_0800; // breakpoint hit

// Exception flags — upper bits
pub const EXEC_IS_EXCEPTION:       ExecStatus = 1 << 27;     // 0x0800_0000
pub const EXEC_IS_TLB_REFILL:      ExecStatus = 1 << 28;     // 0x1000_0000 — use 32-bit UTLB vector
pub const EXEC_IS_XTLB_REFILL:     ExecStatus = 1 << 29;     // 0x2000_0000 — use 64-bit XTLB vector
```

Exception status values are built with helpers:

```rust
exec_exception(code)   // IS_EXCEPTION | (code << CAUSE_EXCCODE_SHIFT)
exec_tlb_miss(code)    // IS_EXCEPTION | IS_TLB_REFILL | code
exec_xtlb_miss(code)   // IS_EXCEPTION | IS_TLB_REFILL | IS_XTLB_REFILL | code
```

The EXC code lives in bits [6:2] of the status word (same position as CAUSE.ExcCode).

### Exception Codes

| Constant    | Value | Meaning                              |
|-------------|-------|--------------------------------------|
| `EXC_INT`   | 0     | Interrupt                            |
| `EXC_TLBL`  | 2     | TLB miss (load / instruction fetch)  |
| `EXC_TLBS`  | 3     | TLB miss (store)                     |
| `EXC_ADEL`  | 4     | Address error (load / fetch)         |
| `EXC_ADES`  | 5     | Address error (store)                |
| `EXC_IBE`   | 6     | Bus error (instruction fetch)        |
| `EXC_DBE`   | 7     | Bus error (data reference)           |
| `EXC_SYS`   | 8     | Syscall                              |
| `EXC_BP`    | 9     | Breakpoint                           |
| `EXC_RI`    | 10    | Reserved instruction                 |
| `EXC_CPU`   | 11    | Coprocessor unusable                 |
| `EXC_OV`    | 12    | Arithmetic overflow                  |
| `EXC_TR`    | 13    | Trap                                 |
| `EXC_FPE`   | 15    | Floating-point exception             |
| `EXC_WATCH` | 23    | Watchpoint                           |

### MemoryInterface / MemAccessSize

```rust
pub enum MemAccessSize { Byte = 1, Half = 2, Word = 4, Double = 8 }
```

Memory access goes through three separate paths (I-cache vs D-cache vs debug):
- `fetch_instr()` — instruction fetch (I-cache path)
- `read_data()` / `write_data()` — loads/stores (D-cache path)
- `debug_read()` / `debug_write()` — override privilege to kernel, never mutate CP0,
  ignore breakpoints/watchpoints

`is_64bit` flag selects 32 vs 64-bit addressing mode. The implementation handles
address translation (TLB + segment mapping), alignment checking, and cache simulation.

---

## 7. Building & Debugging

Normal build:
```
cargo run --release
```

Developer build (enables intrusive debug helpers; affects performance):
```
cargo run --release --features developer
```

The `developer` feature enables: undo buffer, pending-write tracking, extra MCP debug
commands, and some additional assertions.

**Breakpoints** don't survive emulator restart.

**Monitor console** — available in the terminal or via telnet to `127.0.0.1:8888`.
Serial ports are on 8880 (port A) and 8881 (port B / IRIX serial terminal).

---

## 8. GDB Stub

Iris includes a GDB Remote Serial Protocol stub (`src/gdb_stub.rs`) that lets you
connect GDB to the running emulator and debug IRIX/guest code with a real debugger.

### Starting the GDB stub

Pass `--gdb-port <port>` on the command line:

```sh
# Developer build — CPU starts paused, GDB can set breakpoints before first instruction
cargo run --profile developer -- --gdb-port 1234

# Release build — CPU starts running; GDB attaches to a live system
cargo run --release -- --gdb-port 1234
```

The stub binds `127.0.0.1:<port>`. One client at a time; breakpoints set by GDB are
automatically removed when the client disconnects.

### Connecting GDB

With `mips64-unknown-linux-gnu-gdb` (recommended):

```
set architecture mips:isa64
set mips abi n64
set mips mask-address off
set heuristic-fence-post 0
set backtrace past-main on
set backtrace limit 0
target remote localhost:1234
```

- `set architecture mips:isa64` — selects the 64-bit MIPS BFD target.
- `set mips abi n64` — required; without it GDB uses 32-bit o32 ABI and GPRs display
  as 32-bit even though the g-packet contains 64-bit values.
- `set mips mask-address off` — prevents GDB from sign-masking 64-bit kernel addresses
  (e.g. `0xffffffff80010000`) down to 32-bit when sending Z0 breakpoint packets.
- `set heuristic-fence-post 0` — suppresses "can't find start of function" warnings
  when stepping without symbol information.

With `gdb-multiarch`:

```
(gdb) set architecture mips:isa64
(gdb) set mips abi n64
(gdb) set mips mask-address off
(gdb) target remote localhost:1234
```

### Debugging tips

Enable GDB's remote protocol log to see every RSP packet exchanged:

```
(gdb) set debug remote 1
```

This is invaluable for diagnosing register layout mismatches, breakpoint address
truncation (Z0 packets), and g-packet size errors.

To set a breakpoint at a 64-bit kernel address (e.g. after IRIX boots):

```
(gdb) break *0xffffffff80010000
```

`set mips mask-address off` is required for this to work — without it GDB strips the
upper 32 bits and sends `0x0000000080010000` in the Z0 packet, which won't match the
kernel virtual address.

### Supported operations

| GDB command | What it does |
|---|---|
| `info registers` | Read all 72 MIPS registers (GPRs, CP0 Status/Cause/BadVAddr, FPRs, FCSR/FIR) |
| `p $pc`, `p $sp` | Read single register |
| `set $pc = 0x...` | Write single register |
| `x/Ni $pc` | Disassemble N instructions at PC |
| `x/Nw 0x...` | Read memory (handles unaligned, byte-granular) |
| `set {int}0x... = N` | Write memory |
| `break *0x<addr>` | Software breakpoint at virtual address |
| `delete` | Remove breakpoint |
| `continue` (or `c`) | Resume execution |
| `stepi` (or `si`) | Single-step one instruction |
| `watch *0x<addr>` | Write watchpoint on virtual address |
| `rwatch *0x<addr>` | Read watchpoint |
| Ctrl-C | Interrupt a running CPU |

### How execution control works

The GDB stub uses `run_debug_loop` (the same path as the monitor's `run`/`step`
commands) for both `continue` and `stepi`. This means:

- Breakpoints set via both GDB and the monitor console coexist — they use separate
  ID ranges (monitor IDs start at 1, GDB IDs start at 10000).
- Single-step is synchronous and blocking: GDB waits for the step to complete before
  returning a response.
- Continue is asynchronous: the CPU runs in a background thread; the event loop polls
  `is_running()` every millisecond, and returns a stop event when the CPU halts.
- Ctrl-C calls `cpu.stop()` immediately.

### CPU locking

The executor mutex (`Arc<Mutex<MipsExecutor>>`) is the single serialisation point:

- Normal run (`start()`): CPU thread holds the mutex continuously, drops it briefly
  every 500K instructions to allow monitor commands.
- Debug run (`run_debug_loop`): same, but with breakpoint checks and instruction
  tracing enabled.
- GDB operations (register read/write, memory access, add breakpoint): acquire the
  executor mutex via `try_lock()` — only safe when the CPU is stopped.
- **Always call `stop()` (or use `stepi`) before reading/writing registers/memory.**
  In developer mode the CPU does not auto-start, so GDB can connect and inspect state
  freely before issuing `continue`.

### Architecture notes for GDB

The g-packet contains 72 registers (576 bytes). A 73rd pseudo-register `fp` is
declared in the `org.gnu.gdb.mips.linux` XML feature to suppress GDB's stack-frame
heuristic, but it is not included in the g-packet.

| GDB reg # | Name | Source |
|---|---|---|
| 0–31 | r0–r31 | GPRs |
| 32 | status | CP0 Status |
| 33 | lo | LO |
| 34 | hi | HI |
| 35 | badvaddr | CP0 BadVAddr |
| 36 | cause | CP0 Cause |
| 37 | pc | PC |
| 38–69 | f0–f31 | FPRs |
| 70 | fcsr | FPU Control/Status |
| 71 | fir | FPU Implementation |

Memory reads/writes are byte-granular via `debug_read`/`debug_write` (kernel privilege
override, no cache side-effects, no breakpoints triggered).

---

## 9. Reference

- SGI Indy hardware manuals see docs/
- [SGI driver programmer's guide — address spaces](https://tqd1.physik.uni-freiburg.de/library/SGI_bookshelves/SGI_Developer/books/DevDriver_PG/sgi_html/ch01.html)
- MAME `newport.cpp` for REX3 drawing engine reference
