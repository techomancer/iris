# HACKING on IRIS

## How does this thing work?

IRIS is an SGI Indy (IP24) and Indigo2 (IP22) emulator written in Rust, with an
R4400 or R5000 CPU chosen at runtime. It is not cycle-accurate anywhere. IRIX
doesn't expect it, and accuracy would only make things slower.

Before re-deriving a gotcha, check `rules/` — it is organised by subsystem
(`jitv2/`, `rex3/`, `irix/`, `testing/`, `snapshot/`, `gui/`, `perf/`, `scsi/`,
`macos/`, `build/`). When you confirm a non-obvious fix, write it up there.

---

## 1. Data path and endianness

**Word-Transparent Architecture.** Host `u32`/`u64` values are bit-containers — no
internal byte-swapping. Endianness is handled only at "The Edge" (PROM/disk I/O) via
`swap_on_load`. The CPU thread handles byte/half-word packing internally via bit-shifts.
The bus and MC always see aligned word/double-word values.

Do not suggest `.to_be()` or `.to_le()` for memory or register logic.

---

## 2. Concurrency model

Every device can run in its own thread. CPU, REX3, SCSI, and ethernet each have their
own thread. `hptimer.rs` provides a repeating-event timer for devices that don't need
a dedicated thread.

Synchronisation is per-device: each device locks its own internal state. Be careful
when calling back up to a parent device (e.g. from SCSI → HPC3) — that is where
deadlocks live. Ethernet had two of them.

Other threads worth knowing about: the REX3 refresh thread (display, owns the GL
context — teardown must run there too), the HAL2 audio pump, the VINO DMA pump
(only when a video source is configured), the NAT engine, the jitv2 compile pool,
the monitor/serial/CI socket listeners, and in iris-gui the GUI's worker thread
that owns the `Machine`. The REX3 processor and the CPU thread both *park* when
idle instead of spinning (`rules/perf/`); the CPU only does so in `idle-pause`
builds.

Memory access is YOLO: all devices can freely read/write DRAM. It works because 32-bit
and 64-bit transactions are halfway atomic (though unordered). Hardware sync points are
maintained where the real hardware requires them.

---

## 3. Device, bus, and port abstraction

The **MC (Memory Controller)** is the central crossbar. All traffic — CPU PIO and HPC3
DMA — passes through it.

```rust
// src/traits.rs
pub trait BusDevice: Send + Sync {
    fn read8 (&self, addr: u32) -> BusRead8;             // .status == BUS_OK on success
    fn write8(&self, addr: u32, val: u8) -> u32;          // BUS_OK / BUS_BUSY / BUS_ERR
    // ... read16/write16, read32/write32, read64/write64
    fn dma_read64 / dma_write64                           // DMA-initiated access (REX3 HOSTRW)
    fn mem_ptr(&self, addr: u32) -> Option<*const u64>;  // direct backing-store pointer
    fn read_block / write_block / write64_masked          // bulk and masked fast paths
    fn gen_ptr(&self, addr: u32) -> *const AtomicU64;    // jitv2 per-page generation counter
}

pub trait Device: Send + Sync {
    fn step(&self, cycles: u64);
    fn stop(&self);
    fn start(&self);
    fn is_running(&self) -> bool;
    fn get_clock(&self) -> u64;
    fn signal(&self, signal: Signal) {}
    fn register_commands(&self) -> Vec<(String, String)>;   // monitor commands
    fn execute_command(&self, cmd: &str, args: &[&str], w: Box<dyn Write + Send>) -> Result<(), String>;
}

pub trait Saveable { fn save_state(&self) -> toml::Value; fn load_state(&self, v: &toml::Value) -> Result<(), String>; }
pub trait Resettable { fn power_on(&self); }
```

Every width has a default that returns a bus error, so a device implements only
the widths it natively supports. Write return values are layout-compatible with
`ExecStatus`. Devices usually implement both traits and expose getters for each.
Devices connect to each other via implementation-specific `connect()` /
`set_phys()` functions. The monitor asks every registered `Device` for its
commands, so a new device gets console commands by implementing
`register_commands`/`execute_command`. Snapshot support means implementing
`Saveable` and adding a save/load/save round-trip test
(`rules/snapshot/per-device-saveloadsave-round-trip-is-the-regression-net.md`).

**Physical address map** (`physical.rs`) uses a 64KB-granularity lookup table for O(1)
address decoding (`device_map`, one entry per `addr >> 16`). There is a lot of unsafe in there — it's intentional and somewhat
cursed.

Address space hierarchy:
- Upstream: SysAd (64-bit, 8-byte enables)
- Downstream: MC-mapped 64KB pages (DRAM, GIO-Bus)
- GIO-Bus: sub-decoder for HPC3 (GIO32) and Newport (GIO64)

---

## 4. Hardware notes

**HPC3** — self-programming DMA; fetches descriptors via the MC bus.

**REX3/Newport** — has a 16-word internal write FIFO in hardware. We enlarged it to
64K entries so it can absorb large DMA transfers and render pixels while the CPU does
other things.

**Machine profiles** — `[machine] profile` selects Indy IP24 ("Guinness" MC/IOC,
one WD33C93A) or Indigo2 IP22 ("Fullhouse" MC/IOC, two SCSI controllers, INT2,
serial EEPROM for NVRAM and MAC). `src/platform.rs`/`machine.rs` wire the
difference; `platform_profile_tests.rs` pins it down.

**Memory** — emulated as `Vec<u32>` (`src/mem.rs`) by default, or as real host
mappings under `--features ppmem` (`src/ppmem/`, `docs/ppmem-design.md`), where
SIMM mirroring is expressed as repeated mappings instead of address masking.
Banks 2 and 3 can be enabled (up to 512MB). PROM is fine with it; IRIX 6.5 uses
384MB, 5.3 uses up to 512MB. Each RAM page has a jitv2 generation counter.

**Cache** (`src/mips_cache_v2.rs`) — fully emulated L2 was a mistake in hindsight but
here we are. The CPU model is the `CpuModel` trait (`MIPS4`, `PRID`, `FIR`,
`TLB_ENTRIES`, `NAME`) implemented by each cache model; `MipsExecutor` is
monomorphised over it, so both CPUs are in every binary with no per-model branch
on the hot path, and `Machine::new` picks one from `[machine] cpu`.
- R4400: 16KB direct-mapped L1I/L1D (16B lines), 1MB L2.
  - L1I and L1D: virtually indexed, physically tagged (VIPT)
  - L2: physically indexed, physically tagged
  - Decoded instructions are cached in L2; L1I entries point into L2 (inclusive).
  - VCE fires when VA[14:12] doesn't match the L2's stored pidx for the same physical line.
- R5000: 32KB 2-way L1I/L1D (32B lines, LRU in a bitmask), no L2 (the PROM is
  told so). Decoded instructions live in L1I. The R5000 secondary-cache variants
  (`r5ksc`, `r5ksc_triton`) are refused at build time until their L1I bugs are
  fixed (`rules/testing/r5k-l1i-cache-bugs.md`).
- `--features tcache` keeps the whole cache state machine but stops copying line
  data for cacheable RAM (`docs/tcache-design.md`).
- Config.K0 switches KSEG0 between cached and uncached modes, including the
  misaligned-fetch behaviour of the reserved modes
  (`rules/irix/cache-attributes-and-fetch-alignment.md`).

**Address translation** (`src/mips_tlb.rs`, `mips_core.rs`) — layered fast paths in
front of the JTLB: a one-entry fetch `nanotlb`, the direct-mapped data-side
`nutlb` (`docs/nutlb-design.md`, always on), and an 8KB-granularity `vmap` that
points VPNs straight at TLB entries (always on; the `tlbvmap` feature name is
vestigial). All are invalidated on ASID changes and TLB writes. `tlbcheck` and
`tlbstats` are the diagnostic features.

**Count/Compare** — CP0 Count is anchored to host wall-clock time and ticks at a
fixed 33 MHz (`DEFAULT_COUNT_HZ`; `[clock] fixed_mhz` overrides). IRIX reports
that as a 66 MHz CPU. The Compare interrupt (IP7) is delivered by a host timer
(`hptimer.rs`), not by counting instructions. There is no calibration or
slow/fast-tick inference any more — that was removed in September 2026 because
a fixed rate is more stable, and these guests are interrupt-driven so running
faster or slower than "real" does no harm. `--features ci_clock` instead derives
Count from retired instructions (10ns each) for deterministic replays.

---

## 5. Interrupts

The CPU thread polls an `AtomicU64` interrupt bitmask (`MipsCore::interrupts`) every
instruction cycle; devices set bits through `MipsCore::set_interrupt`.
INT2/INT3 (Local0/Local1) logic maps to R4000 IP2/IP3; the Indigo2's extra INT2
cascade is handled in `ioc.rs`/`hpc3.rs`. `docs/interrupt_map.md` has the full map.

---

## 6. CPU execution architecture

**MipsCore** (`mips_core.rs`) — register state: GPRs r0–r31 (64-bit), CP0, CP1
FPU registers, interrupt state (`AtomicU64`), delay-slot state (`in_delay_slot`
and its target), the cycle counter, and the fields compiled code reads directly.
Field order is deliberate (hot fields together for cache locality), and jitv2
bakes field offsets and the core's address into emitted code, so reordering is
not free. No execution logic.

**MipsExecutor** (`mips_exec.rs`) — execution engine combining core + memory:

```rust
pub struct MipsExecutor<T: Tlb, C: CpuModel> {
    pub core: MipsCore,
    pub sysad: Arc<dyn BusDevice>,
    pub tlb: T,
    pub cache: C,
    // + breakpoints, traceback, undo buffer [developer], symbol table,
    //   jitv2 state, LL/SC stats, hot-path fn ptrs, ...
}
```

Key methods:
- `exec(instr: u32) -> ExecStatus` — execute one already-fetched instruction
- `step() -> ExecStatus` — fetch from PC and execute

PC advancement and delay slots are managed internally by the executor. The
executor is wrapped in `MipsCpu` behind the object-safe `CpuDevice` trait, which is
what `Machine` holds.

With the `opcodefusion` feature (implied by `lightning`) the interpreter
collapses branch+NOP, LUI+ORI/ADDIU and address-calc+load/store pairs into one
dispatch; a breakpoint on the second instruction of a fused pair never fires.

### ExecStatus

`ExecStatus` is a `u32` bit-field, **not an enum**.

```rust
pub type ExecStatus = u32;

// Normal (non-exception) status
pub const EXEC_COMPLETE:   ExecStatus = 0x0000_0000; // ran fine, no exception/retry/breakpoint
pub const EXEC_RETRY:      ExecStatus = 0x0000_0100; // bus busy, retry same instr
pub const EXEC_FALLBACK:   ExecStatus = 0x0000_0200; // jitv2+lightning's decode-skip fast path missed; caller must decode and dispatch normally
pub const EXEC_BREAKPOINT: ExecStatus = 0x0000_0800; // breakpoint hit

// Exception flags — upper bits
pub const EXEC_IS_EXCEPTION:       ExecStatus = 1 << 27;     // 0x0800_0000
pub const EXEC_IS_TLB_REFILL:      ExecStatus = 1 << 28;     // 0x1000_0000 — use 32-bit UTLB vector
pub const EXEC_IS_XTLB_REFILL:     ExecStatus = 1 << 29;     // 0x2000_0000 — use 64-bit XTLB vector
```

`EXEC_COMPLETE_NO_INC`/`EXEC_BRANCH_DELAY`/`EXEC_BRANCH_LIKELY_SKIP` used to distinguish *why* PC ended up where it did (interpreter PC+=4 vs a JIT/ERET direct-set vs a taken branch vs PC+=8) back when callers needed that to decide whether to advance PC themselves; every handler is now unconditionally responsible for its own PC, so that distinction is gone. A caller that needs "was a branch just taken" (e.g. gdbstub's `step_one`) checks `core.in_delay_slot` instead.

Exception status values are built with helpers:

```rust
exec_exception(code)   // IS_EXCEPTION | (code << CAUSE_EXCCODE_SHIFT)
exec_tlb_miss(code)    // IS_EXCEPTION | IS_TLB_REFILL | code
exec_xtlb_miss(code)   // IS_EXCEPTION | IS_TLB_REFILL | IS_XTLB_REFILL | code
```

The EXC code lives in bits [6:2] of the status word (same position as CAUSE.ExcCode).

### Exception codes

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

## 7. JIT v2 (`--features jitv2`) — experimental

The only MIPS JIT. The original speculative, tiered JIT (with its rollback path
and `IRIS_JIT*` env vars) was removed in August 2026 (commit `33c4e68`). v2
(`src/jitv2/`) compiles physical 4KB pages via Cranelift with memory-resident
registers and no speculation — compiled code is unconditionally correct at
publish time or not published at all. Full design rationale, the
analyzer/codegen block-emission model, and the delay-slot/exception
materialization rules live in `rules/jitv2/jit-v2-design.md` — read that before
touching `analyzer.rs`/`codegen.rs`. `rules/jitv2/codegen-gotchas.md` has
accumulated Cranelift-specific footguns found the hard way, and the rest of
`rules/jitv2/` covers individual bugs.

Shape of it:
- The CPU thread tracks the physical code page it executes from
  (`PhysicalCodePage`, looked up through a flat pfn→slot array) and requests a
  compile when an entry point gets hot. Compilation is triggered on the
  transition into a page, not in the middle of one.
- Requests go over a lock-free queue to a compile pool (`[jitv2] threads`,
  `--jitv2-threads`, default 1). Finished code is published into the page's
  entry table.
- Invalidation is by per-page generation counters owned by the memory device;
  a write to a page bumps its generation and stale code stops dispatching.
  Without `tcache`, a compile is also abandoned if any line of the page is dirty
  in the emulated cache (`rules/jitv2/dirty-cache-page-probe.md`).
- Loads and stores whose L1D line is already cached are inlined into compiled
  code for both CPU models; everything else calls back into Rust. Callouts take
  the core pointer as their first argument and return status in registers — the
  Windows x64 ABI cannot return two values, so reads write straight into the
  destination GPR (`rules/jitv2/callout-arg0-is-core-ptr.md`,
  `read-status-in-registers.md`).
- Anything without an emitter (`opcode_support.rs`) falls back to the
  interpreter. MIPS IV opcodes are only compiled with the `mips4` feature.
- `j2wp` switches to one Cranelift function per page with many entry points.
  It is not production-ready.

Enabled automatically at runtime once compiled in. Tuned via the `j2` monitor
console command:

| Command | Effect | Default (release / `developer`) |
|---|---|---|
| `j2 opt [none\|speed]` | Cranelift opt level, takes effect on next flush | `speed` / `none` |
| `j2 min-instrs [N]` | minimum instructions in a region before it's compiled | `2` / `1` |
| `j2 max-instrs [N]` | cap on instructions per compile | `128` |
| `j2 min-calls [N]` | dispatch count before a hot entry is scheduled to compile | `4` / `0` |
| `j2 inline [on\|off]` | compile synchronously inline vs. on the compile pool | `off` (`on` under lockstep) |
| `j2 dispatch [on\|off]` | main switch for the jitv2 dispatch gate (off = interpreter-only) | `on` |
| `j2 fallback [on\|off]` | keep an unsupported instruction inside a region as an interpreter call instead of ending the region (needs `j2 flush`) | `off` |
| `j2 inline_mem [on\|off]` | inline L1D loads/stores (forced off under lockstep) | `on` |
| `j2 pagewb [on\|off]` | write back L1D/L2 data when moving to a new code page, to flush out stale-data bugs | `off` |
| `j2 <alu\|fpu\|branch\|loadstore\|cop0> [on\|off]` | enable/disable compiling one instruction category | `on` |
| `j2 instrs [category]` | list instructions and whether they have emitters | — |
| `j2 threads` | compile-pool thread count (read-only) | — |
| `j2 flush` | drop all compiled code and reset the arena (stop the CPU first) | — |
| `j2 clear <paddr>` / `j2 deny <paddr>` | reset one physical code page / deny one entry | — |
| `j2 status` (alias `j2 stats`) | arena usage, compile counts, reject reasons | — |
| `j2 pcp` / `j2 dumppcp [addr] [path]` | physical code page introspection / capture for `jitv2_pcp_dump` | — |
| `j2 html [path]` | render the physical code page visualiser | — |
| `j2 lockstep` / `j2 lstate [full] [N]` | lockstep status / recent lockstep history (`jitv2_lockstep`) | — |

`jitcheck <n> [skip]` runs n instructions interpreter-only and through JIT
dispatch from the same captured state and stops at the first divergence.

`developer` builds default every knob toward "compile and see everything"
(no instruction-count/call-count floor, unoptimized Cranelift output) since
that's what you want while chasing a codegen bug; release builds default toward
throughput.

Verification tools, cheapest first: the `jitv2/equiv_test.rs` unit tests (install
the JIT hooks unconditionally, match the ISA level — see `rules/testing/`), the
cpu-tests matrix (`cpu-tests/run/matrix.sh`, R4400/R5000 × interp/jitv2),
`jitcheck`, `--features jitv2_lockstep` (every instruction checked against the
interpreter; incompatible with `lightning` and fusion), `jitv2_smc_check`, and
`--features fetchverify` for stale code that both engines would agree on.

---

## 8. Building, testing & debugging

The toolchain is pinned to nightly in `rust-toolchain.toml`. `cargo build` builds
the `iris` workspace member; the GUI is `-p iris-gui`.

Normal build:
```
cargo run --release
```

Developer build (enables intrusive debug helpers; affects performance):
```
cargo run --release --features developer     # or: cargo run --profile developer
```

The `developer` feature enables: the undo buffer, pending-write tracking,
per-instruction trace recording, `jitcheck`, the `[DEV]` monitor commands, the
CPU starting paused, and some additional assertions. It is mutually exclusive
with `lightning`. `cargo build --profile profiling` gives full debug info for
`perf`/flamegraph.

Binaries:

| Binary | Purpose |
|---|---|
| `iris` | the emulator |
| `iris-ci` | CI socket client (README) |
| `iris-bench` | benchmark driver (`bench/README.md`) |
| `coffdump` | dump MIPS COFF executables |
| `mkvh` | build and inspect SGI volume headers (`src/sgi_vh.rs`) |
| `chd_extract` | extract CHD images (`--features chd`) |
| `jitv2_analyze`, `jitv2_verify`, `jitv2_pcp_dump` | offline jitv2 analyzer/codegen tools (`--features jitv2`; `jitv2_pcp_dump` also needs `j2wp`) |
| `iris-gui` | the egui front-end (`-p iris-gui`) |

Tests:
- `cargo test --workspace` — unit tests, including per-device snapshot round
  trips, `mips_exec_test.rs`, `mips_tlb_test.rs`, `rex3_tests.rs`, and the jitv2
  equivalence tests (with `--features jitv2`). Tests that build a whole machine
  run on threads with enlarged stacks.
- `make -C cpu-tests run` / `cpu-tests/run/matrix.sh` — bare-metal instruction
  correctness (`cpu-tests/README.md`).
- `iris-bench run` / `iris-bench matrix` — bare-metal throughput and accuracy
  (`bench/README.md`).
- `tools/iris-test` with `tools/tests/*.yaml` — PROM/restore/screenshot smoke
  tests driven over `iris-ci`.
- CI (`.github/workflows/`): `rust.yml` builds and tests the workspace,
  `suites.yml` runs cpu-tests and bench across CPU × engine and gates cpu-tests
  on a per-CPU failure baseline, `release.yml` and `appstore.yml` build the
  distributed packages.

**Breakpoints** don't survive emulator restart.

**Monitor console** — available in the terminal or via telnet to `127.0.0.1:8888`.
Serial ports are on 8880 (port A) and 8881 (port B / IRIX serial terminal).
`debug.md` is a guided tour of the debugger; HELP.md has the full command list.
`src/iris_mcp.py` exposes the monitor as MCP tools.

**Crashes on Windows** — `crash_diag.rs` logs otherwise-silent process deaths
(`0xC000041D` and friends) with a symbolised stack to `iris-crash.log`
(`rules/gui/windows-silent-exit-0xc000041d.md`).

---

## 9. GDB stub

Iris includes a GDB Remote Serial Protocol stub (`src/gdb_stub.rs`) that lets you
connect GDB to the running emulator and debug IRIX/guest code with a real debugger.

### Start the GDB stub

Pass `--gdb-port <port>` on the command line:

```sh
# Developer build — CPU starts paused, GDB can set breakpoints before first instruction
cargo run --profile developer -- --gdb-port 1234

# Release build — CPU starts running; GDB attaches to a live system
cargo run --release -- --gdb-port 1234
```

The stub binds `127.0.0.1:<port>`. One client at a time; breakpoints set by GDB are
automatically removed when the client disconnects.

### Connect GDB

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

## 10. Reference

- SGI Indy hardware manuals and chip datasheets: `docs/*.pdf`; per-device notes:
  `docs/*.md` (`hal2`, `rex3`, `wd33c93a`, `interrupt_map`, `indigo2-ip22`, …)
- [SGI driver programmer's guide — address spaces](https://tqd1.physik.uni-freiburg.de/library/SGI_bookshelves/SGI_Developer/books/DevDriver_PG/sgi_html/ch01.html)
- MAME `newport.cpp` for REX3 drawing engine reference
