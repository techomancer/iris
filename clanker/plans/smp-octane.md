# Plan: two CPUs on the Octane (IP30 SMP)

Status: **M0 to M4 implemented, 2026-10-30.** IRIX 6.5 boots and runs on two CPUs
under the shared JIT. Left: part 5's M5. See "What changed from this plan" next.
Facts in part 1 were checked against the Octane PROM and kernel behaviour; the
inventory in parts 2 and 4 was read from the tree at `fabcd64e` and describes the
plan as written. Line numbers will drift.

## What changed from this plan

- **The monitor** is a `CpuMonitor` object registered on every machine, as the user
  asked, with commands `cpus`, `cpu <n>`, `cpu <n> <cmd>`, `r<n>`, `s<n>`. The `@n`
  suffix was dropped. The commands stay generic in `mips_exec/monitor.rs` behind
  `CpuDevice::monitor_execute`.
- **Child modules, not `src/cpu/debug/`.** `mips_exec/{instrument,monitor,gdb}.rs`: a
  child sees the parent's privates, so M0 was a pure move.
- **`pfn` stayed a plain field.** It changes only in a flush (all parked) or before
  `PfnMap::insert` publishes the slot. Only the map went atomic.
- **Masked stores:** a CAS loop, decided with the user; see 2.1.
- **The atomic gen bump is gated** by `JitDcGeometry::atomic_gen` (several CPUs, or
  `IRIS_JIT_ATOMIC_GEN`) until its single-CPU cost is known.
- **Idle-park `wake()` wakes every parked CPU** rather than taking a slot. Each looks at
  its own word. Interrupt sources kept their signatures.
- **`CachedFlashLines`** also holds read lines with several CPUs, filled from the array:
  [cached-flash-is-a-cache-with-two-cpus](../cpu/cached-flash-is-a-cache-with-two-cpus.md).
- **Workers lead a flush with `World::try_lead`**, not a blocking turn: a CPU leading a
  flush joins the workers, so a worker blocking on it would deadlock.
- **Not done (M5):** GDB multi-thread, all-stop on a breakpoint, error-register CPU
  fields, perfetto per-CPU tracks, FR per-mode slots (no thrash seen yet).

Related: `clanker/cpu/mp-llsc-is-a-host-compare-exchange.md` (the SC design
this plan adopts), `clanker/cpu/cpu-count-is-wallclock-derived.md`,
`clanker/jitv2/jit-v2-design.md`, `clanker/jitv2/compile-pool-starts-after-the-windows.md`,
`clanker/devices/ip30-low-memory-aliases-sdram.md`.

## Results, 2026-10-30

Host: 2990WX pinned to NUMA node 0, LTO build, features
`gr4-jit,jitv2,lightning,opcodefusion,rex-jit`, against a clean export of `fabcd64e`.
Guest seconds, 16 runs per row over two boots, 45 s after telnet answered.
`compute` is a 40M-iteration loop; `stores` writes a 64 KB array 4000 times.

| Config | Processes | compute min/median | stores min/median |
|---|---|---|---|
| `fabcd64e`, 1 CPU | 1 | 1.40 / 1.50 | 0.31 / 0.34 |
| this tree, 1 CPU | 1 | 1.42 / 1.48 | 0.32 / 0.34 |
| this tree, 1 CPU, `IRIS_JIT_ATOMIC_GEN` | 1 | 1.43 / 1.54 | 0.42 / 0.44 |
| this tree, 2 CPUs | 1 | 1.40 / 1.44 | 0.44 / 0.45 |
| this tree, 2 CPUs | 2 | 1.40 / 1.51 | 0.42 / 0.43 |

- **One CPU did not regress.**
- **Two CPUs scale about 2x**: two processes take as long as one.
- **The atomic gen bump costs about 30%** on this pure store loop, which is the worst
  case. That is why it stays off for one CPU, leaving the latent race against DMA there
  (4.9 item 2) as it was. Deciding that is still open.
- **Gen counter contention (6.3) did not show** here: the 2-process store time equals
  the 1-process time. The two arrays are in different processes' pages, so this does not
  rule it out for data the CPUs actually share.

Stress, on the fast-profile build: 40 rounds of parallel `cc` + a checksum of
`/usr/include` per worker gave 0 failures and 80/80 matching checksums at 1.78x CPU time,
alongside 27 pool flushes. Forcing the workers' flush (`IRIS_JIT_ARENA_FLUSH_MB=8`) gave 57
of them under the same load, again clean. `/etc/reboot` works through HEART's software
reset. Boot to telnet: about 40 s.

Regressions: unit suites pass (2374 with `jitv2,gpujit`, 1812 default). `cpu-tests` gives 125
failures on this tree and on a clean `fabcd64e`, with identical failure lists. Headless on
overlays, the Indy (still in its PROM) and O2 (in its kernel, no telnet) look the same after
200 s on both builds, so neither was seen reaching a shell on either.

Not exercised: the status bar on screen (formatting has a unit test), any graphics
workload on two CPUs, long uptime, PROM `disable`/`enable`.

## 0. Goal and scope

Run the Octane with two R10000 CPU threads sharing one RAM, one HEART, one
`Physical` and one jitv2 code pool, so that the stock IRIX 6.5 kernel boots as
a 2-processor machine (`hinv` lists two CPUs, `mpadmin -u` shows both busy).

Two, not four: IRIX's Octane kernel and PROM are built for at most two
processors, even though HEART has four processor slots. The HEART model should
carry four slots (masks, outputs, disable bits) because the registers are laid
out that way, but the machine builds `cpus` of them and `cpus` is 1 or 2.

Out of scope: the Indy and O2 (single-CPU machines; the refactor must leave
them working but adds nothing), cycle accuracy, IRIX kernels that are not MP.

## 1. What the hardware, PROM and kernel do

### 1.1 HEART registers that matter (PIU base `0x0ff0_0000`, all 64-bit)

| Offset | Register | MP role |
|---|---|---|
| `0x00` | MODE | bits 63:60 are the per-processor **disable** bits, bit `60+n` for CPU n. Set: HEART stops serving that CPU, which freezes in place; cleared: it continues where it stopped. Bit 22 is software reset (whole machine), bit 23 cold reset, bit 13 interrupt enable. |
| `0x80` | STATUS | bits 7:4 **processor active** (a presence strap, read-only), bit `4+n`; bits 3:0 HEART's widget id (8). One CPU reads `0x18`, two read `0x38`. |
| `0x10000 + 8n` | IMR0..3 | one interrupt mask **per CPU**. |
| `0x10020` / `0x10028` | SET_ISR / CLR_ISR | write-1-to-set / write-1-to-clear over the 64 ISR bits. |
| `0x10030` | ISR | the 64 pending bits, shared. |
| `0x10038` | IMSR | ISR masked by **the reading CPU's** IMR. |
| `0x20000` | COUNT | one 52-bit 12.5 MHz counter, shared; the PROM's and kernel's time base for delays and timeouts. |
| `0x30000` | COMPARE | one register; nothing in the PROM or kernel arms it. Each CPU's clock is its own CP0 Count/Compare. |
| `0x50000` | PRID | the number (0..3) of the CPU **doing the read**. Everything that needs "who am I" reads this. |
| `0x60000` | SYNC | read as a barrier. |

ISR bit assignment, per CPU n: IPI is bit `46+n` (46..49), the debugger stop
interrupt is bit `42+n` (42..45), the processor bus error is bit `59+n`
(59..62). Bit 63 is the HEART exception (in every CPU's IMR), bit 50 the HEART
timer, bit 41 the power switch.

Level routing is per CPU: outputs are computed from `ISR & IMR[n]` for each n.

| HEART level | ISR bits | Cause bit |
|---|---|---|
| 0 | 15:0 | 10 |
| 1 | 31:16 | 11 |
| 2 | 49:32 (IPI, debug) | 12 |
| 3 | 50 (timer) | 13 |
| 4 | 63:51 | 14 |

An IPI is nothing special: the sender writes `1 << (46+dest)` to SET_ISR, the
bit is only in the destination's IMR, so only that CPU's level 2 rises; the
receiver clears it through CLR_ISR. A HEART that computes levels per CPU from
`mask[n]` gets IPIs for free.

Error registers carry the issuing CPU's number in 2-bit fields (BERR_MISC
bits 22:21, PIUR_ACC_ERR 21:20, PIO_RTO 19:18). Low priority.

### 1.2 Reset and the PROM

1. Every active CPU starts at the same reset vector in the flash.
2. Each reads PRID and STATUS. The master is the **lowest-numbered active
   CPU**, not necessarily 0.
3. A slave sets a private stack (`PROM stack base - (id << 15)`, 16 MiB into
   RAM) and **polls ISR uncached, interrupts off, for its own IPI bit**. When
   it sees it, it jumps to the PROM's system init.
4. The master sets the **disable** bit of every other active CPU in MODE,
   then sizes memory and runs its diagnostics. The slaves are frozen mid-poll.
5. System init on the master: for each slave it clears the disable bit, sends
   the IPI with SET_ISR, spins until the slave clears the bit, then waits up to
   1 s (12.5 M HEART counts) for a "diagnostics started" flag in memory. On
   timeout it disables the slave again. The slave runs its cache diagnostics,
   sets flags through uncached addresses, then spins on ISR for its IPI again.
6. The master re-disables everyone, clears memory, builds the GDA (at `0x400`:
   magic `0x58464552`, master CPU number at `+0x10`, NMI vector at `+0x08`),
   reads the `bootmaster` variable and the NVRAM CPU-disable word (2 bytes, top
   12 bits must be `0xB21`, low 4 bits are the disable mask), then re-enables
   and IPIs every good CPU.
7. Every CPU fills its own **MPCONF** block at phys `0x600 + 128*id`:

| Offset | Size | Field |
|---|---|---|
| `0x00` | u32 | magic `0xBADDEED2` (written last) |
| `0x04` | u32 | CP0 PRId |
| `0x08` | u32 | physical id |
| `0x0c` | u32 | virtual id |
| `0x10` | u32 | log2 secondary cache size |
| `0x14` | u16 | fan loads (block 0 only) |
| `0x18` | ptr | launch |
| `0x20` | ptr | rendezvous |
| `0x40` | ptr | stack |
| `0x48` | ptr | launch parameter |
| `0x50` | ptr | rendezvous parameter |
| `0x58` | u32 | idle flag |

8. Slaves enter the PROM slave loop: idle flag 1, check `launch` every 1 ms
   (timed on HEART COUNT); when non-zero and the magic and virtual id match,
   clear it, idle 0, call launch(parameter) on the given stack, then the
   rendezvous routine if any, idle 1 again.
9. Re-entering the PROM (reboot, `init`): the master gives every idle slave a
   launch address and waits for idle to drop; slaves then wait for an IPI with
   a **6 s** timeout and set the software-reset bit if none comes; the master's
   soft init IPIs all slaves with a **10 ms** timeout and hard-resets the
   machine if one misses it.
10. The PROM `disable`/`enable` commands edit the NVRAM mask and set the MODE
    disable bit at once. A CPU counts as usable when active and not disabled.

No MPCONF or GDA emulation is needed: the real PROM builds them. Only a direct
kernel boot that bypasses the PROM would have to fake them.

### 1.3 The kernel

1. Early on, the master writes `launch = kernel slave loop` into every MPCONF
   slot (its own included; nobody consumes that one). The PROM slave picks it
   up on the PROM stack, sets a per-CPU ready flag, and polls `launch` again.
2. Enumeration: a CPU is enabled when its STATUS active bit is set and its MODE
   disable bit is clear. The master's id comes from PRID.
3. HEART init zeroes all four IMRs, then sets in IMR[master]: bit 63, its own
   bus-error, IPI and debug bits. Each slave does the same for IMR[self] when
   it comes up.
4. Slave launch: parameter, rendezvous 0, stack = the CPU's boot stack, then
   `launch = bootstrap` written last; then a handshake with **about 0.5 s**
   timeout per slave on HEART COUNT. A slave that started but did not set its
   own PDA id gets disabled and the kernel **panics**; a slave that never reached the ready flag is disabled with a
   warning only.
5. Device vectors are spread over CPU 0 and 1 by putting each vector bit in
   exactly one CPU's IMR (the kernel asserts a vector is never in two). Power
   switch and AC-fail go to the boot master.
6. Dispatch: on IP2..IP6 the handler loops over IMSR, clears bits above 2
   through CLR_ISR, and routes 59..62 to bus-error handling. IP7 is the CPU's
   own Count/Compare.
7. Stopping: `cpu_disable` sets the MODE disable bit; the debugger uses the
   debug ISR bits; reboot goes back to the PROM which sets software reset; on
   NMI every CPU enters the NMI path and the one whose PRID equals the GDA's
   master number dumps, the others save registers and spin.

### 1.4 What `src/dev/heart.rs` has and lacks

Has: MODE stored (only bit 13 and the ECC bits matter), all four `mask`
entries stored and readable, SET_ISR/CLR_ISR/ISR, COUNT from the host clock
held while stopped, COMPARE stored but never compared (fine), one `cpu_irq`
word, `levels()` over `mask[0]`.

Lacks, in order of necessity:

1. PRID (`0x50000`) returns 0; must return the issuing CPU.
2. STATUS (`0x80`) is `0x18`; must be `0x8 | present_mask << 4`.
3. IMSR (`0x10038`) uses `mask[0]`; must use the reader's IMR.
4. One output set and one interrupt word; needs `out[4]`, `cpu_irq[4]`,
   `set_interrupts(cpu, sink)`, `levels(cpu)` from `mask[cpu]`, and `drive`
   over every CPU whose levels changed.
5. MODE bits 63:60: a write must freeze or resume the matching CPU thread.
6. MODE bit 22 (software reset): reset the whole machine. The PROM uses it
   after a slave timeout, so a bug anywhere in slave start-up shows up as a
   reboot loop; handle it early so that symptom is readable.
7. The 2-bit CPU fields in the error registers. Later.

## 2. The CPU core today: what is single-instance

Almost everything is already per instance. `MipsCore` owns the interrupt word
(`hot.interrupts`), `hot.cycles`, Count/Compare and its hptimer one-shot, the
LLbit (in the cache), the TLBs, the decode cache; there are no
`thread_local!`s in the core, ppmem or devmap; the devmap fault handler works
by faulting address and does not care which thread faulted. What blocks a
second CPU is a short list of **single-slot wiring points** and the
single-CPU memory semantics.

| Where | What | Becomes |
|---|---|---|
| `src/cpu/idle_park.rs:65,69` `PARKER`/`PARKED`, `wake()` with no argument (called from `mips_core.rs` claim_ip7 and set_interrupt, `mips_exec.rs` signal, `dev/ioc.rs`, `dev/heart.rs:297`) | one parked thread slot | a `ParkSlot` per CPU, handed to interrupt sources together with the word |
| `src/dev/heart.rs:189` `cpu_irq`, `:134` `levels()` | one word, mask[0] | see 1.4 |
| `src/ppmem/ppmem.rs:646-689` `SpaceState { sink, sink2 }` behind an `UnsafeCell` whose contract is "only the CPU thread" | one bitmap sink per consumer, no lock | a `Vec` of sinks under a Mutex; remaps run inside the rendezvous (4.2) |
| `src/ip30.rs:61` `LAST_PC` (and the O2's) | one AtomicU64 | `[AtomicU64; 4]` indexed by cpu |
| `src/perfetto/mod.rs:201` `set_guest_cycles` | one cycles pointer | CPU 0, or one counter track per CPU |
| `src/ip30_machine.rs:198,428` `fasttick` | one Arc shared by graphics and the CPU | one per CPU; the status line shows each |
| `src/ip30_machine.rs:259` `mgras.set_cpu_cycles` | one `CyclesPtr` | a slice; the bar sums or lists them |
| `src/thread_affinity.rs:16` `PerfRole::MipsCpu`, thread name `MIPS-CPU` (`mips_exec.rs:11538`) | every CPU thread pins to the same core | `MipsCpu(n)`, `MIPS-CPU{n}` |
| `src/monitor.rs:163-176` | a command goes to the **first** device that registered it | one `CpuSet` device (4.7) |
| `src/gdb_stub.rs:29-33` `SingleThreadBase` | one thread | `MultiThreadBase`, tid = cpu+1 |
| `src/cpu/mips_exec.rs:3089-3116` `install_jit_mem_ptrs` | publishes one core address and one geometry into the shared `Jitv2` | see 4.9 |
| `src/cpu/jitv2/jitv2.rs:2765` `CompileQueue.cpu` | the one CPU a worker-led flush stops | all CPUs, through the rendezvous |
| `src/ip30.rs:268` `CachedFlashLines` | one store of dirty PROM lines | one per CPU: on hardware those lines are private to each CPU's cache, and the PROM keeps per-CPU scratch in them |
| `src/snapshot.rs`, `machine.rs:1640,1883` | one `cpu` state | `cpu`, `cpu1`, ...; a missing `cpu1` reads as a 1-CPU snapshot |
| `MipsExecutor::symbols` | one `SymbolTable` per executor | one shared Arc |

Statistics, env-var `OnceLock`s, `SMC_HITS`, `JIT_FEEDBACK`, `IDLE_ENABLED`,
the inline-memory counters, the jitv2 toggles and test locks can stay global:
they are configuration or sums.

### 2.1 Memory semantics that are single-CPU

- **SC/SCD** test the LLbit and store (`mips_exec.rs:7145-7187, 7213-7245`).
  Adopt `mp-llsc-is-a-host-compare-exchange.md`: LL records the loaded value,
  SC is a host CAS on the ppmem word, ABA accepted. LL/SC stay Excluded in
  jitv2 so the CAS lives in the interpreter only.
- **`write64_masked`** (SWL/SWR/SDL/SDR and the JIT callout) is a plain
  read-modify-write of the host qword on the RAM path
  (`mips_cache_shadow.rs:490-498`, `ppmem/ppmem.rs:553`; check
  `dev/heart_mem.rs:404`). **Decided 2026-10-30: a CAS loop on the guest
  qword.** No lock table of any granularity: a lock only excludes writers that
  take it, and plain stores (inline JIT, `write32`, DMA) do not, so a merge
  under a lock still erases them unless every guest store checks the lock. The
  CAS compares the whole qword, so any intervening writer forces a retry
  without knowing about it. It costs one locked instruction, contends only on
  the host cache line actually being written, and other CPUs never see a
  half-finished write, which per-byte stores would allow. Bus devices stay as
  they are: each implementation decides, and the `BusDevice` default already
  splits into `write8`.
- **Plain window stores** are raw pointer writes (`mips_cache_shadow.rs:467`).
  Make them `AtomicU64::from_ptr(..).store(Relaxed)`: same instruction, but
  declared data-race free.
- **SYNC** is a no-op (`mips_exec.rs:5733`). Correct on x86 (TSO), needs a
  real fence on aarch64, which the macOS build is. Pick the ordering per host
  from day one; same for the SC CAS (acquire-release off x86).
- **Only data-less caches.** `mips_cache_v2` keeps line data and would give
  each CPU a private copy of RAM. MP builds require the shadow cache; the
  Octane already uses `R10000ShadowCache`. Assert `cpus == 1` for any other
  cache model.

## 3. Mechanism everything else hangs on: the park request

Bit 63 of `hot.interrupts` is already a host-side request (soft reset); both
the interpreter preamble and the compiled-code preamble test the whole word
against zero and bail on any set bit (`mips_exec.rs:2618-2672`,
`codegen.rs:2568-2620`). Add **bit 62, `HOST_PARK`**. When set, the CPU
thread, at its next preamble (never inside compiled code), nulls its jitv2
`pcp`, drops the executor lock, acknowledges on its `ParkSlot`, and waits
until the bit is cleared. Four users, one mechanism:

| User | Who sets | Who clears |
|---|---|---|
| HEART MODE disable bit n | the CPU thread that wrote MODE | the thread that clears the bit |
| jitv2 flush rendezvous | the flush leader (any thread) | the leader after `mega_flush` |
| monitor all-stop (`pause`) | the monitor thread | the monitor thread |
| ppmem remap (MEMCFG write) | the writing CPU thread, for the others | the same |

`CpuShared::park_others(reason)` sets the bit on every other CPU, waits for
each acknowledgement (a parked or already-stopped CPU acknowledges at once,
a CPU in WAIT is woken first), runs the closure, clears the bits and wakes
them. Reasons are counted so a HEART-disabled CPU is not released by the end
of a flush. The wait must not hold the `Jitv2` mutex or any executor mutex,
or a CPU blocked on either can never reach its preamble: that is the
lock-order hazard the jitv2 audit found (`jitv2.rs:2586` runs inside the
`Jitv2` lock today).

A HEART-disabled CPU is exactly a parked one: it makes no progress, keeps its
state, resumes where it stopped, which is what the PROM relies on when it
freezes a slave mid-poll.

## 4. Design

### 4.1 `CpuShared` and per-CPU state

One `CpuShared` per machine (an `Arc`): the bus (`sysad`), the shared
`SymbolTable`, the timer manager and `count_hz`, the ppmem space handle, the
jitv2 handle, the `cpus: Vec<CpuHandle>` list where a `CpuHandle` is
`{ interrupts: *const AtomicU64, park: *const ParkSlot, cycles: CyclesPtr,
fasttick: Arc<AtomicU64>, running: Arc<AtomicBool> }`, and the park
machinery of part 3.

Per CPU, unchanged in `MipsExecutor`: core (GPRs, CP0, CP1, interrupts,
cycles, Count/timer), tlb, cache (tags, LLbit, decode cache, tcache bitmap),
nanotlb/nutlb, traceback, undo, stats, idle profiler, `batch_hook`, the
jitv2 per-executor fields (`pcp`, `cur_code_pfn`, `jit_trigger`, inline
analyzer, lockstep scratch). Add `cpu_id: u8` to `MipsExecutor` and mirror it
into `MipsCore` for perfetto and callouts.

Executors stay inside their own `Arc<Mutex<..>>`: the hptimer, HEART, the
status callback and the JIT hold raw pointers into them. Never a `Vec` of
executors by value.

### 4.2 Which CPU is on the bus

`BusDevice` calls carry no initiator. Add one `thread_local! CURRENT_CPU:
Cell<u8>`, set at the top of each CPU thread and debug thread, default 0 for
DMA and monitor threads. Only HEART reads it (PRID, STATUS, IMSR, the error
registers). Every HEART access from a CPU is an uncached MMIO access through
the bus, never the inline JIT memory path (MMIO is not in the ppmem window),
so the thread-local is always right. Far smaller than threading an initiator
id through `BusDevice`.

### 4.3 HEART

- `cpu_irq: [AtomicPtr<AtomicU64>; 4]`, `park: [AtomicPtr<ParkSlot>; 4]`,
  `out: [u8; 4]`, `present: u8` from `HeartConfig`.
- `levels(cpu)` = `pending & mask[cpu]` folded per level (bit 63 still gated
  by MODE bit 13). `update_irq` recomputes all four, drives the ones that
  changed, wakes the park slot of any CPU whose level rose.
- Reads: PRID = `CURRENT_CPU`, STATUS = `8 | present << 4`, IMSR =
  `pending & mask[CURRENT_CPU]`.
- MODE write: for each n, a 0→1 edge of bit `60+n` parks CPU n (reason
  `HeartDisabled`), a 1→0 edge releases it. Writing the bit for the writing
  CPU itself parks the writer after the store completes (the PROM never does
  it; make it work anyway). Bit 22 requests a machine reset through the
  machine's existing reset path, not the per-CPU soft reset.
- Tests in `src/ip30/tests.rs`: two interrupt words, IPI through SET_ISR
  reaches only the target, IMSR differs per reader, a disable edge parks and
  an enable edge releases a fake slot.

### 4.4 ppmem

`SpaceState` goes behind a Mutex and `sink`/`sink2` become `Vec<*mut u64>`
with `add_bitmap_sink`. `Physical::remap_banks` (reached from the MEMCFG
write on whichever CPU wrote it) runs inside `park_others(Remap)`: the other
CPU must not touch the window during a `MAP_FIXED` remap and must drop its
decode-cache, nanotlb and tcache state afterwards. Remaps happen in the PROM
and once at IRIX boot, so this costs nothing at steady state. The `unsafe
impl Sync` justification comment is rewritten to say so.

### 4.5 Debugger extraction (step 0, no behaviour change)

About 5.2k of the 15.7k lines of `mips_exec.rs` are debugger, monitor and
statistics code (6k with the lockstep tooling). Move them to
`src/cpu/debug/`: `cmds.rs` (`register_commands`, `execute_command`),
`breakpoints.rs` (types and the table; the hot-path `check_breakpoint` stays
in exec), `traceback.rs`, `undo.rs`, `introspect.rs` (backtrace, eframe,
putbuf, proc, idle profile report), `gdb_adapter.rs`, `load.rs` (ELF/bin,
digests). `impl<T, C> MipsExecutor<T, C>` blocks can live in another module
of the crate, so this is a move plus `pub(crate)` on some fields. Do this
first and alone so the diff is reviewable and every machine still works.

### 4.6 Monitor and GDB

- One `CpuSet` device registered instead of N CPU devices. It owns the CPU
  list and `current: AtomicUsize`, and adds `cpu <n>` (select) and `cpus`
  (list: id, pc, state, stop reason). Any existing CPU command accepts `@n`
  (`regs@1`, `bt@1`); without it, the selected CPU.
- `pause()` becomes all-stop through `park_others` plus the selected CPU's
  own stop. Stopping one CPU alone leaves the other spinning on a lock the
  stopped one holds while HEART COUNT keeps running.
- Breakpoints are installed into every executor while all are stopped, so the
  hot-path check does not change. Stop notes record the CPU.
- `run`/`step`/`next`/`finish` act on the selected CPU with the others held
  (scheduler locking); `start`/`stop` act on all.
- GDB: `MultiThreadBase`, tid = cpu+1, `get_cpu_debug` returns the set.
- `debug.md` and `clanker/iris_mcp.py` updated in the same change.

### 4.7 Status line

`BarStats` gets `cycles: [Option<CyclesPtr>; 4]` and `fasttick: [..; 4]`,
`StatusBar::render` prints one MIPS and Hz figure per present CPU, for
example ` 95.1|94.8 MIPS 100|100Hz`. Producers: `mgras/disp.rs:235-247` for
the Octane, `ip32_machine.rs:915`, `gr2disp.rs:41`, `rex3.rs:2984` (these
keep one entry).

### 4.8 Machine, config, snapshot

- `[machine] cpus = 2` (default 1; refused above 2 and on machines other
  than the Octane; refused with a data-carrying cache model, with
  `jitv2_lockstep`, with inline compile mode).
- `Ip30Machine.cpus: Vec<Arc<dyn CpuDevice>>`, `build_cpu(id)`: own TLB, CP0
  with the same PRId, own `CachedFlashLines`, own fasttick, own
  `set_tcache_window`/`set_tcache_gen_window`, own `LAST_PC` slot,
  `PerfRole::MipsCpu(id)`. Start, stop, reset and power-on cover all of them;
  both start at the reset vector and the PROM sorts out master and slave.
- Snapshot keys `cpu`, `cpu1`. `CpuStateDigest`, `jitcheck` and
  `machine_state.cpu_running` report CPU 0 or "any running".

### 4.9 jitv2: one shared pool

**Verified: generated code is CPU-agnostic.** Every per-CPU field is reached
as `core_ptr + offset_of!` and every callout takes the biased core pointer as
argument 0 (`codegen.rs:3480-3674`, `clanker/jitv2/callout-arg0-is-core-ptr.md`).
The only per-core address codegen can bake is a hook pointer read through
`JitConsts.core`, and that is gated off unless `IRIS_BAKE_HOOKS` is set
(`codegen.rs:4359-4372`); even then the hooks are monomorphised functions,
identical for every executor of one type. The `dc_geometry` immediates
depend on the CPU model, which is the same for both CPUs. The persistent
cache already requires an address-free blob. So two CPUs can share one
`Jitv2`, one page pool and one arena. In MP builds publish `JitConsts
{ core: None }` and reword the "one MipsCore" doc on `JitConsts`.

**Keep per executor**: `pcp`, `cur_code_pfn`, `jit_trigger`, the inline
analyzer, lockstep scratch. Do not make the pool per CPU: `compiled`, `func`
and `entry_gen` must be shared for cross-CPU invalidation anyway, and the
dispatch/publish protocol was built for "workers publish while the executor
dispatches", which already tolerates several readers.

**Required fixes** (all in the bookkeeping, none in generated code):

1. **`PfnMap` is a plain `Box<[u32]>` read lock-free by the executor**
   (`jitv2.rs:94-136`, `mips_exec.rs:3507-3527`). CPU B can see A's slot index
   before A's plain `pfn`/`gen` writes in `claim`. Entries become
   `AtomicU32` (Release after `claim`, Acquire on the fast lookup);
   `PhysicalCodePage::pfn` becomes atomic (the same-page test at
   `mips_exec.rs:3543` reads it cross-thread). Claiming stays under the
   `Jitv2` mutex; it is cold, a spinlock buys nothing there.
2. **The inline-store generation bump is a plain load/add/store**
   (`codegen.rs:4659-4667`) while every Rust-side bump is a `fetch_add`.
   Two writers lose increments, and a plain store landing after another
   writer's `fetch_add` moves the counter **backwards**, which defeats
   publish's `current_gen() > snap_gen` check (`jitv2.rs:1859`) and lets stale
   code be published and accepted. This is latent today between the CPU's
   inline stores and DMA; with two CPUs it is routine (kernel data sharing a
   page with code). Fix: Cranelift `atomic_rmw add`. That is a `lock xadd`
   on every guest store, so measure it. See 6.3 for the contention this leaves
   open, which is deliberately not being solved up front.
3. **`&mut *self.pcp`** at `mips_exec.rs:4077, 9289, 9331, 9340` and
   `reset_compiled_state(&mut self)`: two executors holding `&mut` to one
   page is aliasing UB even though every field is atomic. Make them `&*`.
4. **Flush**: every flush path (`flush_from_cpu_thread` from pool exhaustion
   at `mips_exec.rs:3630`, the inline-compile thresholds, `install_jit_mem_ptrs`
   on a geometry change, `j2 flush`; and the worker's `run_leader_flush`
   which stops the one CPU in `CompileQueue.cpu`) assumes the single executor
   is the caller or stopped. With a second CPU it can be inside the arena
   that `codegen.release()` unmaps, or between `is_published` and the `func`
   load while the slot is reset (`entry_gen` 0 and `gen` pointing at the
   never-compilable 0 makes `is_runnable` true with a null `func`), or holding
   `pcp` into a slot reclaimed for another pfn. There is no epoch or
   quiescence scheme; code is only ever freed by a full flush. The fix is the
   rendezvous of part 3: the flush leader parks every CPU (each nulls its
   `pcp` on the way in), stops the workers, drains both queues, `mega_flush`,
   new arena, release. Epoch-based freeing is only worth it if partial
   reclamation is wanted later. The CPU-thread flush must leave the `Jitv2`
   lock before parking the others.
5. **`install_jit_mem_ptrs`** runs per CPU and can trigger a flush each
   time; publish geometry once (same model), set the per-core `jit_*`
   pointers per CPU, flush only on a real geometry change.
6. **`CompileQueue::set_cpu`** becomes the CPU set; `worker_loop`'s leader
   flush uses the rendezvous.
7. **FR mode**: the FR pin is per page, FR mode per CPU. Two CPUs running one
   FPU-touching kernel page in different `Status.FR` modes alternate kill,
   re-pin and recompile forever (the guard kills before running wrong-mode
   code, so this is a performance trap, not a correctness bug). Fix later if
   it shows: two `func`/`compiled`/`entry_gen` sets per page indexed by FR.
8. **Inline compile mode** (`jitv2_compile_inline`) takes the single
   `Codegen` by `take()`; the second CPU gets `None`. Refuse it with
   `cpus > 1`.
9. Machine wiring: per-CPU `set_tcache_window`/`set_tcache_gen_window`,
   multi-sink bitmap publication, all CPUs registered with the queue.
10. Stale comments to fix while there: `jitv2.rs:13-18`, `:1147-1152` (the
    snapshot is a Mutex now), `:2265-2276`, and the "a page is never compiled
    twice at once" claim, which the hot lane already breaks with more than
    one worker (both queues are MPMC `ArrayQueue`s; the design doc's SPSC
    description is out of date).

**Scheduling is already safe for a second producer**: `try_schedule_page`
is an atomic swap, so two CPUs racing on one page produce one request;
`requested` is a cumulative OR; a reader that picks up a `func` for an
offset it does not cover gets `EXEC_FALLBACK` and goes to the interpreter.

**Cross-CPU self-modifying code**: CPU A's store bumps the generation while
CPU B is mid-region; B runs stale code until its region exits, the same
window as single-CPU SMC today. Real MIPS SMP needs cache operations plus an
IPI before the new code is guaranteed visible, so this is acceptable.

### 4.10 Interrupt sources other than HEART

`claim_ip7` (the hptimer one-shot), `Device::signal`, and the IP22/O2 IOC
all call the global `wake()`. They get the `ParkSlot` with the word
(`IrqSink { word, park }` from `MipsCpu::irq_sink()`), and `wake()` takes the
slot. The IOC keeps one sink.

## 5. Milestones

Each milestone is a separate, mergeable change; every one keeps the Indy, O2
and single-CPU Octane working (`cargo test --workspace`, a 6.5 boot on each).

**M0. Debugger out of `mips_exec.rs`** (4.5). Pure move. Reviewed by diff
size and `cargo test` only.

**M1. One CPU, MP-shaped plumbing.** `CpuShared`, `cpu_id`, per-CPU
`ParkSlot` and `IrqSink`, the `HOST_PARK` bit and `park_others`, the ppmem
sink list and remap under the rendezvous, the `CpuSet` monitor device with
`cpu`/`cpus`/`@n`, per-CPU `LAST_PC`, thread naming and affinity, status
line arrays, snapshot keys. Still `cpus = 1` everywhere. Acceptance: no
measurable slowdown (benchmarks per `reference_host_perf_variance`: same
host, side by side), monitor and GDB unchanged for one CPU.

**M2. Memory semantics + HEART MP + second CPU, interpreter only.** SC as
CAS, `write64_masked` as CAS loop, atomic window stores, SYNC fence per
host, per-CPU `CachedFlashLines`; HEART 4.3; `[machine] cpus = 2`; jitv2
dispatch refused when `cpus > 1`. Acceptance: the PROM boots with two CPUs
(both run cache diagnostics, `hinv` in the PROM lists two processors), PROM
`disable 1` / `enable 1` work, a PROM-level reboot does not hard-reset
through the 10 ms IPI timeout.

**M3. IRIX MP boot, interpreter.** Expected to be mostly debugging of M2:
slave launch handshake, IPIs, interrupt spreading, both CPUs taking clock
interrupts. Acceptance: multi-user with `hinv` showing 2 CPUs, `mpadmin -u`
listing both, `sysmp(MP_NPROCS)` = 2, a parallel build (`make -j2`) finishing
with both busy, repeated reboots clean. Measure boot time against 1 CPU.

**M4. jitv2 shared pool.** Fixes 1 to 6 and 8 to 10 of 4.9 in that order
(the atomics and `&*` first, with `cpus = 1`; then the rendezvous flush;
then enable dispatch for `cpus = 2`). Acceptance: `j2 status` healthy on
both CPUs, no `jitv2: mega_flush` reboot loops, equal or better M3
throughput per CPU, the gen-bump cost and the 1 against 2 CPU store-heavy
comparison (6.3) measured and recorded in `clanker/jitv2/`.

**M5. Polish.** FR per-mode slots if the thrash shows, GDB multi-thread,
error-register CPU fields, perfetto per-CPU tracks, the 4-slot HEART facts
into `docs/Octane/`.

## 6. Problems foreseen

1. **Everything is timed on host time.** HEART COUNT follows the host clock
   and the PROM's and kernel's handshakes are measured on it: 10 ms for the
   soft-init IPI (miss it and the PROM hard-resets the machine), 0.5 s per
   slave at kernel launch (miss it and the kernel panics), 1 s and 6 s in the
   PROM. The second CPU must run truly concurrently on its own host core.
   A laptop with fewer cores than CPU threads plus compile workers, a
   descheduled thread, a slave parked by mistake, or a debugger pause at the
   wrong moment all look like a dead slave. Give `cpus = 2` its own
   `thread_affinity` roles and keep the compile pool off those cores. If this
   bites, the alternative is to drive COUNT from emulated progress, which
   costs the "a stop costs the guest no time" property.
2. **The jitv2 flush lock order.** `flush_from_cpu_thread` runs inside the
   `Jitv2` mutex; the other CPU may be blocked on that mutex in
   `jitv2_track_pcp` and can never reach its preamble. The rendezvous has to
   be requested with no JIT or executor lock held. Same for the monitor's
   all-stop against the 1000-instruction batch that holds the executor mutex:
   the park check must be at the batch boundary and in the preamble.
3. **Generation counter contention and cache thrashing. Known, not a
   priority unless benchmarks show it.** Atomicity (4.9 item 2) is a correctness
   fix and is required. Contention is a separate cost: every guest store bumps
   its page's counter, counters are 8 bytes, so one 64-byte host cache line
   covers eight guest pages (32 KB of RAM). Two CPUs storing to nearby pages
   bounce that line on every store, atomic or not. Fighting over code pages is
   not expected; **data pages are where it is likely**, since they take the same
   bump and shared kernel data sits close together. Not measured. Revisit if
   benchmarks show it, by comparing per-CPU throughput at 1 and 2 CPUs on
   store-heavy work, with host `perf c2c` to attribute the line. Candidate fix:
   a read-mostly per-page "has code" map tested before the bump, so data pages
   never write the counter (one load and a predictable branch), kept in step
   with `claim`/`free_page`. It would also remove the locked add from those
   stores. Check its ordering against a claim racing a store before relying on it.
4. **ppmem remap from a CPU thread** breaks the `UnsafeCell` single-writer
   contract; the rendezvous fixes it but every path that reaches
   `remap_banks` must be known (the Octane MEMCFG write, the Indy MC, the O2
   CRIME).
5. **A shared `CachedFlashLines`** leaks one CPU's dirty PROM lines (its
   stack set-up and per-CPU scratch) into the other. Per-CPU instances are a
   one-line change but easy to forget.
6. **Masked-store races and `write64_masked`**: `cur & !mask | val & mask` on
   a shared qword loses the other CPU's bytes, silently. Solved by the CAS loop
   in 2.1; listed so the test in part 7 is not dropped.
7. **CPU identity by thread-local** is right for CPU threads and wrong for
   nothing today, but a future device that reads HEART from its own thread
   (a DMA engine checking IMSR) would read as CPU 0. Note it in the HEART
   doc comment.
8. **The idle profiler, lockstep, equiv tests, inline compile mode and the
   instruction trace ring** are single-CPU; refuse `cpus > 1` with them
   rather than let them interleave.
9. **FR-mode thrash** on kernel FPU pages (4.9 item 7): only performance,
   but it would look like a compile storm in `j2 status`.
10. **Count drift.** Each core's Count is wallclock-anchored at its own
    start; the two drift by their start offset and by stops. IRIX's Octane
    clock is per-CPU Count/Compare and its time base is HEART COUNT, so it
    tolerates this, but anything that compares Counts across CPUs (profiling
    tools) will see offsets. Re-anchor both cores from one instant at start.
11. **Snapshots** of a 2-CPU machine need both CPUs parked at a preamble;
    the rendezvous gives that. Resuming a 2-CPU snapshot on a 1-CPU config
    must be refused.
12. **No reference emulator.** MAME and QEMU have no Octane; the only
    oracle for the PROM's MP path is the PROM itself. Make the HEART MP
    registers monitor-visible from the start (`heart` command: per-CPU
    masks, levels, disable bits, park reasons) so a hang shows which CPU
    waits for what.

## 7. Tests to write

- `src/ip30/tests.rs`: HEART per-CPU levels, IPI, IMSR per reader, disable
  edge parks and releases, STATUS/PRID per reader.
- A no-CPU jitv2 test with two executor-like threads claiming and looking up
  pages while a third flushes through the rendezvous (`cargo test --features
  jitv2`), plus a loom-style or stress test of `PfnMap` claim/lookup.
- SC CAS: two threads on one word, exactly one succeeds; `write64_masked`
  from two threads on disjoint bytes of one qword loses nothing.
- Snapshot round trip with `cpus = 2`.
- `cpu-tests` and `bench/` must stay at their baselines with `cpus = 1`.
