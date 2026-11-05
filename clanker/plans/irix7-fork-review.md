# The `irix7/iris` fork — change inventory and triage

**Reviewed:** 2026-10-11. **Range:** `557d46cd..irix7/main` (103 non-merge
commits, 2026-10-08 → 2026-10-09), 139 files, +41,462/−2,696.
**Fork:** `https://github.com/irix7/iris` (remote `irix7`).

A fork of this tree was driven hard by a fleet of agents over about 36 hours.
It is mostly a GitHub-issue-per-change workflow (tickets #1–#77), so nearly
every commit names its ticket and several ship a `clanker/` note or a `docs/`
spike alongside. The quality is uneven but **not** random: the research notes
are unusually good, several refactors are well tested, and two changes are
outright regressions for interactive use.

This document classifies everything and says, per cluster, whether we should
pursue it. It makes no changes to our tree.

## How to read this

**Nothing here should be cherry-picked as code.** The fork is BSD-licensed, so
we legally may, but not one line of it went through human review — it is agent
output reviewed by agents, and §3 and §6 below are what that produced. The
useful output of this fork is its *findings*: the `clanker/` notes, the two
go/no-go spikes, and a handful of designs that are right. Treat every verdict
below as "is this finding worth acting on", and write our own implementation
against our own tests.

Verdicts:

| | meaning |
|---|---|
| **TAKE** | the finding and the design are both right; reimplement here |
| **TAKE NOTE** | the research note or decision doc is the deliverable; copy it, skip the code |
| **REJECT** | wrong for us, or a regression |
| **PARK** | interesting, blocked, or not worth the churn now |

### What was actually verified

A release build of `irix7/main` at default features passes its own unit suite:
**1026 passed / 0 failed / 13 ignored**. That is the whole of it. Specifically:

- **No IRIX boot, ever.** Every test they added runs against the Indy PROM
  command monitor with no IRIX media — `test/fleet/run.sh`, `test/journal/run.sh`
  (whose own comment says "a full threaded IRIX boot remains unverified"), and
  the HAL2 wav capture. The two workflow jobs that *would* boot a guest
  (`profile-irix`, `bench-irix`, added on the last day in `76411910`) are gated
  `if: env.IRIS_IRIX_IMAGE_URL != ''` — and a job-level `if:` cannot read the
  `env` context in GitHub Actions at all, nor is that secret ever placed in
  `env:`. Both jobs are dead and never ran.
- **No SCSI or network benchmarking.** See §0.
- They say so themselves. `d90dcc2d`, the bulk-DMA + NAT commit:
  *"Bulk DMA and NAT poll still need the bare-metal cpu-tests + boot smoke
  (issue #22), which is blocked on the missing MIPS cross-toolchain."*
  And `9d1988a3`: *"Bare-metal cpu-tests and iris-bench still need running on a
  host with the MIPS cross-toolchain."* Neither gap was ever closed.

So the evidence behind every change in this fork is: it compiles, and the
host-side unit tests pass. For the device-threading changes, that is close to no
evidence at all.

---

## 0. Was the I/O work benchmarked? No.

Asked directly, because the SCSI and NAT changes are both justified as
performance work. There is no measurement of either:

1. **The bare-metal suite cannot measure them.** `bench/kernels/` is
   `codec.c`, `fpu.c`, `imaging.c`, `integer.c`, `memory.c`, `sys.c` — CPU,
   FPU, memory, TLB, cache-op, LL/SC. No disk, no network, no filesystem. All
   **412** entries in their `data/bench_history.json` carry only `codec/`,
   `img/`, `vid/`, `fpu/`, `mem/`, `int/` and `sys/` kernels; grepping the file
   for `disk/`, `net/` or `fs/` returns zero hits. So the 44 bench/CI commits
   and ~22,700 lines of plumbing measured exactly the part of the emulator that
   neither change touches.
2. **The suite that could measure them is ours, and they never ran it.**
   `bench/irix/steps.toml` — `disk/write`, `disk/read_cold`, `disk/read_warm`,
   `disk/copy`, `fs/*`, `mem/syscall_copy` — predates the fork (`07d8a3bc`) and
   is byte-identical on `irix7/main`. It needs a booted guest, which they never
   had.
3. **Even that would not have covered the network change.** The only network
   step is `net/loopback_ping`, which never reaches seeq or the NAT engine.
   Nothing in our suite or theirs measures host↔guest throughput, so the NAT
   rewrite has no before/after at any level.

The practical consequence: the bulk-DMA path (§3.2) and the NAT `poll(2)`
rewrite (§2.5) are the two changes in this fork with the largest behavioural
blast radius on real I/O, and they are the two with the least evidence —
unmeasured *and* never run against a guest. That is independently a reason to
reimplement rather than port, separate from the review question.

---

## 1. Headline summary

| Cluster | Commits | Verdict | Note |
|---|---:|---|---|
| Observation-only cache shadow for R4400/R5000 (+ `accurate-cache` opt-in) | 4 | **TAKE** | the only change worth close to a direct port |
| `mc_vdma` BUS_BUSY → resumable re-dispatch | 1 | **TAKE** as correctness | not a perf win; the no-drop/no-duplicate property is the value |
| GR2 FIN2/FIN3 sequence counters | 3 | **TAKE NOTE** | a hardwired special case of the GFIFO fence design; build the general one |
| REX3-JIT hot/cold compile queue | 1 | **TAKE** | fixes prefetch starving live draws |
| NAT `poll(2)` readiness wait | 1 | **TAKE** the finding | unmeasured; 200 ms tick == our RTO |
| HAL2: backends, resamplers, drift control, snapshot | 6 | **TAKE** mostly | one clamp removal needs re-litigating |
| `BlockNode` storage graph (QEMU-shaped) | 3 | **TAKE** | clean deletion of the `DiskBackend` enum |
| ppmem scalar fast path + SIMD word swap | 1 | **TAKE** | `vpshufd 0xB1`, with the `0x4E` trap documented |
| MC decode table by pointer swap + grace | 2 | **TAKE** | removes a latent torn-table window |
| Snapshot per-bank dirty bitmap | 1 | **TAKE** | incremental CAS saves |
| Bulk DMA (`read_block`/`write_block`) | 1 | **TAKE NOTE** | notes are excellent; code is a third built and defective — §3.2 |
| Snapshot `StateDesc` codec + versioning | 3 | **TAKE NOTE** | right idea, very wide blast radius |
| VINO descriptor-engine split | 3 | **TAKE NOTE** | the plan doc is the valuable part |
| Parallel instances (`--instance`, `StatePaths`) | 5 | **TAKE NOTE** | useful for agent fleets; large CLI surface |
| Fault-based MMIO spike | 1 | **TAKE NOTE** | NO-GO, well argued; the prototype is dead code |
| jitv2 arena free-list impossibility note | 1 | **TAKE NOTE** | confirms what we already concluded; good write-up |
| jitv2 callout clobber masks | 3 | **PARK** | measured gain is one reload per store; no wall clock |
| jitv2 generational arenas / AOT spike | 3 | **PARK** | already our own thinking; their notes add little |
| Record/replay journal on guest cycles | 2 | **PARK** | downstream of the clock change below |
| **Cycle-derived clock as the default** | 2 | **REJECT** | breaks guest wall time — §3.1 |
| Guest-time timer queue (as shipped) | 1 | **TAKE** the queue, reject the clock | §3.1 |
| **jitv2 active-region SMC detection** | 3 | **REJECT** | overkill; SMC implies a cache flush, which already bails out — §4.4 |
| Interpreter switch dispatch | 1 | **REJECT** | refuted by their own A/B — §4.3 |
| bench/CI machinery | 44 | **mostly REJECT** | §5 — 42% of commits, ~22k lines, measuring nothing relevant |

## 2. What is genuinely worth taking

### 2.1 Observation-only cache shadow for R4400/R5000 — **TAKE**

`1fc21693` (#34), `8df2ad09` (#35), `1f2c679f`, `7fa9415c`.

This is the one we already wanted (it is the "shadow cache is the future interp
option" item). The implementation is almost embarrassingly small: `ShadowCache`
already existed for the R10000, so they added two geometry aliases and flipped
the selector.

```rust
// src/cpu/mips_cache_shadow.rs
pub type R4400ShadowCache =
    ShadowCache<16384, 16, 16384, 16, 1048576, 128, false, 0x0000_0440, 0x0000_0500, 48, false>;
pub type R5000ShadowCache =
    ShadowCache<32768, 32, 32768, 32, 0, 128, true, 0x0000_2321, 0x0000_2300, 48, false>;
```

Three details they got right and we should keep:

- **Snapshot identity.** `NAME` is derived from `PRID` (`"R4400"` / `"R5000"`,
  with the R10000 keeping the historical `"shadow"`), because
  `cpu_model_mismatch` compares that string alone — a single `"shadow"` would
  let an R5000 restore accept an R4400 snapshot. Existing IP28 snapshots still
  load.
- **Geometry is asserted equal to the functional model** in a unit test
  (`IC_SIZE`/`IC_LINE`/`DC_*`/`L2_*`, `PRID`, `FIR`, `MIPS4`, `TLB_ENTRIES`), so
  CP0 Config and the PROM's cache sizing do not move.
- **`selected_cache_model()`** is the one place the choice lives, so the
  selector is testable without constructing a `Machine` (the process allows
  only one).

`mips_cache_v2` stays in the tree behind an opt-in `accurate-cache` feature that
also re-enables VCE. The cost is explicit and already priced into their CI:
the shadow has no write-back, so `cache/hit_inv_discards` loses its 2 checks and
the cpu-tests baseline moves **124 → 126** on the interpreter cells (R5000:
108 → 110) — exactly the same trade jitv2's implied tcache already makes.
`accurate-cache` runs as its own CI job at 124/108.

Claimed win: 3–4× on memory-bound kernels for the interpreter. Worth
re-measuring on our own `iris-bench` before believing the number, but the shape
is right and the risk is contained by the feature gate.

### 2.2 `mc_vdma`: BUS_BUSY → resumable condvar re-dispatch — **TAKE**, as correctness

`c8e7a491` (#40) — +927/−310 in `mc_vdma.rs`.

**Not a throughput change, and should not be sold as one.** VDMA is not where
the time goes, and removing a busy-wait buys host CPU, not guest spans/s. The
value is that a `BUS_BUSY` from the GIO sink currently either spins or loses the
push, and this makes the refusal a first-class resumable state.

Each transfer nest becomes an explicit state struct (`NestState`,
`ChunkedState`, `LinesState`) so a refusal parks on a condvar and resumes at
exactly the point it stopped. `Gr2::notify_space()` and the REX3 equivalent wake
it when FIFO space appears (`310c5117`, #67, adds the GR2 side).

The tests are the reason it is worth anything: `refused_gio_ops_resume_identically_on_every_path`
and `refused_gio_reads_resume_identically_on_every_path` run a reference transfer
against a `ResumeSink` that refuses the bulk path, the scalar path, and both, and
assert the resulting memory is byte-identical; `busy_gio_write_yields_and_resumes_without_duplicating`
pins the no-duplicate property.

That property is the one we actually want: our pressure-sweep bench has been
mis-reporting because pushes were dropped under BUS_BUSY (the 22 Gpx/s with flat
spans/s artefact), and "a refused push resumes without duplicating or losing
work" is exactly the invariant that makes such a measurement trustworthy. Take
it for that, not for speed, and expect no benchmark movement.

### 2.3 GR2 FIN2/FIN3 sequence counters — **TAKE NOTE**; build the general fence instead

`b8976a01` (#57), `a47c4517`, `310c5117` (#67), plus the two `clanker/gr2/` notes.

What they did: replace `fin3_pending`'s count-up/count-down with a monotonic
sequence. Every HQ2 FIFO word gets the next `hq_queued_seq`; a `0x0A3`/`0x155`
Finish token records its own sequence as `fin3_target`; `version` bit 0 and the
`0x6B000` read show FIN3 only once `hq_executed_seq >= fin3_target`. With
several Finishes in flight the target names the last one. FIN2 and GEDMA
readiness share the same counters via `fin2_target`, and `a47c4517` rolls the
targets back when the FIFO rejects a push.

**This is the fence design, discovered the long way round and then hardwired to
three GR2 registers.** The mechanism already proposed here is the general form:
a fence token travels through the GFIFO and, when the engine executes it, writes
a monotonic value to an atomic in the GE and in the RE. Every "has X completed"
question then becomes a comparison against those atomics — FIN2, FIN3 and GEDMA
readiness are three instances of it, MGRAS's equivalents are three more, and the
copies/readbacks that the parked dual-RE3 experiment needs are the case their
per-register targets cannot express at all (two REs, each owning `y % 2` rows,
need a fence each). Their `hq_queued_seq`/`hq_executed_seq` pair is a
single-engine, single-consumer fence with the token implicit in the word index.

So: do not port this. Build the fence once, in a form GR2 and MGRAS share, and
let FIN2/FIN3 fall out of it. What the fork is worth here is the validation that
the approach works against real IRIX, plus three specifics that cost them real
debugging:

1. **Progress-driven stall limits, not wall-clock.** `FIN2_STALL_LIMIT` counts
   2 s *without the executed-sequence counters moving*. A fixed cap from the
   first poll is what made `gltest --bench 500` (~2.4 s of RE3 work behind a
   context switch) fail: the kernel's FIN2 poll timed out and IRIX 6.5.22
   crashed in `Gr2PcxSwap`'s `0x1E5` clip loop at PC `0x882adcac`. N = 300–400
   passed, hardware passes at 1000. A fence comparison has the same problem and
   needs the same answer.
2. **Sampling order.** Sample "HQ2 working" *before* reading the register — the
   HQ2 raises FIN2 before it consumes the entry and drops `hq_busy`, and
   sampling after lost the race about one run in three.
3. **Roll back on a rejected push.** Whatever assigns the fence value has to
   undo it when the FIFO refuses the word, or the fence names work that was
   never queued. This is what `a47c4517` is, and a fence scheme has it too.
4. **It must be saveable.** `fin3_pending` was not, so a snapshot lost which
   command was done. Fence atomics go in the board state.

### 2.4 REX3-JIT hot/cold compile queue — **TAKE**

`951da794` (#58), new `src/dev/ng1/rex3_jit_queue.rs` (+209).

The shader compiler took work from one bounded `SyncSender`; a full channel
dropped the request, which was deliberate (never block the GFIFO consumer) but
made no distinction between a warm-up/prefetch shape and the shape the guest is
drawing right now. A 263-entry warm-up profile could fill all 256 slots and
starve the live draw, which then ran interpreted until the backlog drained —
their note reports 30-second waits in graphics tests.

Two lanes: a bounded best-effort **cold** lane (256, the old capacity) for
warm-up/prefetch, and an unbounded **hot** lane for the draw path that is never
rejected. The worker pops hot-first and parks on a condvar. The queue carries no
Cranelift types, so it lives outside the `rex-jit` gate and is unit-tested
without a compiler (`hot_draw_request_jumps_a_full_cold_queue`,
`hot_request_is_admitted_ahead_of_a_full_cold_queue`).

The rules note adds the invariant we should keep: **if a cold send fails, remove
the shader key from the map**, or later draws see a phantom queued entry and
never request compilation again.

### 2.5 NAT: `poll(2)` readiness wait — **TAKE the finding**, rebuild it

`d90dcc2d` (#26), `src/net/mod.rs`.

Before: the NAT thread woke on a 1 ms condvar timeout and unconditionally called
`update_snapshot()` — three fresh `Vec`s and a full walk of every NAT table —
about 1000×/s of pure idle overhead, scaling with table size. Host→guest data
then waited for the 50 ms backstop.

After: one `libc::poll` over every host socket (TCP with `POLLOUT` only when
`to_host` is non-empty, so it cannot spin), plus a wake pipe fed by a small
`seeq-nat-wake` bridge thread that forwards `tx_wake` notifications, so a guest
TX frame wakes the poll like a host socket. `update_snapshot()` is gated behind a
`snapshot_dirty` flag. Windows keeps the condvar fallback.

**It was never measured.** See §0: there is no host↔guest throughput step in
either suite, and the one network step (`net/loopback_ping`) does not reach the
NAT engine. The 1 ms → event-driven argument is sound on its face and the idle
cost it removes is real and arithmetic, but "host→guest data is delivered
without the 50 ms backstop" is an assertion, not a result. Our port needs a
throughput step before and after — which means adding one, since neither tree
has it.

**Caveat to fix on rebuild:** the poll timeout is 200 ms and is described as
covering "the TCP retransmit (RTO) and TFTP timers". Our retransmit queue runs
at RTO = 200 ms, so a 200 ms tick can delay a retransmit by up to a full RTO.
Compute the timeout from the earliest pending retransmit deadline (QEMU's
`aio_poll` shape, which their own note cites) rather than using a fixed tick.

The companion note `clanker/perf/nat-and-enet-threads-poll-at-1ms.md` flags the
same pattern in `seeq8003`'s rx loop (1 ms timeout, `SeeqState` taken twice per
wake); only the NAT half was done.

### 2.6 HAL2 — **TAKE** most of it

Six commits: `c3dac873` (#44 snapshot), `63ce732b` (#59), `47ce24fc` (#60
backpressure), `50093f0d` (#61 drift control), `ffe3e04f` (#62 traits),
`3f274b6a` (#72), `4f04cffa` (#73 CI capture). Net +1377/−177 in `hal2.rs`.

What lands:

- **`AudioBackend` / `AudioOutput` / `Resampler` traits** with `CpalBackend`,
  `NullBackend` and a **`WavBackend`**. The WAV backend is what makes audio
  testable at all — `4f04cffa` runs it in CI and compares a captured prefix.
- **Two resamplers**, `CatmullRomResampler` and `SincResampler`, selectable via
  `[audio] resampler`.
- **`RateCtl`** — closed-loop drift control on the rate loop, and `Pacer`.
- **Bounded-buffer backpressure** replacing the ad-hoc clamp.
- **Snapshot support** (HAL2 was previously unsaved; `RollbackCheckpoint` grows
  a `hal2: Option<toml::Value>`, `None` under `--noaudio`).

What to re-litigate: `3f274b6a` **deletes `DmaClient::read_ahead_of_poll`**, the
clamp that stopped codec A reading ahead of the guest's channel-position poll.
Its replacement is `replay_recorded`, a unit test that drives a synthetic PCM
capture through the production drain at a bounded ring with a modelled host
stall. That is a decent test, but it is a *model* of a busy host, not the guest
polling behaviour the clamp existed for. Port the traits/resamplers/drift
control first and treat the clamp removal as a separate decision with a real
IRIX audio check behind it.

### 2.7 `BlockNode` — the storage graph — **TAKE**

`df7f6c75` (#49), `1801c063` (#50), `d9c1953f` (#51). New `src/block_node.rs`
(+1236); `src/scsi.rs` loses the `DiskBackend` enum entirely.

QEMU's block-graph shape: every image is a node with an optional `backing` node,
one implementation per backend (`RawNode`, `CowNode`, `ChdNode`, `ChdCdNode`),
and the query that makes copy-on-write uniform across formats is
`block_status(offset) -> BlockStatus { state: Data | Zero | Allocated, len, backing_offset }`
— "does this layer hold these bytes, and if not, where do I read them from?"
`commit`/`rollback`/`dirty_count`/`is_cow` are methods on the same contract,
defaulting to no-ops for a node with no divergence layer.

The module docs do the thing our own tree does well and define **one** meaning
for dirty across every backend: a node is dirty when its divergence layer holds
at least one sector shadowing the base it was opened over. Raw returns an exact
dirty-sector count; a CHD diff returns 1/0 because MAME only exposes a
whole-image flag. Callers that want the predicate test `dirty_count() > 0`.

`ScsiDevice::backend` becomes `Option<Box<dyn BlockNode>>` and `cow_commit` /
`cow_rollback` collapse from backend-matching bodies to one `node.commit()`.
This is a straight improvement and the direction we would have gone anyway.

### 2.8 MC decode table by pointer swap — **TAKE**

`06a6d38a` + `372c058b` (#41), `src/physical.rs` +467/−99.

The 65536-slot decode table was mutated in place by `remap_banks` while other
threads indexed it. Now it is built off to the side and published with one
`AtomicPtr` swap, with superseded snapshots retained in a `retired` ring
(`RETIRED_KEEP = 2`, "current plus previous generation plus slack") so a reader
holding an old pointer stays valid. The grace period is counted in **remaps**,
not wall time, with the argument spelled out: a reader would have to be
descheduled across all `RETIRED_KEEP` subsequent *real* remaps to outlive it,
and a remap is serialised on the single CPU thread. IP28's frequent refresh-bit
rewrites are explicitly excluded — `apply_plan` diffs the plan and publishes
nothing when every bank stays put.

Reader cost is unchanged in shape: one acquire load, then a plain index into
immutable memory.

### 2.9 Snapshot: per-bank dirty bitmap — **TAKE**

`12e53f07` (#48). `chunk_store.rs` +248, with `put_words_as_chunks_selective`
and `Machine::last_chunk_hashes`. A save reuses the previous CAS manifest's
hashes for chunks the per-bank dirty bitmap reports clean, so an unchanged bank
is neither cloned, re-hashed nor re-written. Cleared on any load (which mutates
RAM wholesale). Straightforward and self-contained.

### 2.10 ppmem scalar fast path + SIMD word swap — **TAKE**

Part of `9d1988a3`; note at `clanker/perf/word-swap-bulk-copy-simd.md`.

Two things. First, the ppmem bitmap test moves to the top of
`Physical::read8/16/32/64` and `write8/16/32/64`, so an uncached (KSEG1) access
or an interpreter cache-miss fill no longer pays two `dyn BusDevice` dispatches
plus `PpMemory`'s `addr_mask` atomic load for what is plainly mapped RAM. Their
note makes the right argument from prior art: **uncached RAM is still RAM**, and
no emulator routes KSEG1 traffic off the direct path.

Second, the bulk word swap. Guest RAM stores each `u64` `rotate_left(32)`;
`read_block`/`write_block` did a scalar rotate loop. The correct instruction is
a dword shuffle — `PSHUFD` with imm **`0xB1`**, SSE2 baseline, `REV64.32` on
aarch64, portably `simd_swizzle!(v, [1,0,3,2,5,4,7,6])`. LLVM lowers
`rotate_left(32)` to `llvm.fshl` → `vpsllq`+`vpsrlq`+`vpor` and never
pattern-matches the half-swap, so the loop is ~3× the ALU work.

Two traps recorded, both worth keeping in the note when we port it: `0x4E` is
the **wrong** immediate (it swaps the two `u64`s in a lane, not the halves of
each — the ppmem unit tests caught it), and `Simd::swap_bytes` is a full byte
reverse, the wrong operation.

Their own measurement attributes the headline "+9% guest MIPS" from `9d1988a3`
to *this*, not to the switch dispatch it shipped with (§4.3).

---

## 3. What is broken

### 3.1 The clock: cycle-derived time became the default — **REJECT**

`e8b264e5` (#42) and `800f7852` (#43).

`ci_clock` — our opt-in deterministic clock — was promoted to unconditional and
the feature reduced to a no-op. The host `Instant` anchor is **gone**:

```rust
// src/cpu/mips_core.rs
pub const NS_PER_GUEST_CYCLE: u64 = 10;
```

CP0 Count is now a pure function of retired cycles at 10 ns each, scaled to
guest ticks by the fixed `count_hz` (33 MHz IP22/IP24, 97.5 MHz IP28). That is a
real determinism win — the same ELF replays identically at any host speed, which
is what their record/replay journal (#47) is built on — and it kills the
`cp0/count_writable` flake for good.

It also means **the guest models a machine that retires exactly 100 MIPS**, so
guest wall time runs at `host_MIPS / 100` × real time:

- interpreter at ~38 guest MIPS (their own measured figure) → guest clock runs
  **~2.6× slow**;
- jitv2 at several hundred MIPS → guest clock runs **fast**;
- and it flips between the two depending on the workload, because idle parking
  is still host-paced: `park` sleeps toward the next guest deadline and then does
  `core.hot.cycles += elapsed_ns / 10`, i.e. exactly 100 MIPS while idle.

So anything a human perceives — `date`, cursor blink, key repeat, X11
animation timing, network and NFS timeouts, `sleep` — is skewed by a
workload-dependent factor. This is the regression that makes the fork unusable
interactively, and it is not a bug in the implementation; it is the design
choice.

**Verdict:** reject the default. The determinism is worth having as the opt-in
it already was. If we want both, the missing piece is what their own
`ci_clock` comment used to say and the new code drops: a calibration loop that
rescales `NS_PER_GUEST_CYCLE` against the host clock, so guest time tracks real
time on average while staying monotonic and derived from cycles within a
window. That is the thing to build, not this.

**What to keep from #43 anyway:** the *timer queue* is good independent of the
clock policy. `src/cpu/guest_timer.rs` (+482) gives one ordered queue of
absolute `hot.cycles` deadlines with a stable same-deadline tie-break; the 8254
PIT channels and the CP0 Compare deadline both register on it; the run loop
drains it once per dispatch batch; named deadlines are serialisable and re-sorted
on load, and transient callbacks are never saved so an anonymous timer cannot
corrupt a snapshot. The PIT reading its count-down from guest time instead of a
host `Instant` is why their latch/readback timing tests became deterministic and
were re-enabled (`pit8254.rs` +176/−165).

Idle parking also stops being a fixed 1 ms slice: it parks to the next queued
deadline, capped at 100 ms. `2db97f58` (#68) measures it — best of five samples
bounded at 500 µs for a 300 µs deadline, with ~60–75 µs intrinsic overshoot on
an idle host — which is a reasonable regression guard.

Port the queue against a host-paced clock; drop the clock change.

### 3.2 SCSI — degraded, and the promised win was not delivered

Two separate things got conflated here.

**(a) The design note is right.** `clanker/scsi/hpc3-dma-is-bulk-not-byte-at-a-time.md`
makes an airtight case from the Linux drivers for this exact chipset:
`sgiwd93.c`'s `fill_hpc_entries()` builds a descriptor *chain* in bulk (one
descriptor per ≤8 KiB chunk — the comment "without magic only 8192 works
correctly" records a hardware quirk), tags the last `HPCDMA_EOX`, and
`dma_setup()` kicks the whole chain once. `wd33c93.c` walks scatter-gather per
*buffer*. The SCSI bus handshake (REQ/ACK per byte) is irreducibly byte-level —
MAME keeps it byte-level too — but the **HPC3 DMA-memcpy** layer should move
descriptor spans. Our `DmaClient` exposes only per-unit `read()`/`write()`, so a
64 KiB SCSI block is thousands of mutex acquires plus thousands of full bus
traversals. That analysis is correct and actionable.

The companion note `clanker/scsi/wd33c93a-must-not-hold-state-lock-across-dma.md`
is also correct and arguably more important: the worker holds the `state` guard
across the whole transfer while the CPU thread takes the same lock on every
register read, *and* the arm-wait inside it is a 100 ms `yield_now()` spin — so
the CPU thread can stall on the mutex for up to 100 ms, and the thread that must
arm the channel can be starved by the thread waiting on it.

**(b) What shipped is a third of that.** `DmaClient` gains
`read_block`/`write_block` returning `(DmaStatus, Vec<(u32,u16)>, usize)` with
default impls that loop the scalar path, and `PdmaClientImpl` overrides them.
But look at what the override actually does:

```rust
// src/dev/hpc3.rs — dma_read_block
let units = (remaining / step).min(dst_left);
for i in 0..units {
    match self.read_unit() { Some(v) => dst[read + i] = v as u8, None => { ok = false; break; } }
}
let (st, w) = self.advance((units * step) as u32, false);
```

`read_unit` still does one `mem.read8/16/32` through the MC bus **per unit**.
The only things actually batched are the channel-lock acquire and the
descriptor bookkeeping in `advance()`. The `BusDevice::read_block`/`write_block`
memcpy across the descriptor span — the actual win the note argues for, and the
one that is already available on the trait — was never wired up. Neither was
the lock-free status poll, nor releasing `state` across the data move: the
`yield_now()` spin became a 50 µs `thread::sleep`, still under the `state` lock.

**(c) And the rewrite introduced defects.** In `receive_data_chunked_from` the
retry path is tangled: `buf` is allocated once outside the inner loop, and the
exit condition is

```rust
if data.len() >= total || !data.is_empty() { break; }
```

whose second clause subsumes the first and makes the "spin up to 100 ms for the
channel to arm" loop a single iteration once any byte arrives. Worse, when
`dma_read_block` returns early with `units == 0` and a clean status — which it
does whenever `remaining < step` (a descriptor byte-count not a multiple of the
unit width; reachable on the 16-bit enet/SCSI paths) — the caller sees
`data.len() < total` with no EOX and no NOT_ACTIVE, and treats it as "chain
exhausted early", raising `UNEXPECTED_RECV_DATA` and pausing a transfer that was
never actually at a boundary. The arm-spin is also gated on `i == 0` rather than
`i == offset`, so a *resumed* transfer never waits for the channel to arm (this
one is pre-existing, carried over).

**Verdict: TAKE IDEA, reject the code.** Port both rules notes verbatim — they
are the most useful SCSI research in the fork. Then do the real thing: a
`BusDevice`-level block memcpy over ≤8 KiB descriptor spans, boundary state
carried back in `DmaStatus` instead of a per-unit `is_last`, IRQ delivered
*after* unlocking (the callback-up-to-a-parent deadlock class HACKING.md warns
about), the endianness swap applied once per descriptor at The Edge, `state`
snapshotted and released across the move, and the arm-wait on a condvar that the
DMA channel notifies. Keep the byte path as the PIO fallback, as the real driver
does.

---

## 4. Half-finished and unproven

### 4.1 Bulk DMA — see §3.2(b). The trait seam is fine; the implementation is not.

Note that `wd33c93a` discards the writeback vector (`let (mut st, _wb, mut n)`),
which is correct today only because writebacks are documented as enet-only. If
we port the trait, that invariant needs an assertion, not a comment.

### 4.2 jitv2 callout clobber masks — **PARK**

`0ebca992` (#36), `e0cf0943` (#37), `f1513f15` (#66), +~1200 lines across
`callout.rs` and `codegen.rs`.

Every JIT→Rust callout gets a declared GPR/FPR read/write mask, consumed by a
per-region GPR forwarding cache. #66 is the honest part: it adds the measurement
that #37 skipped, and the measurement is thin.

The measured delta is **one avoided reload per store-crossing read**
(`masked avoided=2 emitted=5` vs `conservative avoided=1 emitted=6`), and a
`lw` gets nothing at all — `emit_mem_read` follows the callout with
`emit_check_mem_status`, which splits to a fresh block, and `GprForward::put`'s
`sync` clears the whole cache because the block changed and was never `bless`ed.
Their write-up says the quiet part out loud: the original claim ("a smaller
`SpillPlan`") was not evidence of anything, because the spill plan shrinks in
both the store and load cases and only the store case recovers a reload.

There is no corpus (no IRIX boot in that worktree) and no wall-clock figure; the
recovered load was an L1-resident `core.gpr` access already off the critical
path. The A/B switch (`IRIS_CALLOUT_MASKS=0`, `Codegen::set_callout_masks_enabled`)
is nice infrastructure.

The actually interesting follow-up is named in the note and not done:
`emit_check_mem_status` and `emit_trap_if_nonzero` do not `bless` their
continuation blocks (the interrupt and SMC preambles do), so any GPR write after
one wipes the forwarding cache. That is the lead worth chasing, not the masks.

### 4.3 Interpreter switch dispatch — **REJECT (refuted by its own measurement)**

Part of `9d1988a3`; `clanker/perf/interpreter-dispatch-is-call-threading.md`.

Replacing call-threading with a `match`-based `dispatch_instr` was predicted
(from Ertl's numbers) to win on dispatch-bound code. Their own A/B says it did
not:

| build | accuracy | guest MIPS | DMIPS |
|---|---:|---:|---:|
| call-threading (`0aacb75`) | 100% (40/40) | 38.0 | 66.8 |
| switch dispatch (`9d1988a`) | 100% (40/40) | 41.5 | 66.0 |

`int/alu` ~+1.5% both ways, `int/muldiv` and `int/dhrystone` slightly negative.
The +9% MIPS is in the `img/` byte-store kernels — i.e. the ppmem/SIMD memory
changes that shipped in the same commit (§2.10). And the host was at load ~8.7
on 6 cores with ±5–21% same-binary spread, so even that is soft.

Two useful conclusions to keep: a dispatch-only diff on an idle host is still
unmeasured, and **`opcodefusion` is masking per-dispatch cost** — fused pairs
reduce the number of dispatches, so its value should be re-evaluated against any
new dispatch baseline rather than assumed.

### 4.4 jitv2 SMC detection — **REJECT (overkill)**

`2a7889a8` (#39), `f8394e69` (#74), `b8a23931` (#75).

A per-page compiled-line bitmap (16-byte lines, plus a range form for block
writes) correlates a write with the code a region actually executes instead of
the whole 4 KiB page. CPU inline/interpreter stores and DMA/`PpMemory` writes
latch the executing page's `smc_hit`, and a new **per-instruction** compiled-code
preamble force-exits the region to `EXEC_FALLBACK` so the next dispatch sees the
bumped generation and recompiles.

The window it closes — a native self-modifying loop where a backward branch keeps
running stale code — requires a guest that rewrites instructions *and then jumps
back into them without flushing the I-cache*. On MIPS that is not a thing a sane
guest does: the architecture requires a cache flush between the store and the
fetch, and our flush path already bails out of the JIT. So the correct detector
is the one we have; this is machinery for a guest nobody shipped.

The cost is not small either. It adds `jit_active_pfn`, `jit_smc_lines` and
`jit_smc_hit` to `MipsCore` — two of them raw pointers kept permanently valid by
aiming at shared `SMC_IDLE_LINES` / `SMC_IDLE_HIT` statics so a branchless test
never null-derefs — and a preamble on *every compiled instruction*. That is new
hot-path cost in the part of the tree with the least slack, bought to handle an
insane guest.

Keep `b8a23931`'s self-modifying-guest test shape in mind if we ever want to
assert the flush-path bail-out directly. Skip the rest.

### 4.5 jitv2 arenas and AOT — **PARK**, take the notes

- `89671d8c` (#38) split the code arena by lifetime; `8c6265aa` then documents
  **why the page-granular free list it was asked for is impossible**:
  `SharedArena` is a monotonic bump allocator whose `allocate` skips past
  `sealed_up_to`, sealing is one-way RW→RX with no reverse path anywhere, and
  `free_memory`/`Drop` release only the whole reservation. Every byte below the
  watermark is permanently unreachable. A real free list would need RX→RW
  re-protect (an explicit W^X window) or a dual RW/RX mapping. There is also no
  liveness map from arena bytes to guest pages, and packing puts many functions
  on one host page, so "this page is dead" is not cheaply decidable. Excellent
  note; it should supersede that acceptance criterion in our own tracker too.
- `f8308648` (#77) then builds the thing the note points at instead —
  generational arenas, retire a whole arena once its code is dead (+374).
- `8d4b8619` (#54) is an AOT/persistent-cache spike: **GO on feasibility, GO
  for a bounded opt-in offline filler, NO-GO on a from-binaries AOT compiler,
  on making AOT default, and on shipping prewarmed caches.** The prototype
  (`src/bin/jitv2_aot.rs`) re-runs the real compile front half over a
  `j2 corpus` capture and writes production-format blobs. Useful background: PIC
  is already the default and relocation-freedom is already a hard gate
  (`compile_region` only records a blob when `relocs().is_empty()`), with **zero
  relocation refusals** measured over a real IRIX workload.
- `4f817789` (#76) adds `test/aotbench/aotbench.py` (+554), a repeatable
  first-boot AOT measurement harness.

All of this is parked on our side anyway — and we have an open lead that matters
more (the jitv2 compile queue staying full from ~58 s of boot onward).

### 4.6 Snapshot `StateDesc` codec — **TAKE IDEA**, mind the blast radius

`e86b0018` (#45), `ed9bcce6` (#46), `559f3d92` (#69). New `src/state_desc.rs`
(+638); `machine.rs` +571 across the three.

A device describes its state fields once — `Saveable::state_desc` returning a
`StateDesc` of (field name, kind, read/write closure, `since_version`) — and the
postcard payload, a human-readable field list and a schema signature are all
generated from it. The field list and signature go into `snapshot.toml` (the
QEMU "VM Description" analogue); load checks the recorded signature and exact
field set before applying state, so a missing or renamed field **fails the
restore** rather than loading a silent default. That is Dolphin's Verify, with
Measure as the signature without serialising. #46 adds per-device versioning and
a migration registry.

`Saveable`, `SCHEMA_VERSION` and the on-disk format are unchanged, and
unmigrated devices keep their hand-written codec with a value-shape integrity
signature — so it is genuinely incremental. By #69 they had migrated MC, IOC,
HPC3, SCC, PIT, PS/2, WD33C93A, DS1x86 RTC, Seeq, testdev, plus `mips_exec`,
`eeprom_93c56`, GR2, HAL2, MGRAS and REX3.

The idea is right and we should want it. The problem is that #69 alone rewrites
the save path of six devices including GR2, MGRAS and REX3 — the three we are
actively changing. Port the `state_desc` module and the signature/Verify
machinery, migrate devices one at a time on our own schedule, and keep the
graphics boards for last.

### 4.7 VINO — **TAKE IDEA** (the plan doc)

`f446354c` (#55) is a plan-only spike with a single `#[ignore]`d target test
encoding the acceptance criterion ("IRIX 6.5 `vidtomem` capture yields a clean
frame"). It tabulates exactly where the current model lives — `render_and_pump`
is pixel-driven and `dma_emit_dword` is the only place STOP is consumed — and
argues the shipped 6.5 path works via two 6.5-only heuristics layered on that
pixel-driven loop. `0f15003b` (#56) and `f84d43ab` (#71) then split pixel
production from descriptor fetch (+224/−86); `dfc1f0b6` (#70) adds 5.3-shaped
validation coverage and extends `clanker/video/vino-verification-checklist.md`.

Not our priority, but if VINO comes up again the plan doc is a better starting
point than re-deriving it.

### 4.8 Parallel instances — **TAKE IDEA**

`df5df542`, `947a60c1`, `1bd24a1c`, `ac957abc`, `8ea03119`, plus `1528d703`.

`--instance N` / `--state-dir` / `--port-base` / `--print-instance` derive a
whole isolated VM: a private state dir (`iris-instance-N`), NVRAM/EEPROM paths,
a monitor/serial/CI port block from `9000 + N*10`, a CI socket under the state
dir, a stable FNV-1a-derived SGI-OUI (`08:00:69`) MAC, and a distinct NAT /24.
Two seams own all of it and nothing else may invent a path or port:
`src/state.rs`'s `StatePaths` (with `neutral()` reproducing today's paths
exactly) and `config::derive_instance()`, which is pure and unit-tested.
Explicit user values always win over derivation. `test/fleet/run.sh` boots two
instances and drives each over its own CI serial socket using the Indy PROM
command monitor, so it needs no IRIX media.

The architectural decision is stated and is the right one: **one `Machine` per
process**, because idle parking, CPU affinity, the dev log and the CI socket
path are process-global. A fleet is N processes.

This is genuinely useful for running several agents against separate guests,
which is a problem we have. Worth taking, at our own pace — `StatePaths` first
(it is additive and nothing consumes it until wired), then the CLI.

### 4.9 Fault-based MMIO — **TAKE IDEA** (doc only)

`322ad0bd` (#52): **NO-GO**, well argued. Device apertures are *already*
`PROT_NONE` inside the 4 GiB ppmem reservation, so "map device pages trapped"
is already true; and the branch is not where the cost is, because jitv2 already
emits plain host loads/stores for L1D-cached RAM and already calls back into
Rust for everything else. Fault-based MMIO would substitute a host fault plus a
handler for a callout that is far cheaper (~µs vs ~1 ns for a predicted branch),
block the faulting thread, and put signal handlers next to per-device locks.

The 625-line `src/faultmmio.rs` prototype is behind an off-by-default
`faultmmio` feature and deliberately not wired into anything. Take
`docs/ppmem-fault-mmio-spike.md` and `clanker/ppmem/fault-mmio-restartability.md`;
leave the prototype.

### 4.10 Record/replay journal — **PARK**

`6628353a` + `2f5b89e1` (#47), `src/cpu/journal.rs` (+443), `test/journal/run.sh`.
A journal keyed on guest cycles, driven over the real run loop with a CI
assertion. It only means anything if the clock is cycle-derived, so it is
downstream of §3.1. Revisit if we ever take a calibrated cycle clock.

### 4.11 Build config — **TAKE** selectively

`clanker/build/allocator-pgo-native-defaults.md` is a good note. `mimalloc` went
in as the global allocator and their measurement says **~35% peak RSS cut with
no wall-clock regression** — note that this is the *opposite* sign from the
rustc experiment the note itself cites (which saw up to +35% RSS), so measure
locally before trusting either. `release-pgo` and `release-native` profiles
are added, with the correct caveat that Cargo profiles cannot set
`-C target-cpu=native` and that a `.cargo/config.toml` doing it unconditionally
would silently apply it to distributed binaries.

`flake.nix` pins their rig's toolchain under `/mnt/europa/...` — host-specific,
not portable; skip it.

---

## 5. The bench/CI machinery — mostly churn

**44 of 103 commits (42%)** are `bench:` or `ci:`. They account for roughly
22,700 of the ~41,500 inserted lines, including:

- `data/bench_history.json` — **13,427 lines** of generated data,
- `data/bench_history.svg` (1,831), `bench_history_eff.svg` (1,326),
  `bench_speedup.svg` (682), `bench_heatmap.svg` (366), and three more SVGs,
- `data/bench_history.md` (418), `BENCHMARKS.md` (181),
- `tools/bench_graphs.py` (1,059), `tools/bench_history.py` (470),
  `tools/bench-backfill.sh` (149),
- three new workflows (`bench-ab.yml`, `bench-backfill.yml`,
  `bench-profile.yml` at 358 lines) and `bench/backfill-commits.txt` (201).

The history itself was rebuilt at least six times in a row
(`55cd2fe7` → `fe5ba69a` → `c29ff7f5` → `8877a691` → `29b9a950` → `cee12d65`
→ `345e94d4` → `c4a62cd4` → `f1b23f43` → `7d85dd61` → `1255896b` → `6e3ef56e`
→ `71c5e4fc` → `31170c70`), with eight `bench: record CI results and
regenerate graphs [skip ci]` auto-commits interleaved. Checking generated SVGs
and a 13k-line JSON into the repo and regenerating them from CI on every push is
exactly the kind of thing we should not import.

**Three things in here are worth salvaging:**

1. **`iris-bench --repeat N` with pooling and CV** (`258dee07`).
   `average_runs` averages N samples, reports `mips_cv_pct`, and — the good part
   — *refuses* to average if any repeat disagrees on accuracy, kernel count, or
   a per-kernel checksum, so an average can never hide a divergence. `Run` gains
   `samples` and `mips_cv_pct`. This is real benchmark hygiene, and it is what
   makes their own ±5–21% host-noise observation visible at all.
2. **Per-cell host baseline** (`71758983`, `1255896b`). Each CI cell runs
   `make -C bench hostbench` plus `iris-bench host` on *its own* runner, and the
   report divides the emulated cell by its own `<cell>-host` run — so
   host-normalised efficiency is comparable across four runners of unknown
   relative speed. Worth copying if we ever care about cross-runner comparison.
3. **The `[skip ci]` gotcha** (`06b45eec`, `a4d6b7fb`): GitHub honours
   `[skip ci]` / `[ci skip]` / `[no ci]` / `skip-checks` anywhere in a commit
   **message**, subject or body, *even inside backticks as prose*. Writing about
   the directive in a commit message skips that commit's workflows. That one
   belongs in `clanker/build/`.

Also here: `2aafdc8b` auto-refreshes `bench/prebuilt/` on main (so a missed
`make -C bench prebuilt` cannot ship a stale image scored against golden
checksums compiled into it), and `aea36137` fixes a glob that handed the report
the host baseline instead of the guest result.

---

## 6. Recommended order of work

Everything here means "reimplement against our tests", not "cherry-pick". The
sequence minimises conflict with what we have in flight.

0. **Add the missing measurements first**, because three items below are
   justified entirely by unmeasured claims. A host↔guest TCP throughput step
   (neither tree has one) and a run of our existing `bench/irix/steps.toml`
   `disk/*` steps give us the before/after that the fork never took. Without
   them we would be repeating its mistake.
1. **Cache shadow for R4400/R5000** + `accurate-cache` opt-in (§2.1). Smallest
   diff, largest claimed win, feature-gated escape hatch. Expect the cpu-tests
   baseline to move 124 → 126 / 108 → 110; confirm the 3–4× on `iris-bench`
   before committing to it.
2. **ppmem scalar fast path + `vpshufd 0xB1` word swap** (§2.10). Self-contained,
   and their own A/B attributes the real `img/` kernel gains to this rather than
   to the dispatch change it shipped with.
3. **MC decode-table pointer swap** (§2.8). Closes a latent torn-table window.
4. **The GFIFO fence** (§2.3) — the general mechanism, atomics in GE and RE,
   shared by GR2 and MGRAS, with FIN2/FIN3/GEDMA readiness falling out as
   queries against it. Carry over their four hard-won specifics: progress-driven
   stall limits, sample-before-read, roll back on a rejected push, and keep the
   fence state in the snapshot. This also unblocks the dual-RE3 copies/readbacks
   case, which per-register targets cannot express.
5. **REX3-JIT hot/cold queue** (§2.4). Small, lands in code we are touching, and
   the "drop the map key when a cold send fails" invariant is worth having
   written down either way.
6. **`mc_vdma` resumable re-dispatch** (§2.2), as a correctness fix. Expect no
   benchmark movement; the payoff is that the pressure sweep stops lying.
7. **HAL2 traits / resamplers / drift control / snapshot** (§2.6), leaving
   `read_ahead_of_poll` alone until there is a real IRIX audio check.
8. **`BlockNode`** (§2.7) and **per-bank dirty bitmap** (§2.9).
9. **SCSI bulk DMA, done properly** (§3.2) — take both rules notes, then build
   the `BusDevice`-level descriptor-span memcpy, release `state` across the data
   move, and put the arm-wait on a condvar. Gate it on step 0's `disk/*` numbers.
10. **Guest-time timer queue** against a host-paced clock (§3.1), and
    `StatePaths` (§4.8) whenever parallel guests become a real need.

Explicitly not pursuing: the cycle-derived clock as default, active-region SMC
detection, the switch dispatch, the bench history/graph machinery, `flake.nix`,
and the `faultmmio` prototype.

## 7. Process observations

Worth noting separately from the content, because it is what made the useful
half useful:

- **One ticket per change, named in the commit subject.** Easy to triage a
  103-commit dump when every commit says `(#42)`.
- **A `clanker/` note or `docs/` spike per finding**, including the negative
  results — `ci_clock`'s predecessor note was marked *Superseded*, the switch
  dispatch's note records its own refutation, and #38's note explains why the
  requested feature is impossible. We already do this; the fork shows the
  payoff at scale.
- **Measurement commits that exist only to check an earlier claim** (#66
  measuring #37, #68 measuring #43, #72 validating #60). Several of those
  measurements came back negative or thin, and the notes say so. That is the
  habit worth copying.
- The failure mode is equally visible, and it is structural rather than
  careless. 42% of commits went into benchmark plumbing that measures neither
  subsystem the fork actually changed. The two worst outcomes — the clock
  (interactive use) and SCSI (a degraded subsystem) — are both a correct
  analysis followed by an implementation that did the easy third and declared
  the ticket done, with "needs a boot smoke, blocked on the toolchain" written
  into the commit message and never revisited. And #39 is the third shape:
  a real but vanishingly rare hazard engineered at full cost, where a moment's
  thought about what the architecture already requires (a cache flush between
  the store and the fetch, which bails out of the JIT anyway) would have closed
  the ticket with a note.
- Which is the argument for not porting any of it directly. Nothing in those
  103 commits was read by a person before it landed, so every one of them
  carries the same risk profile as §3.2's `units == 0` stall: plausible,
  well-commented, unit-tested, and wrong in a way only a booted guest would
  show. The findings are separable from that; the code is not.
