# SMP idle parking: an idle IRIX CPU polls its run queue and gets no interrupt for new work

Status: rules confirmed 2026-11-08 on a 4-CPU Tezro (IRIX 6.5.30, 1 GHz
R16000, `lightning,idle-pause`, M4 Max host); the jitv2 section has its own
status line. The numbers below were measured
on our own tree before it was rebased onto upstream; the rules were ported
unchanged. A multiprocessor is any core with `core.smp` set: IP27, IP30 and
IP35 machines with more than one CPU.

**Symptom.** With `idle-pause` and four CPUs the boot sat at "The system is
coming up." for over eight minutes with all four CPUs parked 98% of the time
(13% host CPU), the guest clock still at its correct 100 Hz, and every CPU in
`idler`/`idlerunq`. `idle off` from the monitor let it reach the login prompt
within seconds. Without `idle-pause` an idle guest burns one host core per CPU.

**Cause, two parts.**

1. IRIX 6.5's idle loop polls the run queues (its own, then a scan of the
   other CPUs' for work to steal) and marks itself "idle polling", so a CPU
   that queues work for it sends no IPI. The CPU that takes the clock tick
   runs the timeouts and queues the threads they wake on *other* CPUs' run
   queues, and the IP35 kernel stops an idle CPU's own clock (`stopclocks`
   from `clkreld`). A park that waits for an interrupt therefore waits for
   nothing: `wake_irq` was a few hundred over minutes.
2. A park ended by an interrupt ran one 1000-instruction batch and was parked
   again at once: the PC ring (32 samples, ~10,000 instructions) still held
   the idle loop's samples from before the sleep, so the window looked idle
   while the CPU was in the clock handler. The handler progressed 1000
   instructions per tick.

**Rules** (`src/cpu/idle_park.rs`).

- On a multiprocessor every slice ends the park: the peer slice (100 µs
  doubling to 800 µs, `IRIS_IDLE_PEER_US`, counted as `wake_peer`) while
  another CPU is busy, the all-idle slice (`wake_slice`) while every CPU is
  idle: 1 ms at first, 10 ms once kicks do the polling (last section). Only
  a uniprocessor sleeps until a line. "Busy" is the detector's per-CPU
  idle flag (`IDLE`, over the CPUs in `SEEN`), which a timer interrupt must
  not clear (only a full window of other work does), or the CPUs keep each
  other on short slices.
- The window counts as idle only if the newest batch's samples
  (`RECENT_SAMPLES`, all but one) lie in the dominant cluster's PC range.
  After a park the ring still holds the idle loop; the newest samples say
  whether the CPU is back in it or in a handler or in work it found. One
  idle batch parks again (cheap slices); a handler or a woken thread runs on.
  This rule applies to a uniprocessor too.
- On a multiprocessor a CPU parks only where another CPU's idle-looking
  cluster (`IDLE_CLUSTER`) lies within `SHARED_LOOP_BYTES` (2 KiB) of its
  own (`alone` in `idle status` counts the refusals). The idle loop is one
  piece of kernel code every idle CPU runs; a busy kernel loop that passes
  the window test runs on one CPU alone. Found with the kernel's
  `melf_findaddr` scanning module symbols with `strcmp` during the rc
  scripts (kernel mode, IE=1, 12+ distinct PCs): parked, it ran 1000
  instructions per slice and the rc phase took 88 s where the no-parking boot
  takes 35 s. The first CPU to idle spins until a second reaches the loop; a
  uniprocessor is unchanged.
- Do not widen `CLUSTER_GAP_BYTES` to catch the run-queue scan (about 0x230
  bytes from the queue check it calls): with 1 KiB, busy boot loops look idle
  and the boot crawls again.

**Measured** on our tree (4 × R16000 1 GHz, 4 GiB, NAT, no graphics,
interpreter):

| build | PROM | IRIX boot | idle host CPU | awk 3M loop | ls -lR |
|---|---:|---:|---:|---:|---:|
| `IRIS_NO_IDLE=1` | 73.5 s | 52.1 s | 401% | 28.4 s | 20.0 s |
| park until a line | — | stalled > 8 min | 13% | — | — |
| slice wakes + newest-batch test | 78.9 s | 121.2 s | 49% | 23.7 s | 17.3 s |
| same, `IRIS_IDLE_PEER_US=25`, a flash reader lock (since reverted) | 68.0 s | 103.6 s | 46% | 23.3 s | 16.6 s |
| + shared-loop rule (default slice) | 71.0 s | 47.0 s | 48% | 23.3 s | 16.4 s |

The PROM column also moved with Bedrock changes not part of this port. Four
awk loops in parallel take as long as one (23.2 s), 32 MiB through `gzip -1`
9.2 s, 400 forks 4.0 s. The 121 s boot was not hand-off latency (a 25 µs
peer slice barely moved it) but the `melf_findaddr` scan parked as idle; PC
samples of the four CPUs once a second during the rc phase found it
(`cpu N regs`, `sym` from the monitor).

## Parking under jitv2: interpreted probe runs and learned exit signatures

Status: probe rules confirmed 2026-11-08, learned signatures 2026-11-11, on
the same Tezro (`lightning,jitv2,idle-pause`, four 1 GiB banks), measured on
our tree before the rebase.

**Symptom.** The jitv2 build ran the guest 3–7× faster but never parked: at
the login prompt `idle status` showed `idle=0b0000`, `largest=32 distinct=1`
and 400% host CPU. The run loop samples the PC between `step_jit` calls, and
a compiled loop keeps its back-edges inside the region (with the interrupt
check), so the samples see only region exits: one or two PCs for the whole
idle path, under `CLUSTER_MIN_DISTINCT`.

**Rules** (`src/cpu/idle_park.rs`, the run loop in `mips_exec.rs`:
`probe_next`, `probing`).

- A full ring dominated by a stationary kernel cluster of too few distinct
  PCs is *unknown* (`stationary_unknown`): a compiled loop seen at its
  exits, or a short poll loop. It is neither idle nor busy: the idle flag
  and the published cluster stay as they were.
- An unknown ring starts a *probe run* (`wants_probe`): the ring is cleared
  and the next `PROBE_RUN` (16) batches run interpreted (`step_int`), so the
  ordinary test sees the loop's own PCs. A park gives the run a fresh budget,
  so the batches after a wake stay interpreted while the clock handler
  returns to the idle loop; a run that spends its budget without a park
  backs off `PROBE_BACKOFF` (256) compiled batches. `probes` in
  `idle status` counts the interpreted batches.
- A window whose dominant cluster is the idle loop but whose newest batch
  left it (the handler, or work the loop found) is neither idle nor busy:
  the idle flag stays. Counting it busy flipped the flag on every clock
  tick, and the peers, seeing a CPU running, cut their parks to the peer
  slice (`wake_peer` was 97% of the parks, idle host CPU 300%).
- Only probe runs cost compiled speed: a busy kernel loop with a single
  region exit loses 16 interpreted batches in 272; a user-mode loop is
  never probed (`users > PC_RING / 4`); a loop the detector keeps resetting
  (interrupts masked, the PROM) never fills the ring and is never probed.
- With probes alone every wake ran its batches interpreted (`probes`
  3,100–3,700 a second on an idle Tezro). A probe run that parks within its
  first two batches therefore learns the (PC, return address) pairs of the
  stationary ring that started it (`learned`). A later ring whose dominant
  cluster holds only learned pairs, with the newest samples among them, is
  the idle loop without a probe (`known_parks`), and once a CPU knows a
  signature the batches after a park run compiled. The return address
  separates the idle loop from a busy loop that exits through the same
  helper; the PC alone could not. A park on a learned signature that wakes
  on a masked line forgets the table, and so does a reset (`hot.cycles`
  going back).

**Wrong turns.** One probe batch at a time with a 16-batch back-off: the
probe's ring was still 29/32 exits, so it never decided idle (286%). Probe
runs that withdrew the cluster on an unknown ring: two CPUs had to probe at
the same moment to satisfy the shared-loop rule, 442 parks in three minutes
(403%). Back-off after a post-park probe that ran the clock handler: 100 ms
of compiled idle batches every tick (307%).

**Measured** on our tree (same machine and workloads as above, probe runs
without learned signatures):

| build | PROM | IRIX boot | idle host CPU | awk 3M | ls -lR | gzip 32M | 400 forks |
|---|---:|---:|---:|---:|---:|---:|---:|
| jitv2, no parking | 72.7 s | 34.8 s | 402% | 4.3 s | 4.3 s | 1.2 s | 1.6 s |
| jitv2, probe runs | 84.5 s | 38.0 s | 51% | 4.2 s | 4.5 s | 1.5 s | 1.8 s |
| interpreter, same rules | 66.8 s | 47.0 s | 42% | 29.1 s | 16.7 s | 9.4 s | 4.1 s |

The interpreter's awk time varied 23–29 s across the day's runs with
otherwise identical results, so the last row's 29.1 s is taken as noise,
not a cost of the rules (its probes stay at 0: the ring is never unknown).
The learned signatures were measured only together with the kicks (next
section), so they have no row of their own here.

## Kicks instead of millisecond polls

Status: confirmed 2026-11-11, same Tezro (`lightning,idle-pause` and
`lightning,idle-pause,jitv2`, IRIX 6.5.30, loaded M4 Max host), paired
concurrent runs against the build before kicks and learned signatures, in
both slot orders, on our tree before the rebase.

**Symptom.** An idle 4-CPU Tezro took about 2,700–3,000 parks a second:
every all-idle park ended after its 1 ms slice, and under jitv2 each wake
ran its batches interpreted. Idle host CPU 130–137% (jitv2), 98–119%
(interpreter).

**Rules** (`src/cpu/idle_park.rs`).

- While every CPU is idle, work reaches a CPU only after some CPU has run an
  interrupt handler or work of its own. That CPU *kicks* the others
  (`kick_peers`: a `KICKED` bit, then an unpark) when it parks again after a
  device or inter-processor interrupt (`MipsCore::irqs_taken` moved) or
  after `KICK_AFTER_BATCHES` (4) batches, and when its idle flag clears. A
  kicked CPU leaves its park and polls once. A CPU woken to poll parks again
  within two batches and kicks no one, so kicks do not chain. The all-idle
  slice (10 ms, `IRIS_IDLE_SLICE_US`) only bounds a hand-off nothing
  announced.
- The CP0 timer's tick alone (IP7) does not count in `irqs_taken`. On IP35
  `counter_intr` calls `clock()`: Count/Compare is every CPU's scheduling
  clock, and each idle CPU takes it about 77 times a second, so each polls on
  its own tick anyway. Kicking on it tripled the parks (1,060/s against
  445/s); what a tick wakes is due only at tick granularity.
- `wake()` unparks every parked CPU on any device interrupt. A
  multiprocessor's park sleeps on to its deadline when the wake was not for
  it (no line of its own, no kick).

**Counting trap.** The old park loop broke out right after `park_timeout`
returned, so a CPU woken by its own clock tick counted as `wake_slice`, not
`wake_irq`: the old builds show `wake_irq` near 0 although the per-CPU clocks
tick. `idle status` breaks `wake_irq` down by line and CPU.

**Measured** on our tree (300 s idle windows, rates per second; workload
geomean of grep, sort, awk, fork, cc1, cc4 against the concurrent baseline;
"new" is kicks plus the learned signatures of the jitv2 section):

| run | idle host CPU | parks | wake_slice | wake_kick | wake_irq | probes | workload |
|---|---:|---:|---:|---:|---:|---:|---:|
| jitv2, before, slot A / B | 137.3% / 132.3% | 2,665 | 2,048–2,123 | — | 0 | 3,142–3,196 | — |
| jitv2, new, slot B / A | 73.0% / 53.3% | 441–446 | 17–20 | 73–83 | 334–353 | 99–137 | +4.7% / −0.9% |
| interpreter, before, slot A / B | 118.9% / 97.7% | 2,724–2,975 | 2,428–2,721 | — | 0 | 0 | — |
| interpreter, new, slot B / A | 88.9% / 67.2% | 682–738 | 22–30 | 120–126 | 307–329 | 0 | −2.1% / +0.4% |

The workload differences have no consistent sign across slot orders and lie
within the ±5% run-to-run spread of this host. The intermediate build that
also kicked on the timer tick took 1,040–1,070 parks a second (jitv2, 59%
and 80% idle).
