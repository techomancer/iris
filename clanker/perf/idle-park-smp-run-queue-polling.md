# SMP idle parking: an idle IRIX CPU polls its run queue and gets no interrupt for new work

Status: rules confirmed 2026-11-08 on a 4-CPU Tezro (IRIX 6.5.30, 1 GHz
R16000, `lightning,idle-pause`, M4 Max host). The numbers below were measured
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
  another CPU is busy, the 1 ms slice (`wake_slice`) while every CPU is idle.
  Only a uniprocessor sleeps until a line. "Busy" is the detector's per-CPU
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

**Open.** jitv2 does not park with these rules: compiled code keeps a loop's
back-edges inside its region, so the run loop samples only region exits (one
or two PCs, under `CLUSTER_MIN_DISTINCT`). The same Tezro under
`lightning,jitv2` ran the guest 3–7× faster but idled at 402% host CPU.
