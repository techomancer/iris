# CP0 does not have to end a region

Landed 2026-09-22, ported from an earlier experiment without its Cargo
feature or `IRIS_JIT_ATOMIZED` env policy — those existed to A/B the policy while it was being established, and
it has been.

`OP_COP0` classifies as `Excluded`, and an excluded word used to be a hard
region boundary: every `mfc0`/`mtc0`/`eret` cut a region in two. A
whole-kernel census found **677 COP0 sites** in `unix.B`, concentrated in
exactly the exception, timer and interrupt paths that run most often.

The safe subset now stays **in** the region as an interpreter-fallback head
calling the real `exec_cop0`. There is deliberately no native CP0 emitter and
there must not be one — a second implementation would drift. Nothing changes
what a CP0 instruction does; only whether the region has to end on it.

## What it is worth, and why three measurements said "nothing" first

**~13-15% wall clock on a syscall-bound workload**, with the guest's own CPU
counter dropping ~40%.

Getting that number took four attempts, and the three failures are the
instructive part:

| measurement | result | why it was blind |
|---|---|---|
| Dhrystone / Whetstone | nothing | userland never executes a CP0 instruction at all |
| boot to login, 5 rounds, alternating arms | 52 s vs 52 s | boot is latency-bound — fixed `rc` delays and disk I/O, not CPU |
| `j2 status` mean region length | 348.34 vs 348.46 instrs | aggregate is dominated by userland pages; the kernel's shorter regions vanish in it |
| **syscall loop (`times()` x 2M)** | **15 s -> 13 s** | every iteration traps, runs the mfc0-dense exception path, and ERETs |

The decisive run is a **single boot with the runtime toggle**, which removes
host drift entirely — this host had a background `mediaanalysisd` taking 2.5
cores, and two separate boots could never have excluded it:

```
j2 cop0 on   ->  13, 12, 13 s      guest acc 58-63M
j2 cop0 off  ->  16, 14, 15 s      guest acc 95-107M
j2 cop0 on   ->  13, 14, 12 s      guest acc 58-63M
```

Same binary, same guest, same warmed page cache; only the toggle moves. The
`acc` column is the benchmark's own accumulated `tms_utime` and is a second,
independent instrument that agrees and reverses. (Treat its *magnitude* as
directional only — `tms_utime` is the counter that undercounts by ~3x, see
`../perf/guest-cpu-time-accounting-undercounts.md`. Its *ratio* within one
guest still tracks work done.)

**Reversibility is what makes this evidence rather than a warm-up artifact.**
A one-way improvement across a flush could just be the cache warming; one that
goes back when you toggle off, and returns when you toggle on, cannot be.

## What is admitted, and what is not

`jitv2::cop0::stays_in_region`:

- **MFC0/DMFC0, every `rd`.** `read_cp0` mutates only CP0-internal
  bookkeeping (`update_random`, the Count memoization). The Status.CU0
  privilege gate is `exec_cop0`'s and still runs — this is a fallback head,
  not a native emitter.
- **MTC0/DMTC0 on `MTC0_SAFE_REGS`**, plus **Status (12)** under the codegen
  gate below.
- **ERET.**

Deliberately excluded, each for its own reason:

- **Cause (13)** — the one that looks safest and is not. `emit_pending_interrupt_preamble`
  samples `core.hot.interrupts != 0`; the interpreter's `step_preamble!` tests
  `(pending | core.cp0_cause) != 0`. So the interpreter can deliver on Cause
  IP bits that never reach `hot.interrupts` — the two **software** interrupts
  IP0/IP1, writable only by `MTC0 Cause`. Compiling through it would let a
  region run arbitrarily far past an interrupt the interpreter delivers at once.
- **TLB ops** (TLBR/TLBWI/TLBWR/TLBP) — complex, rare, and a subtle bug there
  corrupts memory in an unrelated address space minutes later.
- **EntryHi (10) / Config (16)** — safe but worthless; no hot-path site writes them.
- **CFC0/CTC0/BC0** — only outcome is Reserved Instruction; an instruction
  that never has a successor is not worth compiling through.

## The Status gate, which is the whole safety argument

A region bakes in exactly two things: **FR mode** and the
**pending-interrupt sample**. Cause covers the second. Status carries
`STATUS_FR`, and `emit_fr_mode_guard` checks it **once** per region entry on
behalf of every FPR access in the region — valid only while nothing in the
region can move FR.

So `compile_region_uncommitted` **declines any region that combines `has_fpu`
with a Status-writing fallback head** (`cop0::writes_cp0_status`). A region
with no CP1 instruction has no FPR-access emitter and no guard at all, so a
Status write there has nothing to invalidate — and that is exactly the shape
of the kernel exception and timer paths this exists for.

That check is **not** conditional on anything, deliberately: the blanket
`j2 fallback on` toggle admits `MTC0 Status` as a fallback head too, and has
done since interpreter fallback landed, so without it the same stale-FR
hazard already existed on that path. One comparison closes it for both.

## The toggle

`j2 cop0 off` (then `j2 flush`) puts every CP0 word back to ending its region,
live, so a divergence found on a boot can be bisected without relaunching. It
works through the ordinary per-`InstrKind` ENABLED table — `Mfc0`/`Dmfc0`/
`Mtc0`/`Dmtc0`/`Eret` are listed in `has_jitv2_support` purely so they default
enabled there, *not* because they have emitters. They do not, and must not.

Related: [`instructions-jitv2-still-interprets.md`](instructions-jitv2-still-interprets.md),
The guest-CPU-time accounting note cited by the original investigation was
not committed; use host wall-clock timing for these comparisons.
