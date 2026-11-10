# An IRIX boot is bound by how much jitv2 compiles, not by how fast each compile is

Status: measured 2026-11-10 on `a044b7b2` plus the two switches below. Nothing
fixed. Follow-up to [gpr-cache.md](gpr-cache.md), whose suite gain did not
show up in the guest.

## The compile workers never go idle

Indy, R4400, IRIX 6.5.22, `lightning,jitv2,rex-jit`, `[jitv2] threads = 4`,
cold boot from a fresh overlay, then a cc build, `find`/`tar`/`compress` and a
compute program over telnet: about 2.5 minutes of host time.

- The exit line (`jitv2 compiles: ...`) reads 5,000-5,450 compiles, mean
  50-55 ms, 270-290 s of compile time: four workers busy for essentially the
  whole run.
- `perf record` over the first 75 s: the four `jitv2-compile-*` threads take
  62% of all samples, `MIPS-CPU` 26% (about half of that in compiled code),
  `hptimer` 9%.
- `j2 corpus` about a minute after the login prompt held 3,505 pages, so
  the 5,000-odd compiles are not one per page; the table below shows how
  far from it they can get.

Where a compile goes (same profile, compile threads only):

| Phase | Share |
|---|---:|
| regalloc2 (backtracking allocator) | 37% |
| Cranelift other (CFG, dominators, DFG, layout, constant phis) | 18% |
| egraph optimizer (`opt_level=speed`) | 20% |
| lowering / instruction selection | 14% |
| vcode build and emit | 3% |
| frontend (IR build, SSA construction) | 1.4% |
| iris's own analyzer and emitters | 0.5% |

Our code is a rounding error. The time is Cranelift working on page-sized
functions.

## Making each compile cheaper does not help

`IRIS_JIT_OPT=none|speed` seeds the optimisation level for a whole run and
`IRIS_JIT_REGALLOC=single_pass` selects Cranelift's one-pass allocator. Same
workload, two cold boots each:

| Build of the code | Mean compile | Compiles in the run | Compile time | Program, warm |
|---|---:|---:|---:|---:|
| `speed`, backtracking (default) | 50-55 ms | 5,041-5,421 | 272-279 s | 1.8 s |
| `none`, backtracking | 38-44 ms | 5,984-6,184 | 226-272 s | 1.85 s |
| `speed`, single pass | 35-39 ms | 7,308-8,074 | 281-283 s | 3.3 s |
| `none`, single pass | 21-24 ms | 11,622-12,624 | 267-282 s | 3.7 s |

- Compiling 2.4 times faster got 2.4 times as many compiles and the same
  total: the workers are still saturated. **Demand is not a fixed list of
  pages.** A page is compiled again each time an entry is added, each
  compile covers the whole page, and the faster the workers are the more of
  those recompiles they get through.
- The single-pass allocator's code is about half as fast (3.3 s against
  1.8 s). It is not usable as the only tier.
- `opt_level=none` with the backtracking allocator ran this program as fast
  as `speed`. On the benchmark suite's loops it will not; it was not run
  there. Boot time to the login prompt is bimodal on this host (25 s or
  34 s, unrelated to the setting), so it cannot rank these.

## What this points at

- **Recompiles, not compile speed.** Compile a page's later entries without
  redoing the page (revenir's 0008 does this and merges once the page
  settles), or wait for a page to settle before paying for a whole-page
  compile. `j2 status` already counts "had to recompile"; the queue-full
  lead in memory (`project_jitv2_queue_starvation_lead`) is the same thing
  seen from the queue.
- A cheap first tier (`none` + single pass, 21 ms) under a second, optimised
  compile of pages that stay hot would get code in front of the guest
  sooner, but only pays once recompiles are under control; today it would
  just feed the churn.
- The first run of a program after boot is 3 to 4 times slower than its
  third (6 s, 2.1 s, 1.8 s for the test program). That gap is the size of
  the prize.

## Reproducing

`target/gprfwd/` held the scripts for this session (not in the tree): boot
with a config whose disk has `overlay = true`, run a fixed workload over
`tools/irix-sh.py`, kill iris, read the exit line. For the profile,
`perf record -F 299 -g -p <iris pid> -- sleep 75` from just after launch and
`perf report --no-children -g none --sort comm` (then `--comm
jitv2-compile-0,... --sort symbol`).
