# Hot-path inlining must be stated, not left to the inliner

`CpuCache::fetch` runs once per interpreted instruction. It used to inline into
`step_int` because the inliner happened to judge it small enough, with its miss
path `fill_l1i_line` inlined into it.

1431f560 added a 4-word `ic_raw` snapshot to `fill_l1i_line`, a cold path. That
grew `fetch` past the inliner's budget, and `fetch` became an out-of-line call
on every instruction. Bare-metal iris-bench (lightning, R4400, full cache) fell
to ×0.84 of `main` (`sys/uncached` ×0.51, `int/branch` ×0.75). IRIX dhrystone
went 71428 → 58823.

Found by bisecting the bare-metal suite, then confirmed with
`nm -C --size-sort … | grep 'CpuCache<16384.*fetch'`: the symbol exists only
in the slow build.

Rules:
- A per-instruction hit path gets `#[inline(always)]` (`CpuCache::fetch`,
  `ShadowCache::fetch`, `DecodeCache::fetch`).
- Its miss path gets `#[inline(never)] #[cold]` (`fill_l1i_line`,
  `DecodeCache::fetch_slow`), so changes there cannot move the hot path.
- After touching anything reachable from `fetch`, `read`, `write` or
  `step_int`, check the symbol table: a new out-of-line symbol for a hot
  function is the regression.

Fixed in 28d8c01a (back to ×1.02 of `main`).

Broken again by aded5989, this time without losing inlining: it put
`eprintln!` bus-error reports with a formatted `String` into the hot fetch,
load and store functions. They never ran on a good access, but the formatting
code compiled into those functions, and the lightning interpreter lost 8% of
its suite speed. Fixed by moving the reports into one `#[cold]
#[inline(never)]` method.

`tools/hot-path-check.sh BINARY` reads a lightning release binary built with
fat LTO, and runs in CI on the Linux one. It fails when any `MipsExecutor` code
outside the cold methods named in the script calls a formatting function, or
when a hit-path `fetch` has its own symbol. It cannot see a layout or inlining
regression with no formatting in it, or allocation without formatting. Without
fat LTO, calls into std go through the GOT and it reports the binary
unreadable. ELF and GNU binutils only.

Open: `handle_exception` prints on its bus-error, address-error and
coprocessor-unusable branches, and it runs on every guest exception. It is on
the script's cold list; whether that costs anything on exception-heavy guests
(IRIX user space: TLB refills, syscalls, interrupts) has not been measured.
