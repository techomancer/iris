# Writing benchmark kernels for `bench/` — things that bit

The suite scores itself on accuracy: every kernel's result checksum is compared
against a golden value computed by building the same C natively. That comparison
turned out to be a much better bug detector than intended, and everything below
is something it caught. Read this before adding a kernel.

## A checksum must depend on nothing but the arithmetic

**Never fold a multi-byte array in as raw bytes.** `cksum_bytes(h, (const
unsigned char *)coeffs, n * sizeof(short))` compares *byte order*, not results.
The golden values come from a little-endian host; the guest is big-endian; the
mismatch reads as an emulator fault. `img/dct8x8` and `vid/motion_est` both did
this. Use `cksum_u64` / `cksum_f64`, which are defined on values.

**Initialise everything you checksum.** `mem/copy`'s verify wrote 5 KB of an
8 KB region and checksummed all of it. The tail was whatever the allocator last
held — different on the host, different between two host builds, different
between a first and second run.

**Do not read past the end of a buffer.** `mem/unaligned` passed the buffer size
as the readable length while starting one byte in, so the last load reached one
byte beyond. On the guest that byte was a previous kernel's leftovers; on the
host it was fresh malloc. There is no value either side could agree on.

**Anything endian-sensitive by nature needs a host-side equivalent, not a
host-side copy.** `mem/unaligned` genuinely wants an unaligned big-endian load;
the host assembles the word from bytes under `#if BENCH_HOST` so the *value*
matches while the guest still executes the instruction being measured.

## `work_alloc` is a bump allocator with no free

Call it **once per `run()`, outside the iteration loop**. The runner resets the
allocator between benchmarks, so a kernel that allocates per iteration only
exhausts the 24 MB work area at whatever iteration count the autoscaler picked —
which makes a plain bug look like a data-dependent failure. `codec/rle` called
`src_build()` inside its loop and died on the sixth pass.

The flip side is useful: because the allocator resets to the same base every
time, a kernel can cache expensive setup behind `if (built && ptr == cached)`
and it will hit every time.

## A faulting kernel does not crash — it reports a number

The shared exception dispatcher records the fault and steps over the
instruction, so a kernel that faults keeps running and produces a throughput
figure for taking exceptions. `mem/unaligned` scored a plausible 871 k
accesses/s while taking an address error on three loads in four.

`*(const u32 *)(const void *)p` on an unaligned `p` is the trap: the cast
promises an alignment the pointer does not have, and GCC emits a plain `lw`. If
you want an unaligned load on MIPS, write `lwl`/`lwr` yourself.

The harness now counts exceptions across every timed run and prints them on the
line (`exc:N`); mark a kernel that faults on purpose with `BENCH_EXC` so it is
not flagged. Add that flag rather than ignoring the count.

## The guest must probe the test device, not trust it

An emulator built before `TESTDEV_HOST_NS`/`ICOUNT`/`CAPS` existed decodes only
16 bytes and repeats, so `0x20` (CAPS) aliases onto `SIGNATURE` — and `'IRIS'`
has bit 0 set, so a naive `caps & CAP_TIMEBASE` says yes. Every timing then
comes back from a frozen clock, and the suite reported the assumed 100 MHz Count
rate as if it had measured it.

`probe_timebase()` rejects a CAPS word equal to the signature *and* requires the
clock to advance before trusting it. Also: never **write** an unprobed offset in
that window — on an old device `0x1C` aliases onto `EXIT`.

## CP0 Count is not a stopwatch

Under IRIS, Count is virtual: materialised from a wall-clock anchor at a
`count_hz` that `infer_count_hz` learns from the guest's own Compare writes. A
bare-metal binary never writes a plausible one — `start.S` sets Compare to
`0xFFFFFFFF`, whose delta is outside the plausible-tick bounds and is ignored —
so it sits at `DEFAULT_COUNT_HZ`, 33 MHz, not the ~100 MHz a real 200 MHz R4400
would show. Time with the test device's host clock and *measure* the Count rate;
the ratio between them is a report on the timer model, not a nuisance.

## Assembly that shares a word with C

`bench_pfn_delta` was `.space 8` in the assembler and `u32` in C, read back with
`lw`. On a big-endian machine `lw` from the base of a 64-bit object returns the
**high** half — zero — so the TLB refill handler built an identity map onto
unmapped physical memory and `sys/tlb_miss` measured 780 000 bus errors instead.
It is `.space 4` now. Keep the width the same on both sides, and make a kernel
that depends on a mapping *prove* the mapping before timing it.

## The feature banner has to name the CPU

`print_build_features()` in `src/main.rs` did not list `r5k`, `jitv2`, or
`mips4`, so an R5000 build announced itself as `build features: tlbvmap` and a
benchmark result recorded from it was indistinguishable from an R4400 one. It
lists them now. `iris-bench matrix` still cross-checks the guest's own
`#machine cpu=` line, read from PRId — same guard, and for the same reason, as
`cpu-tests/run/matrix.sh`.
