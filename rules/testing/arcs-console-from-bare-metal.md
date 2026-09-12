# Calling ARCS firmware from the bare-metal harness

The suite drives SCC channel B directly, which is invisible on an Indy whose
PROM console is graphics — see
[running-cpu-tests-on-real-hardware.md](running-cpu-tests-on-real-hardware.md).
`harness/arcs.c` fixes that by calling the PROM's own console routine, so output
goes wherever the PROM's console goes.

## The addresses, and how to get them without a datasheet

Boot any PROM-booted run, drop to the IRIS monitor on port 8888, and read
memory. Everything below was obtained that way in about ten commands; none of it
needs a specification or a web search.

```
mm 0x1000        -> 53435241   SPB signature 'ARCS'
mm 0x1004        -> 00000048   SPB length
mm 0x1010        -> 00000000   DebugBlock
mm 0x101c        -> 0000008c   FirmwareVectorLength = 140
mm 0x1020        -> a0001800   FirmwareVector
mm 0x186c        -> 9fc1030c   entry 27 = Write()
```

Note `mm` takes a **physical** address. `mm 0xa0001000` reads as `ffffffff` and
looks like an empty SPB.

Two independent cross-checks that the layout really is the ARC one, rather than
a coincidence:

- `FirmwareVectorLength` is `0x8c` = 140 = **exactly** 35 four-byte entries,
  which is the ARC firmware vector count. `Write` is entry 27.
- `DebugBlock` (SPB + 0x10) is 0, which is what the IRIX source quoted in
  `src/debug.md` tests for — `if (SPB->DebugBlock && ...)` — when symmon is not
  loaded.

Every vector entry points into PROM space (`0x9fc.....`), so `arcs_probe()`
range-checks that before calling: a bad layout would otherwise be a jump into
nothing, on a machine with no console to report it.

## It survives the whole run

`exc_install()` writes three trampolines and flushes `0x200` bytes from
`0x80000000`. The SPB is at `0x1000` and the vector table at `0x1800`, so the
harness never touches either. The vectors live in KSEG0 and everything is read
through KSEG1, so nothing depends on TLB state — the firmware stays callable
after the TLB tests have rewritten all 48 entries and the cache tests have run.

## Two things that bit, both only visible once it worked

**The firmware clobbers FP state.** With ARCS output enabled the suite reported
`2096 passed / 66 failed` instead of `2101 / 61`. The same 15 tests failed —
five extra *checks* inside them, all under `fpu/`. Those are the tests that
print diagnostics with `observe_s`/`observe_d` while the test is still in
flight, so the firmware call landed in the middle of one. Saving and restoring
Status and FCSR around the call restores the baseline exactly. Guard the FCSR
access on `ST_CU1`, or `cfc1` faults when a test has coprocessor 1 disabled.

**Output doubles unless you silence the SCC.** ARCS writes to the PROM console,
which on a `console=d` run *is* SCC channel B — so every line appeared twice,
once from `scc_putc` and once from the firmware. `con_disable_scc()` sets the
existing `scc_dead` latch when `arcs_probe()` succeeds. That also removes the
unprogrammed-SCC spin cost on a graphics console for free, since nothing writes
the port at all any more.

## Calling convention

A plain C function pointer works. The PROM is o32 and the harness is n32, but
the first four integer arguments are `$a0`-`$a3` in both, and `-mno-abicalls
-fno-pic -G0` means there is no `$gp` for the callee to disturb. Sign-extend
every pointer argument through `(long)(s32)` — a 32-bit address zero-extended
into a 64-bit register is xkuseg, not KSEG0.

Buffer per line. One firmware call per character would be ruinous over a
19,000-byte run, and `Write(FileId, Buffer, Length, &Count)` wants a counted
buffer anyway.

## The disk write goes through ARCS too

`harness/scsilog.c`'s own WD33C93A driver works under IRIS and **fails on real
hardware** — a real Indy returns `rc=-3`, meaning the command completed with a
status other than `SELECT_TRANSFER_SUCCESS`. Selection and command phase worked;
the transfer did not.

The reason is almost certainly that the PROM negotiated **synchronous** SCSI
with the drive when it read the boot file, and **IRIS models no such thing** —
`SYNC_TRANSFER` is a bare register constant that nothing reads, and the SDTR
bytes are received and discarded. A driver proven under the emulator has only
ever run asynchronously.

`arcs_disk_write()` sidesteps all of it by using the firmware's own driver:
`Open`(23) / `Seek`(28) / `Write`(27) / `Close`(24) on `dksc(0,2,10)`.

- **Partition 10, not 8.** Slot 8 is `PT_VOLHDR` and only 1265 blocks, which
  does not reach `SCSILOG_LBA` (8192). Slot 10 is `PT_VOLUME`, the whole disk
  from block 0, so the byte offset is the absolute one and `extract-log.py`
  needs no change. No new partition is required — `mkvh` already emits both.
- **Seek takes a 64-bit offset by pointer**, not as an immediate argument.

### The buffer must be cache-coherent, and uncached stores are not enough

The first working write came back with a **128-byte hole** in the middle of an
otherwise perfect log. `scsilog` had been writing the buffer through its KSEG1
alias, which is right for its own DMA but wrong here: the firmware reads the
buffer through a *cached* mapping, so lines still resident from `.bss` being
zeroed shadowed the uncached stores, and those bytes read back as zero.

Write the buffer cached and call `dcache_wb_invalidate_range()` before handing
it to anything — that satisfies the firmware's cached reads and the DMA engine
both. Invalidate-without-writeback is the wrong tool here: it would discard the
data rather than publish it.
