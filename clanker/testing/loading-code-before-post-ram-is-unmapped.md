# Loading code before POST: RAM is unmapped, and the write is silent

`MEMCFG0/1` are **zero at reset** (`mc.rs`: "all banks invalid at reset (VLD=0
per spec)"), so on a cold machine *no RAM is mapped anywhere*. The PROM
configures the banks during POST. Anything that writes to RAM before that —
`--load-elf`, a scripted `mw`, a test harness poking memory — is writing into
the void.

Two things make that failure completely silent, and both have to be handled:

1. **`UnmappedRam` accepts every write and reads back 0** (`mem.rs`), and
   `Physical` routes unmapped lomem/himem slots to it. No bus error, no status
   code — `debug_write` returns `EXEC_COMPLETE`. The only symptom is a
   `MC: CPU Error at <phys>` line on stderr, which nothing checks.
2. **A cached probe cannot detect it.** KSEG0 is cacheable, so a
   write-then-read-back through `debug_write`/`debug_read` is absorbed by L1D
   and returns the pattern you just wrote even with nothing behind it. Probe on
   the bus (`exec.sysad.read32`/`write32` at the *translated* physical address)
   and restore the original word.

`load_range()` in `mips_exec.rs` probes both ends of every range that way before
committing an image, and `Machine::load_elf` calls `MemoryController::post_map_banks()`
first, which programs MEMCFG0 exactly as POST does (bank 0 at `LOMEM_BASE`,
bank 1 at `LOMEM_BASE + BANK_SIZE`; banks 2/3 are synthesised at HIMEM by the
MEMCFG write itself). That is what makes a cold-start `--load-elf` land in real
RAM with no PROM involved.

Related: **Indy RAM starts at physical 0x08000000**, not 0. A bare-metal binary
therefore links at **0x88000000+** (KSEG0), not 0x80000000 — physical 0 is a
512 KB alias window (`ALIAS_BASE`..`ALIAS_END`) that mirrors the *bottom* of
lomem and nothing else, so an image at 0x80100000 lands nowhere.
And KSEG1 for a device at physical `0x1FBD9830` is `0xBFBD9830` — KSEG1 masks
the top three bits, so `0xA1BD9830` silently becomes physical `0x01BD9830`.
