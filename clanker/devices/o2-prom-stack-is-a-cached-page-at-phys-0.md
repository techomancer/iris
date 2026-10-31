# O2 PROM stack is one cached page at physical 0, and RAM behind it is a different thing

Status: landed 2026-10-30. Both O2 CPUs run the shadow cache from reset; the handover
([o2-cache-handover-full-to-shadow.md](o2-cache-handover-full-to-shadow.md)) is gone. Code:
`O2Memory::stack_tag_store` in `src/dev/crime_mem.rs`, the tag-store hook in
`src/cpu/mips_cache_shadow.rs`. Sections below are the measurements that chose the design; the
prototype results used earlier binaries.

## Where the stack is

KSEG0 `0x8000_0000..0x8000_1000`, physical `0..0x1000`, `sp` starts at
`0x8000_1000`. This is the low alias of bank 0, so it overlaps regular memory.
It exists only as cache lines:

1. `0xbfc006dc` sets `s0 = 0x8000_0000`, `sp = s0 + 0x1000`, calls `0xbfc008f0`.
2. That sweeps the page by index with PD Index Store Tag, no fill, tag page 0,
   state dirty: TagLo `0xc8` on R10000 (also L2 tags with CACHE `0xb`), `0xc0` on R5000.
3. `0xbfc00a58` fills the page with the complement of each address, cached.

## Why memory cannot back it

During the stack's life the PROM uses the same physical bytes as RAM, uncached:

| PC | What | Touches |
|---|---|---|
| `0xbfc05a2c`, `0xbfc05ad4` | bank sizing through an uncached TLB mapping at VA 0 | physical `0..0x28` of this page, nothing higher |
| `0xbfc061f0(0x4000_0000, 0x8000)` | MTE clear | not this page, see below |
| `0xbfc05110(0, 0x1000)` | walking-bit and pattern test at `0xa000_0000` | physical `0..0x1000`, all of it |
| `0xbfc060c0` loop | copies `0x8000_xxxx` to `0xa000_xxxx` up to `0x1000` | the handoff to RAM |
| `0xbfc06b28` | whole-cache tag invalidate, no writeback | ends the cached page |

`0xbfc044ac` then ORs `sp` with `0xa000_0000` and POST continues on the uncached copy.

Measured at `0xbfc060c0` on the full model: KSEG0 `0xff4` reads `0xbfc04400` (the
stack), KSEG1 reads `0xa0000ff0` (the memory test's pattern). A cache that writes
through to RAM loses the frame to the memory test, and only to that: with the shadow
model and no page, the frames survive sizing and the MTE clear, then the test replaces
them and POST takes an XTLB miss on a test pattern (BadVAddr `0x5ffff0efa0000f10`).

## What is and is not in the page

- Stack only. At `0xbfc060c0` KSEG0 differs from the fill at `0xed0..0xff8` (38
  dwords, 304 bytes) and at offset 0, which the fill loop stops short of. Same on
  R10000 and R5000. No PROM variable lives in the page: all 512 dwords of RAM behind
  it hold the memory test's pattern at that point.
- The state POST keeps is just above it: `sp` is saved at `0xa000_1008`, and the
  relocated POST code runs from `0xa000_4000`, copied after the page is gone.
- Sizing does not reach the stack. Canary over the page before `0xbfc05ad4`, diff at
  `0xbfc04484`: 8 words change, all in `0..0x27`, on 32, 64, 128, 128+64+32 and 1 GB.
- The MTE clear skips the page because of the firmware. It builds each linear TLB
  pair with an OR of a sign-extended low half, so the `sd` writes
  `0xffffffff_80040001`: every even-page entry is all ones. The emulator drops those
  (page `0xfffff` is outside memory), so only pages 1, 3, 5 and 7 clear. What real
  hardware does with that entry is unknown.

## Bank controls

Other banks do move through the low 256 MB during sizing and stay there after it
(128+64+32 ends as bank 0, 0, 0, 0, 2, 3, 4), but bank 0 never leaves physical 0.
Traced on 32, 64, 128, 256, 64+32, 128+64+32, 128+128, 4x32 and 1 GB: `ctrl[0]` is
only ever `0x100` then, for a 32 MB bank, `0`. Over an IRIX boot and reboot at 1 GB
every write was in POST and slot 0 stayed bank 0. So forcing page 0 to bank 0 changes
nothing; it is what the decode already gives, and the failure above happens anyway.

## A scratch page works only on the cached path

Mapping scratch memory into the bus decode table would break POST: the memory test
would overwrite the stack through KSEG1. The two paths are already separate, so
swap **only ppmem's low-alias window**:

- cached (`mips_cache_shadow`, jitv2 inline): `window + phys` sees scratch;
- uncached: decode table to `BankView` sees the bank;
- MTE: `RamWindow` at `RAM_BASE + off` sees the bank.

PD Index Store Tag with a valid state and tag page 0 maps it in. With state
invalid it maps it out (rebuild the window). Neither is on a hot path.

## What landed: guard the uncached path

Keep ppmem, the cached path and the JIT on the bank permanently. While the stack is
live, point the decode table's 4 KB entry for bank 0's offset 0 (`SLOT_SHIFT` is 12) at
a separate `RamBank` that wraps at 4 KB. There is no guard device and no range check. The stack lives in the bank; the memory test, the copy loop
and everything else uncached use the side page.

- On: PD Index Store Tag with a valid state and tag page 0. Copy bank to side.
- Off: the first PD Index Store Tag with an invalid state. Copy side to bank, restore
  the normal entry. The copy is needed: `sp` moves to KSEG1 before the sweep, whose
  first frame goes to the side page.
- Not on reset: IRIX reboots by jumping into the PROM, so no `power_on` runs.
- Guard every place offset 0 answers, in both the window and the low alias. A 32 MB
  bank under a 128 MB setting repeats and sizing relies on it; guarding only the first
  hung every 32 MB fitting in sizing.
- At cold reset, map bank 0 again over low alias slot 1 for the CPU only, to fill the
  bitmap hole below. No scratch bank is needed.
- No ppmem mapping changes for the page, so host page size does not matter.
- If the MTE ever cleared page 0 it would hit the stack, as it uses the window. The
  clear skips it on both PROMs today (the TLB pair code is the same).

## Traps

- `recompute_bitmap` sets a bit only when the whole 64 MB region is covered. At
  cold reset every bank control is 0, so only 32 MB is mapped, bit 0 is clear and
  the cached path takes the bus. Stack frames land in the bank, then sizing sets
  the bit and reads come from empty scratch: `ra` = 0, `jr` to 0, TLBL, hang at
  `0xbfc003a0`. The prototype maps scratch over `32..64 MB` while live.
- **Test from a cold start.** After the monitor's `reset` the stale copy of the
  stack in RAM hides that bug. `Crime::power_on` zeroes CRIME's registers but not
  `O2Memory`'s `ctrl`, so RAM and the decode both survive it.
- `remap` runs `clear_mappings` on each bank control write, and sizing writes many.
  The page must be reinstalled at the end of every `remap`.
- `map_range` gets no generation mapping for 4 KB (8 bytes of counters, below host
  granularity). Harmless here since the page is never code, but it trips a
  `debug_assert` in debug builds; the low-alias path shows how to round up.

## Results of the landed code

Linux, 2990WX, one NUMA node, jitv2, 1 GB, IRIX 6.5.22 on an overlay, PROM 4.18. Cold boot,
guest load below 0.15, best of 3 in each of 2 boots per build. Base is 24c28aed.

| CPU | Build | `iris_bench` 20, MB/s | Dhrystone 10M, /s | Whetstone 100k, host s |
|---|---|---|---|---|
| R5000 | base | 4.60, 4.56 | 250000, 250000 | 19.26, 18.86 |
| R5000 | new | 4.74, 4.76 | 277778, 277778 | 18.97, 20.03 |
| R10000 | base | 4.42, 4.61 | 250000, 250000 | 18.65, 18.04 |
| R10000 | new | 4.50, 4.59 | 270270, 277778 | 19.67, 19.60 |

Whetstone's own figure counts whole seconds, so use host time. Run first in the boot, away
from the flush below, R10000 is level: best 19.42, 19.03 base, 19.21, 19.00 new.

Interpreter, 1 GB, best of 2 in 2 boots: `iris_bench` 3 went from 0.39 to 0.40-0.41 MB/s on R10000,
0.41 to 0.42 on R5000.

R10000 new reaches the JIT's 4096 pages during these runs and takes one mega-flush there. Base
took its flush at the handover and stays just under the limit. It is a one-off recompile.

Boots on the final code: login and `/etc/reboot` to login again on PROM 4.18 and 4.3, R10000 and
R5000; login at 32, 64, 128+64+32 and 4x32 MB on R10000; interpreter login on R10000.

The removed branch: the interpreter's load and store each lost 2 conditional branches
(`read_data_impl` 99 to 97, `write_data_impl` 70 to 68). The new instantiations fold into
the same code as the other machines' shadow cache, so look for them under `ShadowCache<16384, ...`
in a disassembly. Compiled code reaches them only through its helpers.

Removing the handover's flush exposed a startup-order bug:
[../jitv2/compile-pool-starts-after-the-windows.md](../jitv2/compile-pool-starts-after-the-windows.md).

## Not done

- Snapshots carry the page and its live flag, but no round trip was run in a live guest (unit
  test only). The manifest's cache name changed, so older O2 snapshots are refused unless
  `IRIS_SNAPSHOT_SKIP_CHECK=1`.
- Not run: other guests (NetBSD, OpenBSD, Linux), Diagnostics and Install paths, Mac and Windows
  hosts. Host page size does not matter here because ppmem does not change for the page.
- The "valid tag, page 0" trigger fits these PROMs. A guest storing a valid tag for page 0
  would start the page; the next invalid tag ends it.
