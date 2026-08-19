# Where the suite lives in memory

## Physical RAM does not start at 0

On IP22/IP24, physical RAM begins at **`LOMEM_BASE = 0x08000000`**
(`src/physical.rs:136`). The only RAM below that is a **512 KB alias** at
`0x00000000..0x0007ffff`, which mirrors `0x08000000..0x0807ffff`
(`src/physical.rs:165-177`). Everything between `0x00080000` and `0x08000000`
is unmapped.

Unmapped physical addresses are backed by `UnmappedRam`, which **accepts every
write and reads back zero** — so writing an image into that hole is completely
silent. `--load-elf` probes the bus at both ends of each segment before
committing precisely because of this; without the probe, the first symptom is a
CPU that fetches zeros.

That makes the naive choice wrong:

| KSEG0 address | Physical | What is there |
|---|---|---|
| `0x80000000` | `0x00000000` | the 512 KB alias → real RAM at `0x08000000` |
| `0x80200000` | `0x00200000` | **nothing** — writes vanish |
| `0x88000000` | `0x08000000` | RAM, first byte — same bytes as `0x80000000` |
| `0x88200000` | `0x08200000` | RAM, 2 MB in ← **where the suite links** |

## The layout the suite uses

```
KSEG0 0x80000000  exception vectors      (physical 0x00000000, via the alias,
0x80000000  TLB refill                    so the same RAM as 0x88000000)
0x80000080  XTLB refill
0x80000180  general

KSEG0 0x88200000  _ftext   suite image starts here
                  .init  .text  .rodata  .data
                  .bss
                  _stack_top     64 KB stack
                  _scratch_start 256 KB scratch for mem/tlb/cache tests
                  _end
```

Note the vectors and the image base alias to the same physical RAM through
different windows: `0x80000000` is physical `0x08000000` and so is `0x88000000`.
Linking at `0x88200000` keeps the image 2 MB clear of the 512 KB alias window,
so the vectors and the suite can never overwrite each other.

## Why KSEG0 rather than where the PROM loads us

`sashARCS` on the IRIX 6.5.22 CD links at `0x10000000` — **kuseg, which is
TLB-mapped**. A suite whose job includes rewriting all 48 TLB entries cannot
execute from a mapped region: the first `tlbwi` would unmap the code running
it. So `start.S` computes its load bias with a `bal`, copies the image to the
link address, flushes I and D, and jumps. From then on:

- execution is unmapped, so the TLB tests cannot pull the rug out;
- cache tests can reach their own data through KSEG1 by clearing bit 29, and
  the KSEG0/KSEG1 views are deliberate rather than accidental;
- the suite is independent of *how* it was loaded — `--load-elf`,
  `boot -f dksc(...)` and `boot -f bootp()` all converge on the same layout.
