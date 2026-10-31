# With two CPUs, cached flash reads must hold their lines

Status: rule, found 2026-10-30 on the first two-CPU Octane boot.

## Symptom

The PROM screen shows `NESTED EXCEPTION #2 (vid 1)` and CPU 1 sits in the PROM's
one-instruction hang loop. The fault differs per run: a TLBL at `0xffffffffffff8081`
one time, a TLBS at 0 the next. CP0 misleads, because EXL was set for the second
exception, so EPC is the first one's. CPU 1's traceback ends with an impossible
instruction:

```
0xffffffff9fc6c648: 0x80808080 lb zero, -32640(a0)
```

## Cause

`0x80` on every byte is the flash's ready status. CPU 0 had put the part in status
mode to probe it. CPU 1 was executing from flash, and the shadow cache has no
contents, so its fetch reached the chip. A real R10000 hits its instruction cache.

## Fix

`CachedFlashLines::for_one_of_several_cpus`: cached reads hold their line too, filled
with `sflash::reading_the_array`, which reads the array whatever command the part is in.
One instance per CPU, because `Config.K0` is 3 (non-coherent), so each CPU's dirty PROM
lines are its own. One CPU keeps the old read-through path.

## Finding it

`cpu 1 bp add` on the hang function and on the RAM vector `0xffffffff80000180`, then
`cpu 1 dt`. Do not break on the flash vectors: BEV is off by then.
