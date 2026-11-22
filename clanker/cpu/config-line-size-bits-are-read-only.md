# Config IB/DB are read-only: only K0 and CU take an MTC0

Status: fixed 2026-11-20 (techomancer/iris#773).

## Symptom

NetBSD/sgimips (11.1 and 11.99.10) on an Indy with `--cpu r5000` panics right
after the kernel starts, before any device probe:

```
panic: r5k picache line size 16
```

The same CD boots to the installer on an R4400 Indy and on the R5000 O2.

## Cause

The IP22/IP24 PROM writes Config while it sets up the caches, with bits 5:4
(IB, DB) clear: at the PROM menu Config reads `0x00c286c3`, where reset gave
`0x00c286f3`. The MTC0 Config handler in `mips_core.rs` took bits 5:0 from the
written value, so IB and DB stuck at 0.

NetBSD's `mips3_get_cache_config` derives the primary line sizes from IB and
DB (`MIPS3_CONFIG_CACHE_L1_LSIZE`: set = 32 bytes, clear = 16), and the R5000
case in `mips_config_cache_prehistoric` accepts only a 32-byte I-cache line.
IRIX and the PROM never read the line size back, so they were unaffected. The
O2 PROM does not clear the bits, so the O2 never showed it. An R4400 resets
with IB/DB clear, so the write changed nothing there.

On the R4000/R4400 and the R5000 the line sizes are fixed by the part (or the
boot-mode bits on the R4000); software can write only K0 (2:0) and CU (3).

## Fix

The Config write mask is `0xF` (`0x100F` with `r5ksc_triton`, adding SE).
Nothing in the emulator read IB/DB after reset; cache geometry comes from the
model type.

Tests: `config_line_size_bits_ignore_mtc0` (unit, R4400 and R5000) and
`identity/config_line_size_ro` in `cpu-tests` (R4400/R5000/R4600; the R10000
lays Config out differently). The existing `identity/config_k0` wrote the
original Config back with only K0 changed, so it could not see this.

## Reproduce

Indy, `--cpu r5000`, headless, the NetBSD sgimips ISO as `[scsi.4]` CD-ROM.
From the PROM command monitor (serial channel B):
`boot -f dksc(0,4,8)ip2xboot ip2x` (the CD's root holds the install kernel as
`ip2x`; with no argument ip2xboot looks for `/netbsd` and fails). A failed
`boot -f` leaves the PROM short of memory ("doesn't fit in a FreeMemory
area"); `init` and enter the monitor again before retrying.

`tools/netbsd-cd-boot.py ISO --cpu r5000` does all of this unattended and
exits 1 with the panic line before the fix, 0 at the sysinst menu after.
