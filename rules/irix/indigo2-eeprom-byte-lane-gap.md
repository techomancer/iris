# Indigo2 motherboard EEPROM: PROM writes via byte access, HPC3 only handled 32-bit

## Symptom

Indigo2 (IP22 fullhouse) boots IRIX with no Ethernet MAC set, even after
adding a backdoor that writes `08:00:69:...` into `Eeprom93c56` words
`0x7D-0x7F` at machine construction (before the CPU ever runs). `nveeprom
dump` after boot showed the backdoor value was gone/never took effect, and
`IRIS_DEBUG_LOG=nveeprom` (enabled before CPU start) showed **zero**
read/write log lines during boot — looked like PROM never touched the chip.

## Root cause

`Hpc3::read8`/`write8` (`src/hpc3.rs`) never had a case for `MISC_BASE`
addresses at all — only `read32`/`write32` handled `MISC_EEPROM_DATA` (the
bit-banged EEPROM control register at physical `0x1fbb0008`). PROM actually
drives this register **8 bits at a time**, writing to `0x1fbb000b` — the
bottom byte lane of the big-endian 32-bit word (`offset & 3 == 3`, same
convention already used for the RTC's `PBUS_BBRAM` sparse packing). Every
one of those byte-wide writes fell through to the `read8`/`write8` handlers'
final "not found" branch and was logged as:

```
[hpc3] HPC3: Unexpected write8 at offset 3000b (addr 1fbb000b) val=00
[hpc3] HPC3: Unexpected write8 at offset 3000b (addr 1fbb000b) val=02
[hpc3] HPC3: Unexpected write8 at offset 3000b (addr 1fbb000b) val=04
...
```

— a clear CS/SK/DI bit-toggle pattern (values 0x00/0x02/0x04/0x08 = bits
1/2/3), confirming PROM *was* actively driving the chip. It just never
reached `Eeprom93c56`'s state machine, so the chip's internal `tick()` never
advanced and no READ/WRITE opcode ever completed — hence the `nveeprom` log
module staying silent even though logging was correctly enabled before CPU
start.

## Fix

Added the `MISC_EEPROM_DATA` byte-lane case to both `read8` and `write8`,
mirroring the 32-bit handlers' pin logic (CS=bit1, DI=bit3, SK=bit2, DO=bit4
on read). Placed **before** the unconditional `self.state.lock()` further
down in each function (matching where the RTC's byte-lane case sits) —
placing it after would double-lock `self.state` (`parking_lot::Mutex` is not
reentrant) and deadlock on the very first EEPROM byte access. This was an
easy mistake to make once, since `write8`'s existing structure has the
generic lock *after* several early-return device-specific blocks, but
`read8`'s early PBUS-PIO/RTC blocks are interleaved differently — always
check where the surrounding function's generic `self.state.lock()` sits
before adding a new arm that also needs `self.state`.

## Lesson

When a device's log module stays completely silent despite being enabled
before the CPU starts, don't assume "guest never touches this register" —
check whether the guest is hitting it via a **narrower bus width** than the
one your handler actually implements (8-bit vs 32-bit access to the same
byte-packed register is an easy gap, especially for registers modeled as
sparse/byte-lane-packed like `PBUS_BBRAM` and `MISC_EEPROM_DATA`). The
"Unexpected read8/write8" catch-all log line at the bottom of `read8`/
`write8` is the tell — it was firing the whole time, just not being watched
under the assumption that "no eeprom log = no eeprom access."

See also [[project_ethernet_mac_backdoor]] for the two-EEPROM split
(`eeprom_mc` vs `eeprom_hpc3`) this bug was found while debugging.
