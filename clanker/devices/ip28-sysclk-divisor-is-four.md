# The IP28 runs SysAD at a quarter of the CPU clock (Config.EC 7)

Status: rule, 2026-11-24.

**Symptom.** IRIX on the IP28 logged "Old IP28 prom installed -- memory
refresh was incorrectly configured [fixed]" at every boot, and the PROM's
`system` command printed the SysAD divisor as "?".

**Cause.** The generic R10000 Config left EC (bits 12:9, the SysClk divisor)
at 0, which the IP28 tables treat as 1:1 and the PROM prints as undefined.
The PROM's frequency probe sets the MC refresh preload CTRLD to
`mhz * n * 125 / (2 * d)` from EC's n/d; with 1:1 that is 0x2F9B, and the
kernel takes any CTRLD of 0xD00 or more for an old PROM and rewrites it.

**Rule.** The IP28 sets Config.EC = 7 (divide by 4: 195 MHz / 4 = 48.75 MHz
SysClk) in `machine.rs`, and the MC counts at the same SysClk
(`2 * count_hz / 4`). Keep the two together: IRIX derives the RPSS cycle
counter's period from the CPU MHz and the same EC, so changing one without
the other skews every RPSS-based timing. With EC 7 the PROM writes 0xBD7 and
the kernel leaves it alone.

**Watching it.** `mc regs` lists CTRLD's write count and last 16 values. A
healthy IP28 boot reads `c30 186 c30 bd7 17a bd7`: the csu reset value, the
keyboard-bell watchdog bump (>>3) and restore, the frequency probe, and a
second bump and restore. A seventh write during kernel start-up means
check_ctrld fired.
