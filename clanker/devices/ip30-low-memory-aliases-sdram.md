# The Octane's 16 KiB at physical 0 is the first 16 KiB of SDRAM

Status: rule (HEART revision B and later, which covers every Octane we model;
DMA reach is unknown).

**What to know.** Physical 0..0x4000 is the first 16 KiB of SDRAM, offset 0
of the memory that starts at 0x2000_0000. The processor reaches those bytes
only through the low window: its access to 0x2000_0000..0x2000_4000 is a bus
error, and SDRAM is reachable at its own address from 0x2000_4000. The stock
IDE diagnostics' expectations are exactly this. 0x4000 up to the 64 KiB decode
slot is a bus error too.

In `src/dev/heart_mem.rs` (on `Physical` since 2026-10-28), `HeartRam::rebuild`
puts the bus's default (bus error) over 0x2000_0000..+16 KiB in the table and
points 0..0x4000 at a `bank_view` of the bank behind the unit at 0x2000_0000,
so both reach one bank and one set of jitv2 page generations. The kernel
writes its exception vectors at physical 0, so a store must advance the
counter of the SDRAM page behind it or the JIT keeps running stale code.

**The guard is enforced on the table path only.** ppmem's window maps whole
32 MiB units, guard included, and its bitmap is 64 MiB-granular, so
`Physical`'s fast paths (`mem_ptr`, `read_block`, `write_block`, `gen_ptr`)
and the CPU's cached accesses through the window reach the guard's bytes.
Uncached processor loads and stores (`read32` and friends) go through the
table and bus-error. This was already true of cached accesses before the
move (the tcache read the same window); keeping the guard out of the window
would push the kernel's first 64 MiB off the fast path.

**Unknown.** Whether bus masters can reach 0x2000_0000..+16 KiB. The model
leaves DMA as it was: the `dma_view` reaches those bytes, the same ones the
low window shows. Public software only uses 0x400 (the GDA, OpenBSD's
`ip30.h`) and 0x600 (the CPUs' MPCONF records, Linux's `ip30-smp.c`).
