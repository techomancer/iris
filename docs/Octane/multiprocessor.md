# Two processors on the Octane

`[machine] cpus = 2` builds an Octane with a dual-processor module. Each
processor is a host thread running at full speed, so the guest gets real
parallelism: a two-process workload finishes in about the time one process
does. IRIX reports `2 195 MHZ IP30 Processors`, schedules across both and
spreads device interrupts over them.

```toml
[machine]
profile = "octane_ip30"
cpu     = "r10000"
cpus    = 2
```

The emulator needs two free host cores for this to be useful, and more for the
compile pool. Two is the limit: HEART has four processor slots, but the Octane's
PROM and IRIX use two.

## How a second processor starts

Nothing is faked. The real PROM and kernel do it all through HEART, whose
registers the emulator answers per processor.

Both processors start at the same reset vector. Each reads its number from
HEART's `PRID` register and the fitted processors from the active bits of
`STATUS`. The lowest-numbered active processor is the master. The other polls
the interrupt status register, uncached, for its own inter-processor interrupt
bit. The master freezes it with a processor-disable bit in `MODE` while it sizes
memory, then releases and wakes it. The slave runs its cache diagnostics,
writes its `MPCONF` block at physical `0x600 + 128 * id`, and waits for a launch
address there. The kernel later writes one, and the slave is running IRIX.

The PROM also checks that the processor module's name agrees with the active
bits: a `PM10` has one processor and a `PM20` two. The emulator reports the
module that matches `cpus`.

## HEART, per processor

| Register | Answer |
|---|---|
| `PRID` | the number of the processor reading it |
| `STATUS` | bits 7:4 are the fitted processors, so `0x18` for one and `0x38` for two |
| `IMR0`..`IMR3` | one interrupt mask each |
| `IMSR` | pending bits under the reader's own mask |
| `MODE` bits 63:60 | disable processor n: it freezes in place, and continues where it stopped when the bit clears |
| `MODE` bit 22 | software reset, of the whole machine |

Interrupt outputs are computed per processor from the pending bits and that
processor's mask. An inter-processor interrupt is nothing special: the sender
sets the destination's bit (`46 + n`) in `SET_ISR`, and only the destination's
mask has it. HEART's counter is shared and is the guest's time base. Each
processor's clock interrupt is its own CP0 Count/Compare.

## Memory

Both processors see one RAM. There is no cache model holding data, so coherency
is what the host provides. Two things are not plain stores:

- **`SC` and `SCD`** are a host compare-exchange against the value `LL` read. A
  store by the other processor cannot clear this processor's LLbit, so the
  store itself has to notice.
- **`SWL`, `SWR`, `SDL` and `SDR`** merge bytes with a compare-exchange on the
  doubleword, so they cannot lose what the other processor stored beside them.

The PROM runs from the boot flash with its cached data held privately per
processor, as the hardware's non-coherent caches hold it. That matters for more
than data. While one processor has the flash in status mode to program or probe
it, the other keeps executing from its cached lines and never sees the status
byte.

## Compiled code

Both processors share one jitv2 page pool and one body of compiled code. Code
reaches each processor's registers through the core pointer it is called with,
so nothing in it belongs to either one. A page compiled for one runs on the
other, and a store by either invalidates it for both.

A flush stops both. A processor stops at a batch boundary, outside compiled
code, with its executor unlocked.

## Debugging

See [debug.md](../../debug.md#several-cpus) for the monitor commands: `cpus`,
`cpu <n>`, `r0`, `s1` and stepping one processor while the other holds still.

## What is not there yet

- The GDB stub sees processor 0 only.
- A breakpoint stops the processor that hit it. The other keeps running until
  `stop`.
- Snapshots are not available on the Octane at all.
