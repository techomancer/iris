# A PIC interrupt packet can target memory, with any 8-bit vector

Status: rule, 2026-11-18. Fixed in `src/dev/pic.rs` (`Head::packet`).

## Symptom

Tezro with `[tezro] audio = true`: running amesh (or another audio program)
hangs the machine. The ICE1724 keeps playing its ring, so the sound repeats,
while telnet and the desktop freeze. `envy` showed the card idle, with no
register access for many seconds.

## Cause

IRIX makes a PIO read safe on a PIC (and on Bridge) like this:

1. It points an interrupt input's host address register at a word in memory,
   with vector 0xed. On Tezro that is input 7: `0x00ed000000afc890`.
2. It clears the word, does the PIO, and writes the input's force-always
   register (0x180 + 8n).
3. It spins until the word reads 0x1ed (the vector with the set bit).

The interrupt write is ordered behind the PIO, so seeing it proves the PIO
has completed. The ICE1724 driver does this for every position read (MT24,
MT10).

The model refused every packet whose level was outside 7..127, because it
assumed a packet always goes to a PI's pending-modify register. So vector
0xed never arrived, and the CPU looped forever in the kernel's wait (a
`ld`/`beq` loop comparing against 0x1ed).

## Rule

A packet is the vector field (host address register bits 55:48,
`PIC_INT_ADDR_FLD` in the public headers) plus bit 8, written to the host
address, whatever is there.

The 7..127 limit applies only to packets addressed to the XIO register (I/O)
view, address bit 47, where the PI is. A packet to memory carries any vector.

## How it was found

`envy` showed the card no longer polled. Sampling the CPU PCs from the
monitor showed one CPU stuck in a tight loop. Its a0+80 was the flag, and
the PIC's input 7 host register held that address with vector 0xed.

Writing 0x1ed there by hand (`cpu N mw <a0+80> 0x1ed d`) released it until
the next read.

Tests: `a_forced_input_aimed_at_memory_writes_its_vector_there` and
`only_levels_a_pi_can_take_leave_the_chip_for_a_pi` (`src/dev/pic/tests.rs`).
