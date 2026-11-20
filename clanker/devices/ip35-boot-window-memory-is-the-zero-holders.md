# IP35: memory in the boot window belongs to the zero holder, on every node

Status: fixed 2026-11-19.

## Symptom

A two-node IP35 machine (Onyx 3000, Origin 3200 with `c_bricks = 2`) with
512 MiB or more per node booted IRIX and then fell apart: `/etc/rc2[55]`
died with `Memory fault(coredump)`, later processes faulted at random
(`Bus error`, shell scripts printing nonsense), raw disk reads returned
different data each time, and under load the kernel panicked (`KERNEL FAULT,
PC: 0x0`) or hung. Same with jitv2 dispatch off; same without the V-brick.
One node was clean, and so were two nodes with the second brick's CPU
disabled in the PROM.

The corruption itself, captured by piping a known disk region through two
processes and comparing with the image: whole 16 KiB pages (the IRIX page
size) of zeros where data belonged.

## Cause

`NodeDecode::map_banks` kept "the boot window at node field zero" on the
reader for a renumbered Bedrock node. Where the node has no memory behind
the window that shows its own boot flash, which the PROM needs. But it kept
the node's own *banks* there too: with 512 MiB or more, physical
0x1fc0_0000..0x2000_0000 at node field zero was the second node's own RAM
for its processors, and the zero holder's RAM for the first node's
processors, for DMA and for the block transfer engine. IRIX hands those
four megabytes out as ordinary pages of node 0.

On the hardware only *uncached* references to 0x1fc0_0000 are the boot
vector, and they reach the local flash through the hub-special space. The
PROM's copy in memory is at a low address, mapped at virtual
0xc000_0000_1fc0_0000 through the TLB. A cached reference to physical
0x1fc0_0000 is plain memory by node number.

## Rule

At node field zero a renumbered node keeps its alias (the first bytes its
chip says) and, on the Bedrock, its boot *flash* where it has no memory
behind the window. Memory is never the reader's own there: it is the zero
holder's. Every path to a physical address has to agree: each node's
`Physical` and ppmem window (what its processors see), and memory by number
(`NodeDecode` as a bus: DMA, block transfers).

## How it was found

The order matters more than the result; each step halved the search:

1. One node, same Fibre Channel disk: byte-exact against the host
   (`dd ... | cksum` in the guest and on the image). Not the ISP2200.
2. Two nodes, `IRIS_JIT_DISPATCH=off`: still fails. Not the JIT.
3. Two nodes, no graphics: still fails. Not the V-brick.
4. Two nodes, second CPU disabled (PROM menu 5, `disable -m 001c07 -cpu a`,
   `reset`): clean with all memory in use. Not node-1 memory or DMA.
5. Pipe a raw region to the NFS share and `cmp` with the image: 16 KiB
   pages of zeros. A page seen as two different pages.
6. Compare what each CPU reads at the same physical address from the
   monitor (`cpu N mem 0xa8...`), and CPU 1's view of a node-0 address
   against its view of its own node at the same offset. Sample every
   64 KiB; a coarse stride over the whole of memory missed the 4 MiB.

Traps on the way: `0x9000...` is the hub-special space, where a per-node
alias is legitimate (uncached memory is `0x9600...`); breakpoints do nothing
in a `lightning` build; the Tezro image has no network on these machines
until `ifconfig ef0` by hand.

Not a problem: bank 0 smaller than 512 MiB (`banks = [256, 256]`). Each bank
sits in its own 1 GiB slot and every C-brick gets the same banks, so then no
node has memory at 0x1fc0_0000: both show flash there and IRIX has no pages
in the window to hand out. Two nodes of 256 MiB banks, both CPUs, jitv2 on:
clean boot, raw read and `/usr/lib` sweeps match (2026-11-19). A node showing
flash over the zero holder's memory would need bricks of different sizes,
which the configuration does not allow; revisit if it ever does.
