# SGI Origin 3200 (IP35)

The Origin 3200 is built from bricks. A C-brick is a compute node: a Bedrock
node controller with its processors and memory. An I-brick carries the base
I/O on an XBridge. One or two C-bricks, each a 400 MHz R12000 with an 8 MiB
secondary cache and 512 MiB in two 256 MiB banks, are cabled to an I-brick on
the first. Two C-bricks are joined by a direct NUMAlink cable with no router.

## Configuration

Select `machine.profile = "origin3200_ip35"` and give the PROM payload as for
the [Fuel](fuel-ip34.md). `[origin3200]` takes `c_bricks` (1 or 2), `flash`,
`console`, `l1_port`, `l1_log`, `l1_revision` and `brick_serial`; see
[FEATURES.md](../FEATURES.md#origin3200). `banks` is per C-brick: left unset it
is two 256 MiB banks. Banks of 256 or 512 MiB are accepted, up to eight, in pairs of one size.
Graphics are `none`. The I-brick's Fibre Channel adapter takes its disks from
`[fibre_channel.disk.N]`, N the disk's loop ID (see [the Fibre Channel guide](fibre-channel.md));
`board`, `slot` and `bridge` name an Octane's option board and are refused. [origin3200.toml.example](../origin3200.toml.example)
is a complete file.

Each C-brick has its own L1; the first carries the console
(`console = "l1"`, port `l1_port`). Each C-brick runs one processor; the other
three positions are listed disabled in the node's flash, which spares the PROM
three power cycles for each.

## The I-brick

The I-brick is data (`IBRICK` in `src/ip35_machine/origin3200.rs`): an XBridge
whose heads E and F are widgets 14 and 15, host port A (widget 10) cabled to the
first C-brick, and on head F's bus (the IO7) a Fibre Channel adapter at device 1,
the IOC3 at device 4 and the USB controller at device 5. Port B (widget 11) is
declared and empty. The Fibre Channel adapter is always fitted, with or without
disks, and its slot runs 64-bit data phases (inferred from the PROM's behaviour:
the control block and rings it builds are right only if the bridge reverses
each doubleword of the adapter's DMA). A `[scsi.N]` table is refused.

## Nodes and the link

Address bits 39:33 name a node. Each node's local block holds the number the PROM
stores in its revision word, and the node is reachable at that number from the
other node while the cable is up (`src/dev/bedrock/route.rs`). The vector
engine's route 0 reads and writes the peer's registers, which is how the PROM
discovers the neighbour and numbers it; port status bit 1 reads link-up. Each
node's ppmem window shows its own memory and the other's at its number
(`src/node_windows.rs`, node shift 33), so a processor reads either node's
memory without the bus. Processors are numbered across the machine, and one
`World` holds them all still when a bank configuration write moves memory.

## Observed boot

PROM 6.210 reaches the System Maintenance Menu with one C-brick (1 CPU, 512 MB)
and with two (2 CPUs, 1024 MB), both `ALL DIAGS PASSED`, and Start System boots
IRIX 6.5 from a disk on loop ID 1 to a login on both. The I-brick's IEEE 1394
controller is absent, so the first diagnostic prints `No 1394 Firewire found on
BASEIO` and carries on.

## Approximations and limits

- One processor per C-brick; the other positions are disabled in the flash.
- Which low nibble of the board identification word a C-brick presents is a
  choice (2, the 400 MHz, 8 MiB row). The L1's node board and PIMM records are
  placeholders (`NOTOBSERVED0`), with synthetic serials.
- The I-brick's own L1 is a stub: the first C-brick's L1 answers its bay and
  letter and serves every other request as its own.
- The IO7's IOC3 is the Fuel's with the placeholder subsystem id `0xffff`;
  the clock part is unidentified and is the Fuel's M48T35-compatible one.
- A reset or power request from either node resets the whole machine.
- No cache coherence runs between the nodes; both see the same host memory.
- A cached reference to the reset PROM's address window always stays on the
  node. A 512 MiB bank among 0 to 3 covers that window, so the processor
  presents the PROM's uncached fetches at the PROM's own address
  (`uncached_fetch_alias`); the split between the two is inferred, not
  established (`src/ip35.rs`, `RESET_WINDOW_FETCH`).
- Under `jitv2` each processor keeps a page pool of its own, and the two-node
  machine hands the compiler the banks' own page counters, not the chip's
  logical ones. The PROM's uncached-fetch copy sits at the top of the second
  node's address range (`RESET_WINDOW_FETCH`), clear of any RAM. Both C-bricks
  boot IRIX to a login this way. A bank configuration change after a page was
  compiled is not tracked by the banks' counters; the PROM makes it before
  any kernel code runs.
