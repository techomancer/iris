# SGI Origin 3200 (IP35)

The Origin 3200 is built from bricks. A C-brick is a compute node: a Bedrock
node controller with its processors and memory. An I-brick carries the base
I/O on an XBridge. One or two C-bricks, each with one, two or four 400 MHz
R12000s (8 MiB secondary cache) and 512 MiB in two 256 MiB banks, are cabled to
an I-brick on the first. Two C-bricks are joined by a direct NUMAlink cable with no router.

## Configuration

Select `machine.profile = "origin3200_ip35"`. The IP35 PROM 6.210 is built
in; `prom` overrides it as on the [Fuel](fuel-ip34.md). `[machine]` takes `nodes` (1 or 2 C-bricks; `c_bricks` is another name for it), `flash`,
`console`, `l1_port`, `l1_log`, `l1_revision` and `brick_serial`; see
[FEATURES.md](../FEATURES.md#origin3200). The `[origin3200]` table of older configs is still read. `banks` is per C-brick: left unset it
is two 256 MiB banks. Banks of 256 or 512 MiB are accepted, up to eight, in pairs of one size.
Graphics are `none`. The I-brick's Fibre Channel adapter takes its disks as `hdd` devices at
`xio15/pci0/1/loopN`, N the disk's loop ID (see [the Fibre Channel guide](fibre-channel.md)). [origin3200.toml.example](../origin3200.toml.example)
is a complete file.

Each C-brick has its own L1; the first carries the console
(`console = "l1"`, port `l1_port`).

`[machine] cpus` is the processors on each C-brick: 1 (the default), 2 or 4, so
two C-bricks run up to eight, each on its own host thread. The positions left
empty are listed disabled in a new flash, which spares the PROM three power
cycles for each. Raising `cpus` on a machine with a kept `flash` file ends the
disable records iris wrote for the positions now filled, and the PROM reports
them as newly present and enabled. Lowering it leaves the PROM to find the
empty positions itself on the next start.

## The I-brick

The I-brick is data (`IBRICK` in `src/ip35_machine.rs`): an XBridge
whose heads E and F are widgets 14 and 15, host port A (widget 10) cabled to the
first C-brick, and on head F's bus (the IO7) a Fibre Channel adapter at device 1,
the IOC3 at device 4 and the USB controller at device 5. The controller's two
root ports hold a USB HID keyboard (port 1) and mouse (port 2): boot-protocol,
low-speed functions that the host's keyboard and mouse reach through the same
`ps2` and `key`/`ptr` monitor commands as on the other machines, so the PROM
and IRIX graphics consoles have the input a real I-brick gives them. Which
port holds which is inferred. Port B (widget 11) is
declared and empty. The Fibre Channel adapter is always fitted, with or without
disks, and its slot runs 64-bit data phases (inferred from the PROM's behaviour:
the control block and rings it builds are right only if the bridge reverses
each doubleword of the adapter's DMA). A disk at a SCSI slot is refused.

An `ice1724` device fits an audio card in one of the I-brick's PCI slots (the first free one, `bus2-slot1`, unless `at` names another; `iris --list-slots` shows them): the IC
Ensemble ICE1724 with the DMediaPro DM8's converters, the same model as the
[Tezro's](tezro-ip53.md). The device number is the model's choice; which
devices the I-brick's option slots are is not established. On an Onyx 3000,
IRIX 6.5.30 lists it as `Iris Audio Processor: version MAD revision 1` and
plays through it; it has not been run in an Origin 3200 guest. It plays to the
host as `[audio]` sets, or silently with `no_audio`.

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
IRIX 6.5 from a disk on loop ID 1 to a login on both. Two C-bricks of two and
of four processors (4 and 8 CPUs) boot the same way, and IRIX runs all of them. The I-brick's IEEE 1394
controller is absent, so the first diagnostic prints `No 1394 Firewire found on
BASEIO` and carries on.

## Approximations and limits

- The L1 names one processor module for each processor interface in use; the
  real module count for a population is not established.
- Which low nibble of the board identification word a C-brick presents is a
  choice (2, the 400 MHz, 8 MiB row). The L1's node board and PIMM records are
  placeholders (`NOTOBSERVED0`), with synthetic serials.
- The I-brick's own L1 is a stub: the first C-brick's L1 answers its bay and
  letter and serves every other request as its own.
- The IO7's IOC3 is the Fuel's with the placeholder subsystem id `0xffff`;
  the clock part is unidentified and is the Fuel's M48T35-compatible one.
- A reset or a power cycle requested from either node resets the whole
  machine; only a shutdown (`halt -p` in IRIX) powers it off.
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

