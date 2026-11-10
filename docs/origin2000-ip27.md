# SGI Origin 2000 deskside (IP27 node boards)

The Origin 2000 deskside is one or two IP27 node boards on the module
midplane. Each node is a Hub with its memory, boot flash and one R10000 or
R12000, and the host of its own Crossbow. The first Crossbow carries the BaseIO
card (a Bridge with two ISP1040s and an IOC3). With two nodes the Hubs are
joined by a direct cable, the Null Router. The machine runs headless on the
IOC3's first serial port.

## Configuration

Copy [`origin2000.toml.example`](../origin2000.toml.example) to
`origin2000.toml`, provide the PROM and disk named there, and run
`iris --config origin2000.toml`. The sample selects
`machine.profile = "origin2000_ip27"`.

- `[origin2000] nodes` is 1 or 2. The memory in `banks` is fitted to every
  node.
- `[origin2000] io_prom` names the I/O PROM image (`io6prom.img`, beside the
  node PROM on the IRIX 6.5.30 media) to fit in the BaseIO's boot flash;
  without it the node PROM runs its own internal copy.
- `machine.cpu` is `r10000` or `r12000`. Each node has one processor.
- `[origin2000] flash` and `controller_store` keep the boot flash (the second
  node's at the same name with `.1` added) and the module controller's store
  between runs.
- `prom` takes the stock IP27 PROM, as for the [Origin 200](origin200-ip29.md).

Node slot 1 drives raw slot identity 7 into its Hub and slot 2 drives 6. The
first node's Crossbow has the BaseIO at widget 8 and the Hub at widget 9; the
second's has the Hub at widget 10. The module has one controller, reached
through an I2C adapter per node.

## Limits

- One node boots the stock PROM to the System Maintenance Menu and IRIX
  6.5.30 to a serial login.
- Two nodes: the PROM finds both Hubs, numbers them, chooses the first as the
  global master, passes its diagnostics with both processors and 256 MB and
  reaches the menu. IRIX then stops with "Processor #1 did not start": the
  second node's processor takes a TLB exception in the PROM (the master
  reports "has taken an exception"), so it never waits to be launched.
- The BaseIO's boot flash is fitted only when `[origin2000] io_prom` names an
  I/O PROM image; without one the PROM falls back to its own copy of the I/O
  PROM.
- The BaseIO is the only board; the second Crossbow has no widgets but its
  node.
- The module's midplane one-wire part is not fitted.
- A node that holds a nonzero number sends cached references to node zero's
  window beyond its cached alias to the Hub that holds zero, and its own RAM
  is then reached through its own number. Each processor's ppmem window shows
  both nodes' memory at the same addresses, so both nodes run at the same
  speed. With two nodes it counts 128 MiB regions: a 64 or 256 MiB bank
  leaves its regions part empty, and those take the slower bus path.
- Snapshots are not supported.
