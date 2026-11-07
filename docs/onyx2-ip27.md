# SGI Onyx2 deskside (IP27 node boards)

The Onyx2 deskside is one or two IP27 node boards on the 4P1G5 midplane. Both
Hubs are hosts of the one Crossbow, which carries the BaseIO-G at widget 15 and
the graphics slot at widget 8. Two nodes are joined by the direct cable. The
machine runs headless on the IOC3's first serial port. There is no
InfiniteReality pipe: the graphics slot holds a test widget that answers its
identification word and nothing else.

## Configuration

Use [`origin2000.toml.example`](../origin2000.toml.example) with
`machine.profile = "onyx2_ip27"`; `[origin2000]` (`nodes`, `flash`,
`controller_store`) configures this machine too. Node slot 1 drives raw slot
identity 3 into its Hub and takes widget 10; slot 2 drives 2 and takes widget 9.
With one node, the board is in slot 1.

A disk installed on an Onyx2 boots; the machine cannot install without a pipe.

## Limits

- One node boots the stock PROM to the System Maintenance Menu and an
  installed IRIX 6.5.30 to a serial login.
- Two nodes: the PROM finds both Hubs, numbers them, makes the second slot's
  board the global master, passes its diagnostics with both processors and
  256 MB, and IRIX boots to a root login with both processors.
- Either Hub's processor reaches the BaseIO-G through the Crossbow. The Bridge
  delivers DMA and interrupts to the Hub at the widget the guest programs into
  it, and both Hubs are hosts, each behind its Crossbow port's link. With two
  nodes the PROM and IRIX program widget 9, the master's Hub; with one, widget
  10.
- The BaseIO-G is its base board: no MediaIO or audio.
- The module controller, the cable and the node windows are the
  [Origin 2000 deskside's](origin2000-ip27.md), with its limits.
- Snapshots are not supported.
