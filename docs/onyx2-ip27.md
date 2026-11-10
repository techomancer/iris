# SGI Onyx2 deskside (IP27 node boards)

The Onyx2 deskside is one or two IP27 node boards on the 4P1G5 midplane. Both
Hubs are hosts of the one Crossbow, which carries the BaseIO-G at widget 15 and
the graphics slot at widget 8. Two nodes are joined by the direct cable. The
machine runs on the IOC3's first serial port. With `[graphics] board = "none"`
(the default) the graphics slot holds a test widget that answers its
identification word and nothing else; with `"ir2"`, `"ir2:2rm"`, `"ir2e"` or `"ir2e:2rm"` it holds an
InfiniteReality2 or 2E pipe with one or two raster managers (`src/dev/kona`), whose
window is the machine's display. `[graphics] flash` names a file that keeps the
pipe's flash EEPROM between runs.

## Configuration

Copy [`onyx2.toml.example`](../onyx2.toml.example) to `onyx2.toml`, provide the
PROM and disk named there, and run `iris --config onyx2.toml`. The sample
selects `machine.profile = "onyx2_ip27"`; `[onyx2]` (`nodes`, `flash`,
`controller_store`, `io_prom`) holds its own keys. `io_prom` names the I/O PROM
image the IRIX 6.5.30 media ship beside the node PROM, fitted in the BaseIO's
boot flash; the graphics console needs it, since the node PROM's own internal
copy has none. A file that still names `[origin2000]`
for an Onyx2 is refused, with the table to rename it to. Node slot 1 drives raw slot
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
- The BaseIO-G carries the MediaIO daughtercard: a second IOC3 at PCI device 6
  and the RAD1 audio chip at device 7, with the daughtercard's own board
  record. The MediaIO's serial ports have no host backend, and the host's keys
  and pointer go to the board IOC3's controller only.
- The module controller, the cable and the node windows are the
  [Origin 2000 deskside's](origin2000-ip27.md), with its limits.
- The pipe's interrupt packets and DMA reach the Hub at the widget the guest
  programs into it (widget 9 with two nodes, 10 with one), behind that
  widget's Crossbow link.
- Snapshots are not supported. The pipe can save its own state, but the Hubs
  and node memory cannot yet.
