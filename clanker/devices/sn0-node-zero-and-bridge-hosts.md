# SN0 machines: node zero's window and a Bridge's host (rule)

Two rules about the Hub machines (Origin 200, Origin 2000, Onyx2) that are easy
to get wrong, because each looks like something the code could simply assume.

## A renumbered node reroutes its cached references

Cached references use the node's `Physical` directly: RAM at node-relative
offsets, with the ppmem window the cache and the JIT read. That is right while
the node holds node number zero, which is the one-node Origin 200 and the first
node of a deskside.

The stock PROM numbers the second node 1 after reset. A cached reference from
that node's processor to its own memory may still name node zero, and then the
reference is not the node's own: it belongs to whichever Hub holds zero.
`NodeZero` (`src/ip27.rs`) applies this when the node's number or its cached
alias size changes:

- Within the cached alias (`PI_CALIAS_SIZE`) the reference stays on the node.
- Beyond it, node zero's window is sent through the Hub's node-qualified memory
  to the Hub that holds zero, across the cable.
- The node's own memory is then reached by its own number.

The ppmem window, which the cache and the JIT read without the bus, must show
the same memory at the same addresses, or the renumbered node's processor takes
the bus path for every cached reference (about 2.2 times as long for a
CPU-bound loop when only the first node had a window). `Windows`
(`src/ip27.rs`) lays every node's window from the numbers the Hubs publish,
whenever any node's number or alias changes:

- node zero's window as above, from the node's own banks or the zero holder's;
- each node's memory at its number (`n << 32`), its own and the other's;
- only the difference, so a page that stays mapped is replaced in one host
  mapping call and a renumbering never clears the window a running processor
  executes from.

A store to any Hub's number or alias publishes the numbers (`Routing`) and
lays every node's window at once, from the storing thread: the hardware takes
the new number at once, so no window trails the decode table. The other node's
processor can be running meanwhile, and its CACHE instructions hold and release
line pages in the same `PpMemSpace`. The space has one small mutex, taken by
every remap (`map_bank`, `map_alias`, `map_range`, `unmap_range`,
`clear_mappings`) and by `add_cached_view`, `hold_line_page` and
`release_line_page`. A load, store or fetch never takes it: they read through
the processor's inline bitmap and its view's base pointer. Without it a
renumber rebuilds the mapping list under a `hold_line_page` that is reading it
(`a_renumber_on_one_processor_does_not_disturb_the_others_line_pages`). Lock
order is `Windows`, then the space.

A two-node window is 8 GiB, so its region bitmap counts 128 MiB regions; a 64
or 256 MiB bank leaves regions part empty, and those take the bus path. Do not
add a decision to the access path for any of this: the rare state change lays
a different mapping, and nothing is tested per access.

## A Bridge behind a Crossbow picks its host by the widget the guest programs

A Bridge sends DMA and interrupt packets to a host chip addressed by widget
number. Which widget is the guest's to program (`DIRECT_MAP` and `INT_DEST` in
the Bridge's registers), not the board's:

- On the Onyx2 both node boards are hosts of the one Crossbow. The Bridge's
  `Upstream` holds a `HostPort` for each, at its widget (10 and 9), and the stock
  PROM programs the widget of the board it makes the global master. DMA and
  interrupts go to that Hub.
- On the Origin 2000 the Bridge sits behind the first node's own Crossbow and
  reaches only that node's Hub, at widget 9.
- Behind a Crossbow, `Xbow::upstream` gates each host on its port's link, so a
  packet to a host with no live link is dropped where the Crossbow drops it. The
  Origin 200 has no Crossbow, so its host is not gated.

Build the upstream from `Hub::host_port(widget)` for every Hub the Bridge can
reach, never with a fixed host. `Upstream::get(widget)` is one indexed load, so
nothing is searched per transfer.
