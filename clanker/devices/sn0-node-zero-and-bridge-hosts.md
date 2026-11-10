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

A cabled board's window spans the node numbers up to the next power of two
above its node count (8 GiB for two nodes), reserved as inaccessible address
space that commits nothing. A number past it stays on the bus path. Its
one-word region bitmap, the test on every load, store and fetch, counts 64
regions of 128 MiB, numbers 0 and 1. A region it does not hold whole, a 64 or
256 MiB bank leaves half of each, or another number's, is found in the window's
fine table (`PpMemSpace::fine_table`, one bit per 64 MiB) on the bitmap's miss
branch, before the bus. The decode cache keeps window pointers for both, so it
keys them on a value the space moves with every change of either
(`add_key_sink`), not on the bitmap. Do not add a decision to the hit path for
any of this: the rare state change lays a different mapping.

Limit: a CPU's lines are private to it (`hold_line_page`) only for pages the
bitmap names, its first 64 regions. A page past them returns `None`, so on a
three- or four-node board the lines of node numbers 2 and 3 are not modelled as
the CPU's own, and a store there is visible to the other CPUs at once. Numbers
0 and 1, which the stock PROM gives, are covered. Nothing observed depends on
it.

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
