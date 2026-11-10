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

Sending the window elsewhere takes away the ppmem mapping, so every cached
reference of that node takes the bus path while its number is nonzero. A node
that holds zero, and any board that routes nothing, never leaves the fast path.
Do not add a decision to the access path for this: the rare state changes which
mapping is installed, and nothing is tested per access.

The cost of the bus path for the second node is measured in
`docs/origin2000-ip27.md`, not assumed.

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
