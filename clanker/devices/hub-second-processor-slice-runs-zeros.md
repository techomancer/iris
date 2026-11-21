# IP27: the second processor of a Hub node board runs into zeroed memory in the PROM

Status: open 2026-11-20. `[machine] cpus = 2` is refused on the Hub machines
(Origin 200, Origin 2000, Onyx2) until this is fixed.

The builders are ready: `ip27::machine` builds `HubConfig::cpus` processors
per node, numbered from the node's first, and gives each its PI slice's
interrupt word. Only `MachineProfile::max_cpus` holds the count at one.

## Symptom

One or two nodes with both slices populated: the console stays empty.

- Slice A sits in a delay loop reading the Hub's real-time counter
  (PC near 0xc000_0000_1fc5_add4 in PROM 6.x), apparently waiting for B.
- Slice B executes all-zero words (NOPs) through the PROM's mapped image
  from about 0xc000_0000_1fc1_aa4c upward, until it leaves the mapping at
  0xc000_0000_1fe0_0000 and takes a TLB refill there, again and again.

So B runs the PROM's relocated copy through a mapping whose memory holds no
code for it.

## Where to look

`src/ip27.rs` lists the approximation that fits: the low 128 KiB of the
Hub-special memory view is not private to each processor of the pair. Check
what B's TLB entry for 0xc000_0000_1fc0_0000 points at, and whether that
physical range is one each slice should see its own copy of.

One processor per node, one or two nodes, is unaffected.
