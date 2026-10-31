# Octane memory past 4 GB takes a 16 GB ppmem window, 256 MB per bitmap bit

Status: implemented 2026-10-31.

RAM runs from 0x2000_0000, so 8 GB ends at 0x2_2000_0000. The 4 GB window held
only the first 3.5 GB; the rest took the bus and its code never reached the JIT.

The bitmap must stay one u64 (the JIT tests it on every guest load and store), so
a larger window means larger regions, and that cannot be global: small IP22 banks
would no longer cover a region. So the shift is per machine:
`PpMemSpace::over_window`, `ppmem::window_shift`, `Physical::with_window`.
`HeartRam::try_new` picks 16 GB when RAM_BASE + fitted memory passes 4 GB. Machines
that stay under keep 4 GB and shift 26.

Where the shift lives: the shadow cache and decode cache hold it
(`set_tcache_shift`, before the CPU runs); the JIT bakes it from
`JitDcGeometry::window_shift`, which is in the persistent cache's fingerprint.

`PfnMap` is sized per machine from its `MemoryMap` (`frames_for`): about 8.5 MiB
at 8 GB, 768 KiB on an Indy. Frames above the map's end get no page and run in
the interpreter (`pfn_of`). `Pfn` stays u32: a
40-bit address has a 28-bit frame. `page_base`/`phys_base` are u64; the analyzer
still takes u32 because it only uses the offset in the page.

Regions not wholly covered (a small bank at the top) take the bus. That path is
correct, including SC: see [mp-llsc](../cpu/mp-llsc-is-a-host-compare-exchange.md).

`j2` subcommands taking a physical address still refuse anything past HIMEM_END.
