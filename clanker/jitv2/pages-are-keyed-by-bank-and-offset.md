# A code page is keyed by bank:offset, not by physical address

Status: current (2026-11-19, issue #779).

## Rule

`PhysicalCodePage.key` is a `CodeKey { bank, page }`: the device that holds the
bytes (`CodeBank`, an id from one atomic counter, taken by every RAM bank, PROM
and flash part when built) and the 4 KiB page within it. The bus answers
`BusDevice::code_page(addr)` with the key, the bank's own generation counter
and, for RAM and PROM, a pointer to the page's words in the bank's storage.
Call it with the start of the page.

- Never take a page's counter from `gen_ptr` on a `Physical`: with ppmem that
  is the *window's* counter for the address, which a remap points at another
  bank. `code_page().gen` is the bank's own.
- A bus that forwards `gen_ptr` should forward `code_page`. Where the bus
  answers `None` the page is keyed by address in its space
  (`CodeKey::at_address`, counter from `gen_ptr`), which is what every page
  was before: it compiles, but a remap does not follow it and other nodes do
  not share it. That is a fallback for MMIO, not a place for real code: every
  device code runs from sits behind `BusDevice`s, so forward the question.
  The Bridge did not (the Octane's PROM flash is on its chip select 0, the
  BaseIO flash likewise), nor did it forward `gen_ptr`, so on main that code
  ran against a counter that never moved. An earlier draft returned "not
  code" for unnamed pages and the Octane PROM ran interpreted: its 2 GB
  memory clear took minutes.
- A test bus with no banks answers `CodeKey::flat_page(addr, gen)` (bank 0).
  Two devices behind one bus must not both do that for the same address.
- The key assumes the bank offset of a page start is page-aligned.

## What sits on top

- `Jitv2::by_key`: key -> slot, pre-sized, touched only on a frame-table miss.
- `Jitv2::spaces`: one `PfnMap` (frame -> slot) per address space, the
  lock-free page-switch path. A cache; `forget_addresses` empties it.
- `PhysicalCodePage::addr[cpu]`: frame (and uncached bit) each CPU of the pool
  last reached the page at. Display, and the bus read below. May be stale
  after a remap; nothing trusts it without re-asking the bus for the key.
- The executor keeps its own `cur_code_pfn` + `pcp_uncached` for the
  same-page test; it never compares against the page.
- Compile read (`comp::snapshot_page`): bank storage when `page.words()` is
  not null; else `CodeBuses::locate` (a CPU's bus + address, key checked
  before and after the read).

## Remaps

`World::layout_changed` sets `REQUEST_JIT_REMAP` on every CPU; each, before
running guest code again, calls `jitv2_layout_changed`: clears its nanotlb and
`pcp`, its space's frame table and its `addr` slots. No flush, no world stop.
Measured on PROM memory sizing: Octane 2 CPUs 94 requests, Tezro 4 CPUs 16,
Indy 12, all with 0 mega-flushes (each was a full flush before).

## One pool per machine

`jitv2::share_pool` (IP27, IP35 family) puts every CPU on one `Jitv2`: CPU
numbers in the pool are dense (not the board's CPU ids), each node is a space.
Nodes with different `CpuModule`s get separate pools: `JitConsts`, the ISA
level and the D-cache geometry are per pool.

## Open

- An address-keyed page survives a remap with its code. Remaps move RAM
  banks, which are always bank-keyed, so nothing known breaks; a controller
  that swapped two unnamed devices would.
- Flash parts (`CfiFlash`, bedrock `Flash`) hand out no storage pointer (their
  `Vec<u8>` lifetime and mode-dependent reads were not audited), so their
  pages still read through the bus.
- `POOL_CPUS` is `smp::MAX_CPUS` (4).
- `j2 status` prints `shared pages`: pages more than one CPU reached, and
  how many at different addresses. IRIX replicates kernel text per node, so
  each node's CPUs run their own copy (different banks, no sharing there);
  what two nodes share is user and library code.
- Seen once in five Tezro boots, on the bus-read build: all CPUs spinning in
  the PROM near 0xc00000001fc33614 with 67 pages claimed. Not reproduced, not
  yet compared with main.
