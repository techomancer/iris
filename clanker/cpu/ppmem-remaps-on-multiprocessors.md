# On a multiprocessor a ppmem remap parks the other CPUs and lays around held pages

Status: rule, Nov 2026. The code follows it; `docs/ppmem-design.md` §5.1
("no locking, no stop-the-world") predates it and holds for one CPU only.

**Why §5.1 is not enough.** Its two reasons are that the remap runs inside
the storing CPU's own instruction and that nothing else uses memory while the
PROM sizes it. Both assume one CPU. On an Octane, Origin, Fuel or Tezro with
several, the others keep running while one stores to the memory controller,
and each reads memory through a cached view of its own in which the pages its
cache holds lines for are its private bytes
(`cache-lines-live-in-the-cpus-view.md`). The IP35 PROM sets up every CPU's
caches at once, so CACHE instructions on all of them reach one `PpMemSpace`
while a remap may be due.

**Rules.**

- A whole remap runs with the other CPUs parked: HEART's MEMCFG
  (`ip30_machine.rs`, `remap_world.stop_others`) and Bedrock MD's bank
  configuration write (`dev/bedrock/md.rs`, `world.stop_others`), parked
  before taking `remapping`, which a CPU waiting to park could hold. A new
  board whose memory controller remaps must do the same.
- Every path that changes the space's state takes its `guard`: the remaps,
  `add_cached_view`, `hold_line_page`, `release_line_page`,
  `held_line_pages`. None takes it inside another. No load, store or fetch
  takes it: those read the window through the CPU's base pointer and inline
  bitmap. The contract is written out above `unsafe impl Sync for
  PpMemSpace` in `src/ppmem/ppmem.rs`; keep it current.
- A remap never touches a page a view holds (`free_runs`): it is left as it
  is and the shared window shows what the remap laid. Laying over a held page
  and putting it back afterwards would show memory for an instant to a CPU
  running from that page, and would lose a store made in between.
- A range that goes away is scrubbed to zero pages, never unmapped, so a CPU
  holding a pointer into the window always lands on mapped memory.

**Where this came from.** A downstream tree whose IP35 memory controller
changed pages without parking the others had all four Tezro CPUs in the
space at once while the PROM set up its caches, which is why every entry
that reads or changes the mapping list, the bitmap or the held pages takes
the lock. The same tree first laid a remap over a held page and put the
page back afterwards; the page then showed memory for an instant to a CPU
running from it.
