# The TLB decodes EntryLo C with the CPU model's table

*Fixed 2026-10-28. Supersedes T2 in
[cache-attributes-and-fetch-alignment.md](cache-attributes-and-fetch-alignment.md).*

**Symptom.** On the Octane every user-mode fetch, load and store went to the
bus: no decode cache, no ppmem window, no jitv2 inline path. Dhrystone ran at
111k/s against 417k/s on the IP28 with the same binary; a profile of the CPU
thread was topped by `Physical::read32` and `BankView::read32` from
`fetch_instr_impl`'s uncached branch.

**Cause.** `ShadowEntry::from_entry` had a model-agnostic table that decoded
only C = 2, 3 and 5 and read everything else as uncached. That table had been
widened once ("every C but 2 is cacheable") and narrowed again because 0/1 and
4/6 mean opposite things on the R5000 and the R4400. The Octane's IRIX maps
every user page C = 4 (cacheable coherent exclusive) and reaches memory through
xkphys `0xa000...` (also C = 4); xkphys had its own `3 | 5` match. The IP28 and
the Indy use C = 3, so only the Octane paid.

**Rule.** `MipsExecutor::decode_cache_attr` is the one per-model table (R4400,
VR5000, R10000). `MipsExecutor::new` installs it in the TLB
(`Tlb::set_cache_attrs`, from `cache_attr_table()`), and xkphys calls it
directly. `MipsTlb` keeps a 2/3/5 default only until the executor installs its
table. Do not add another C decode: route it through `decode_cache_attr`.

Tests: `test_a_mapped_page_uses_the_models_cache_attr_table`,
`test_xkphys_uses_the_models_cache_attr_table`. The first clears Status.ERL:
out of reset kuseg is an uncached identity window, which passes "not cached"
assertions without touching the TLB.
