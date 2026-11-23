# IP27: the second processor of a Hub node board ran into zeroed memory in the PROM

Status: resolved 2026-11-22. `[machine] cpus = 2` runs on the Origin 200, Origin 2000 and
Onyx2.

## Symptom

One or two nodes with both slices populated printed nothing. Slice A sat in a
pair barrier on PI pending bits 49/48; slice B ran zeroed words in the PROM's
RAM-copy mapping (ASID 2, 0xc000_0000_1fc0_0000 onto 0x1a0_0000) until it
took a TLB refill at 0xc000_0000_1fe0_0000.

## Causes

1. B had taken an address error fetching 0xffff_ffff_ffff_ffff (a `jr ra`
   whose `ra` came back all ones) and entered POD mode, which maps the RAM
   copy before A has made it. The stack `ra` came from is the PROM's early
   stack near 0x103000, kept in cache lines each slice tags itself.
   Both slices use the same addresses. With `connect_cpu(.., false)` their
   lines were shared memory, and A's memory test overwrote B's stack. The
   IP27 processors now keep their lines in their own cached view, as the IP35
   ones do (`clanker/cpu/cache-lines-live-in-the-cpus-view.md`).
2. Under host load a boot then often stopped with each slice having disabled
   the other. PROM 6.156 (the routine at 0xc000_0000_1fc5_878c) posts code
   0x24 to its own PI_RT_COMPARE, reads the other slice's for it at most
   200,000 times, and on a miss clears the other's PI_CPU_ENABLE ("The other
   CPU on this node died"). Threads that run 100-200 ms apart miss it. The
   Hub PI now holds the polling slice (`Pi::read_compare`), as
   `src/ip35/presence.rs` does for the IP35 PROM's same check. At host load
   about 250, 6 of 12 two-node, four-processor boots hung this way without
   the hold; 24 of 24 reached the menu with it (a separate run from the
   counts under 3).
3. After the match, PROM 6.156 has slice A store its liveness bit in the I/O flag
   word (IIO_SCRATCH_REG0, Hub 0x400150, at 0x1fc58898-0x1fc588b8) while slice B
   only waits 1 ms (a delay routine at 0x1fc588c4); B's next barrier
   (0x1fc58690-0x1fc586a8) takes a clear bit as "A is dead", and B then runs as
   a second local master beside A. The 1 ms is on the Hub real-time counter,
   which runs on host time, so a slice A descheduled for a millisecond
   loses. The two slices then disagree about the barrier they are in; the
   visible results were a "took an exception! Disabling and continuing" on the
   node, or a hang after "Copying PROM code". Slice B is therefore also held,
   after the match, until slice A stores the flag word. This is a software
   expectation from the listing, not from hardware. With only the first hold,
   4 of 8 pinned four-processor boots failed this way; with the second step,
   0 of 60.

The hold (`Pi::poll_read`, `Pi::flag_word_stored`) is an approximation: the
PROM relies on both slices running concurrently, which the hardware does and
host threads do not. It ends when the slice waited for has retired 2 billion
cycles without doing its part (`ip35::presence::HOLD_CYCLES`), or after 120 s
of machine time. It does not cover a partner that is itself parked for long,
and it is shaped by this PROM's two steps; another PROM revision's pair check
would need its own look. A mutual deadlock (each slice polling before it
posts) does not arise on 6.156, which posts first; it would be ended only by
the machine-time bound.

## Not this bug: a node board missing under load

Boots pinned to two host cores each, two at a time, sometimes reach the menu
with "2 CPUs on 1 nodes found" (or wait at "waiting for node ... at global
barrier" past the harness limit): the global master stops waiting for the
other node board. It happened in 3 of 30 boots of the two-node, one-processor
machine, which has no pair and no hold, and in 3 of 60 of the four-processor
one. Both are the PROM's host-time waits meeting starved threads, not the pair
check, and the Hub PI hold does not touch them.
