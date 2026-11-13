# Cache lines live in the CPU's own view of memory

The R10000-family PROMs use the caches as memory before memory works. They make
lines valid with Index Store Tag and Index Store Data and then load, store and
fetch through them, while uncached accesses, which bypass the caches (NEC
VR10000 user's manual p. 75), and every other master see memory. What the shipped PROMs do on iris:

| Machine | Lines | Memory behind them |
|---|---|---|
| O2 | secondary and primary data lines for page 0 and 0x1000 hold POST's stack for about 50-65 ms, while POST tests page 0 uncached. On the R5000, primary data lines for page 0 (TagLo 0xc0) | yes |
| Octane | a tag and data RAM test on secondary set 0, read back only with Index Load Tag/Data | no: the patterns name no memory |
| IP28 | a secondary line for page 0, once while the PROM starts, then released | yes |
| Fuel | primary data lines for 0x100000-0x107fff hold the stack | no, until memory is configured mid-phase |
| Origin 200 | none held in a boot to the IRIX login | - |
| Indy, Indigo2 (R4400) | none: an R4400 line never holds a page | - |

In these runs, measured while only Index Store Tag ended lines, every line ended
by Index Store Tag, none was written back, and none was read back with Index Load
Data while it held a page. The invalidating operations end lines too now (see
Limits); a PROM that invalidates a line before rewriting its tag releases it a
little earlier and writes it back if it is dirty.

## How

- Each CPU reads memory through a window of its own (`PpMemSpace::add_cached_view`).
  Every mapping laid in the shared window is laid in each view too, and the view's
  region bitmap goes to that CPU's inline fields. The shared window, which
  `Physical`'s direct paths, CRIME's rendering engine and HEART read, never holds a
  line.
- When Index Store Tag makes a line valid, its page becomes private memory in that
  view, a copy of memory, and the line's data array is written into it
  (`ShadowCache::line_enters`). An invalid or different tag ends the line, as
  the invalidating CACHE operations do (`ShadowCache::invalidate`), and the
  page goes with its last line. The shadow's tags stay the only line state, with a
  held bit beside each. Index Store Data into a held line writes through.
- The R5000 does the same for a valid primary data line, whose tag has the
  R10000's PTag and PState fields (NEC VR5000 user's manual U11761EJ6V0UM p. 133).
  It has no Index Store Data, so the page starts as memory. Its instruction lines
  hold nothing, since one could hold only what memory holds.
- A page with no memory behind it holds nothing, and the line stays in the arrays,
  as a tag test's patterns do, unless the machine allows lines there
  (`PpMemSpace::allow_unbacked_lines`, as the Fuel does for its bank slots). The
  region bitmap is all a CPU tests, so a held page's whole region is the view's
  while it stands, reading zeros wherever nothing is mapped. The O2's POST holds lines for page 0 before CRIME maps the rest of that
  region.
- A remap saves and restores held pages, so memory configuration mid-phase keeps
  the stack.
- Loads, stores, fetches, the decode cache and compiled code are untouched: they
  read the view through the base pointer and bitmap they always had. The decode
  cache compares every word it serves, so it sees a page change hands.

## CACHE operands

On the R10000 family every CACHE operand but Cache Barrier's is translated as a
data read before the operation (NEC U12754EJ1V0UMJ1 pp. 70, 72), index
operations included: a mapped operand can take TLB Refill, TLB Invalid or a
privilege Address Error, never TLB Modified. The primaries index by the virtual
address; the secondary takes its set and way from the physical one
(`MipsCache::set_cache_op_pa`), so a line made valid through a mapped page
holds the frame's page. The kseg0, xkphys and ERL kuseg operands the PROMs use
translate without a fault and to the same set as before. An operand on an
uncached page is undefined on the R10000 and goes ahead as if cached.
op[1:0] = 10, a secondary instruction cache the R10000 does not have, does
nothing. The R4400 and R5000 translate hit operations only.

## Limits

Each is an approximation, stated where it is made.

- Page, not line: the rest of a held page shows memory as it was when the page was
  taken, uncached writes to it meanwhile are not seen by cached reads, cached stores
  to it outside a line that is written back are dropped at release, and a line
  that ends leaves its bytes until the page's last line goes. The page is the host
  granule: 4 KB on Linux and x86 macOS, 16 KB on Apple silicon, 64 KB on Windows.
- A line ends by Index Store Tag; by Index Writeback Invalidate (D) zeroing the
  tag from the R12000 on; by Index Writeback Invalidate, Hit Invalidate and Hit
  Writeback Invalidate (S); and by Index Invalidate and Hit Invalidate (I) (NEC
  U12754EJ1V0UMJ1 pp. 17-19, 25-27). An invalidation sets the tag's state to
  Invalid, for every valid line, held or not, and keeps the rest of the tag, which
  Index Load Tag then reads; the manual also rewrites the tag from the operand and
  regenerates its check bits, which the shadow does not. An invalidated secondary
  line invalidates the primary instruction lines inside it; one Index Store Tag
  ends does not. The primary data cache's hit operations, and its Index Writeback
  Invalidate on the R10000, leave a line untouched.
- Only Index Writeback Invalidate (S) and Hit Writeback Invalidate (S) write back,
  and only a held line: its bytes go from the page to memory before the page is
  released, when its tag is DirtyExclusive or its bytes differ from its data array.
  Stores go straight to the page and keep no dirty state, and only a cached store
  changes the page without the array (Index Store Data writes both), so the
  compare stands in for the DirtyExclusive a store would make (NEC U10278EJ4V0UM
  p. 246). A store of the bytes already there is missed. Writing every held line
  back instead would put a clean line's staged bytes in memory where hardware
  leaves memory alone. The compare is one line, at the rare CACHE instruction
  that ends a held line; loads and stores pay nothing. A line that holds no page
  names no memory and is not written. Every other end drops the line unwritten.
- Index Load Data reads the data array, not the page, so it does not see cached
  stores.
- A region holding a page reads zeros in its unmapped gaps for that CPU's cached
  accesses, where the bus would answer.
- A secondary doubleword puts TagLo at the lower address. The manuals say only
  that TagHi is the high word and TagLo the low word (MIPS R10000 UM p. 201, NEC
  VR10000 UM p. 289); which address each pairs with is inferred. The IP35 PROM,
  6.210, stages code with the word at A in TagLo and the word at A + 4 in TagHi,
  and its staged code equals its image only this way. The IP35 boot-flash PROM's
  staging pairs the words the same way: simulated with the other order, 378 of its
  562 staged fetches break, every trampoline fetch among them.
- No replacement, parity or ECC, and no coherence with other CPUs or DMA.
- A snapshot taken while lines hold pages is refused at restore.

## Compiled code

A page some CPU holds is not compiled: compiles read memory, and compiled code is
shared. Each CPU's view puts its own `GEN_HELD` on the page's generation counter
at its first hold and takes it off at its last release, so no CPU reads another's
view. ppmem marks a page only while memory backs it, so one held with nothing
mapped behind it carries no mark until memory arrives. A remap takes every view's
marks off and puts them back on the counters that back the pages now
(`PpMemSpace::mark`). The compile
that finds one publishes nothing and denies every offset until the last view lets
go. ppmem's counters are the bus's for memory. Where the bus maps a device over
memory, as the IP35 boards map their reset PROM, the compile checks that device's
counter instead, so each CPU's cache puts the same mark there when a page becomes
held in its view and takes it off with the page's last line (`mark_foreign`). That
work is at CACHE instructions only; a load, store or fetch is unchanged.

## Primary instruction lines

Only the Tezro's PROM uses these. A valid primary instruction line
writes each 36-bit predecoded word (p. 288) into its page as the instruction it
predecodes, through an eleven-class table inferred from PROM 6.210's words
(`l1i_decode`). A word of another class leaves memory's word, where hardware would
run it. A page held only for code also answers cached loads with code.

The model never refills on a fetch. The PROM runs a staged secondary line
and reads the primary words back with Index Load Data (I) to stage them itself, so
Index Load Data (I) in a set a held secondary line maps to answers with that line's
words predecoded (`refilled_word`), until Index Store Data (I) writes the set. When
two held lines map one set, the later wins. An instruction no class holds comes back
in the model's own form (bits 35:32 set, which takes classes 0x78-0x7f), which
hardware would not produce. Index Load Tag (I) reads the stored tag.

## Seeing it

`l2 probe 0` in the monitor prints how many lines hold pages and which. Every
machine should show none at the IRIX login. `log l2c mask op` logs each page held
and released.
