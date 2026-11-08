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

Every line ends by Index Store Tag. None is written back, and none is read back
with Index Load Data while it holds a page.

## How

- Each CPU reads memory through a window of its own (`PpMemSpace::add_cached_view`).
  Every mapping laid in the shared window is laid in each view too, and the view's
  region bitmap goes to that CPU's inline fields. The shared window, which
  `Physical`'s direct paths, CRIME's rendering engine and HEART read, never holds a
  line.
- When Index Store Tag makes a line valid, its page becomes private memory in that
  view, a copy of memory, and the line's data array is written into it
  (`ShadowCache::line_enters`). An invalid or different tag ends the line, and the
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

## Limits

Each is an approximation, stated where it is made.

- Page, not line: the rest of a held page shows memory as it was when the page was
  taken, uncached writes to it meanwhile are not seen by cached reads, cached stores
  to it are dropped at release, and a line that ends leaves its bytes until the
  page's last line goes. The page is the host granule: 4 KB on Linux and x86 macOS,
  16 KB on Apple silicon, 64 KB on Windows.
- Lines end only by a tag change the shadow already models: Index Store Tag, and
  Index Writeback Invalidate (D) zeroing the tag from the R12000 on. Nothing is
  written back. The hit operations and the other index invalidates leave a held
  line as they leave every shadow tag, untouched.
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
