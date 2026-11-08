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
- A page with no memory behind it holds nothing: the line stays in the arrays, as
  a tag test's patterns do. The region bitmap is all a CPU tests, so a held page's
  whole region is the view's while it stands, reading zeros wherever nothing is
  mapped. The O2's POST holds lines for page 0 before CRIME maps the rest of that
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
shared. ppmem puts `GEN_HELD` on the page's generation counter at the first hold
and takes it off at the last release, moving it with the page on a remap
(`PpMemSpace::mark`). The compile that finds it publishes nothing and denies every
offset until the release. ppmem's counters are the bus's for memory; a machine whose bus
maps a device over memory must mark that device's counter as well.

## Seeing it

`l2 probe 0` in the monitor prints how many lines hold pages and which. Every
machine should show none at the IRIX login. `log l2c mask op` logs each page held
and released.
