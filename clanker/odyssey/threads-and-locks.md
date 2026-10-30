# VPro threads, the lock-free state and the two queues

Status: reference (iris `src/dev/odyssey/{interface,fifo,worker,raster_unit,mod}.rs`,
2026-10-30).

**Why it is shaped this way.** The board decodes and draws behind the
processor: the driver stores more commands and polls registers while a
primitive is drawn. The model lets the processor go on. Where the board's
own units share memory it shares it the same way: the display and the
rendering engine read and write the planes, the lookup RAM and the mode
registers at the same time, with no lock or borrow between them. Software
orders them with buffer swaps, fences, query sync words, flag bits and
waiting reads, and the model keeps exactly those. A synchronisation fault
in a guest, or in the model, shows as a flash of wrong planes or colours
for a frame, as it does on the hardware; a persistent fault means a
behaviour is not modelled. Do not add a lock or a borrow to hide one.

## Chips and threads

Each thread stands for a chip, or a part of one, of the board:

| Thread | Hardware | Grade and source |
| --- | --- | --- |
| CPU thread | the processor's widget accesses | |
| `odyssey-work` | BUZZ front end: command parse, register writes, transform, lighting, primitive setup | BUZZ as the one transform and raster chip: established (`007-4271-001` p. 5, `108-0314-003` p. 1) |
| `odyssey-raster` | BUZZ raster unit: raster, texture, shading, framebuffer writes | one raster unit: inferred (US 6,259,460) |
| `odyssey-display` | PB&J back end: retrace, colour map, cursor, scanout | PB&J as the display back end with no drawing role: established (`108-0314-003` p. 1) |

The command FIFO lives in board SDRAM. Its depth, in any unit, is unknown;
the model gives it 32,768 words,
a model choice independent of the `0x6b1` the kernel stores.
The split between the two BUZZ threads follows US 6,417,858, which puts an
output FIFO between the transform engine and the rasteriser (inferred to
apply to BUZZ). Nothing models more than one raster unit.

## What crosses each queue

- **Command FIFO** (`fifo.rs`, CPU to `odyssey-work`): the processor's
  stores to the two command-FIFO ports and the context-switch stores, each
  as (offset, size, value) with its credit state. The level the guest sees
  is in words, against the model's 32,768.
- **Raster queue** (`raster_unit.rs`, `odyssey-work` to `odyssey-raster`):
  jobs, each owning what its raster work reads, captured when the front
  end set it up. A GL primitive is a `GlDrawJob`: its raster rules, draw
  state, vertices, texture chain and the texture tables it uses. The
  raster unit hands a drawn one back through a small return ring and the
  front end fills it again, keeping its vertex buffer and texture chain,
  so a draw allocates nothing (allocating per draw on one thread and
  freeing on the other had both contending for the allocator). The rest
  are closures: a GL clear's or pattern rectangle's raster ops; a fence's
  address and token; a query's snapshot and sync word. This encoding is
  the model's own: the hardware's is unknown. Capturing state at its place
  in the stream is how raster state reaches the raster unit in order.
- **The planes** stay where `Board` keeps them. The raster unit owns them
  while it has jobs and reaches them through raw spans, as the IMPACT
  model's engines reach their pixel memory (`clanker/mgras/DESIGN.md`). The
  front end's own plane work (integer and image drawing, pixel DMA, the
  object store, reset) waits for the raster queue to empty first
  (`Board::raster_wait`). That wait is the command stream's order, not a
  lock: the board's pipeline runs a plane access behind the drawing before
  it. A DMA into an object that already has its rows does not wait: its
  texels go to the raster unit as a job, behind the draws that may still
  read the old ones (`Board::object_stage`). A texture upload otherwise
  drained the whole raster queue, up to 180 ms of drawing in Smoke &
  Mirrors.
- **The display** takes nothing from any of them. It reads the planes
  through raw pointers (`PlaneView`, reader.rs), as the MGRAS, REX3 and GR2
  displays read theirs. See "The display" below.

Both rings are the shared `GFifo`, 64K entries. The IMPACT model's raster
FIFO has a head per raster engine with scanline interleave; the VPro's has
one head, as the board has one raster unit.

The front end is bounded by estimated work, not jobs. Each job carries a
cost in pixels (a draw's area from its vertices' window positions, a
clear's or block fill's pixels at an eighth, plus a small fixed overhead),
and the raster queue keeps their running total. A front end that finds
more than `RASTER_WORK_BUDGET` (2^17 pixels, a few milliseconds of shaded
fill) queued waits on a condition variable until the raster unit has
worked it down to half, and a stop wakes it. Smoke & Mirrors queues up to
about 11,700 small triangles, so a bound of 256, 1,024 or 4,096 jobs made
the front end, and through the full command FIFO the processor, wait for
the raster unit, and cut mirrors from about 4 to 3.0, 3.2 and 3.6 swaps a
second; perfly did not change.

## Locks and who owns what

There is no board lock and no raster lock. As on the other iris boards
(GR2's `regs`, `re3` and `hq_engine`, MGRAS's `eng` and `rss`), the state of
a unit is an `UnsafeCell` owned by the unit's thread, and a processor thread
touches it only after the unit's FIFO has drained, with the contract written
at the `unsafe impl Sync` (`BoardCell` in `mmio.rs`, the raster queue's
`ctx`). Unlike GR2, which has one writer, VPro supports two processors
(`[machine] cpus = 2`), so the processors take a small lock against each
other, as MGRAS's producers take its `submit` lock (below).

- **`Board`** (everything but the host registers, `BoardCell`) belongs to
  the front end's thread while it runs stores. A processor's access reaches
  it by one of three ways: a store goes into the command FIFO
  (`CommandFifo::push`); an access that must see the drawing is answered
  busy by the bus until the FIFO and the raster queue are empty
  (`BoardLocks::bus_read`, `bus_write`, rule 2 below), and then runs on the
  processor's thread with the front end idle; and a few accesses that do
  not wait (the level word, the cursor and back-end read windows, an
  unmodelled offset) read whole words the front end writes only as whole
  words or atomics, and show the old or the new value (`BoardCell::get` is
  `unsafe`, with its contract; it is visible to the board's module only).
  With the front end stopped (tests, a stopped machine) the caller runs
  what is queued (`Board::work_drain`); with the front end running it runs
  nothing.
- **The processors' lock** (`BoardLocks::submit`) serialises one processor's
  access against the other's, from its busy check to the end of the access
  and across a command push, so two processors cannot both pass the check
  and borrow the board, or wake the front end under each other. The front
  end's thread and the raster thread never take it, so it adds no wait for
  drawing. A processor access to a host-interface register does not take it.
- **`HostInterface`** is behind the front lock, a register file's own
  lock held for a few instructions: identity, control, the interrupt bank,
  the routing, the swap bank and the FIFO status, plus the interrupt state
  and the credit return. A front access (those registers) takes only it
  and never waits for a draw. It is a leaf; the raster thread takes it for
  a fence's interrupt, and the display takes it to latch swaps.
- **The raster unit's state** (`RasterCtx`: the stipple phase and the rows
  drawn) belongs to the raster thread while it runs. The front end runs a
  job itself only with no raster thread (tests, a stopped machine); the
  raster thread can still publish its rows after `pending` falls to 0, and
  nothing else touches the state then.
- **The display** takes no board or raster state and waits for no drawing.
- What remains are the queues' own locks: the command ring's, the ring
  consumer's accounting against reset (`CommandFifo::pop`), the raster
  queue's wake and idle condition variables, and the thread handles.

A guest that stores to a register while it waits on a drain, or two
processors that program the board without their own ordering, can race each
other's register values (the processors' lock keeps the borrows apart, not
the guest's order). The hardware has no interlock there either; a fault shows as a
wrong frame or a lost register store, not as a model that hides it.

## The display

The display thread (`Odyssey::frame`) polls the retrace, stages, resolves
and presents with nothing held against the board. A frame reads:

| What | How it is shared | Ordering it relies on |
| --- | --- | --- |
| Colour, id and overlay planes | raw pointers (`PlaneView::copy_row`), the one function that reads memory a writer may be storing to | the writer marks a row after its last store; the frame takes the marks before it copies |
| Dirty and unnamed rows | atomic bitsets (`RowMarks`) | store, then mark; the frame takes the marks first |
| Rows the raster unit drew | atomic words (`PublishedRows`) | published before a fence, query or sync word the guest can see, when the queue drains and at the end of a slice |
| Lookup RAM, display modes, overlay modes, cursor | atomic words (`SharedRegs`) and a generation | the store marks every line after it, and the frame copies the registers after it takes the marks; the generation is read before the copy |
| Scanout mode, retrace restart | atomic words (`ModeCell`) and two counters | counters move after the fields and are read before them |
| Swap latch, displayed-buffer bits, retrace interrupt | the host interface, under the front lock | see below |

Sync points kept, each because the hardware has it:

- **The buffer swap.** The swap arm is latched at the vertical retrace, and
  only the display thread latches. The displayed-buffer bits are read once
  per frame, after the latch, and used for the flip, the lines staged and
  the resolve. A guest that draws into the buffer that is not on the screen
  and arms the swap after its fence never shows a mix of two frames. The
  host-interface lock only serialises the register file the swap bank lives
  in; the display holds it for a few instructions.
- **Fences, queries and sync words.** The raster unit publishes the rows it
  drew before it runs a completion job, so a guest that waits for one before
  it arms a swap has its rows known to the display when the swap latches.
- **Waiting reads.** A register read that waits drains the command FIFO and
  the raster queue (rule 2 below).

What is not kept, because the hardware has none: a frame can copy a row the
raster unit is drawing, a lookup entry the front end is storing, or the old
and new halves of a mode change. The writer marks after it stores, so the
next frame copies the row or entry again. A guest that arms a swap before
its drawing is done sees the half-drawn frame for as long as the queue
takes, as it would on the board.

## The front end's run

The worker parks (`park_timeout`, 50 ms, and an unpark from a store that
joins an empty queue), waits `GAP` (100 us) so a burst of stores lands
first, then runs the queue until it is empty. It needs no time slices:
nothing waits for it to let go of the board (the `slice` flag and the
`odyssey-work` `slice` trace event mark a run). Devices start before the
processors, so no processor store is in flight when `work_start` sets
`running`. A processor access that must see the stores is told busy and retries,
taking its interrupts meanwhile, so the guest's 10 ms clock tick is never
held behind a long draw. A run always runs at least one store.

## The command ring

The PROM's and IRIX's textports send hundreds of thousands of FIFO words per
scrolled line, so a store must cost little. Stores go into iris's shared
`GFifo` (`src/gfifo.rs`, 64K entries; `fifo.rs` wraps it). The processor
pushes without the front lock and wakes the worker only when it sleeps. With
two processors (`[machine] cpus = 2`) a second producer pushes the same way:
the ring's own lock orders them, with no lock of ours above it. A push that
loses that lock to the other processor, or finds the last place taken between
the room check and the push, is refused, and its store retries the push behind the thread's run
(`BoardLocks::write_command`), not around it. The ring is deeper than the board's FIFO, so a second producer's
overshoot of the level by a store is absorbed. The consumer (the worker, or a draining access with the worker stopped) looks
at up to 32 stores, runs them, and takes the ones it ran out together
(`CommandFifo::take_run`, `retire`): the level, the store count and the
credits move once a run, not once a store, because each store costs the
worker little and the processor reads those words. A store that has run but
not retired still counts in the level, so the level never reads below what the
board has left to take. A push counts its store (level, store count, counted
stores) before the ring shows it and takes the counts back off if the ring
refuses it: the worker can take a store and retire it the instant it lands,
before the push returns, and a retire must find every store it takes already
counted or the level goes below zero. While a counted store waits a run is one store.
While the worker runs every command store queues, so decoding and drawing
never run on the processor's thread.

The guest sees the model's FIFO, not the ring: 32,768 words, a model choice
whatever the kernel's `0x6b1` set-up stores. The level is the words waiting, a doubleword store counting
two (also a model choice). STATUS0 bit 16 is empty, bit 17 below the model's
16-word low-water threshold, bit 15 at or above the stored `0x6b2` (flags bit
25 rises on reaching it). Which register is a high-water mark is unknown:
applying `0x6b2` there is a model choice. A processor store is refused with the bus busy
(`BoardLocks::bus_write`) only when the 64K-entry host ring is full, as
IMPACT's and GR2's are: the processor retries the instruction, taking
interrupts meanwhile, and the board's thread, which then starts its next
run without the pause, makes room. Past 32,768 words the guest reads a
full FIFO (high-water bit up once a `0x6b2` is set) while its stores still
succeed.
A caller other than the bus, with the board's thread stopped, still runs
the queue and then its store. The flow-control target word `+0x1074` is stored, not routed, so
a store to it goes in at once even while the FIFO holds work.

Counted stores (machines whose processor interface counts graphics credits;
the Octane has none and the calls are no-ops) are credited as they enter the
FIFO's first 16,384 places (the model's depth, a model choice), counted in
stores: at the store when there is room,
or when the stores ahead have run, with the node holding the credit as owed
meanwhile.

## The FIFO ordering rules

These are guest-visible, and hold across both queues; any change to either
must keep them.

1. One in-order stream: stores run in arrival order, and raster jobs in the
   order the front end set them up.
2. A register read that waits (`access_waits`) drains the command FIFO and
   then the raster queue, so it sees the drawing of every store before it.
   The memory word `0x106c` never waits (`level_word`): its fields are
   straps and FIFO levels, none a drawing result. The kernel reads it at
   every flow-control attach, which each GL make-current runs; waiting
   there stalled the processor behind the drawing of each frame. A waiting
   store is held back for the command FIFO only (`BoardLocks::bus_write`);
   the raster unit never shared state with it, and a store that touches the
   planes waits for the raster queue itself (`Board::raster_wait`). The
   registers that do not wait answer from the front at once. On the bus
   (`BoardLocks::bus_read`, `bus_write`) such an access answers busy while
   that work is still queued and the board's threads run: the processor
   retries it, taking interrupts, and the board's threads run the work,
   as IMPACT's waiting reads do. Callers other than the bus, with the board's thread stopped, drain it
   themselves.
3. STATUS0's empty, low-water and high-water bits follow the words in the
   command FIFO, not the raster queue: they are the board's FIFO.
4. The low-water flag latches as a store retires below the model's threshold (16 words),
   whatever the enable mask says; it reaches the driver only while enabled.
5. Completion words are written when the raster unit is done with the work
   before them: fences and a current-state query's publication (snapshot,
   counter, sync word) run as raster jobs; the context-switch done word
   waits for the raster queue to empty.
6. Stopping the machine finishes the command FIFO and then the raster queue;
   a reset abandons the waiting stores and waits for the raster queue before
   it clears the planes.
7. The display waits for nothing: it copies what is in the planes and the
   registers when the frame runs. Rows and entries it copies half-written
   are copied again at the next frame.
8. The back end's display-FIFO level (`0x106c` bits 6:0) stays 0: back-end
   writes apply when the front end runs them.

## Tracing it

`docs/perfetto.md`: `VPro cfifo level` is sampled at run edges and STATUS0
reads, not per store. The `slice` slice on `odyssey-work` against `drain` on
the CPU thread shows who is running the commands, and `raster slice` on
`odyssey-raster` who is drawing. A long `raster wait` on `odyssey-work`
points at front-end plane work behind queued drawing; a long `slice` with
the CPU answered busy on a read, at the front end still running stores.
