# VPro threads, the locks and the two queues

Status: design (iris `src/dev/odyssey/{interface,fifo,worker,raster_unit,mod}.rs`,
2026-10-30). Check against the code before relying on details.

**Why it is shaped this way.** The board decodes and draws behind the
processor: the driver stores more commands and polls registers while a
primitive is drawn. The model must let the processor go on, yet never let a
draw and a processor access touch the same board state.

## Chips and threads

Each thread stands for a chip, or a part of one, from the Odyssey datasheet
(origami `docs/datasheets/odyssey.md`, "Chip roles from the vendor
documents" and "Board and kernel"):

| Thread | Hardware | Grade and source |
| --- | --- | --- |
| CPU thread | the processor's widget accesses | |
| `odyssey-work` | BUZZ front end: command parse, register writes, transform, lighting, primitive setup | BUZZ as the one transform and raster chip: established (`007-4271-001` p. 5, `108-0314-003` p. 1) |
| `odyssey-raster` | BUZZ raster unit: raster, texture, shading, framebuffer writes | one raster unit: inferred (US 6,259,460) |
| `odyssey-display` | PB&J back end: retrace, colour map, cursor, scanout | PB&J as the display back end with no drawing role: established (`108-0314-003` p. 1) |

The command FIFO lives in board SDRAM, 32,768 words as the kernel sets it
up (established, `0x6b1`; the porting guide gives about 1 ms of commands).
The split between the two BUZZ threads follows US 6,417,858, which puts an
output FIFO between the transform engine and the rasteriser (inferred to
apply to BUZZ). Nothing models more than one raster unit.

## What crosses each queue

- **Command FIFO** (`fifo.rs`, CPU to `odyssey-work`): the processor's
  stores to the two command-FIFO ports and the context-switch stores, each
  as (offset, size, value) with its credit state. The level the guest sees
  is in words, against the board's 32,768.
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
  while it has jobs and reaches them through raw spans, as the IMPACT model's
  engines reach their pixel memory (`clanker/mgras/DESIGN.md`). The front
  end's own plane work (integer and image drawing, pixel DMA, the object
  store, reset) waits for the raster queue to empty first
  (`Board::raster_wait`). A DMA into an object that already has its rows
  does not: its texels go to the raster unit as a job, behind the draws
  that may still read the old ones (`Board::object_stage`). A texture
  upload otherwise drained the whole raster queue, up to 180 ms of drawing
  in Smoke & Mirrors. The display holds the raster unit's context lock
  while it copies a frame's dirty lines aside, and resolves the copy with
  no lock held.

Both rings are the shared `GFifo`, 64K entries. The IMPACT model's raster
FIFO has a head per raster engine with scanline interleave; the VPro's has
one head, as the board has one raster unit. The raster queue is not
bounded below the ring: Smoke & Mirrors runs up to about 11,700 jobs
ahead of the raster unit. Bounding it at 256, 1,024 or 4,096 jobs made
the front end, and through the full command FIFO the processor, wait
for the raster unit, and cut mirrors from about 4 to 3.0, 3.2 and 3.6
swaps a second; perfly did not change.

## Locks

`Board` (everything but the host registers) is behind the board lock.
`HostInterface` is behind the front lock and holds the registers that do not
wait for drawing: identity, control, the interrupt bank, the routing, the
swap bank and the FIFO status, plus the interrupt state and the credit
return.

- Lock order: board, then front. Nothing takes the board lock while holding
  the front lock.
- A front access (those registers, and command stores) takes only the front
  lock and never waits for a draw.
- Every other access takes the board lock, which waits out the current
  slice, then runs whatever is still queued before it (`work_drain`), so it
  sees all earlier stores. A processor's access on the bus answers busy
  instead while stores it must see are queued (rule 2 below), so the drain
  normally finds nothing to run.
- The display thread takes the board lock, then the raster unit's context
  lock, which the raster thread holds for each slice of jobs (at most 1 ms,
  checked between jobs). A frame therefore waits out a raster slice in
  flight and never reads a half-drawn primitive. It holds both locks only
  while it copies the dirty lines' plane bytes aside
  (`Board::scanout_stage`), lets both go while it resolves the copy
  (`ScanoutJob::resolve`), and takes the board lock again to put the
  lines in the presented frame with the cursor over them. Resolving a
  whole frame under both locks held the raster unit up about 6 ms of
  every 16.7 ms frame while a GL program drew. The retrace is counted
  under the board lock and can be up to one slice late. A swap does not
  repaint the frame: the frame update compares the displayed-buffer bits
  with the ones it last staged and repaints only the rows where the
  flipped windows' two buffers differ (`Scanout::differ`, kept from the
  rows each draw reports and measured from the planes after a writer
  that does not report rows).
- Lock order: board, then raster context, then front. The raster thread
  takes only its context lock and, for a fence's interrupt, the front lock.
  The front end waits for the raster queue to empty with the board lock
  held; the display, which also needs the board lock first, cannot then be
  holding the raster unit up.

## The 1 ms slice

The worker sleeps on a condvar until a store joins an empty queue, waits
`GAP` (100 us, both locks free), then holds the board lock for a slice of
at most `SLICE` (1 ms of host time, checked every 16 stores) and releases it
with `unlock_fair`. The gap exists because a thread that drops a lock and
takes it straight back usually wins it again; without the gap a processor
waiting for the board could wait out many slices. A slice never lets go
part-way: a draw holds `&mut` into the board. 1 ms stays well under the
guest's 10 ms clock tick. A slice always runs at least one store.

## The command ring

The PROM's and IRIX's textports send hundreds of thousands of FIFO words per
scrolled line, so a store must cost little. Stores go into iris's shared
`GFifo` (`src/gfifo.rs`, 64K entries; `fifo.rs` wraps it). The processor
pushes without the front lock and wakes the worker only when it sleeps. The
consumer, whoever holds the board lock (worker or a draining access), looks
at up to 32 stores, runs them, and takes the ones it ran out together
(`CommandFifo::take_run`, `retire`): the level, the store count and the
credits move once a run, not once a store, because each store costs the
worker little and the processor reads those words. A store that has run but
not retired still counts in the level, so the level never reads below what the
board has left to take. While a counted store waits a run is one store.
While the worker runs every command store queues, so decoding and drawing
never run on the processor's thread.

The guest sees the board's FIFO, not the ring: 32,768 words, the size the
kernel's `0x6b1` set-up gives it (datasheet, "Command FIFO level and
credits"). The level is the words waiting, a doubleword store counting two.
STATUS0 bit 16 is empty, bit 17 below the model's 16-word low mark, bit 15
at or above the kernel's `0x6b2` high mark (flags bit 25 rises on reaching
it). A processor store that would overfill the 32,768 words is refused
with the bus busy (`BoardLocks::bus_write`), as IMPACT's full FIFO
refuses one: the processor retries the instruction, taking interrupts
meanwhile, and the board's thread, which then starts its next slice
without the pause, makes room. Before, the storing processor ran the
whole queue itself under the board lock, about 20 ms each time in Smoke
& Mirrors. A caller other than the bus still runs the queue and then its
store.

Counted stores (machines whose processor interface counts graphics credits;
the Octane has none and the calls are no-ops) are credited as they enter the
FIFO's first 64 places, counted in stores: at the store when there is room,
or when the stores ahead have run, with the node holding the credit as owed
meanwhile.

## The FIFO ordering rules

These are guest-visible, and hold across both queues; any change to either
must keep them.

1. One in-order stream: stores run in arrival order, and raster jobs in the
   order the front end set them up.
2. A register read that waits (`access_waits`) drains the command FIFO and
   then the raster queue, so it sees the drawing of every store before it.
   The memory word `0x106c` drains only the command FIFO: its fields are
   straps and FIFO levels, none a drawing result
   (`read_waits_for_raster`). The kernel reads it at every flow-control
   attach, which each GL make-current runs; waiting for the raster queue
   there stalled the processor behind the drawing of each frame. The
   registers that do not wait answer from the front at once. On the bus
   (`BoardLocks::bus_read`, `bus_write`) such an access answers busy while
   that work is still queued and the board's threads run: the processor
   retries it, taking interrupts, and the board's threads run the work,
   as IMPACT's waiting reads do. Callers other than the bus drain it
   themselves.
3. STATUS0's empty, low-water and high-water bits follow the words in the
   command FIFO, not the raster queue: they are the board's FIFO.
4. The low-water flag latches as a store retires below the mark (16 words),
   whatever the enable mask says; it reaches the driver only while enabled.
5. Completion words are written when the raster unit is done with the work
   before them: fences and a current-state query's publication (snapshot,
   counter, sync word) run as raster jobs; the context-switch done word
   waits for the raster queue to empty.
6. Stopping the machine finishes the command FIFO and then the raster queue;
   a reset abandons the waiting stores and waits for the raster queue before
   it clears the planes.
7. The display waits for a raster slice in flight (the context lock).
8. The back end's display-FIFO level (`0x106c` bits 6:0) stays 0: back-end
   writes apply when the front end runs them.

## Tracing it

`docs/perfetto.md`: `VPro cfifo level` is sampled at slice edges and STATUS0
reads, not per store. The `slice` slice on `odyssey-work` against `drain` on
the CPU thread shows who is running the commands, and `raster slice` on
`odyssey-raster` who is drawing. A long `raster wait` on `odyssey-work`
points at front-end plane work behind queued drawing; a long `slice` with
the CPU blocked on a read, at the front end holding the board lock.
