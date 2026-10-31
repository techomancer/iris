# Perfetto tracing

Build with `--features perfetto` to record traces that open in
<https://ui.perfetto.dev> (or `trace_processor` for SQL). The feature
works with `lightning`, so a trace can be taken at full speed.

## Recording

At startup:

```
iris --perfetto boot.pftrace                         # everything, host timeline
iris --perfetto gfx.pftrace --perfetto-cats irq,gfx  # only some categories
iris --perfetto g.pftrace --perfetto-clock guest     # guest-instruction timeline
```

From the monitor, at any point:

```
perfetto start FILE [cats=LIST] [clock=host|guest]
perfetto cats irq,gfx,-jit      # change categories while recording
perfetto status
perfetto stop
```

Categories select what is recorded. `LIST` is comma-separated names,
`all` for every category, and a leading `-` removes one (`all,-jit`). An
event that belongs to several categories is recorded when any of them is on.

| Category | What it covers |
|---|---|
| `cpu` | Interrupts taken by the CPU (also `irq`). |
| `irq` | IOC/INT2 interrupt lines, the CPU's IP2-IP7 pins, interrupts taken. |
| `timer` | The 8254 slow and fast ticks and Count/Compare (IP7); these are also `irq`. |
| `dma` | HPC3 PDMA channels, VDMA (also `gfx`), IMPACT host DMA (also `gfx`). |
| `scsi` | WD33C93A commands, CDBs and chip interrupts. |
| `net` | Ethernet frames sent, received, refused and filtered. |
| `audio` | HAL2 codec arm/disarm, codec A pacing, host underruns. |
| `gfx` | vblank, flips, guest busy-waits, FIFO-full stalls, FIFO level, REX3 present, IMPACT engine busy. |
| `prim` | One REX3 GO per primitive: up to millions per second under a busy X server. Drop it (`all,-prim`) for long captures. |
| `jit` | jitv2 compile requests, compiles, compile queue full. |
| `misc` | Reserved; nothing uses it yet. |

The file is flushed every couple of milliseconds and has no footer. A run
that ends with `quit`, Ctrl-C or a crash still leaves a loadable trace, at
most one batch short. `perfetto stop` and a normal shutdown write everything.

## Timelines

- **host** (default): host time in ns. On Linux this is `CLOCK_BOOTTIME`,
  Perfetto's own trace clock, so an IRIS trace can be loaded next to a
  system-wide Perfetto or `perf` capture of the same run. Events record raw
  TSC ticks (CNTVCT on aarch64). The writer maps ticks to ns by
  interpolating between clock anchors it takes at every drain.
- **guest**: the CPU's retired-instruction count (`Hot::cycles`), drawn as
  1 instruction = 1 ns. "1 ms" on screen is a million guest instructions.
  Use it when the question is "how much guest work happened between A and
  B", independent of host speed or JIT warm-up. Device threads read the
  CPU's counter, so their events land at the instruction count current
  when they happened. While the CPU is stopped, the guest clock stands
  still.

The `guest MIPS` counter track (sampled every 10 ms by the writer) is in
every trace.

## What is traced

Named tracks are grouped under the `iris` process. Thread tracks carry the
host thread's name.

| Where | Track / thread | Events |
|---|---|---|
| Interrupts | `L0 *`, `L1 *`, `map *` | One slice per assertion of each IOC/INT2 status bit (raw, before masking), from raise to clear. `slow tick (8254 timer 0)` and `fast tick (8254 timer 1)` are latches, so their slices measure tick-to-acknowledge latency. |
| CPU interrupt pins | `IP2`…`IP6`, `IP7 (count/compare)` | The CPU's interrupt inputs. IP7 runs from the Count/Compare match to the guest's Compare write. |
| `MIPS-CPU` | thread | `interrupt taken` (arg: pending & enabled IP bits); jitv2 `compile request` and `hot request` (arg: PC). |
| DMA | `PDMA pbus0-7`, `PDMA scsi0/1`, `PDMA enet rx/tx` | HPC3 channel `active` slices (arg: first descriptor), `desc` per data descriptor (arg: byte count/flags), `int` on completion interrupts. PBUS channels carry HAL2 audio. |
| DMA | `VDMA` | MC graphics DMA: `mem -> gio`, `gio -> mem`, `fill` (arg: bytes). |
| `WD33C93A` | thread | `wd cmd` (arg: command), nested `cdb` (arg: target << 8 \| opcode), `chip int` (arg: status). |
| `seeq-enet` | thread | `tx frame`, `rx frame` (arg: length), `rx no buffer`, `rx filtered`. |
| Audio | `HAL2`, `hptimer` | Codec `arm`/`disarm`, `host underrun` (the host audio callback found the ring empty), and a `codec A pace` slice per pacing wake on the timer thread. |
| Graphics | `REX3 display`, `REX3 busy wait`, `REX3 gfifo full`, `REX3 gfifo level` | `vblank`; `xmap mode write` (how IRIX swaps Newport buffers); the guest reading GFXBUSY from STATUS (waiting for idle); CPU stores retried on a full GFIFO; the FIFO level as STATUS reports it. `REX3-Processor` has a `GO` slice per primitive (`prim`); the refresh thread has `present`. |
| Graphics | `GR2 display`, `GR2 busy wait`, `GR2 fifo full` | `vblank`; reads and stores the board answered with bus-busy. |
| Graphics | `IMPACT display`, `IMPACT busy wait`, `IMPACT fifo full` | `schedule swap` (GL's flip request reaching the GE), `draw bank` (the bank GL draws into, from VALIDATE_BANKS or a context image), `buf select` (the kernel's flip at retrace), `xmap mode write` (a window ID's display mode), `retrace`; reads waiting for idle; stores stalled on the FIFO. `MGRAS-HQ3` / `MGRAS-RSS*` threads show `HQ3 busy` / `RSS busy` and host DMA slices. |
| Graphics | `IMPACT capture`, `IMPACT present` | The refresh thread copying the shown frame out of the framebuffer (arg: raster entries still queued), and handing it to the window. |
| Graphics | `IMPACT swap wait` | SCHEDULE_SWAP held until the raster engines have drawn the frame before it (arg: raster entries queued), so a flip never shows a half-drawn frame. |
| Graphics | `IMPACT status`, `IMPACT flags`, `IMPACT vc3` | Every STATUS read with its value (bit 2 is vertical blank); flags raised (arg: raster entries queued << 32 \| bits) and cleared by the host; VC3 register and SRAM writes (the window-ID tables) and current-line reads. |
| Graphics | `IMPACT irq` | The board's three interrupt lines (arg: line << 1 \| level: 0 FIFO, 1 general, 2 retrace). On the Octane, the interrupt packets sent to HEART (arg: vector << 1 \| set) and the kernel's acknowledgements (arg: offset << 32 \| bits). |
| Graphics | `VPro cfifo level` | The command-FIFO level STATUS0 is built from (words queued, not yet run), sampled at slice start and end and at STATUS0 reads. |
| Graphics | `VPro irq` | `cfifo low water` (the level fell below the mark as a store retired), `dma done`, and an `interrupt packet` per packet sent to HEART (arg: routed-source shift << 9 \| vector << 1 \| set). |
| Graphics | `VPro display`, `VPro capture`, `VPro present` | `swap arm` (the driver's flip request, arg: mask), `retrace` (flips latched, arg: displayed-buffer bits); the refresh thread resolving the frame and handing it to the window. Inside a capture, `frame planes` on `odyssey-display` is the part that holds the planes against the raster unit while it copies the dirty lines aside. |
| `odyssey-work` | thread | `slice` per run of queued stores under the board lock (arg: stores waiting), `dma descriptor` per page-table transfer (arg: table address). A processor access that drains the queue shows `drain` on its own thread. |
| `odyssey-raster` | thread | `raster slice` per run of raster jobs under the raster unit's context lock (arg: jobs pending). A front end that waits for the raster unit before it touches the planes shows `raster wait` on its own thread (arg: jobs pending). |
| jitv2 | `jitv2 compile queue`, `jitv2-compile-*` | `compile queue full` from the first dropped request to the next that fits; a `compile` slice per page compile (on the CPU thread in inline-compile mode). |

## Adding instrumentation

1. Declare the event in `src/perfetto/events.rs`:

   ```rust
   pub static VSYNC: EventDef = EventDef::new("vsync", cat::GFX).on_track("Display");
   pub static JIT_COMPILE: EventDef = EventDef::new("compile", cat::JIT).with_arg("page");
   pub static GFIFO_DEPTH: EventDef = EventDef::new("gfifo depth", cat::GFX);
   ```

   - Without `.on_track`, the event goes on the emitting thread's track: use
     this for work the thread is doing.
   - `.on_track("Name")` puts it on a named track shared by all threads: use
     this for interrupt lines, DMA channels and FIFO states, and for slices
     that begin on one thread and end on another.
   - `.in_group("Name")` puts a named track or counter under a collapsible
     group ("Interrupts", "DMA", "Graphics", ...).
   - `.with_arg("name")` shows the event's `u64` argument in the details
     panel.
   - An array of defs (`[EventDef; N]`) gives per-channel or per-bit tracks;
     index it at the call site (`events::PDMA_ACTIVE[id]`).

2. Emit it:

   ```rust
   crate::pf_instant!(crate::perfetto::events::VSYNC);
   crate::pf_scope!(crate::perfetto::events::JIT_COMPILE, pa);   // slice to end of block
   crate::pf_begin!(EV, arg);  /* ... */  crate::pf_end!(EV);    // explicit pair
   crate::pf_counter!(crate::perfetto::events::GFIFO_DEPTH, depth);
   ```

   For state that is a level rather than an event, keep a tracker and
   report the level every time it is known. Only edges are emitted:

   ```rust
   pf_wait: crate::PfLevel,                   // in the device struct
   crate::pf_level!(self.pf_wait, events::REX3_WAIT_IDLE, busy);
   crate::pf_bits!(IOC_PF_LINES, events::IOC_LINES, status_word);   // 32 lines
   crate::pf_value!(self.pf_level, events::REX3_GFIFO_LEVEL, level); // counter on change
   ```

   The trackers record which slices were opened in the current session.
   A session started while a level is high opens its slice at the next
   rising edge (`PfBits`: at the next update), never with an unmatched end.
   They are `const`-constructible and zero-valid, and zero-sized without
   the feature. So they can live in statics, in structs built from zeroed
   memory (GR2, MGRAS), or as locals in a thread loop.

Without the feature the macros expand to nothing and their arguments are not
evaluated. Paths into `crate::perfetto` are only resolved inside the macro,
so call sites need no `#[cfg]`.

## Cost and limits

- Category off: one relaxed load and a branch (~0.3 ns).
- Recorded: ~20 ns (a TSC read, a read of the guest counter, a 32-byte
  store into the thread's ring). Measure with
  `cargo test --release --features perfetto --lib -- --ignored emit_cost --nocapture`.
- Each thread gets a 64Ki-slot (2 MiB) ring the first time it emits. The
  writer drains every 2 ms. A full ring drops events and never blocks.
  Drops show on the `perfetto dropped events` counter track and in
  `perfetto status`. Instants and counters may only fill 7/8 of a ring; the
  rest is reserved for slice begins and ends. Without that reserve, a flood
  of instants would drop some slice's end, and every later slice on that
  track would nest inside it.
- A full IRIX 6.5 boot to the login prompt with every category on
  (`lightning,rex-jit,jitv2`) is about 55 MB over 90 s.
- About 40 bytes per event on disk, so a million events a second is
  roughly 40 MB/s. Narrow the categories for long captures.
- Reading the guest counter from a device thread pulls the CPU's hot cache
  line. That is harmless at device event rates, but don't put an event in
  a loop that runs millions of times per second off the CPU thread.

## Format notes

`src/perfetto/proto.rs` writes the protobuf by hand. Perfetto never
renumbers fields, and the ~30 we use are listed there with the upstream
`.proto` files they come from. The file is a stream of `Trace.packet`
records on one packet sequence. Event names, categories and argument names
are interned the first time they appear. Thread tracks carry the host
thread's name and tid.

To check a trace outside the UI:

```
curl -LO https://get.perfetto.dev/trace_processor && chmod +x trace_processor
./trace_processor -q query.sql trace.pftrace
```

`IRIS_PERFETTO_DEMO=/tmp/demo.pftrace cargo test --features perfetto --lib -- --ignored demo_trace`
writes a small multi-thread sample trace.
