# REX3 proper GPU mode: implementation plan

Status: upstream integration planned 2026-11-13. Upstream
[#569](https://github.com/techomancer/iris/pull/569) merged M0, the gated
test-only M1 resident slice and a minimum M2 scanout slice. The remaining
M2-M10 work exists in an unmerged candidate stack, not in upstream `main`.
No production GPU-authoritative Newport mode has landed. The tile-binned
`IRIS_REX3_GPU` prototype is separate and CPU-canonical, not the target
architecture.

## Goal

Add a proper Newport/REX3 GPU mode with the same overall shape as the MGRAS
and Odyssey wgpu modes:

- every REX3 framebuffer effect runs through one command-driven wgpu
  uber-shader;
- the wgpu compositor reads the same GPU-resident framebuffer resources;
- ordinary drawing and presentation perform no CPU framebuffer work and no
  framebuffer readback;
- the CPU and GL paths remain available when GPU mode is not selected;
- CPU raster plus the wgpu compositor remains a supported combination; and
- explicit CPU consumers may request a synchronized readback.

The target is correctness and complete GPU ownership first. Optimize batching,
tile size and command encoding only after representative guest workloads run
without fallback or routine readback.

## Upstream delivery and existing work

Build new, reviewable PRs against upstream `main` on top of the foundation
that has already merged; do not start a second implementation or replay the
old foundation commits. As of 2026-11-13, the baseline is `1c9d16ad`, which
contains the squash merge `bbbe5de0` of #569. Refresh the baseline before
preparing each PR.

The local `feature/gpu-m2-display-state` through
`feature/gpu-m10-frontend` branches contain the rebased candidate work. The
25 existing PRs, Tzeentchnet/iris#4 through Tzeentchnet/iris#28, are still
open against other branches in the fork, not upstream `main`; their GitHub
heads have not been rebased. They are source and review references, not
evidence that their changes landed upstream. Do not change or close them as
part of updating this plan.

The rebased M10 candidate passed the 130-case resident gate on Windows with
Vulkan and D3D12 on an NVIDIA RTX 5070, plus constructor-error and
production-machine tests. Five intermediate branches passed `cargo check`
with wgpu. These are candidate results, not tests of merged upstream
functionality; re-run the relevant checks after adapting each PR. Repeat live
application and window acceptance on the final integrated code.

### Proposed PR sequence

Use the existing patches and tests as candidates, splitting or combining them
for review as appropriate. The milestones and checkboxes below describe what
has landed on upstream `main`, not what has been implemented in a local branch.

1. **Finish M2:** GPU DID/display state, same-device common presentation and
   explicit readback compatibility. Start with Tzeentchnet/iris#4 through
   Tzeentchnet/iris#6, keeping the resident path gated.
2. **M3 raster coverage:** command ABI and ordered blocks, lines, clipping,
   pixel operations, blending, patterns, shading, copies and continuation.
   Adapt Tzeentchnet/iris#7 through Tzeentchnet/iris#16 in reviewable slices;
   keep strict CPU/JIT/GPU equivalence and no-fallback tests with each slice.
3. **M4 HostW:** staged payloads, segment semantics, lines, zoom, producer
   retry and fast clears. Adapt Tzeentchnet/iris#17 through
   Tzeentchnet/iris#22, preserving ordering and all-or-nothing `BUS_BUSY`.
4. **M5-M7 observation and lifecycle:** HOSTR, explicit observers,
   snapshot/reset and failure handling. Adapt Tzeentchnet/iris#23 through
   Tzeentchnet/iris#25, with exact observation and restore tests.
5. **M8 telemetry:** use Tzeentchnet/iris#26 for counters and latency
   bounds; tune only after representative live execution is correct.
6. **M9/M10 activation and frontends:** adapt Tzeentchnet/iris#27 and
   Tzeentchnet/iris#28 only after the required raster, HostRW, observer and
   lifecycle coverage has landed. Enable the direct resident path explicitly;
   keep readback compatibility labeled and never silently fall back.

Each upstream-facing PR should start from the latest upstream `main`, include
only its incremental change and tests, and remain buildable while production
activation is gated. Land prerequisites first, then rebase the next PR onto
the updated `main`; use an unmerged PR as a base only if the maintainer agrees
to that review arrangement. Reconcile shared code and documentation instead
of copying a stale fork version over upstream changes.

### Pending upstream PRs to coordinate with

- [#698](https://github.com/techomancer/iris/pull/698) is an open draft for
  Odyssey/VPro changed-frame composition and a shared offscreen wgpu
  presenter plus `ci-gpu` coverage. It is not Newport resident raster
  coverage. Its `src/main.rs` and `src/ui.rs` changes overlap the candidate
  stack, and `src/wgpu/present.rs` is a shared presenter API. If it lands,
  adapt M2/M10 to that API rather than adding a competing offscreen path.
  Preserve the same-device handoff, explicit resident failure behavior and
  zero routine raw-plane readback. Its VPro tests count frames; add Newport
  pixel, ordering and no-readback assertions rather than treating those counts
  as proof of Newport correctness. Its software Vulkan and Metal CI are
  useful additional coverage, not a replacement for the real Vulkan and
  D3D12 resident hardware gate or a visible-window test.
- [#679](https://github.com/techomancer/iris/pull/679) is a broad pending
  documentation and GUI pass with shared Newport, configuration, GUI and
  documentation files. The inspected REX3 hunks are documentation edits,
  not the resident implementation. Reconcile its changes if it lands; it is
  not a prerequisite for the GPU stack.
- The pending audio/configuration work in
  [#690](https://github.com/techomancer/iris/pull/690) and
  [#694](https://github.com/techomancer/iris/pull/694), and the CI-clock work
  in [#696](https://github.com/techomancer/iris/pull/696), also touch machine
  or configuration files used by the candidate. They are not graphics
  dependencies; preserve their changes, upstream constructor and PROM errors,
  and explicit GPU errors when preparing activation.

Recheck PR status and actual diffs before each submission: pending work may
change or merge. Do not assume a pending PR has landed or require it to land
before an otherwise independent Newport PR.

## Non-negotiable decisions

1. **GPU raster mode makes GPU memory authoritative.** Once activated, normal
   REX3 draws, clears, copies and HostW operations do not read or write
   `fb_rgb` or `fb_aux` in CPU memory.
2. **All REX3 pixel operations use one stable raster pipeline.** Commands and
   state select behavior inside an uber-shader. Do not switch specialized
   pipelines per primitive.
3. **Proper GPU raster mode uses the wgpu compositor.** Raster and scanout
   share one wgpu device, queue and set of framebuffer resources.
4. **Do not build GL/wgpu texture interop.** Sharing native textures with the
   GL context would require backend-specific work below normal wgpu APIs.
   Wgpu-raster plus GL/CPU composition may exist only as an explicit
   readback-based validation or compatibility mode.
5. **CPU raster plus wgpu composition is valid.** It uploads CPU framebuffer
   damage and does not require the wgpu rasterizer.
6. **Keep geometry and graphics-front-end processing on CPU.** Existing
   profiles put this work at about 9% of one core. Even eliminating it
   completely has a theoretical serial ceiling of about 1.10x and would add
   synchronization and latency.
7. **No unsupported-draw CPU framebuffer fallback in GPU mode.** Missing
   opcode coverage means GPU mode is incomplete. It must not checkpoint and
   silently execute the draw against the CPU framebuffer.
8. **Readback is explicit.** HOSTR, snapshots, screenshots, VNC/CI capture,
   monitor reads and the CPU-compositor validation mode may synchronize GPU
   output. Idle, display fences, batch limits and normal presentation may not.

## Backend matrix

| Rasterizer | Compositor | Contract |
|---|---|---|
| CPU | GL/CPU | Existing compatibility path; CPU framebuffer authoritative. |
| CPU | wgpu | Upload CPU framebuffer damage and compose/present with wgpu. |
| wgpu | wgpu | Proper GPU mode; direct shared resources and no routine readback. |
| wgpu | GL/CPU | Explicit readback validation/compatibility path; not optimized. |

The renderer and rasterizer selections remain independent, but a user asking
for the proper accelerated path receives wgpu for both.

This is the target matrix, not a current Newport selector: normal Newport
supports CPU raster with GL/CPU or wgpu composition. Its separate
`IRIS_REX3_GPU` experiment is CPU-canonical. The resident wgpu raster and
direct scanout are internal tests only, and the readback compatibility mode
is not implemented as a live option.

## Existing evidence and prior art

### Current REX3 experiment

The current implementation and measurements are described in
[rex3-gpu-parallelization.md](rex3-gpu-parallelization.md).

Useful pieces to retain:

- strict CPU/JIT/GPU equivalence corpus;
- tile bins and ordered per-tile command lists;
- barrier-free pixel ownership within a tile;
- telemetry and timestamp accounting;
- backend preflight and lifecycle tests;
- device-loss fault injection;
- command matching and state capture;
- large-fill and tiny-fill performance gates.

Results worth preserving as baselines:

- 1,280-pixel spans reached roughly 2.5x the CPU renderer;
- 8-pixel draws reached roughly 0.96x on Vulkan and 0.95x on D3D12;
- most tiny-draw time is shared GFIFO and command handling, not GPU compute;
- a large batch amortizes submission cost but can add latency; and
- tile binning avoids the primitive-per-dispatch failure mode.

Parts that do not carry into the proper design:

- a GPU plane covering only the visible 1,344 x 1,024 RGB region;
- tile-major storage chosen primarily for damage-limited CPU checkpoints;
- CPU framebuffer reseeding after fallback;
- periodic journal checkpoints into CPU memory;
- display, idle and wait-idle readbacks;
- unsupported draw fallback to the CPU framebuffer; and
- recovery that depends on a continuously current CPU plane.

### MGRAS and Odyssey wgpu work

The common architecture from the `wgpuopt` branch is already in upstream
`main`. Its original commits remain useful as prior art:

- [common wgpu branch](https://github.com/techomancer/iris/tree/wgpuopt)
- [VPro raster and wgpu window](https://github.com/techomancer/iris/commit/60a770076b3a30df421bbfeed149a3a0244b452a)
- [MGRAS raster on wgpu](https://github.com/techomancer/iris/commit/f65908b57f68ef3eb03b8fd4166a297ecf840936)
- [MGRAS direct GPU scanout](https://github.com/techomancer/iris/commit/a899ccc57daee4657276ed305677be7ed6a44bbe)

Reuse or port:

- shared `Gpu` device ownership;
- independent CPU/wgpu raster and GL/wgpu renderer configuration;
- `takes_gpu_frames` / `present_gpu_frame` style handoff;
- common batch recorder, state slots and tile bins;
- fixed host-visible upload sets;
- fixed readback staging;
- board-specific scanout kernels over the rasterizer's real buffers; and
- wgpu presentation on the board's device.

Do not blindly cherry-pick the branch or replace the merged common layer.
Build on the current upstream APIs and reconcile later presenter, lifecycle
and device-loss changes as they land.

## Target ownership model

### CPU-owned state

The CPU continues to own:

- the REX3 register file and architectural context;
- GFIFO publication and consumption;
- register decode and GO recognition;
- command/state record construction;
- CPU tile bin construction unless measurement justifies moving it;
- HostRW input gathering and architectural walker state;
- guest-visible status and interrupt state; and
- submission, fences and explicit observation requests.

Advancing deterministic architectural context when a command is accepted is
allowed. It must happen exactly once; the GPU must not independently publish a
second conflicting context transition.

### GPU-owned state

In wgpu raster mode the GPU owns:

- the full RGB framebuffer plane;
- the full auxiliary/CID/overlay plane;
- any separate depth or other pixel storage required by the model;
- queued HostW payloads until consumed;
- all framebuffer read-modify-write effects; and
- the composed display frame.

The original CPU framebuffer allocations remain for compatibility, snapshot
formats and explicit readback consumers. They are stale unless synchronized
at a recorded GPU stream position.

### Full plane size

Start with full 2048 x 1024 planes, not only the 1,344-pixel visible region.
Offscreen pixmaps and screen-to-screen operations are part of normal Newport
use and must not force CPU fallback.

## GPU memory layout

Default to the REX-native linear layout:

- 2048 pixels per row;
- 1024 rows;
- one 32-bit word per RGB or auxiliary pixel.

Tile binning does not require tile-major framebuffer storage. Linear storage
simplifies:

- scanout;
- full-plane HOSTR and snapshot readback;
- CPU-compositor validation;
- screen-to-screen coordinates;
- debugging and framebuffer dumps; and
- restore uploads.

The current tile-major layout was primarily valuable because it reduced
damage-limited CPU checkpoints, which the proper path removes. Keep it only if
a controlled benchmark proves a material shader benefit that outweighs
readback conversion and complexity.

## Command and state ABI

Use fixed buffers and stable bind groups for the backend lifetime. A conceptual
command record contains:

```text
RexCommand:
    opcode / primitive kind
    state-slot index
    destination bounds
    source bounds where applicable
    payload offset and length
    ordering / hazard flags
```

A state slot contains the REX3 state needed by the pixel pipeline, normalized
into a GPU-safe plain-data layout. Do not upload the entire 1 MiB CPU
`Rex3Context` for every draw.

Payloads contain HostRW words, CPU-prepared segment descriptors or other
variable data. They live in persistent host-visible staging sets and remain
owned until the corresponding GPU submission completes.

## Uber-shader

The raster shader must cover every framebuffer-affecting REX3 operation:

- spans and blocks;
- integer, fractional and antialiased lines;
- clears;
- RGB and auxiliary/CID/overlay planes;
- Z operations;
- logic operations and write masks;
- blending and dithering;
- clipping and stipples;
- screen-to-screen copies;
- HostW unpacking, zoom and drawing;
- HOSTR source semantics where the CPU read walker needs a snapshot;
- continuation/residual HostRW state; and
- empty/no-op commands with correct context advancement.

The CPU packages commands. The shader performs all framebuffer reads, tests,
blending, logic operations and stores.

### Tile scheduling

Retain ordered tile lists:

- one tile owner processes commands affecting that tile in GFIFO order;
- a pixel has one writer within a dispatch;
- bins preserve overlapping-draw order;
- empty tiles should eventually be skipped through an active-tile list or
  indirect dispatch if that measures better than dispatching the whole screen.

The initial 32 x 32 tile size is a measured starting point, not a permanent
contract.

### Global hazards

Wgpu has no global barrier inside one compute dispatch. Split submissions or
dispatches when an operation requires results written by another tile, for
example:

- overlapping screen-to-screen copies;
- source/destination aliases;
- cross-tile framebuffer reads;
- explicit guest observation;
- display swap/fence semantics; or
- a payload/resource upload that must land between draws.

Keep the same raster pipeline across these dispatches. A hazard boundary is not
a reason to introduce per-primitive pipeline variants.

## Wgpu compositor and presentation

The REX3 scanout kernel binds the authoritative GPU planes directly and
combines:

- RGB and auxiliary data;
- DID/XMAP mode state;
- colormap entries;
- Bt445 gamma state;
- VC2 timing, cursor position and cursor glyph;
- topscan and visible dimensions.

Raster and scanout use the same `Gpu`, device and queue. A normal frame is:

1. close or submit the pending REX3 batch when required;
2. submit scanout after it on the same queue;
3. present the composed GPU frame through the wgpu presenter.

There is no CPU framebuffer fence between these steps.

### Frontends

- The CLI/window frontend uses the common wgpu presenter.
- `iris-gui` needs an eframe wgpu path before it can consume direct GPU frames
  without readback. Its current Glow path remains for CPU raster mode; the
  candidate's explicit readback mode is a separate, labeled compatibility
  option, not direct presentation.
- CPU raster plus wgpu composition uploads dirty CPU framebuffer regions.
- Wgpu raster plus GL/CPU composition explicitly reads back and is labeled a
  slow validation/compatibility mode.
- Do not add native GL/wgpu texture-sharing code.

## HostRW

HostRW is deliberately asymmetric.

### HostW: guest/host memory to framebuffer

All HostW framebuffer effects remain on GPU. The CPU stages source data and
constructs commands.

#### Contiguous VDMA pixmaps

The common case should be:

1. MC gathers one translated DMA run.
2. REX reserves a persistent staging range.
3. The raw HostRW qwords are copied into it.
4. One ordered command references the range and its REX state.
5. The uber-shader consumes the words and runs the pixel pipeline.

One DMA block should not become one GFIFO payload entry or one GPU operation
per qword.

Reuse the existing bulk bus boundary described in
[hostrw-batching-design.md](../rex3/hostrw-batching-design.md), but replace
payload-in-GFIFO storage with a staging reference for GPU mode.

#### PIO and zoomed pixmaps

Repeated CPU HostRW writes require the CPU architectural walker to retain:

- `host_count` and `host_shift`;
- word and row boundaries;
- STOPX/STOPY behavior;
- source progression and zoom;
- partial-word state; and
- the exact point where a segment/GO completes.

Walking this state on CPU does not permit CPU framebuffer stores. The walker
produces ordered HostW segment descriptors and stages their source words.
The uber-shader applies the REX pixel pipeline and stores the destination.

Many architectural records may share one GPU submission. Avoid expanding
zoomed output into one CPU-generated record per destination pixel unless
measurement proves that representation best.

### HOSTR: framebuffer to guest/host memory

The first implementation should prefer a simple full selected-plane readback
over a complex GPU-generated HostRW result pipeline:

1. submit and order all preceding GPU work;
2. copy the selected full authoritative plane into a reusable MAP_READ buffer;
3. wait and map once;
4. copy into the linear CPU framebuffer shadow;
5. run the existing CPU HOSTR walker against that snapshot;
6. fill the CPU HostRW output array for the whole PIO/DMA transfer; and
7. let the guest consume the result without another GPU readback.

A 2048 x 1024 32-bit plane is 8 MiB. One explicit transfer plus mapping latency
is expected to be far cheaper and simpler than a GPU/CPU round trip per qword.
Read RGB, auxiliary, or both according to what the operation observes.

Associate the CPU snapshot with a GPU framebuffer generation or stream
position:

- the first HOSTR after GPU writes synchronizes;
- continuation words consume the same snapshot;
- another readback occurs only after GPU framebuffer state changes and another
  CPU read is requested.

Start with full selected-plane readback. Add bounding-box or dirty-region
readback only if measurements show HOSTR synchronization is material.

### Staging rules

Follow the prior wgpu host-buffer rules:

- do not call `Queue::write_buffer` or create a temporary staging buffer per
  payload;
- use persistent mapped host-visible buffers where supported;
- use a bounded number of upload sets;
- reuse a set only after its submission completes;
- submit a full set rather than allocating unbounded replacements;
- keep a fixed reusable readback buffer; and
- keep GPU-only framebuffer buffers free of MAP usage.

A staging reservation and its GFIFO command are all-or-nothing. If either the
queue or staging set lacks capacity, return `BUS_BUSY` without committing
anything so a retry cannot duplicate a partial pixmap.

Preserve two-producer ordering between CPU PIO and MC VDMA. Payload bytes may
be staged early, but the GPU command becomes visible only at its GFIFO stream
position.

## Explicit CPU observations

The following operations may synchronize GPU data:

- HOSTR and guest DMA reads;
- snapshots/checkpoints;
- screenshots;
- VNC or CI frame capture;
- monitor framebuffer reads/dumps/hashes;
- differential test checkpoints; and
- the wgpu-raster plus GL/CPU-compositor mode.

Prefer reading the composed output for screenshots/VNC rather than both raw
planes when raw board state is not required.

Snapshots perform a defined full synchronization, serialize the resulting
planes and restore by uploading them before GPU execution resumes.

No ordinary display, idle transition, GFIFO status read or batch limit may
implicitly make the CPU framebuffer current.

## Lifecycle and failure policy

The current experimental recovery journal depends on CPU-current checkpoints.
That is incompatible with a proper GPU-authoritative mode.

The gated resident slice uses a sticky fatal device-loss policy: notify the
machine controller once, stop the machine and refuse reset/restart without
recovery from stale CPU planes. This policy must carry through production
integration; it does not yet provide snapshot/recreation or full lifecycle
coverage.

Do not retain an unbounded command journal and do not silently claim recovery
from a stale CPU framebuffer.

Keep the current work's useful lifecycle coverage:

- activation and Stop ordering;
- bounded adapter/device initialization;
- callback error handling;
- frontend teardown;
- snapshot/reset barriers;
- no stale restart after Stop;
- process-wide device/lease ownership if still required; and
- backend-specific tests on Vulkan, D3D12 and eventually Metal.

## Performance model

### What GPU acceleration can improve

- clears and large block fills;
- long interpolated spans;
- large Z/colour operations;
- bulk pixmap transfers once HostW is staged;
- removal of CPU raster-thread work;
- direct GPU scanout without CPU copy/upload.

### What may remain CPU-limited

- guest command generation;
- GFIFO/register traffic;
- tiny UI primitives;
- single-word PIO HostRW traffic;
- emulated guest CPU throughput;
- host command construction and queue submission.

GLQuake/sgiquake evidence shows the expected endpoint: after GPU rasterization,
the guest CPU can limit frame rate while a low-end discrete GPU remains lightly
used and does not clock up. Low GPU utilization is not itself a failure.

Do not spend effort moving geometry/front-end processing to GPU without new
evidence. Profiles put it near 9% of one core, and it may overlap raster work.

### Benchmark discipline

Separate shader measurements from live application measurements.

For shader work:

- use GPU timestamp queries;
- record or pin GPU clocks when comparing small deltas;
- measure bin, raster and transfer stages independently;
- record active tiles, candidate pixel tests and useful writes;
- inspect shader register pressure; and
- compare several interleaved runs.

For live workloads:

- record guest FPS and guest MIPS;
- record GFIFO occupancy histogram, maximum and full/backpressure events;
- measure guest cycles blocked by graphics;
- measure host command-generation and submission time;
- count submissions and primitives per batch;
- measure p50/p95 presentation latency;
- report upload/readback reasons and bytes; and
- report graphics-thread CPU use.

Watching the status-bar GFIFO sample is useful triage, but acceptance needs
instrumented occupancy over time. A filling GFIFO proves consumer backlog. A
mostly empty GFIFO suggests that faster raster alone will not improve
throughput.

## Milestones

### M0 - Rebase the architecture, not the prototype

- [x] Reconcile the current branch with the common wgpu layer on `wgpuopt`.
- [x] Preserve the current vendored wgpu-core fixes or replace them only with
      proven upstream behavior.
- [x] Add independent CPU/wgpu raster and GL/wgpu renderer configuration.
- [x] Keep all existing CPU/GL behavior unchanged by default.
- [x] Establish shared `Gpu` ownership through the display/renderer boundary.

Complete when an unchanged CPU raster build can use either legacy GL/CPU
presentation or the common wgpu presenter.

Completed 2026-11-10. The referenced `wgpuopt` commits were already ancestors
of the updated main branch, so none were cherry-picked. Main uses upstream
wgpu 30.0.1 and has no active vendored wgpu-core override to preserve. The
legacy `rex-gpu` prototype now builds on the common `Arc<Gpu>` and exposes it
through `GfxDisplay`, so a wgpu window reuses the same device. The prototype
still synchronizes through its CPU-canonical framebuffer. Newport explicitly
rejects `raster = "wgpu"` rather than silently treating that selection as
CPU raster. Keep it unavailable until the coverage and activation gates are
met.

### M1 - Full authoritative GPU planes

- [x] Allocate full 2048 x 1024 RGB and auxiliary GPU buffers.
- [x] Choose linear storage unless a controlled benchmark justifies tile-major.
- [x] Upload reset/restored contents once when GPU authority begins.
- [x] Add a GPU framebuffer generation/stream-position counter.
- [x] Gate CPU framebuffer access in tests so an ordinary GPU draw fails if it
      touches the CPU planes.
- [x] Port flat fills to the new buffers without periodic checkpoint readback.

Complete when flat fills remain GPU-resident across idle and display, and an
explicit observation is the only operation that updates the CPU shadow.

Complete within the gated test-only scope: under
`cfg(all(test, feature = "wgpu"))`, the real GFIFO drives full linear RGB and
auxiliary GPU planes seeded once on an injected common device. Supported flat
fills use bounded batches, preserve offscreen columns, and track typed
epoch/stream positions and per-plane shadow currency; unsupported work fails
without CPU fallback. Page-protected shadows and counters cover fills, idle,
fences and direct test frames. Explicit selected-plane observation returns a
snapshot; the caller separately installs and acknowledges a CPU shadow.
Quiescent reset/reseed and partial restore are tested. This does not expose a
live raster selector or implement board-level HOSTR, snapshots or other raw
consumers; the legacy `rex-gpu` replay/live prototype remains CPU-canonical.

### M2 - Direct wgpu scanout

- [x] Add a REX3 scanout kernel over the authoritative GPU planes.
- [ ] Upload only small display state/tables needed for each changed frame.
- [ ] Present through the common wgpu presenter on the same device.
- [x] Add CPU-raster plus wgpu-compositor uploads.
- [x] Add an eframe wgpu path or explicitly keep proper GPU mode unavailable in
      `iris-gui` until one exists.
- [ ] Retain readback composition as a labeled validation mode.

Complete when a GPU-rendered live frame reaches the window with zero
framebuffer readback and without a CPU plane upload.

Partial, test-only: a read-only same-device binding and GFIFO frame marker
order resident fills, direct scanout and a GPU frame consumer. Screenshot,
frame-request, headless and VNC tests capture composed output only. Ordinary
frames after seeding have zero raw-plane upload/readback, CPU-plane access
and CPU fallback. The `WgpuRenderer` has a test-only common-presenter handoff,
but the marker/presentation tests use a same-device test presenter, not a
real live window. Changed decoded DID metadata can still upload a cached
2 MiB map; small-state-only/GPU DID expansion, live integration and the
readback compositor mode remain open. `iris-gui` cannot select this mode.

The Windows gate `.\scripts\test-resident-gate.ps1` checks 25 resident CPU,
3 flat-fill and 1 display-generation tests, plus 10 resident GPU and 5
direct-scanout tests on each of Vulkan and D3D12. It requires the named
backend on an integrated/discrete adapter and fails rather than skipping
when unavailable; a single-backend run is not the complete gate. See
`TESTING.md` for commands and prerequisites.

### M3 - Command ABI and complete uber-shader

- [ ] Define plain-data command, state-slot and payload layouts.
- [ ] Use fixed buffers/bind groups and bounded reusable batch sets.
- [ ] Preserve ordered per-tile command lists.
- [ ] Implement blocks, spans and clears.
- [ ] Implement integer, fractional and antialiased lines.
- [ ] Implement RGB, auxiliary/CID/overlay and Z effects.
- [ ] Implement clipping, stipples, masks, logic operations, blending and
      dithering.
- [ ] Implement screen-to-screen copies and overlap hazards.
- [ ] Implement continuation and empty/no-op context behavior.
- [ ] Remove the production CPU framebuffer fallback path.

Complete when the strict corpus and representative live workloads report zero
unsupported or CPU-fallback draws.

### M4 - HostW staging

- [ ] Port persistent host-visible upload sets and fixed ownership tracking.
- [ ] Make one VDMA block one staged payload plus one command where possible.
- [ ] Preserve all-or-nothing `BUS_BUSY` behavior.
- [ ] Add CPU-walked PIO/zoom segment descriptors.
- [ ] Preserve packing, residual shifts, row flushes and STOPX/STOPY.
- [ ] Stress CPU and VDMA producer interleaving and staging wraparound.

Complete when ordinary and zoomed pixmaps match CPU mode with no CPU
framebuffer store and no per-qword GPU submission.

### M5 - HOSTR and CPU observation

- [ ] Add a fixed reusable full-plane readback path.
- [ ] Synchronize only the selected source plane(s).
- [ ] Run the existing CPU HOSTR walker against the synchronized snapshot.
- [ ] Reuse one snapshot for the full PIO/DMA transfer generation.
- [ ] Preserve PIO read-then-advance and DMA prime-then-read behavior.
- [ ] Reuse the synchronization primitive for snapshots and monitor raw-plane
      reads.
- [ ] Measure before considering region readback.

Complete when HOSTR matches CPU mode and performs at most one plane readback
per transfer generation, never one per qword.

The test-only resident backend already has reusable, explicitly selected raw
plane observation and shadow acknowledgement; none of these checkboxes imply
that the board's HOSTR, snapshot or monitor consumers are wired to it.

### M6 - Explicit observers and frontend matrix

- [ ] Read back composed output for screenshots and VNC/CI capture.
- [ ] Keep raw-plane reads separate from composed-frame reads.
- [ ] Validate CPU+GL, CPU+wgpu, wgpu+wgpu and readback wgpu+GL/CPU modes.
- [ ] Label the readback compositor mode as compatibility/validation only.
- [ ] Ensure capture users cannot silently force the proper GPU path into
      continuous raw-plane readback without reporting the mode change.

Complete when all four combinations produce matching images and only the
documented combinations/readers increment readback counters.

Composed-output screenshot, headless and VNC capture are exercised by the
test-only resident display; live frontend integration remains open.

### M7 - Snapshot, reset and failure lifecycle

- [ ] Define a quiescent GPU snapshot boundary.
- [ ] Serialize full planes and required command/context state.
- [ ] Restore by uploading before command processing resumes.
- [x] Define device-loss behavior without CPU-current checkpoints.
- [ ] Re-run Stop/reset/restore/debugger/frontend lifecycle tests.
- [ ] Validate Vulkan and D3D12; unblock Metal only after safe completion and
      destruction are demonstrated.

Complete when snapshot round trips are exact and GPU failure cannot return
success with stale or partially recovered framebuffer state.

Device loss is sticky and fatal in the gated resident path, with one machine
notification and no restart from stale CPU planes. Production lifecycle and
snapshot support remain open.

### M8 - Optimize only after complete live execution

- [ ] Capture representative live Newport workloads.
- [ ] Record primitive, mode, area and HostRW transfer distributions.
- [ ] Add GFIFO occupancy/backpressure telemetry.
- [ ] Tune tile size, active-tile dispatch and batch limits.
- [ ] Bound batching by both capacity and presentation latency.
- [ ] Compare full-plane and region HOSTR readback only if HOSTR is material.
- [ ] Re-run correctness after every scheduling/layout change.

Complete when the selected design passes correctness and end-to-end gates on
representative applications, not only synthetic fills.

### M9/M10 - Production activation and frontends

- [ ] Compile the resident path in normal wgpu builds after its prerequisites
      land, and enable Newport GPU raster only through an explicit selection.
- [ ] Share the validated device across raster and direct presentation; fail
      explicitly on initialization, presentation or device loss.
- [ ] Offer readback as an explicit slow compatibility mode, never a fallback;
      keep `iris-gui` direct mode unavailable until it has a wgpu presenter.
- [ ] Preserve upstream machine-construction, PROM and configuration errors
      while wiring GPU construction and lifecycle.
- [ ] Re-run the resident gate, production lifecycle, visible-window and
      representative IRIX application tests on the integrated code.

The candidate stack implements and tests an activation path, but these
checkboxes remain open until the corresponding changes land upstream.

## Validation requirements

### Correctness

- CPU interpreter, REX JIT and wgpu results match bit-for-bit.
- Tests assert that the intended engine actually ran; a green comparison that
  dispatched the CPU path twice is invalid.
- Full RGB and auxiliary plane hashes match at explicit checkpoints.
- Overlapping draws retain GFIFO order.
- Cross-tile hazards split at correct boundaries.
- Display output matches the existing compositor, including DID, palettes,
  overlays and cursor.
- Reset, resolution changes and context continuation remain exact.
- An offscreen presenter exercises the same-device Newport path, including
  changed and unchanged frames; visible-window acquisition and presentation
  are tested separately.
- Software-adapter CI augments rather than weakens the hardware-adapter gate.

### HostRW

- One DMA pixmap becomes one staged block/command where capacity permits.
- Zoomed multi-write PIO images match partial rows and residual shifter state.
- HOSTR sees all prior GPU draws.
- Later draws do not contaminate the synchronized HOSTR snapshot.
- One readback serves a complete transfer generation.
- Staging exhaustion never partially commits a retried transfer.
- CPU and VDMA producers cannot overwrite each other's payloads.

### No-readback contract

For wgpu raster plus wgpu compositor during an ordinary workload:

- CPU framebuffer write count is zero after activation;
- raw-plane readback count and bytes are zero;
- display fences and idle cause no mapping;
- journal/batch limits cause no mapping;
- compositor consumes the raster buffers directly; and
- there are zero CPU raster fallbacks.

Explicit HOSTR, snapshot, screenshot/VNC and monitor operations must report
their reason and transferred bytes separately.

### Performance

Retain the existing synthetic gates as component tests, but add end-to-end
criteria:

- large spans remain materially faster than CPU/JIT;
- tiny draws regress no more than the agreed tolerance;
- GPU mode does not reduce a CPU-limited application's frame rate materially;
- p95 presentation latency remains bounded;
- GFIFO backlog decreases on workloads that previously filled it;
- host graphics-thread CPU use decreases; and
- normal presentation transfers zero framebuffer bytes to CPU.

Do not require high GPU utilization. A lightly used GPU at base clock is an
expected result when the emulated guest CPU is the producer bottleneck.

## Open implementation questions

These are engineering choices to settle with prototypes or measurements, not
reasons to change the architecture:

- how to use the current or future shared offscreen presenter for Newport
  while retaining explicit failure and hardware-gate semantics;
- linear versus tile-major framebuffer storage after routine checkpoints are
  removed;
- command and state-slot packing;
- active-tile list versus full-screen dispatch;
- batch time limit that balances submission overhead and latency;
- full-plane versus region HOSTR readback after real frequency measurements;
- how `iris-gui` selects or hosts its wgpu renderer;
- carrying the gated fatal/restart policy through the production lifecycle; and
- which backend becomes the first supported default after validation.

Geometry offload and native GL/wgpu texture sharing are not open questions.
They are out of scope unless new evidence changes the maintainer's decision.

## Definition of done

REX3 GPU mode is ready to consider for normal use only when:

1. every framebuffer-affecting operation used by supported workloads executes
   through the GPU command stream;
2. wgpu raster plus wgpu scanout shares authoritative resources and performs
   no routine readback;
3. HostW is staged and HOSTR synchronizes no more than once per transfer
   generation;
4. CPU and GPU engines agree bit-for-bit across the strict corpus and live
   captures;
5. snapshots, reset, Stop and explicit observers have defined exact behavior;
6. no silent CPU fallback or stale-frame success path remains;
7. representative applications meet correctness, latency and performance
   gates; and
8. user documentation states the available raster/compositor combinations,
   fallback behavior and explicit readback costs.
