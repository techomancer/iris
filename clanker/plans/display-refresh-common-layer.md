# One display refresh thread in the common layer (issue #742)

Status: in progress on branch `display-refresh` (2026-11-14). Phase 1
(the common layer, `src/display/`, 18 unit tests) is done. Phase 2: GR2,
IMPACT, VPro and Newport are migrated and checked at the PROM (Indy XZ,
IP28 High IMPACT, Octane V12, Indy Newport; `--ci`, screenshot through the
monitor). The O2's display is migrated too, with its timing interrupts raised by
the refresh thread as line events (IRIX 6.5.22 desktop checked,
2026-11-15), see "O2 interrupts" below. InfiniteReality is migrated but not run: there
is no Onyx2 PROM or config on this machine, only its unit tests. Phases 3-7 not started. See
"Progress" at the end for what the code looks like and how it differs from
the sketch in section 3.

Goal: one refresh thread for all graphics boards. A board registers its
heads and requests a resolution and a video timing; the thread keeps the
timing, calls the board's vblank and compose callbacks, and blits the
result. The refresh rate becomes the one the guest programmed, not a fixed
60 Hz. The design leaves room for stereo (SBS output) and for more than one
head, and for choosing a host monitor, including SBS display glasses.

## 1. What exists today (HEAD 83c8dddc)

Six private display threads, each with a hardcoded 16.667 ms period. None
uses the refresh rate the guest programmed.

| Board | Thread | Timing | Vblank to guest | Compose output |
|---|---|---|---|---|
| Newport | `REX3-Refresh`, one per head (`ng1/rex3.rs:3718`, spawn `:4562`) | host sleep; respawned on every stop/start | STATUS VRINT + IOC line, before present, cleared by STATUS read | in the renderer: `SwCompositor` (memory), `GlCompositor` (GL texture), `Ng1Scanout` (wgpu buffer) |
| GR2 | `GR2-Display` (`gr2/gr2disp.rs:24`) | host sleep | after present: line up, `sleep(700 µs)`, line down; IOC latch, `SG_STAT_0` level on Indigo2 | `gr2comp::compose` to memory |
| IMPACT | `MGRAS-Refresh` (`mgras/disp.rs:208`) | host sleep | after snapshot: pulse with `sleep(500 µs)`; STATUS vblank bit = "under 1 ms since frame start" on the host clock | memory, or wgpu buffer |
| VPro | `odyssey-display` (`odyssey/mod.rs:377`) | 60 Hz tick polls a `RetraceClock` (host monotonic, catches up missed retraces, `scanout.rs:144`) | retrace latches swaps, then sends the routed vector | memory, or wgpu buffer |
| O2 | `O2-Display` (`ip32_machine.rs:921`); interrupts on `O2-Service` | display: host sleep, composes every frame; VT interrupts and beam position from guest CPU cycles x dot clock (`o2gfx.rs:253`) | CRIME inputs 16-19 at four programmable lines | memory |
| InfiniteReality | `kona-display` (`kona/back/display/clock.rs:121`) | `park_timeout`, with a quiesce handshake for snapshots | video interrupt packet | memory |

Shared pieces already present: `GfxDisplay` (`src/gfx_display.rs`, hand-off
points only), `Renderer` (defined in `ng1/rex3.rs:35`, Newport arguments in
its signature, used by every board), `Rex3Screen` (`src/disp.rs`, the frame
container for all boards), `DisplayStop`, and the status bar and debug
overlay, of which each thread owns a copy. Every loop repeats the same body:
status snapshot, idle heartbeat (every 6th frame), `renderer.present`,
screenshot PNG thread, `DisplayStop::finish`.

Three facts shape the design:

- **Host vsync sits on the guest's vblank thread.** GL uses
  `SwapInterval::Wait(1)` (`ui.rs:216`) and wgpu uses Fifo
  (`wgpu/present.rs:127`), so `present` can block for a host frame on the
  thread that raises the guest retrace. On one shared thread this would
  delay every board, and it would cap a 76 Hz guest at the host's 60 Hz.
- **GR2 and IMPACT sleep inside the thread** to hold the retrace level. A
  shared thread cannot sleep per board; the falling edge must be a
  scheduled event.
- **A GL context belongs to the thread that presents**, and its teardown
  must run there (`clanker/gui/gl-teardown-must-run-on-the-refresh-thread.md`).
  One long-lived refresh thread makes this simpler than today's respawning
  threads, but every window's context then lives on that one thread.

Dual head today: two independent Newport boards, each with its own
unsynchronised thread. The `iris` window shows only head 0 (`main.rs:215`);
iris-gui shows both heads side by side in one window; VNC and host input are
head 0 only. No monitor selection exists: fullscreen is
`Fullscreen::Borderless(None)` and nothing calls `available_monitors`.

Stereo today: none. VPro folds right-eye drawing into the left buffers
(`odyssey/command.rs:818`), IMPACT only knows the width of the 1280x492
format (`mgras/frame.rs:283`).

## 2. What the hardware does

Sources: the datasheets in `docs/datasheets`, the headers in `ignore/gr2`,
`ignore/gr4` and `ignore/odsy`, and the video format files that ship with
IRIX (`.sdb` / `.vfo`), which decode to exact totals and pixel clocks.

### Timing

Every board runs a timing generator from a frame table and a line table,
with a programmable pixel clock. Refresh = pixel clock / (total pixels x
total lines).

| Board | Tables | Pixel clock | Formats seen |
|---|---|---|---|
| Newport | VC2 frame table of (line pointer, count); line runs in units of 2 pixel clocks | Bt445 PLL (reference frequency not in the repo) | 1024x768 at 60/70, 1280x1024 at 60/72/76 |
| GR2 | VC1 SRAM frame table at 0x0800, line table at 0x0000 | crystal (107.352 or 132 MHz) or serial PLL with two known byte tables | 1280x1024 at 60/72, STR_RECT stereo, NTSC, PAL |
| IMPACT | VC3 line table at 0x000, frame table at 0x380 | DAC PLL (R, V, control) | 1024x768 at 60/72/76/96s, 1280x1024 at 49-76, 1280x492_120s, 1600x1200_60, 1920x1035_60 |
| VPro | frame table at 0x2400, line table at 0x2000, written through the display FIFO | I2C PLL; the format carries the clock in Hz | 71 formats, 640x480 to 1920x1200 |
| O2 | counters with max x / max y compares | dot clock M/N/P | 640x480 to 1600x1024 |

Example: 1280x1024_76 on IMPACT is 1712x1079 at 140.25 MHz = 75.92 Hz.
Guest-readable position: VC2 and VC3 have a line counter (register 0x0B),
VC1 has line and frame counters, GBE has live x/y. VPro has none documented.

### Stereo

Two kinds exist.

- **Full-screen, field-sequential ("STR_RECT").** The 1280x1024 buffer is
  scanned as two 492-line fields, top half then bottom half, with a vertical
  blank between them: 60 frames/s, 120 fields/s. The format marks each field
  with an eye (field 1 left, field 2 right). GR2 (monitor ID 4), IMPACT,
  VPro and O2 all have it, and so does Newport: `setmon` accepts
  `STR_RECT`, `STR_BOT` and `STR_TOP` on XL graphics, and the VC2's
  odd-field timing output is the left/right signal for the glasses. This
  kind needs no extra buffers and no eye select in the XMAP9: the
  application draws the left eye in one half of the ordinary (double
  buffered) framebuffer and the right eye in the other, at half vertical
  resolution, and the timing table alone turns the halves into fields. It
  costs nothing in frame rate. Newport has no windowed stereo.
  `STR_RECT` leaves the X server unaware (the desktop is drawn across both
  halves and looks doubled through the glasses); `STR_TOP` / `STR_BOT` tell
  the X server which half is the primary one.
- **Stereo in a window (quad buffer).** Formats `1024x768_96s`,
  `1280x1024_96s`, `1280x1024_100s`: two full-height fields per frame, 48 or
  50 frames/s. Each window's display mode says whether it is stereo.
  - IMPACT: two visible buffer banks; the window record has a stereo bit and
    a second origin and second set of buffer pointers.
  - VPro: left and right are separate byte fields inside one wide pixel
    (for a 16-byte pixel: left front/back at 7/3, right at 6/2); the
    display-mode word at 0x2900+id has a stereo bit; the timing generator
    drives a STEREO output.
  - O2: software swaps the tile-list pointers once per field.

For a compositor this means: the eye is a property of the field, and a
stereo compose takes an eye argument. For full-screen stereo the eye picks a
half of the buffer; for windowed stereo it picks, per window, which buffer a
stereo display ID reads. Mono windows show the same pixels to both eyes.

### More than one head

| Hardware | Shape |
|---|---|
| Two Newport boards, two GR2 boards, two IMPACT boards | independent boards, each with its own timing and interrupt. Indigo2 XL dual head can frame-lock the second board to the first through the backplane (`setmon -c`), so two heads are not always free-running |
| VPro Dual Channel Display | one framebuffer twice as wide (2560x1024), shown as two channels; the `2@` formats scan two fields per frame, one per channel, 60 Hz each |
| IMPACT channel option | one framebuffer, 2 or 4 rectangles, each with its own DAC and PLL |
| O2 dual channel | exists; how it maps the framebuffer is not known |

So "head" has to mean a channel: a rectangle of some board's framebuffer
with its own timing. Independent boards are the case where each board has
one channel.

## 3. Design

### 3.1 The refresh thread

New module `src/display/` (clock, head registration, frame types, present
targets). One thread, `Display-Refresh`, per machine, created with the
machine and alive until it is dropped; it is not respawned on device
stop/start. It keeps time on the host monotonic clock with absolute
deadlines, in the style of VPro's `RetraceClock`: missed retraces are
caught up one at a time and a mode change restarts the phase.

Every board gets the same frame, in this order. The refresh thread calls a
board callback for each step; boards no longer choose their own order.

1. **vblank start.** The board latches pending swaps and raises its retrace
   interrupt and status.
2. **blank period.** The thread waits until the end of the blank on its
   deadline clock (about 0.4-0.7 ms for the usual formats). No board sleeps.
3. **vblank end.** The board drops its level and status.
4. **capture**, when the picture changed, then **compose**. Capture is its
   own step so the two can be moved apart later; a board that composes
   straight from its memory has no capture. Both come in display time,
   after the blank, because of how the IRIX kernel swaps (checked in the
   GR2, IMPACT and Newport kernel modules): a swap request only schedules
   the swap and takes the client's access away, and the retrace handler
   writes the new display mode and gives the access back. Once the retrace
   is serviced the visible buffer is finished and the client draws into
   the other one. No retry, no second look: a frame once started is shown.
   The blank is the format's own (0.66 ms at 1280x1024, 60 or 76 Hz; 0.79
   ms at 1024x768_60), which is the time the guest has to service the
   retrace before the capture.
5. **host swap**, non-blocking.
6. Screenshot, frame request and VNC hand-off, done by the thread.
7. Sleep until the next vblank start.

With stereo fields or several heads the steps repeat per field and per
head; heads with different rates each have their own deadlines, and the
thread serves whichever is due.

**Why compose does not start before vblank end.** The frame the guest
means to show exists only once its retrace handler has run, and the blank
is the time it has to do that. Starting earlier would read the buffers
before the flip. In host-locked mode (3.3) the whole sequence is placed
against the host's swap instead, which is where the latency gain is.

### 3.2 What a board provides

```rust
pub struct VideoTiming {
    pub width: u32, pub height: u32,          // visible, per field
    pub h_total: u32, pub v_total: u32,
    pub pixel_clock_hz: u64,                  // refresh = clock / (h_total*v_total)
    pub vblank_lines: u32,
    pub fields: Fields,                       // Mono | StereoHalves | StereoPages | Channels(n)
}

pub trait Head: Send + Sync {
    fn vblank_start(&self, field: Field, t: Instant);
    fn vblank_end(&self, field: Field, t: Instant);
    fn compose(&self, eye: Eye, target: &mut Target) -> Composed;
}

pub enum Composed { Frame, Unchanged, NotReady }
```

The board calls `handle.set_timing(VideoTiming)` whenever the guest
reprograms its timing generator; `None` means blanked (keep the last size,
keep interrupts running at the last rate). The common layer resizes the
window, restarts the phase, and exposes `handle.beam(now)` (line, in vblank,
field) so status bits and line counters come from the same clock instead of
ad-hoc wall-clock windows.

`Target` is one of: a memory buffer, the GL context (compose into a GL
texture), or a wgpu device (compose into a buffer or texture). The thread
picks the target from the front-end; the head composes into it. This covers
the software path, egui, Newport's GL compositor, and the wgpu scan-out
units.

**Unchanged frames.** `Composed::Unchanged` means the last frame still
stands. The thread keeps the last composed frame per head and redisplays it
itself: for the status bar heartbeat, a paused machine, an expose or
resize, a screenshot, a new VNC client. The board is not asked again and
keeps no idle counter. A board can also mark a head dirty from any thread
(`handle.damage()`), and the thread may skip calling `compose` entirely
while a head is clean and nothing asks for a frame. While the machine is
stopped in the monitor the thread keeps running and redisplaying; vblank
callbacks stop.

**`NotReady` and the Newport fence.** Newport holds compose back until the
drawing queued before a display-state change has drained (the XMAP fence
against the GFIFO fence, `rex3.rs:3748`, capped at 50 ms today). This is
our own invention and it must survive: without it a palette or mode change
is shown against pixels that were drawn for the old one. In the common
layer it becomes: Newport's `compose` checks the fence and returns
`NotReady` if the queue has not caught up; the thread shows the previous
frame and retries at the next retrace, and a board-supplied cap (the
existing 50 ms) forces the compose so a stuck queue cannot freeze the
screen. The shared thread never blocks on the fence. A unit test in phase 2
pins this: a fenced state change with undrained drawing must not reach the
screen until the drain or the cap. The mechanism is generic, so GR2, IMPACT
and VPro can use it for the same class of problem.

**Screenshots, frame requests and VNC** move to the thread. It holds the
last frame per head, serves `request_screenshot`, `request_frame`,
`set_capture` and the PNG writer from it, and asks for a memory read-back
only when a consumer needs one and the frame lives on the GPU. Boards lose
their copies of this code, and the differences between them (GR2 and O2
ignore frame requests today, InfiniteReality counts them its own way)
disappear. VNC and `iris-ci` take a head index.

What stays in the board: how its interrupt is raised and acknowledged, swap
latching, the fence test, and everything about the framebuffer.

### 3.3 Present and the host display

The refresh thread owns the present targets and blits there, so compose and
blit share a context. Free-running (the default), presents never block:

- GL: swap interval 0. On composited desktops (Wayland, macOS, Windows,
  X11 with a compositor) this does not tear.
- wgpu: `Mailbox` where offered, else `AutoNoVsync`, frame latency 1.
- Compose at most once per guest frame; vblank events keep the guest's
  rate regardless.

**Host-locked mode (planned, not in the first phases).** When the guest's
refresh rate equals the host monitor's (within a small tolerance, say
0.5%), the thread can lock to the host display instead of free-running:

- The host swap becomes the time base. Guest vblank start is placed so
  that steps 1-4 finish just before the host's swap deadline: vblank start
  at `host_vsync - (blank + compose budget + margin)`, with the compose
  budget measured as a running maximum of recent composes.
- The swap is vsynced (interval 1 / Fifo) and its return is the timing
  reference; the guest's period is slewed to the host's measured period, so
  a 60.00 Hz guest on a 59.94 Hz panel runs 0.1% slow and never drops or
  repeats a frame.
- Result: every guest frame shown exactly once, no tearing, latency of
  blank + compose instead of up to a host frame.
- It must drop back to free-running when the rates stop matching (guest
  mode change, window moved to another monitor, monitor mode change), when
  a compose overruns its budget repeatedly, or when more than one head
  with different rates is shown.
- Integer ratios (120 Hz stereo fields on a 60 Hz panel, 60 Hz guest on a
  120 Hz panel) lock the same way with one swap per N retraces or N swaps
  per retrace.

What phase 1 must leave room for: the clock takes its period and phase from
a source that can be swapped (fixed period now, host swap later), and the
frame steps are scheduled relative to a deadline, not run back to back.
`[display] sync = "free" | "host" | "auto"` selects it; `auto` locks only
when the rates match.

`Renderer` moves out of `ng1/rex3.rs` into `src/display/` and loses its
Newport-specific arguments.

### 3.4 Heads, windows and stereo output

The common layer keeps a list of heads. A front-end maps heads to windows:

- `iris` window: one window per head (today head 1 has no window).
  Window calls stay on the event thread through the existing deferred
  request path.
- iris-gui: unchanged, both heads in one window.
- VNC and host input take a head index.
- A VPro Dual Channel board registers two heads that are two halves of one
  framebuffer, phase-locked.

Stereo: a head whose timing has stereo fields gets `compose` called with
`Eye::Left` and `Eye::Right` once per frame, and the pair is published
together (never a left from one frame and a right from the next). Vblank
callbacks still fire per field (120 per second for `1280x492_120s`). Output
modes: left eye only (default on a normal monitor), full SBS, half SBS,
top/bottom, anaglyph (useful as a no-hardware test). Full-screen stereo
halves are stretched 2x vertically, as the monitor does.

### 3.5 Host monitors and glasses

- Enumerate with winit `available_monitors()` on the event thread.
- Identify each monitor by EDID manufacturer ID, product code and name.
  winit gives only the connector name, so this is a small per-platform
  lookup: `/sys/class/drm/*/edid` on Linux, `DisplayConfigGetDeviceInfo` on
  Windows, CoreGraphics vendor/model on macOS. Three EDID fields; no crate.
- Display glasses (XREAL, Rokid, RayNeo, VITURE, Lenovo, ASUS) are ordinary
  monitors. In 3D mode they offer 3840x1080 (3840x1200 on 1200p models) and
  split it into two eyes; in 2D they offer 1920x1080. The switch is usually
  a button on the glasses and the display re-enumerates.
- Detection: EDID name on a built-in list (`Rokid Max`, `SmartGlasses` for
  RayNeo, `VITURE`, XREAL `Air`/`One` only together with manufacturer `MRG`)
  plus a mode at least 3.1 times wider than tall. Aspect alone is not
  enough: 32:9 desktop monitors exist. No EDID data is public for Lenovo
  or ASUS, so the list is extensible from config.
- Re-evaluate on every monitor change; resolve monitors by identity, not
  index.
- No programmatic 2D/3D switching at first. An MIT-licensed driver covers
  XREAL Air and Rokid if wanted later; the GPL and vendor-blob routes are
  out.
- Glasses in SBS run at 60 Hz (XREAL Air also 72 and 90). `1280x492_120s`
  gives 60 pairs/s, an exact match. `_96s` and `_100s` give 48 and 50 pairs
  and will judder at 60 Hz.

Config sketch:

```toml
[display]
sync = "free"              # "free" | "host" | "auto" (host lock when rates match)
monitor = "auto"           # index | connector ("DP-2") | EDID name substring
fullscreen = "off"         # "off" | "borderless"
refresh = "guest"          # "guest" | fixed Hz, as an escape hatch

[stereo]
output = "auto"            # "off" | "auto" | "sbs" | "half-sbs" | "top-bottom" | "anaglyph"
monitor = "auto"           # "auto" = first known glasses in wide mode
swap_eyes = false
known_names = []
```

A monitor command lists heads with their timing and host monitors with
their identity and classification (documented in `debug.md` when added).

## 4. Phases

Each phase lands on its own and leaves every board working.

1. **Common layer, no behaviour change.** `src/display/`: scheduler,
   `Head`, `VideoTiming`, `Target`, the shared loop body (status bar,
   heartbeat, screenshots, requests, stop, quiesce). Unit tests drive it
   with a manual clock and a fake head: edge order, catch-up, mode change,
   two heads at different rates, stop from the main thread. Move `Renderer`
   out of `ng1`.
2. **Migrate boards at a fixed 60 Hz timing**, one commit each, simplest
   first: GR2, IMPACT, VPro (its `RetraceClock` folds into the scheduler),
   InfiniteReality (checked at the PROM prompt), O2, Newport last (fence,
   respawn, the test-only resident display thread, and the wgpu present
   path patched to the new thread; see section 6).

   O2: the GBE has four programmable line interrupts and a guest-readable
   beam position, not one retrace. The head interface therefore takes line
   events: a head may register extra scan lines and gets a callback when
   the beam crosses each, on the same deadline clock; vblank start is the
   common case of one such line. `VT_XY` reads `handle.beam()`, and the
   shadow registers latch at vblank start. The `O2-Service` thread stops
   polling the GBE. Present
   becomes non-blocking in this phase.
3. **Real timing.** Each board decodes its tables and clock into
   `VideoTiming`: VPro and IMPACT first, then GR2, O2, Newport. Status bits
   and line counters read `handle.beam()`. Newport needs the Bt445
   reference frequency; until it is known, match the decoded totals against
   the known formats.

   Method, per board: in the running guest, list the format files
   (`/usr/gfx/ucode/<board>/vof`), switch formats with `/usr/gfx/setmon`
   (and `xsetmon`), and after each switch dump what the guest wrote: the
   frame and line tables, the PLL bytes, the mode bits. Compare the decoded
   total and clock with the values in the format file, which are exact.
   That gives a table of format -> registers -> rate to turn into unit
   tests, and it exercises the mode-change path (resize, phase restart)
   for real. Include the stereo and `2@` formats so their field and channel
   bits are seen even before they are displayed.

   Do the VPro pass as a V10 or V12. The V6/V8 boards have a PLL that
   cannot make every clock (the driver refuses 109-194 MHz on the early
   BUZZ revision), so IRIX hides formats there; as V10/V12 it offers all
   71. Then repeat a few formats as V6/V8 to see the other PLL's bytes.
   Newport formats are in `/usr/gfx/ucode/NG1/vof` in the guest; the
   `[graphics] resolution` presets that pre-load a table must keep working.
4. **Host lock** (section 3.3), once real timing is in: `sync = "auto"`.
5. **Heads and host monitors.** One window per head in `iris`, head index
   for VNC and input, monitor enumeration and identity, `[display]` keys,
   the monitor command.
6. **Stereo.** The common side first (eye-pair frames, SBS and anaglyph
   output, glasses detection), tested with a synthetic head. Then
   full-screen stereo on Newport, GR2 and IMPACT, which needs no new
   buffers: the two eyes are halves of the buffer that is already there,
   and the timing decode from phase 3 already reports the two fields. Then windowed
   stereo on IMPACT and VPro, which needs right-eye buffers in the raster
   path and is the large item.
7. **VPro Dual Channel Display**, if wanted: two heads from one wide
   framebuffer.

Checks per phase: `cargo test --workspace`, the power-on replay goldens,
the IRIX boot test on panther (10/10, graphics login near 259 s), and from
phase 3 a guest-side retrace count over 10 s against the format's rate
(`1280x1024_76` must give about 759).

Documents to update when code lands: `HACKING.md` (threads and locks),
`FEATURES.md` and `HELP.md` (new keys), `debug.md` (monitor command),
`CHANGELOG.md` (refresh rate follows the guest; second head window; stereo),
and the per-board thread notes in `clanker/`.

## 5. Risks

- **Guest speed changes.** A guest that paces itself on retrace runs 20-27%
  faster at 72/76 Hz. Benchmarks and boot-test timings will move in phase 3;
  `refresh = 60` is the escape hatch and the comparison baseline.
- **Tearing** with non-blocking GL swaps on uncomposited X11.
- **GL contexts for several windows on one thread**: a `make_current` per
  window per present; macOS has known traps here.
- **Newport fence** must be kept, and must not stall other heads (3.2).
- **Unified order changes GR2 and IMPACT.** They took their frame before
  the retrace pulse and now take it after the blank
  (`clanker/mgras/display-snapshot-before-retrace.md` needs rewriting).
  The old order was a fix for tearing on Octane (octahedra), so that is
  the regression to test under a desktop; if it tears, the guest is
  servicing the retrace later than the blank lasts.
- **Lock order**: the thread must not hold one board's screen lock while it
  composes or presents another head.
- **Overlap with open work**: the Newport GPU presenter follow-up and the
  shared offscreen presenter draft touch `ui.rs`, `main.rs` and
  `wgpu/present.rs`. We migrate first and carry the wgpu present path
  over, so those branches will need a rebase onto the new thread.
- **O2 on host time**: VT interrupts no longer slow down with the guest.
  Guest code that counts instructions between line interrupts sees a
  different ratio when emulation is slower than real time.
- **Glasses**: no EDID captured in 3D mode for any model, so the name or
  product code may differ there; Wayland cannot set the 3840x1080 mode
  itself; a kernel patch that marks XREAL as non-desktop would hide it.

## 6. Decisions

Taken (2026-11-14):

- **Time base: the host clock.** No general lock to the host display, but
  an exact lock when the guest and host rates match, planned for later
  (3.3).
- **One order for every board**: vblank start, blank, vblank end, compose,
  non-blocking host swap, sleep. The thread calls the board for each step.
- **The thread owns screenshots, frame requests and VNC.** Boards only
  compose, and can answer "unchanged".
- **Newport's fence stays.**
- **Timing is verified in the guest** with `setmon` / `xsetmon` against the
  format files, VPro as V10/V12.

- **Second head: its own window** in `iris`.
- **InfiniteReality migrates with the others.** There is no IRIX image for
  it yet, so its check is the PROM prompt on screen.
- **Newport migrates now**, not after the GPU presenter work. The wgpu
  present path (`src/wgpu/present.rs`, `WgpuRenderer`, the test-only
  resident frame path and its M2 tests) is patched to run on the new
  thread in the same change, so the contributor can continue on the new
  framework. Update `GPUTodo.md` with what moved.
- **O2 vertical interrupts move to the host clock.** The refresh thread
  calls the GBE's interrupt functions like any other board's.

Nothing is open.

## 7. Progress

What is in the tree, where it differs from section 3:

- `src/display/mod.rs`: `Refresh` (the thread, `Display-Refresh`),
  `HeadHandle` (screen, renderer slot, `DisplayStop`, screenshot / frame /
  capture requests, `damage()`, `set_timing()`, `beam()`, `want()`,
  `start()` / `stop()`), the `Head` trait (`capture`, `vblank_start`,
  `vblank_end`, `changed`, `compose`, `live_planes`, `gfifo_pending`),
  `VideoTiming`, `Composed`, `Want`. `Refresh::manual(clock)` runs the same
  schedule with no thread on a clock the test moves (`src/display/tests.rs`).
- The machine's `Refresh` hangs off `StatusSource` (`status.refresh()`),
  because the status source is the one thing every board of a machine is
  already given. A board creates its handle in its constructor
  (`status.refresh().head(status.clone())`) and attaches itself once it is
  in its `Arc` (`head.attach(Arc::downgrade(..))`).
- The frame is still `Rex3Screen`, and `compose` takes `(screen, Want)`,
  not `(eye, target)`: a head answers `Composed::Frame` (finished pixels in
  `fb_rgb`, or Newport display state for the renderer's compositor) or
  `Composed::Gpu { frame, width, height }`. The eye argument comes with
  the stereo phase.
- `Renderer` lives in `src/display/renderer.rs` (re-exported from
  `ng1::rex3`); its signature is unchanged so far.
- The thread fills `screen.rgba` only when a reader asked (screenshot,
  `request_frame`, capture), not on every frame as GR2 and IMPACT did.
  Every reader in the tree asks first.
- The thread returns to its loop after each frame, so a head that is slow
  to compose (debug builds) gets fewer frames, every retrace is still
  delivered, and other heads are not held up.
- Newport's kernel module was taken from the IRIX 6.5.22 guest
  (`ignore/ip24/guest-files/`: the whole boot directory, `/usr/gfx`, and
  `ng1.a` unpacked). Its retrace handler has the same shape as GR2's and
  IMPACT's (`newportSchedSwapBuf`, then `rrmValidateBuffer` from
  `newportRetraceHandler`); not decompiled yet.
- Phase 3 so far (2026-11-15): IMPACT and VPro decode their timing and
  run at the guest's rate, checked with `setmon -n` against the format
  files (`clanker/mgras/video-timing-from-vc3-and-dac-pll.md`,
  `clanker/odyssey/video-timing-from-the-vtg-tables-and-clock-part.md`).
  Newport, GR2 and the O2 followed
  (`clanker/rex3/video-timing-from-vc2-and-bt445-pll.md`,
  `clanker/gr2/video-timing-from-vc1-tables.md`); InfiniteReality runs
  `VideoTiming::HZ60`.
- Compose only when the shown picture changed (2026-11-15):
  `clanker/devices/draw-marks-tell-the-display-what-shows.md`. IMPACT
  composes straight from the pixel pages (6.5 ms), which fits 76 Hz.

VPro: the retrace (swap latch and interrupt packet) is `vblank_start`, and
the frame is composed after the blank with no capture, because the latch
itself picks the displayed buffer. `changed()` is always true (the reader
finds out what changed while composing) and `compose` answers `Unchanged`.
`RetraceClock::next_retrace` and `ScanoutReader::poll` remain for the
tests, which run frames with no refresh thread (`Odyssey::frame`).

Newport: `Rex3` is not built in an `Arc`, so its head is a small shim
(`Rex3Head`) made at the first `start`, when the board is at its final
address; `Drop for Rex3` stops the head. Vblank start raises VRINT and the
line and latches the cursor's working Y. `changed()` peeks the damage and
dirty flags; `capture` is `Rex3Screen::refresh` behind the fence (refused
while `gfifo_fence` is behind `xmap_fence`, for at most 50 ms); `compose`
only answers `Frame`, since the renderer's compositor makes the picture.
The common layer took over the software compose for a reader with no
renderer (`--ci`), forcing the software compositor while capture is on,
and the debug overlay hook. The PROM screenshot is pixel-identical to the
old thread's. The test-only resident display thread still runs its own
loop on `refresh_thread`, reading the head's screen and renderer; the wgpu
tests compile but were not run (the desktop card was in use).

O2: `O2Display` composes GBE's scan-out every frame (`changed()` is always
true: GBE reads RAM with no mark to test).

**No catch-up (2026-11-15, user decision).** The rate is the guest's
timing or the compose time, whichever is slower. A retrace reached more
than a sixteenth of a period late happens then and the schedule slips to
it; missed retraces are not delivered afterwards. This replaces the
catch-up of up to four fields described above.

**O2 interrupts (done 2026-11-15).** GBE's four timing interrupts are
line compares, raised by the refresh thread: `HeadHandle::set_line_events`
and `Head::line` in the common layer, `Gbe::retrace` (shadow latch, beam
anchor, the lines) and `Gbe::timing_interrupt`. The old reason for keeping
them on the CPU's instruction count (a dispatcher that never returned) did
not reproduce: see `clanker/devices/o2-timing-interrupts-are-line-compares.md`.

InfiniteReality: its `Display` makes its head when first used, from the
status source it has by then (the back end is built detached and given the
machine's status right after). `DisplayThread` keeps its name and its
`spawn`/`stop`, and is now a head: vblank start completes the retrace
requests and raises the video interrupt (`Clock::frame`), `changed()` is
the store's touched flag, `compose` composes into the screen. `quiesce`
sets a flag the callbacks honour and waits out one in flight. A head can
be attached again (each start makes a new one) and re-statused.

Not done yet in phase 2: a live InfiniteReality run,
the wgpu resident-frame tests run and the resident path folded into the
head (`swap interval 0`, wgpu `Mailbox`), and the
`clanker/mgras/display-snapshot-before-retrace.md`, `HACKING.md` and
per-board thread notes.

Seen on the way, not investigated: an Indigo2 IP22 with Solid IMPACT and a
fresh NVRAM/EEPROM never initialises the board in two minutes (the same
board on IP28 with its saved NVRAM reaches the PROM screen). Check against
main before blaming this branch.
