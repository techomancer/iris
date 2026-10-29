# Built-in VNC server — plan

Status: implemented 2026-10-29 on branch `vnc` (phases 1-5): `src/vnc/`,
`src/hostinput.rs`, `src/keysym.rs`, monitor `screen`/`ptr`/`key`, MCP
tools, docs. Tested on Indy Newport (window and `--ci`), XZ, IP28 IMPACT,
Octane, O2, with TigerVNC and `tools/vnc-check.py`. Not done: VNC password,
second head, CI-socket input commands. Design notes in `clanker/vnc/`.

## Goal

An optional RFB (VNC) server in the `iris` binary (not iris-gui) that works:

- windowed (alongside the winit window; input from both is merged),
- with `--ci` (no window; offscreen rendering already runs),
- for every board: Newport (REX3), GR2, IMPACT (Indigo2 + Octane), O2 GBE.

It also gives LLM agents and CI a "see the screen, click at x,y" path through
the monitor/MCP/CI socket, reusing the same frame tap and pointer logic.

Hard constraints:

- **No instrumentation in drawing code.** No damage tracking in REX3/RE3/RSS
  draw paths. Damage comes from diffing composed frames.
- **One path for all boards.** VNC consumes the composed frame
  (`Rex3Screen::rgba`, stride 2048, `0xAABBGGRR`) and nothing else.
- REX3 is forced onto the software compositor while VNC is on.
- Off by default; costs nothing when off.

## Where frames come from today (survey summary)

- `GfxDisplay` (`src/gfx_display.rs`) is common to all boards; `screen()`
  gives `Mutex<Rex3Screen>` with `rgba`, `width`, `height`.
- GR2, IMPACT, O2 compose on their own refresh threads and keep `rgba`
  current (`prebuilt = true`).
- REX3 composes inside the installed `Renderer` (`SwCompositor` or
  `GlCompositor`) and fills `rgba` only when `need_readback` (screenshot).
  With `--ci` there is no renderer, so `ci.rs` composes inline.
- Input: everything goes through `Ps2Controller::push_kb(KeyCode, bool)` and
  `push_mouse_input(buttons, dx, dy, dz)`. **Mouse is relative only**, and X
  applies pointer acceleration.
- Each board knows its hardware cursor position: REX3 VC2 cursor X/Y regs
  (+`cursor_x_adjust`), GR2 `Vc1::cursor_pos`, IMPACT `Dcb::cursor`, O2 GBE
  cursor regs.

## Design

### 1. Frame tap (core, board-agnostic)

Extend `GfxDisplay`:

```rust
fn frame_seq(&self) -> u64;                   // bumps whenever rgba is rewritten
fn request_frame(&self);                      // "please make rgba current", no PNG
fn cursor_pos(&self) -> Option<(i32, i32)>;   // live HW cursor, screen coords; None if disabled
```

- `Rex3Screen` gets `frame_seq: u64`, bumped by whoever writes `rgba`: the
  three prebuilt refresh loops (GR2, IMPACT, O2) and the REX3 renderer
  readback. These are refresh-loop / present-time sites, not draw code.
- REX3: a separate `frame_wanted: AtomicBool` (the existing
  `screenshot_pending` also writes a PNG file, so it can't be reused). The
  refresh loop ORs it into `need_readback` and forces a present on the next
  tick when content changed.
- REX3 forced SW: while a frame consumer is registered, `GlRenderer` uses
  `SwCompositor`; `disp compositor gl` is refused with a message. Readback is
  then `read_pixels()`, a slice copy, no GL readback.
- No-window REX3: move iris-gui's `CaptureRenderer` idea into core as
  `SwCaptureRenderer` (SwCompositor → `rgba`) and install it when VNC (or CI)
  needs frames and no renderer is present. `ci.rs`'s inline compose then goes
  away.
- `cursor_pos` reads live device state (VC2/VC1/DCB/GBE), not the 60 Hz
  snapshot, so the pointer loop sees updates promptly.

### 2. Damage from diffs

One grabber thread per served head:

- Tick at `max_fps` (default 30). If `frame_seq` unchanged, skip.
  Otherwise `request_frame()`, lock `screen`, copy the visible
  `width × height` into a tight `cur` buffer, unlock (~1–2 ms at 1280×1024).
- Compare `cur` with `prev` in 64×64 tiles (row slices compared as `[u64]`,
  early exit on first mismatch). Changed tiles get `tile_gen[t] = frame_gen`.
- Optionally shrink each dirty tile to its exact bounding box before sending.
- Resolution change → new buffers, every tile dirty, DesktopSize to clients.
- Swap `cur`/`prev`, notify clients (Condvar).

Clients keep `seen_gen`; an incremental FramebufferUpdateRequest blocks until
some `tile_gen > seen_gen`, then sends dirty tiles inside the requested rect,
merging horizontal runs into wider rects. Non-incremental requests send the
whole rect. Slow clients just skip intermediate frames (no queue).

### 3. RFB server (`src/vnc/`)

Own implementation, std::net, thread per client like the monitor/CI socket
(no RFB crate worth depending on; no tokio in the tree).

- RFB 3.8 (accept 3.3/3.7), security type None. Bind `127.0.0.1:5900` by
  default; warn loudly on a non-loopback bind. VNC password auth later.
- Pixel formats: server native 32bpp; honour SetPixelFormat for 32/16/8bpp
  true-colour by shift/max conversion. No colour-map mode.
- Encodings: Raw (required), ZRLE (per-connection zlib stream, `flate2` as
  a direct dep; supported by TigerVNC, RealVNC and noVNC), pseudo-encodings
  DesktopSize and QEMU Extended Key Event. Tight/JPEG and CopyRect later, if
  ever.
- Cursor: drawn into the frame by the board compositors, so no cursor
  pseudo-encoding. The viewer's local dot cursor works.
- Max clients 8; on disconnect, release that client's held keys/buttons.

### 4. Input

**Keyboard.** Prefer the QEMU Extended Key Event (client sends XT scancodes
→ map to `KeyCode`, layout-independent; TigerVNC and noVNC send it). Fallback:
X keysym → `KeyCode` table for a US layout (`src/keysym.rs`), mapping a
shifted symbol to its physical key. Clients already send Shift
themselves. The same table backs monitor `key NAME` and can replace
`ip32_screen::keycode`.

**Pointer: closed loop on the hardware cursor.** VNC gives absolute
positions; the guest wants relative PS/2 packets that X then accelerates.
Rather than model acceleration, steer:

- A `PointerSeek` (`src/hostinput.rs`) holds the target. Every ~8 ms:
  `err = target - (cursor_pos() + hotspot)`. Send a delta clamped to a
  step size that shrinks as `|err|` falls (big accelerated steps far away,
  ≤ accel-threshold steps near). Stop at `err == 0` or after N stalled ticks.
- Button/wheel events queue behind the pending move: send them once the move
  converges, or after a timeout (~150 ms), so a click lands where it was
  aimed.
- **Fallback, open loop**: if `cursor_pos()` is `None` (cursor disabled) or
  the cursor stops moving while we push (an app grabbed and warps the
  pointer), switch to relative mode: send the VNC position delta as is.
- **Hotspot**: the hardware position is the glyph's top-left, X sets it to
  `pointer - hotspot`, and the hotspot isn't in hardware. Start with the
  first opaque pixel of the cursor glyph (the arrow tip), then calibrate per
  board with `xclick`/`xgetpix` (open item below).
- Both the window and VNC feed the same `Ps2Controller`; the seek only runs
  while VNC pointer events arrive, so the two don't fight unless both are
  used at once.

### 5. LLM / CI path (same machinery, no VNC client needed)

Monitor commands, registered by the host-input module (document in
`debug.md`):

- `screen shot PATH [x y w h]`, `screen wait [ms]` (block until the frame is
  stable or has changed), `screen seq`
- `ptr move X Y`, `ptr click X Y [left|middle|right] [double]`,
  `ptr drag X1 Y1 X2 Y2`, `ptr pos`
- `key NAME [down|up]`, `key ctrl+alt+F1`, `key type TEXT`

MCP (`clanker/iris_mcp.py`): `screenshot` returning image content (optional
crop/scale), `click`, `move_pointer`, `type_text`, `press_key`,
`wait_screen`. CI socket and `iris-ci`: `click`, `key`, `type`, `screenshot
--head`. Skills (`irix-gfx-loop`, `irix-guest-loop`): replace the
corner-pushing and `xclick` guidance with these.

### 6. Configuration

```toml
[vnc]
listen  = "127.0.0.1:5900"   # absent = off
max_fps = 30
head    = 0                  # 1 = second Newport head (later: both, 5900/5901)
```

CLI: `--vnc [ADDR]`. Runtime option, no Cargo feature (the only new
dependency is `flate2`, already in the lockfile). `--headless` with Newport
has no display: refuse `--vnc` with a hint to use `--ci`.

## Phases

1. **Frame tap**: `GfxDisplay` additions, `frame_seq`, REX3 `frame_wanted`
   + forced SW, core `SwCaptureRenderer`, `cursor_pos` for all four boards.
   ci.rs screenshot moves onto it. Unit tests: seq bumps, rgba current
   under `--ci` for REX3.
2. **RFB output**: handshake, Raw, ZRLE, tile diff, DesktopSize. Test with
   an in-process minimal RFB client (handshake, full + incremental update,
   pixel check) and manually with TigerVNC and noVNC.
3. **Input**: Extended Key Event + keysym table, `PointerSeek`, button
   ordering, release on disconnect. Measure click accuracy on each board.
4. **LLM path**: monitor `screen`/`ptr`/`key`, MCP tools, CI socket
   commands, skill updates.
5. **Docs**: HELP.md, FEATURES.md, debug.md, HACKING.md (`GfxDisplay`
   change, frame tap), CHANGELOG.md.

## Open items

- Hotspot offset per board; does IRIX X use the HW cursor on every board
  (GR2/IMPACT/O2)? If any uses a software cursor, `cursor_pos` is `None`
  there and the seek falls back to open loop. Verify.
- IRIX default `xset m` acceleration/threshold: sets the "near" step size.
- Grab cost at 1600×1200 × 30 fps under the screen lock; if it shows in
  profiles, double-buffer `rgba` at the producer instead.
- Two heads: one listener per head, or one with a head switch.
