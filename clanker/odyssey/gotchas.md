# VPro gotchas

Status: rule (2026-10-30).

- **An IMPACT install has no VPro driver.** An Octane disk installed with an
  IMPACT board boots, but X finds no screen on the VPro. Use a disk
  installed on a VPro machine (see `bringup.md`). Do not patch a disk by
  hand; install with the install tooling.

- **Textport output costs per FIFO word.** A scrolled line in the PROM or an
  IRIX textport is hundreds of thousands of command-FIFO words, so slow
  scrolling is store cost, not draw cost. Take the host interface's lock per
  store and the textport crawls; the ring in `fifo.rs` exists for this. Do not
  add per-store work (logging, perfetto events, extra atomics beyond the
  level and wake flag) to the push path. `VPro cfifo level` is sampled for
  that reason.

- **The depth path is double precision, on purpose.** The clip, window and
  depth maths accumulate in `f64`. In `f32` a vertex just beyond the far plane
  under a perspective model-view rounds onto it, and the keep-or-clip
  decision flips. Lighting uses `dev::gl`'s single-precision core because the
  board's microcode does; transform, clip planes and the window map do not.
  Do not move them onto `dev::gl` without a far-plane test, and expect pixel
  changes if you do.

- **Revision code 4 vs 5.** A V10 or V12 needs the Octane2's Crossbow
  (revision code 5) with its reset fences and resource lock. With code 4 the
  PROM's fence writes hit a reserved range and the machine never reaches the
  graphics init.

- **PROM too old.** V10/V12 need PROM 4.5, V6/V8 4.0; iris warns at start-up.
  Heed the warning before reporting a hang.

- **Board profile or X format surprises.** V6/V8 start at 1280x720_72 because
  the guest refuses 1280x1024_60 at 16-byte depth on BUZZ A.1. That is not a
  scanout bug.

- **Retrace can be late by a slice.** It is counted under the board lock,
  which the worker holds for up to 1 ms. Swap timing measured to better than
  that is noise.

- **Approximations to remember.** The FIFO holds 64 stores in the model; the
  kernel's set-up asks for far more. The low-water mark (16) is the model's.
  The level is stores, not words. All are listed with their tests in the
  datasheet's "Command FIFO level and credits".

- **No Mesa renderer.** The host-Mesa path from the QEMU model was left out;
  everything is drawn by the scalar rasteriser.
