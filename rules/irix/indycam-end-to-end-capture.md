# IndyCam end-to-end capture on iris — campaign notes

**Keywords:** vino, indycam, vidtomem, vlinfo, capture, camera, end-to-end, campaign
**Category:** vino, irix
**Status:** Working on both IRIX 5.3 and 6.5 (PR #28, commit `751a566`).
`vidtomem -f /tmp/cap -v 0` produces a 640×480 RGB frame. Still-image capture
in the media capture GUI app also works. Movie/continuous capture does not yet
work (multi-buffer DMA ring not fully supported). Linux V4L2 camera backend
(`camera_v4l.rs`) confirmed working end-to-end on Linux with Huddly GO.
One known visual artifact — see "Open issues" below.

## The path that finally works

```bash
# host
cp iris.toml ...   # ensure [vino] source = "camera" (or "test_pattern")
                   # ensure [scsi.2] scratch = true, size_mb = 16+
./target/release/iris --ci
./target/release/iris-ci start
# … usual PROM → Start System → boot → root login dance …

# guest, on the IRIX shell
/usr/etc/videod &
vidtomem -f /tmp/cap -v 0 && \
  dd if=/tmp/cap-00000.rgb of=/dev/rdsk/dks0d2s0 bs=512 conv=sync

# host: retrieve via iris-ci scratch read (fast — local Unix socket, no
# uuencode-over-serial). Must use s0 not vol; see src/ci.rs comment.
./target/release/iris-ci scratch read /tmp/grab.rgb --length $(wc -c </tmp/cap-00000.rgb)
file /tmp/grab.rgb   # → SGI image data, RLE, 3-D, 640 x 480, 3 channels
```

## The eight things that had to be fixed (in order)

Each of these silently broke the previous one's success. The order is the
order they fail in if you walk the IRIX vino driver's init path.

1. **`vino_gio_alias` offset** (`src/physical.rs`, commit `d1808b9`).
   GIO64-aperture reads at `0x1F080000` were being routed to `0x1E080000`
   (gio-error region) instead of `0x00080000`. Wrapping-add offset was
   `0xFF000000` (= `-0x01000000`); had to be `0xE1000000` (=
   `-0x1F000000`). Pure off-by-an-F mistake.

2. **MC SYSID bit 4** (`src/mc.rs`, commit `4cf7c67`).
   The IRIX 5.3 vino driver's `vino_init()` reads MC SYSID at `0xBFA0001C`
   and silently bails if bit 4 is clear. Per the MC datasheet bit 4 = "EISA
   present"; on Indy that's clear per spec. The vino driver uses it as a
   vino-board-present gate anyway, so iris has to report it set. Without
   this, `videod` prints "no boards found".

3. **CDMC I2C address `0xAE/0xAF` → `0x56/0x57`** (`src/cdmc.rs`,
   commit `2af7ead`). iris was using a stale 0xAE assumption from
   `IRIX 6.5 indycam.h`. The actual IRIX 5.3 driver writes 0x56/0x57 (=
   7-bit address 0x2B << 1) — verified by literal scan of `vino_*.o`
   immediates.

4. **I2C_DATA writes trigger transfer when NOT_IDLE set** (`src/vino.rs`,
   commit `2af7ead`). Real VINO sends a byte to the slave on every
   I2C_DATA write while NOT_IDLE is asserted. iris was only triggering on
   I2C_CONTROL writes, so only the very first byte of each I2C transaction
   ever reached the slave.

5. **I2C_CONTROL trigger only on rising edge of NOT_IDLE** (`src/vino.rs`,
   commit `2af7ead`). The flip-side of #4: once you trigger on every
   I2C_DATA write, you must NOT also trigger on every I2C_CONTROL poll,
   otherwise every byte goes out twice and CDMC state gets scrambled.

6. **CDMC repeated-start recognition** (`src/cdmc.rs`, commit `2af7ead`).
   IRIX reads CDMC registers via `START → 0x56 → subaddr → REPEATED-START
   → 0x57 → read`. The CDMC state machine has to recognise its own slave
   address re-arriving mid-transaction as a repeated-start, not as a
   data byte.

7. **CDMC `CAMERA_ID` register at subaddr 0x0E = 0x10** (`src/cdmc.rs`,
   commit `2db20fb`). `vinoCameraAttached()` reads CDMC subaddr 0x0E and
   requires *exactly* 0x10 to consider the camera present (disassembly in
   vino_main.o); otherwise the kernel prints "IndyCam not attached.
   [HELP=VINONOCAMERA_WARN]" and refuses to schedule frame DMA. iris's
   CDMC register space had to grow from 9 slots to 15 to expose this
   register.

8. **HPC1 region (`0x1FB00000..0x1FB80000`) black-holed** (`src/physical.rs`,
   commit `2db20fb`). iris doesn't emulate HPC1. The CPU-bus-error path
   that catches accesses there is tolerated during boot but escalates to
   a hard kernel panic during vino capture-pipeline setup (`PANIC: IRIX
   Killed due to Bus Error` at `0x1FB02000`). Mapping the region to
   BlackHoleRegion (read-zero, write-eat) silences it without implementing
   HPC1 semantics.

9. **Vino sub-word access (`read8/16, write8/16`)** (`src/vino.rs`,
   commit `2db20fb`). IRIX issues at least one 16-bit read at
   `VINO_BASE + 0x16`; iris's BusDevice impl only had 32/64-bit, so the
   default trait err triggered a `PANIC: KERNEL FAULT … Bad addr:
   0xa0080016`. Implementations extract the appropriate sub-field of the
   underlying 32-bit register.

10. **Interlace: rewind DMA cursor at field boundaries** (`src/vino.rs`,
    commit `e72ba22`). `pump_field()` reloads descriptors from
    `start_desc_ptr` and sets `page_index` to either 0 (Even field) or
    `line_size + 8` (Odd field, = actual row stride since `CH_LINE_SIZE`
    is encoded as stride-minus-8) at the start of each captured field.
    Without this the Odd field continued past the end of the frame
    buffer and the captured image's odd memory rows stayed zero.

11. **Interlace skip-trigger condition `>=` → `>`** (`src/vino.rs`,
    commit `e72ba22`). The line-size register encodes the *last dword's*
    offset within the line — one dword short of the actual stride. The
    trigger must fire after `line_counter` exceeds `line_size`, not after
    it equals it. With `>=` the last dword of every row went to the next
    row's territory and a 2-px-per-row drift accumulated.

12. **nokhwa-on-macOS YUYV is YVYU byte-order** (`src/camera.rs`,
    commit `e72ba22`). The nokhwa AVFoundation backend delivers Cr at
    byte 1 and Cb at byte 3 (not the textbook Cb-then-Cr YUYV order).
    Reading them straight to `yuv_to_rgb` swapped U and V in every pixel,
    producing the washed-out turquoise/red cast in the first end-to-end
    camera capture. Swap on read at the camera-source layer.

## Open issues

- **Diagonal artifact in Even-field rows.** A thin dark diagonal line
  drifts at -1 col per output row from col 638 at row 0 toward col 0 at
  row 478. Cosmetic — geometry and colour are otherwise right. Likely a
  stride/descriptor-layout mismatch we don't yet reproduce from
  `line_size` alone. Worth instrumenting the kernel-side memory writes
  (e.g. log every CPU write into the frame buffer region) to find where
  the drift originates.
- **Two captured rows short of full frame.** Clip register from the
  driver covers source y = 2..240 (Even) and y = 1..239 (Odd) — 239 rows
  per field, 478 total instead of 480. The last two output rows stay
  zero. Not the same bug as the diagonal.

## Mechanics worth keeping

- **inst-watch.py** (`tools/inst-watch.py`) — classifies stall events
  from `irix-install-console.log` so the harness can react fast.
- **Scratch SCSI exchange.** Add `[scsi.2] scratch = true, size_mb = 16`
  to `iris.toml` for fast host↔guest file exchange (~500 KB/s vs ~10 KB/s
  over the serial uuencode pipe). Use `dks0d2s0` from the guest, not
  `dks0d2vol` — `vol` overlaps the SGI VH and writing to it corrupts the
  partition table. `iris-ci scratch read --length N <path>` retrieves
  the payload.
- **Disassembling `vino_*.o`** — `uuencode` over serial → `bsdtar xf` on
  the host → `capstone` (Python `pip install capstone`) with the
  big-endian MIPS-II mode. ECOFF EXTR records are 16 bytes (`u32`
  reserved, `u32 iss`, `u32 value`, `u32 bits`); SYMR offsets in HDRR
  are at byte offsets 64/68 (issExtMax/cbSsExtOffset) and 88/92
  (iextMax/cbExtOffset). A `tools/disasm-vino.py` helper was used for this but never checked in; the guest-side VINO tools that were are in `tools/vino/`.

## See also

- [vino-gio-alias-offset.md](vino-gio-alias-offset.md)
- [vino-attach-via-sysid-bit4.md](vino-attach-via-sysid-bit4.md)
- `rules/irix/irix-install.md` — install procedure + lessons
- `tools/inst-watch.py` — prompt-classifying tail for the serial log
