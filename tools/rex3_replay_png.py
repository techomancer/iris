#!/usr/bin/env python3
"""Replay a `log rex <file>` REX3 trace into PNG images of what the HOST sent.

This deliberately does NOT model the emulator's pixel pipeline (no blending, no
dithering, no 12bpp quantisation, no alpha test). It just plots, per drawn pixel,
the RGBA the host loaded into COLORRED/GRN/BLUE/ALPHA. The point is to see the
application's intent — where it thinks the rock is opaque and where transparent —
so pipeline bugs can be told apart from bad input.

Key trace facts this relies on (see rules/rex3/blendalpha-and-alpha-blending.md):
  * STOPONX=0 on every billboard draw, so ONE GO == ONE PIXEL at (xstart, ystart).
    xend is the remaining distance to the polygon edge, NOT a span width.
  * Colour registers are IEEE floats biased by +4096; the hardware masks off the
    exponent. COLORRED keeps 24 bits, the others 20; integer part is bits 22:11.

Outputs (default prefix "rock"):
  <p>_rgb.png    source colour as sent, alpha ignored
  <p>_alpha.png  source alpha as greyscale (black=0, white=255)
  <p>_rgba.png   colour composited over a checkerboard using the sent alpha —
                 i.e. what the app intends you to see
  <p>_lowalpha.png  magenta where alpha < THRESHOLD, matching `rex alphadebug`

Usage:
    python3 tools/rex3_replay_png.py raw.log [--prefix rock] [--frame N]
                                     [--threshold 16] [--dm1 3165d011,3165d031]
"""
import re, sys, os
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    sys.exit("needs Pillow:  pip install Pillow")


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


LOG       = sys.argv[1] if len(sys.argv) > 1 else "raw.log"
PREFIX    = arg("--prefix", "rock")
THRESHOLD = arg("--threshold", 16, int)
FRAME     = arg("--frame", None, int)
DM1_FILTER = arg("--dm1", None)
if DM1_FILTER:
    DM1_FILTER = {int(v, 16) for v in DM1_FILTER.split(",")}

P = re.compile(r'REX3 Process: Offset ([0-9a-f]+) \(([A-Z0-9_]+)\) Val ([0-9a-f]+)')
D = re.compile(r'REX3 Draw: (\S+) (\S+) Mode0=([0-9a-f]+) Mode1=([0-9a-f]+)')
C = re.compile(r'Coords: Start\(([-\d.]+), ([-\d.]+)\) End\(([-\d.]+), ([-\d.]+)\)')

R_RED, R_ALPHA, R_GRN, R_BLUE = 0x0200, 0x0204, 0x0208, 0x020c


def clamp_component(c):
    """o12.11 DDA register -> 0..255, matching Rex3Context::clamp_color_component."""
    if c & (1 << 31):
        return 0
    v = (c >> 11) & 0x1FF
    if v >= 0x180:
        return 0
    return 0xFF if v > 0xFF else v


def main():
    st = {}
    pending = None
    pixels = []          # (x, y, r, g, b, a)
    frame = 0
    # A frame boundary is hard to detect reliably; use FASTCLEAR/BLOCK clears as a
    # hint, falling back to "all draws" when the caller doesn't ask for one frame.
    for line in open(LOG):
        m = P.search(line)
        if m:
            st[int(m.group(1), 16)] = int(m.group(3), 16)
            continue
        m = D.search(line)
        if m:
            adr, dm1 = m.group(1), int(m.group(4), 16)
            if adr == 'BLOCK':
                frame += 1
            if DM1_FILTER is not None and dm1 not in DM1_FILTER:
                pending = None
                continue
            if FRAME is not None and frame != FRAME:
                pending = None
                continue
            pending = (
                clamp_component(st.get(R_RED, 0) & 0xFFFFFF),
                clamp_component(st.get(R_GRN, 0) & 0xFFFFF),
                clamp_component(st.get(R_BLUE, 0) & 0xFFFFF),
                clamp_component(st.get(R_ALPHA, 0) & 0xFFFFF),
            )
            continue
        m = C.search(line)
        if m and pending is not None:
            # STOPONX=0 -> one pixel per GO, located at (xstart, ystart).
            x = int(float(m.group(1)))
            y = int(float(m.group(2)))
            r, g, b, a = pending
            pixels.append((x, y, r, g, b, a))
            pending = None

    if not pixels:
        sys.exit("no matching draws found")

    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    print(f"{len(pixels)} drawn pixels   bbox x {x0}..{x1}  y {y0}..{y1}   -> {w}x{h}")

    rgb   = Image.new("RGB",  (w, h), (0, 0, 0))
    alpha = Image.new("L",    (w, h), 0)
    over  = Image.new("RGB",  (w, h), (0, 0, 0))
    low   = Image.new("RGB",  (w, h), (0, 0, 0))
    prgb, palpha, pover, plow = rgb.load(), alpha.load(), over.load(), low.load()

    # checkerboard so "transparent" is visually obvious in the composited view
    for yy in range(h):
        for xx in range(w):
            c = 96 if ((xx >> 3) + (yy >> 3)) & 1 else 160
            pover[xx, yy] = (c, c, c)
            plow[xx, yy] = (c, c, c)

    n_low = 0
    for x, y, r, g, b, a in pixels:
        xx, yy = x - x0, y - y0
        prgb[xx, yy] = (r, g, b)
        palpha[xx, yy] = a
        # composite source over whatever the checkerboard/previous pixel holds
        br, bg, bb = pover[xx, yy]
        pover[xx, yy] = ((r * a + br * (255 - a)) // 255,
                         (g * a + bg * (255 - a)) // 255,
                         (b * a + bb * (255 - a)) // 255)
        if a < THRESHOLD:
            plow[xx, yy] = (255, 0, 255)
            n_low += 1
        else:
            plow[xx, yy] = (r, g, b)

    for img, suffix in ((rgb, "rgb"), (alpha, "alpha"), (over, "rgba"), (low, "lowalpha")):
        path = f"{PREFIX}_{suffix}.png"
        img.save(path)
        print(f"  wrote {path}")
    print(f"  pixels with alpha < {THRESHOLD}: {n_low} ({100*n_low/len(pixels):.1f}%)")


if __name__ == "__main__":
    main()
