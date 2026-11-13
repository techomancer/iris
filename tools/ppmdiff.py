#!/usr/bin/env python3
"""Compare two directories of glprim read-backs (PREFIX_<mode>.ppm/.pgm),
case by case: a real board's against the emulator's.

  tools/ppmdiff.py HW_DIR EMU_DIR [--out DIR] [--zoom N]

For each image present in both: the pixels that differ, the largest
channel difference, and with --out a side-by-side PNG (board | emulator |
difference in red) scaled by --zoom, for the cases that differ. Rows are
printed bottom-up as GL reads them, so the images are flipped to the
screen's orientation.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image


def load(path):
    im = Image.open(path)
    a = np.asarray(im)
    if a.ndim == 2:
        a = a[:, :, None]
    return a.astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hw")
    ap.add_argument("emu")
    ap.add_argument("--out")
    ap.add_argument("--zoom", type=int, default=6)
    a = ap.parse_args()
    names = sorted(n for n in os.listdir(a.hw) if n.endswith((".ppm", ".pgm")) and os.path.exists(os.path.join(a.emu, n)))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    same = 0
    for n in names:
        h = load(os.path.join(a.hw, n))
        e = load(os.path.join(a.emu, n))
        if h.shape != e.shape:
            print(f"{n}: size {h.shape} vs {e.shape}")
            continue
        d = np.abs(h - e)
        diff = (d.max(axis=2) > 0)
        count = int(diff.sum())
        if count == 0:
            same += 1
            print(f"{n}: identical")
            continue
        only_hw = int(((h.max(axis=2) > 0) & (e.max(axis=2) == 0)).sum())
        only_emu = int(((e.max(axis=2) > 0) & (h.max(axis=2) == 0)).sum())
        print(f"{n}: {count} pixels differ, max {int(d.max())}; lit on the board only {only_hw}, in the emulator only {only_emu}")
        if a.out:
            hh, ww = diff.shape
            rgb = lambda x: x if x.shape[2] == 3 else np.repeat(x, 3, axis=2)
            red = np.zeros((hh, ww, 3), np.int32)
            red[diff] = [255, 0, 0]
            gap = np.full((hh, 2, 3), 80, np.int32)
            row = np.concatenate([rgb(h), gap, rgb(e), gap, red], axis=1).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(row).resize((row.shape[1] * a.zoom, hh * a.zoom), Image.NEAREST)
            img.save(os.path.join(a.out, n.rsplit(".", 1)[0] + ".png"))
    print(f"{same} of {len(names)} identical")


if __name__ == "__main__":
    sys.exit(main())
