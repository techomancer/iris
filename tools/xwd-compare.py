#!/usr/bin/env python3
"""Compare an X window dump (xwd -root, i.e. XGetImage of the screen) with
the IMPACT framebuffer the emulator holds (`mgras fbdump`'s fb.bin).

  xwd-compare.py root.xwd mgrasdump/fb.bin [--diff out.png]

For an 8-bit (PseudoColor) dump each pixel must equal the low byte of the
framebuffer word at the same screen position (fb.bin is big-endian u32,
2048x2048, row 0 at the bottom; the screen is its bottom-left corner, so
screen row y is fb row height - 1 - y, height being the dump's). Prints the header, the match rate and
where mismatches are.
"""
import argparse
import struct
import sys
import zlib

W, H = 2048, 2048  # fb.bin: the whole framebuffer


def read_xwd(path):
    d = open(path, 'rb').read()
    f = struct.unpack('>25I', d[:100])
    names = ['header_size', 'file_version', 'pixmap_format', 'pixmap_depth', 'pixmap_width', 'pixmap_height',
             'xoffset', 'byte_order', 'bitmap_unit', 'bitmap_bit_order', 'bitmap_pad', 'bits_per_pixel',
             'bytes_per_line', 'visual_class', 'red_mask', 'green_mask', 'blue_mask', 'bits_per_rgb',
             'colormap_entries', 'ncolors', 'window_width', 'window_height', 'window_x', 'window_y',
             'window_bdrwidth']
    h = dict(zip(names, f))
    colors = []
    off = h['header_size']
    for i in range(h['ncolors']):
        pix, r, g, b, flags, pad = struct.unpack('>IHHHBB', d[off + 12 * i: off + 12 * i + 12])
        colors.append((pix, r >> 8, g >> 8, b >> 8))
    data = d[off + 12 * h['ncolors']:]
    return h, colors, data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xwd')
    ap.add_argument('fb')
    ap.add_argument('--diff')
    a = ap.parse_args()
    h, colors, data = read_xwd(a.xwd)
    print({k: h[k] for k in ('pixmap_format', 'pixmap_depth', 'pixmap_width', 'pixmap_height', 'bits_per_pixel',
                             'bytes_per_line', 'visual_class', 'ncolors', 'byte_order')})
    fb = open(a.fb, 'rb').read()
    if h['bits_per_pixel'] != 8:
        sys.exit("only 8-bit dumps are compared so far")
    w, hh, bpl = h['pixmap_width'], h['pixmap_height'], h['bytes_per_line']
    word = lambda x, y: struct.unpack_from('>I', fb, ((hh - 1 - y) * W + x) * 4)[0]
    bad = []
    for y in range(hh):
        row = data[y * bpl: y * bpl + w]
        for x in range(w):
            if row[x] != word(x, y) & 0xFF:
                bad.append((x, y, row[x], word(x, y)))
    total = w * hh
    print(f"{total - len(bad)}/{total} pixels match ({100.0 * (total - len(bad)) / total:.3f}%)")
    if bad:
        xs = [b[0] for b in bad]
        ys = [b[1] for b in bad]
        print("mismatch bbox", (min(xs), min(ys), max(xs), max(ys)))
        print("first mismatches (x, y, xwd, fb):", [(x, y, hex(p), hex(q)) for x, y, p, q in bad[:12]])
    if a.diff:
        rows = []
        badset = {(x, y) for x, y, _, _ in bad}
        for y in range(hh):
            r = bytearray()
            for x in range(w):
                r += b'\xff\x00\x00' if (x, y) in badset else bytes([data[y * bpl + x]] * 3)
            rows.append(b'\x00' + bytes(r))

        def chunk(t, dd):
            return struct.pack('>I', len(dd)) + t + dd + struct.pack('>I', zlib.crc32(t + dd) & 0xffffffff)
        open(a.diff, 'wb').write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, hh, 8, 2, 0, 0, 0))
                                 + chunk(b'IDAT', zlib.compress(b''.join(rows))) + chunk(b'IEND', b''))


if __name__ == '__main__':
    main()
