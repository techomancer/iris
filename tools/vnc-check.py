#!/usr/bin/env python3
"""Minimal RFB client for checking IRIS's VNC server from scripts.

  tools/vnc-check.py [--addr 127.0.0.1:5900] [--zrle] shot OUT.png
  tools/vnc-check.py ... watch SECONDS       # incremental updates: rects, pixels
                                             # (--no-decode with --zrle times the server)
  tools/vnc-check.py ... click X Y [BUTTON]  # 1 left, 2 middle, 4 right
  tools/vnc-check.py ... move X Y
  tools/vnc-check.py ... key KEYSYM          # e.g. 0xff0d (Return), 0x61 (a)

Needs Pillow for `shot`. Decodes Raw and ZRLE (with --zrle).
"""
import argparse
import socket
import struct
import sys
import time
import zlib


def recvn(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise EOFError("server closed")
        b += c
    return bytes(b)


class Rfb:
    def __init__(self, addr, zrle, decode=True):
        self.decode = decode
        host, port = addr.rsplit(":", 1)
        self.s = socket.create_connection((host, int(port)), timeout=20)
        ver = recvn(self.s, 12)
        assert ver.startswith(b"RFB "), ver
        self.s.sendall(b"RFB 003.008\n")
        n = recvn(self.s, 1)[0]
        types = recvn(self.s, n)
        assert 1 in types, types
        self.s.sendall(b"\x01")
        assert struct.unpack(">I", recvn(self.s, 4))[0] == 0
        self.s.sendall(b"\x01")  # shared
        self.w, self.h = struct.unpack(">HH", recvn(self.s, 4))
        recvn(self.s, 16)
        (nl,) = struct.unpack(">I", recvn(self.s, 4))
        self.name = recvn(self.s, nl).decode()
        # 32 bpp, R at shift 16 (the usual viewer format) to exercise conversion.
        pf = struct.pack(">BBBBHHHBBBxxx", 32, 24, 0, 1, 255, 255, 255, 16, 8, 0)
        self.s.sendall(b"\x00\x00\x00\x00" + pf)
        encs = ([16] if zrle else []) + [0, -223, -258]
        self.s.sendall(struct.pack(">BxH", 2, len(encs)) + b"".join(struct.pack(">i", e) for e in encs))
        self.fb = bytearray(self.w * self.h * 4)
        self.z = zlib.decompressobj()

    def request(self, incremental, x=0, y=0, w=None, h=None):
        self.s.sendall(struct.pack(">BBHHHH", 3, incremental, x, y, w or self.w, h or self.h))

    def pointer(self, x, y, mask):
        self.s.sendall(struct.pack(">BBHH", 5, mask, x, y))

    def key(self, sym, down):
        self.s.sendall(struct.pack(">BBxxI", 4, down, sym))

    def read_update(self):
        """One server message; returns list of (x, y, w, h, enc) for an update."""
        t = recvn(self.s, 1)[0]
        if t != 0:
            raise RuntimeError(f"message type {t}")
        (n,) = struct.unpack(">xH", recvn(self.s, 3))
        rects = []
        for _ in range(n):
            x, y, w, h, enc = struct.unpack(">HHHHi", recvn(self.s, 12))
            rects.append((x, y, w, h, enc))
            if enc == 0:
                data = recvn(self.s, w * h * 4)
                for r in range(h):
                    o = ((y + r) * self.w + x) * 4
                    self.fb[o:o + w * 4] = data[r * w * 4:(r + 1) * w * 4]
            elif enc == 16:
                (ln,) = struct.unpack(">I", recvn(self.s, 4))
                d = self.z.decompress(recvn(self.s, ln))
                if self.decode:
                    self.unzrle(d, x, y, w, h)
            elif enc == -223:
                self.w, self.h = w, h
                self.fb = bytearray(w * h * 4)
            elif enc == -258:
                pass
            else:
                raise RuntimeError(f"encoding {enc}")
        return rects

    def unzrle(self, d, x, y, w, h):
        p = 0

        def cpix():
            nonlocal p
            v = d[p:p + 3] + b"\x00"
            p += 3
            return v

        def rl():
            nonlocal p
            n = 1
            while True:
                b = d[p]
                p += 1
                n += b
                if b != 255:
                    return n

        for ty in range(y, y + h, 64):
            th = min(64, y + h - ty)
            for tx in range(x, x + w, 64):
                tw = min(64, x + w - tx)
                sub = d[p]
                p += 1
                pal = []
                if 2 <= sub <= 16 or sub >= 130:
                    pal = [cpix() for _ in range(sub if sub <= 16 else sub - 128)]
                tile = []
                if sub == 0:
                    tile = [cpix() for _ in range(tw * th)]
                elif sub == 1:
                    tile = [cpix()] * (tw * th)
                elif sub <= 16:
                    bits = 1 if sub == 2 else 2 if sub <= 4 else 4
                    rowb = (tw * bits + 7) // 8
                    for _ in range(th):
                        for i in range(tw):
                            b = d[p + (i * bits) // 8]
                            idx = (b >> (8 - bits - (i * bits) % 8)) & ((1 << bits) - 1)
                            tile.append(pal[idx])
                        p += rowb
                elif sub == 128:
                    while len(tile) < tw * th:
                        v = cpix()
                        tile += [v] * rl()
                else:
                    while len(tile) < tw * th:
                        b = d[p]
                        p += 1
                        tile += [pal[b & 127]] * (rl() if b & 128 else 1)
                for r in range(th):
                    for c in range(tw):
                        o = ((ty + r) * self.w + tx + c) * 4
                        self.fb[o:o + 4] = tile[r * tw + c]
        assert p == len(d), (p, len(d))

    def save(self, path):
        from PIL import Image
        # Pixels are little-endian with R at shift 16: bytes B, G, R, 0.
        Image.frombytes("RGBX", (self.w, self.h), bytes(self.fb), "raw", "BGRX").convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:5900")
    ap.add_argument("--zrle", action="store_true")
    ap.add_argument("--no-decode", action="store_true", help="watch: inflate ZRLE but do not decode tiles (times the server)")
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    c = Rfb(a.addr, a.zrle, not a.no_decode)
    print(f"connected: {c.name} {c.w}x{c.h}")
    if a.cmd == "shot":
        c.request(0)
        t = time.time()
        rects = c.read_update()
        # The ext-key ack or a resize may come first; make sure we have pixels.
        while not any(r[4] in (0, 16) for r in rects):
            c.request(0)
            rects = c.read_update()
        c.save(a.args[0])
        print(f"saved {a.args[0]} {c.w}x{c.h} in {time.time() - t:.2f}s")
    elif a.cmd == "watch":
        end = time.time() + float(a.args[0])
        c.request(0)
        c.read_update()
        c.request(1)
        n = 0
        c.s.settimeout(max(0.1, end - time.time()))
        try:
            while time.time() < end:
                rects = c.read_update()
                n += 1
                px = sum(r[2] * r[3] for r in rects)
                print(f"update {n}: {len(rects)} rects, {px} px: {rects[:4]}")
                c.request(1)
        except socket.timeout:
            pass
        print(f"{n} updates")
    elif a.cmd in ("click", "move"):
        x, y = int(a.args[0]), int(a.args[1])
        b = int(a.args[2]) if len(a.args) > 2 else 1
        c.pointer(x, y, 0)
        if a.cmd == "click":
            time.sleep(0.05)
            c.pointer(x, y, b)
            time.sleep(0.05)
            c.pointer(x, y, 0)
        time.sleep(0.8)  # let the server steer before we hang up
    elif a.cmd == "key":
        sym = int(a.args[0], 0)
        c.key(sym, 1)
        c.key(sym, 0)
        time.sleep(0.2)
    else:
        sys.exit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
