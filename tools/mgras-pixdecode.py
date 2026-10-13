#!/usr/bin/env python3
"""Decode the GL command stream of an IMPACT (MGRAS) recording, split by
pixtest case markers, with the GE pixel-pipeline state annotated.

  tools/mgras-pixdecode.py REC [--ucode mgras_ge.u] [--case N|NAME] [--results pixtest.txt]

REC is an `mgras rec` recording (binary or text). Words pushed to the user
command FIFO (0x70080/0x70084/0x74080) are parsed as the HQ3 does: control
words (bit 31 clear: command bits 20:8, byte count 7:0) and pixel data
commands (bit 31 set: byte count 19:0, sent as doublewords). pixtest marks
each case with RASTER_POS4F (case, 30000).

GE state writes (token 0x080 to GE memory, 0x0CD to ERAM) are shown as
address = values. The pixel pipeline program at GE address 0x13B is a list
of (microcode entry, state address) pairs; with --ucode the entries are
named after their microcode source file (the .u file's line table).
"""
import argparse
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def token_names():
    names = {}
    src = (ROOT / "src/dev/mgras/hq3.rs").read_text()
    for m in re.finditer(r'0x([0-9A-Fa-f]{3}) => "([A-Za-z0-9_]+)"', src):
        names.setdefault(int(m.group(1), 16), m.group(2))
    return names


def load_ucode(path):
    d = open(path, "rb").read()
    h = struct.unpack(">11I", d[:44])
    rec_off, rec_bytes, ftab, nfiles, stroff = h[5], h[6], h[7], h[8] // 2, h[9]
    offs = struct.unpack(">%dH" % nfiles, d[ftab:ftab + 2 * nfiles])

    def s(o):
        e = d.index(b"\0", stroff + o)
        return d[stroff + o:e].decode().split("/")[-1]

    names = [s(o) for o in offs]
    m = {}
    for i in range(rec_bytes // 8):
        pc, line, f = struct.unpack(">IHH", d[rec_off + 8 * i:rec_off + 8 * i + 8])
        if f < len(names):
            m.setdefault(pc, "%s:%d" % (names[f], line))
    return m


def records(path):
    data = open(path, "rb").read()
    if data[:8] == b"MGRASREC":
        i = 12
        while i < len(data):
            t = chr(data[i])
            i += 1
            if i + 32 > len(data):
                return
            if t in "WR":
                bits, off, val = struct.unpack_from("<BIQ", data, i)
                i += 13
                yield t, bits, off, val
            elif t in "Mm":
                bits, addr, val = struct.unpack_from("<BII", data, i)
                i += 9
                yield t, bits, addr, val
            elif t == "H":
                i += 32
            elif t in "FTC":
                pass
            else:
                raise SystemExit("bad record tag %r at %d" % (t, i - 1))
    else:
        for line in data.decode().splitlines():
            f = line.split("#")[0].split()
            if not f:
                continue
            if f[0] in "WRMm":
                yield f[0], int(f[1]), int(f[2], 16), int(f[3], 16)


def fifo_events(path):
    """('word', port, w) for command FIFO words (port 0 user, 1 privileged)
    and ('ctx', slot) at each context switch (the 63-word image that follows
    the switch request carries the incoming context's slot first)."""
    image = 0
    for t, bits, off, val in records(path):
        if t != "W":
            continue
        if off == 0x50050:
            image = 63
            continue
        if off in (0x70080, 0x70084, 0x74080):
            port = 0
        elif off in (0x50080, 0x50084):
            port = 1
        else:
            continue
        ws = [(val >> 32) & 0xFFFFFFFF, val & 0xFFFFFFFF] if bits == 64 else [val & 0xFFFFFFFF]
        for w in ws:
            if image:
                if image == 63:
                    yield ("ctx", w)
                image -= 1
                continue
            yield ("word", port, w)


class Parser:
    """One context's command FIFO parser (the HQ3 keeps one per context)."""

    def __init__(self):
        self.need = 0
        self.w = 0
        self.data = []

    def push(self, w):
        if self.need == 0:
            self.w = w
            self.data = []
            if w & 0x80000000:
                # Data starts (w >> 22) & 7 bytes into the first word.
                self.need = ((w & 0xFFFFF) + ((w >> 22) & 7) + 7) // 8 * 2
            else:
                self.need = ((w & 0xFF) + 3) // 4
            if self.need == 0:
                return self.done()
            return None
        self.data.append(w)
        self.need -= 1
        if self.need == 0:
            return self.done()
        return None

    def done(self):
        return ("pix" if self.w & 0x80000000 else "cmd", self.w, self.data)


def commands(path):
    """(kind, word, data, slot) for every complete user-port command."""
    parsers = {}
    slot = None
    for ev in fifo_events(path):
        if ev[0] == "ctx":
            slot = ev[1]
            continue
        _, port, w = ev
        if port != 0:
            continue
        p = parsers.setdefault(slot, Parser())
        c = p.push(w)
        if c:
            yield c + (slot,)


def f32(v):
    return struct.unpack(">f", struct.pack(">I", v))[0]


def fmt_val(v):
    if 0x30000000 <= v <= 0x4F000000 or 0xB0000000 <= v <= 0xCF000000:
        return "%#x(%g)" % (v, f32(v))
    return "%#x" % v


def ge_write(d):
    """[?, ?, bytes, addr, 0, 0, n, v1..vn, n, v1..vn]: (addr, values)."""
    if len(d) < 7:
        return None, d
    addr, n = d[3], d[6]
    return addr, d[7:7 + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--ucode")
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--results")
    ap.add_argument("--all", action="store_true", help="show every token, not just pixel-path ones")
    a = ap.parse_args()
    names = token_names()
    uc = load_ucode(a.ucode) if a.ucode else {}
    case_names = {}
    if a.results:
        for line in open(a.results):
            m = re.match(r"(PASS|FAIL|SKIP) #(\d+) (\S+?)(:|$)", line.strip())
            if m:
                case_names[int(m.group(2))] = (m.group(3), m.group(1), line.strip())
    want = set()
    for c in a.case:
        if c.isdigit():
            want.add(int(c))
        else:
            want |= {k for k, v in case_names.items() if c in v[0]}
    quiet = {0x2A, 0x2C, 0x35, 0x2B, 0xA2, 0x47, 0x3B, 0x3C, 0x3A, 0x66, 0x6F, 0x65,
             0x64, 0x67, 0x6E, 0xED, 0x02, 0x00, 0x1D, 0x27, 0x15, 0x4C, 0x4A, 0xBA,
             0xBC, 0xBE, 0x12, 0x39, 0x33}
    cur = 0
    show = not want
    pending_pix = None
    for kind, w, d, slot in commands(a.rec):
        if kind == "cmd":
            cmd = (w >> 8) & 0x1FFF
            if cmd == 0x38 and len(d) >= 2 and d[1] == 30000:
                cur = d[0]
                show = not want or cur in want
                if show:
                    info = case_names.get(cur)
                    print("\n==== case #%d %s" % (cur, info[2] if info else ""))
                continue
            if not show:
                continue
            name = names.get(cmd, "tok%03x" % cmd)
            conv = (w >> 23) & 0x7F
            if cmd in (0x80, 0xCD):
                addr, vals = ge_write(d)
                space = "GE" if cmd == 0x80 else "ERAM"
                line = "  %s %s[%#x] = %s" % (name if cmd == 0xCD else "PIXEL_STATE", space,
                                              addr or 0, " ".join(fmt_val(v) for v in vals))
                if addr == 0x13B and uc:
                    mods = []
                    for i in range(0, len(vals) - 1, 2):
                        mods.append("%s@%#x" % (uc.get(vals[i], "%#x" % vals[i]).split(":")[0], vals[i + 1]))
                    line += "\n      pipeline: " + " -> ".join(mods)
                print(line + "   [hdr %s]" % " ".join("%#x" % x for x in d[:3]))
            elif not a.all and cmd in quiet:
                continue
            else:
                cs = " conv=%#x" % conv if conv else ""
                print("  %s [%#x]%s %s" % (name, cmd, cs, " ".join(fmt_val(v) for v in d[:24])))
        else:
            if not show:
                continue
            nb = w & 0xFFFFF
            print("  pixel data %d bytes addr %d align %d%s%s: %s%s" % (
                nb, (w >> 20) & 3, (w >> 22) & 7, " packed" if w & 1 << 25 else "",
                " last" if w & 1 << 26 else "", " ".join("%08x" % x for x in d[:12]),
                " ..." if len(d) > 12 else ""))


if __name__ == "__main__":
    main()
