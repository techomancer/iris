#!/usr/bin/env python3
"""Inspect and convert MGRAS recordings (src/dev/mgras/record.rs).

  mgras-rec.py text IN.rec OUT.txt [--from N] [--to M] [--no-hash]
      write records [N, M) in the text form; --no-hash drops checkpoints
      (a slice starting mid-session cannot match a fresh board's state)
  mgras-rec.py summary IN.rec
      record counts by kind and the busiest offsets
"""
import argparse
import collections
import struct
import sys

SIZES = {ord('W'): 14, ord('R'): 14, ord('M'): 10, ord('m'): 10, ord('H'): 33,
         ord('F'): 1, ord('T'): 1, ord('C'): 1}


def records(path):
    data = open(path, 'rb').read()
    if not data.startswith(b'MGRASREC'):
        # Already text: pass lines through as (kind, line).
        for line in data.decode().splitlines():
            line = line.split('#')[0].strip()
            if line:
                yield line.split()[0], line
        return
    i = 12
    while i < len(data):
        t = data[i]
        n = SIZES.get(t)
        if n is None or i + n > len(data):
            return
        c = chr(t)
        if c in 'WR':
            bits = data[i + 1]
            off, = struct.unpack_from('<I', data, i + 2)
            val, = struct.unpack_from('<Q', data, i + 6)
            yield c, f"{c} {bits} {off:x} {val:x}"
        elif c in 'Mm':
            bits = data[i + 1]
            addr, val = struct.unpack_from('<II', data, i + 2)
            yield c, f"{c} {bits} {addr:x} {val:x}"
        elif c == 'H':
            yield c, "H " + data[i + 1:i + 33].hex()
        else:
            yield c, c
        i += n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('text')
    t.add_argument('inp')
    t.add_argument('out')
    t.add_argument('--from', dest='first', type=int, default=0)
    t.add_argument('--to', dest='last', type=int, default=None)
    t.add_argument('--no-hash', action='store_true')
    s = sub.add_parser('summary')
    s.add_argument('inp')
    a = ap.parse_args()

    if a.cmd == 'text':
        with open(a.out, 'w') as out:
            out.write(f"# MGRAS recording (record.rs text form), from {a.inp} records {a.first}..{a.last or 'end'}\n")
            for n, (kind, line) in enumerate(records(a.inp)):
                if n < a.first:
                    continue
                if a.last is not None and n >= a.last:
                    break
                if a.no_hash and kind == 'H':
                    continue
                out.write(line + "\n")
    else:
        kinds = collections.Counter()
        offs = collections.Counter()
        for kind, line in records(a.inp):
            kinds[kind] += 1
            if kind in 'WR':
                offs[(kind, line.split()[2])] += 1
        print("records by kind:", dict(kinds))
        print("busiest offsets:", [(k, f"0x{o}", n) for (k, o), n in offs.most_common(12)])


if __name__ == '__main__':
    sys.exit(main())
