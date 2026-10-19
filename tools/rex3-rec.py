#!/usr/bin/env python3
"""Inspect and convert REX3 recordings (src/dev/ng1/rex3_rec.rs).

  rex3-rec.py dump IN.rec [--from N] [--to M]
      records [N, M) with register names, numbered from 1 as replay counts
      them (REX3_REPLAY_LIMIT=M replays exactly the first M)
  rex3-rec.py text IN.rec OUT.txt [--from N] [--to M] [--no-hash] [--no-state]
      write records [N, M) in the text form, for testdata/. A slice that
      does not start at record 1 has no start state, so its checkpoints
      cannot match a power-on board: use --no-hash (and replay it with
      REX3_REPLAY_NOCHECK=1 or assert on pixels instead)
  rex3-rec.py summary IN.rec
      record counts by kind and the busiest registers
  rex3-rec.py from-trace TRACE.log OUT.txt
      a `rex trace` turned into a text recording: its GF lines (GFIFO
      register writes, in order) as bus writes. It has no start state and no
      checkpoints, DMA batch payloads are not in a trace (those entries are
      skipped), and CPU-side DCB writes are left out (their access width is
      not traced). Enough to replay a trace's drawing offline:
        REX3_REPLAY=OUT.txt REX3_REPLAY_NOCHECK=1 REX3_REPLAY_OUT=dir \\
          cargo test --release --features gputrace --lib -- rex3_replay_file
"""
import argparse
import collections
import os
import re
import struct
import sys

MAGIC = b'REX3REC\0'
VERSION = 1
REX3_RS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'dev', 'ng1', 'rex3.rs')


def reg_names():
    """offset -> name, from rex3.rs's constants and rex3_reg_name()."""
    src = open(REX3_RS).read()
    consts = {c: int(v, 16) for c, v in re.findall(r'pub const (REX3_\w+): u32 = (0x[0-9A-Fa-f]+);', src)}
    names = {}
    for c, n in re.findall(r'(REX3_\w+) => "(\w+)",', src):
        if c in consts:
            names[consts[c]] = n
    return names


def records(path):
    """Yield (kind, fields) per record; fields as in the text form."""
    data = open(path, 'rb').read()
    if not data.startswith(MAGIC):
        for line in data.decode().splitlines():
            line = line.split('#')[0].strip()
            if line:
                f = line.split()
                yield f[0], f[1:]
        return
    if len(data) < 12:
        sys.exit(f'{path}: truncated REX3 recording header')
    version, = struct.unpack_from('<I', data, 8)
    if version != VERSION:
        sys.exit(f'{path}: unsupported REX3 recording version {version}')
    i = 12
    while i < len(data):
        t = chr(data[i])
        try:
            if t in 'WR':
                bits = data[i + 1]
                off, = struct.unpack_from('<H', data, i + 2)
                if bits > 32:
                    val, = struct.unpack_from('<Q', data, i + 4)
                    n = 12
                else:
                    val, = struct.unpack_from('<I', data, i + 4)
                    n = 8
                fields = [str(bits), f'{off:x}', f'{val:x}']
            elif t == 'D':
                off, val = struct.unpack_from('<HQ', data, i + 1)
                n, fields = 11, [f'{off:x}', f'{val:x}']
            elif t in 'Bb':
                off, cnt = struct.unpack_from('<HI', data, i + 1)
                words = struct.unpack_from(f'<{cnt}Q', data, i + 7)
                n, fields = 7 + 8 * cnt, [f'{off:x}', f'{cnt:x}'] + [f'{w:x}' for w in words]
            elif t == 'H':
                n, fields = 33, [data[i + 1:i + 33].hex()]
            elif t == 'S':
                ln, = struct.unpack_from('<I', data, i + 1)
                n, fields = 5 + ln, [data[i + 5:i + 5 + ln].hex()]
            else:
                sys.exit(f'{path}: bad record tag {data[i]:#x} at byte {i}')
        except struct.error:
            return  # cut off mid-record: keep what is complete
        if i + n > len(data):
            return
        yield t, fields
        i += n


def describe(kind, f, names):
    if kind in 'WR':
        bits, off, val = int(f[0]), int(f[1], 16), int(f[2], 16)
        go = ' GO' if off & 0x800 else ''
        name = names.get(off & ~0x800, f'[{off & ~0x800:#06x}]')
        return f'{kind} {bits:2} {name}{go} = {val:#x}'
    if kind == 'D':
        return f'D    {names.get(int(f[0], 16) & ~0x800, f[0])} -> {int(f[1], 16):#018x}'
    if kind in 'Bb':
        n = int(f[1], 16)
        sample = ' '.join(f[2:6]) + (' ...' if n > 4 else '')
        return f'{kind}    {n} qwords {"to" if kind == "B" else "from"} {names.get(int(f[0], 16) & ~0x800, f[0])}: {sample}'
    if kind == 'H':
        return f'H    checkpoint {f[0][:16]}'
    if kind == 'S':
        return f'S    start state ({len(f[0]) // 2} bytes)'
    return kind + ' ' + ' '.join(f)


def in_range(k, a):
    return k >= a.from_ and (a.to is None or k < a.to)


def cmd_dump(a):
    names = reg_names()
    for k, (kind, f) in enumerate(records(a.input), 1):
        if a.to is not None and k >= a.to:
            break
        if in_range(k, a):
            print(k, describe(kind, f, names))


def cmd_text(a):
    with open(a.output, 'w') as out:
        out.write(f'# REX3 recording: {os.path.basename(a.input)} records [{a.from_}, {a.to or "end"})\n')
        for k, (kind, f) in enumerate(records(a.input), 1):
            if a.to is not None and k >= a.to:
                break
            if not in_range(k, a) or (a.no_hash and kind == 'H') or (a.no_state and kind == 'S'):
                continue
            out.write(' '.join([kind] + f) + '\n')


def cmd_summary(a):
    names = reg_names()
    kinds = collections.Counter()
    regs = collections.Counter()
    for kind, f in records(a.input):
        kinds[kind] += 1
        if kind in 'WR':
            off = int(f[1], 16)
            regs[f"{kind} {names.get(off & ~0x800, hex(off & ~0x800))}{' GO' if off & 0x800 else ''}"] += 1
    print('records:', sum(kinds.values()), dict(kinds))
    for r, n in regs.most_common(20):
        print(f'  {n:9} {r}')


def cmd_from_trace(a):
    names = {v: k for k, v in reg_names().items()}
    line_re = re.compile(r'GF  #\d+ (\w+)( GO)? = (0x[0-9a-f]+)')
    n = skipped = 0
    with open(a.output, 'w') as out:
        out.write(f'# REX3 recording from the GF lines of {os.path.basename(a.trace)} (rex3-rec.py from-trace)\n')
        for line in open(a.trace):
            m = line_re.search(line)
            if not m:
                continue
            name, go, val = m[1], m[2], int(m[3], 16)
            off = names.get(name)
            if off is None or name.startswith('DMA_'):
                skipped += 1
                continue
            off |= 0x800 if go else 0
            if name == 'HOSTRW64':
                out.write(f'W 64 {off & ~1:x} {val:x}\n')
            else:
                out.write(f'W 32 {off:x} {val:x}\n')
            n += 1
    print(f'{n} writes, {skipped} skipped', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('dump', 'text'):
        p = sub.add_parser(name)
        p.add_argument('input')
        if name == 'text':
            p.add_argument('output')
            p.add_argument('--no-hash', action='store_true')
            p.add_argument('--no-state', action='store_true')
        p.add_argument('--from', dest='from_', type=int, default=1)
        p.add_argument('--to', type=int)
    p = sub.add_parser('summary')
    p.add_argument('input')
    p = sub.add_parser('from-trace')
    p.add_argument('trace')
    p.add_argument('output')
    a = ap.parse_args()
    {'dump': cmd_dump, 'text': cmd_text, 'summary': cmd_summary, 'from-trace': cmd_from_trace}[a.cmd](a)


if __name__ == '__main__':
    main()
