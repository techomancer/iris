#!/usr/bin/env python3
"""Find what holds off the guest's clock tick in an IRIS perfetto trace.

The Octane's RAD driver warns "RAD thread held off for more than 32 mS" when
its kernel thread finds the status DMA ring all but full: the thread was not
scheduled for that long, while the card kept writing slots on host time. The
thread is scheduled from the clock tick (IP7), which the vCPU takes only
while it runs and the guest has the tick enabled.

Record with every relevant category on, then run this on the file:

    perfetto start /tmp/rad.pftrace cats=cpu,irq,timer,gfx,audio
    ... reproduce ...
    perfetto stop
    tools/rad-holdoff.py /tmp/rad.pftrace

It reports:
  * the IP7 latency: from the compare timer raising IP7 to the vCPU taking
    an interrupt with IP7 in it;
  * the longest stretches with IP7 masked in Status (IE or IM7 clear);
  * the longest VPro busy streaks (one access answered busy until it
    completed), and those that ran while IP7 was masked: guest poll loops
    at a level the tick cannot interrupt;
  * for each time the RAD ring came within one slot of full, what the
    preceding window held.

Decodes the subset of the trace format IRIS writes (src/perfetto/proto.rs).
"""

import argparse
import bisect
import collections
import sys


def varint(b, i):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        if c < 0x80:
            return v, i
        s += 7


def fields(b):
    """(field, value) pairs: ints for varints, bytes for length-delimited."""
    i, n = 0, len(b)
    while i < n:
        key, i = varint(b, i)
        f, wire = key >> 3, key & 7
        if wire == 0:
            v, i = varint(b, i)
        elif wire == 2:
            ln, i = varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wire == 1:
            v, i = int.from_bytes(b[i:i + 8], "little"), i + 8
        elif wire == 5:
            v, i = int.from_bytes(b[i:i + 4], "little"), i + 4
        else:
            raise ValueError(f"wire type {wire}")
        yield f, v


def signed(v):
    return v - (1 << 64) if v >= 1 << 63 else v


BEGIN, END, INSTANT, COUNTER = 1, 2, 3, 4


def load(path):
    """Events as (ts, kind, track, name, value), sorted by time."""
    data = open(path, "rb").read()
    tracks = {}  # uuid -> name
    seqs = collections.defaultdict(lambda: {"names": {}, "args": {}})
    events = []
    for f, pkt in fields(data):
        if f != 1:
            continue
        ts = seq = None
        te = idata = desc = None
        flags = 0
        for pf, pv in fields(pkt):
            if pf == 8:
                ts = pv
            elif pf == 10:
                seq = pv
            elif pf == 11:
                te = pv
            elif pf == 12:
                idata = pv
            elif pf == 13:
                flags = pv
            elif pf == 60:
                desc = pv
        st = seqs[seq]
        if flags & 1:
            st["names"].clear()
            st["args"].clear()
        if desc is not None:
            uuid, name, tname = None, None, None
            for df, dv in fields(desc):
                if df == 1:
                    uuid = dv
                elif df == 2:
                    name = dv.decode(errors="replace")
                elif df == 4:
                    for tf, tv in fields(dv):
                        if tf == 5:
                            tname = tv.decode(errors="replace")
            if uuid is not None:
                tracks[uuid] = name or tname or f"track {uuid}"
        if idata is not None:
            for nf, nv in fields(idata):
                if nf in (2, 3):
                    iid, nm = None, ""
                    for ef, ev in fields(nv):
                        if ef == 1:
                            iid = ev
                        elif ef == 2:
                            nm = ev.decode(errors="replace")
                    st["names" if nf == 2 else "args"][iid] = nm
        if te is None or ts is None:
            continue
        kind, track, name, value = 0, None, "", None
        for ef, ev in fields(te):
            if ef == 9:
                kind = ev
            elif ef == 11:
                track = ev
            elif ef == 10:
                name = st["names"].get(ev, f"name {ev}")
            elif ef == 30:
                value = signed(ev)
            elif ef == 4:
                for af, av in fields(ev):
                    if af == 3:
                        value = av
        events.append((ts, kind, track, name, value))
    events.sort(key=lambda e: e[0])
    return [(ts, k, tracks.get(t, f"track {t}"), n, v) for ts, k, t, n, v in events]


def slices(events, track):
    """[(begin, end, arg)] of the slices on a named track."""
    out, open_ = [], None
    for ts, k, t, _, v in events:
        if t != track:
            continue
        if k == BEGIN:
            open_ = (ts, v)
        elif k == END and open_ is not None:
            out.append((open_[0], ts, open_[1]))
            open_ = None
    return out


def ms(ns):
    return ns / 1e6


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--window", type=float, default=40.0, help="ms looked back from a near-full ring (default 40)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--from", dest="start", type=float, default=0.0, help="skip the first seconds of the trace (the boot, before the kernel enables its clock)")
    a = ap.parse_args()

    ev = load(a.trace)
    if not ev:
        sys.exit("no events")
    t0 = ev[0][0]
    ev = [e for e in ev if e[0] >= t0 + a.start * 1e9]
    rel = lambda ts: ms(ts - t0)
    print(f"{len(ev)} events from {a.start:.0f} s to {ms(ev[-1][0] - t0) / 1e3:.0f} s; times below are ms from the trace's start")

    masked = slices(ev, "IP7 masked (Status)")
    busy = slices(ev, "VPro access busy")
    raised = [b for b, _, _ in slices(ev, "IP7 (count/compare)")]
    taken = [ts for ts, k, _, n, v in ev if k == INSTANT and n == "interrupt taken" and v is not None and v & 0x80]
    unread = [(ts, v) for ts, k, t, _, v in ev if k == COUNTER and t == "RAD status slots unread"]
    late = [(ts, v) for ts, k, _, n, v in ev if k == INSTANT and n == "late completion"]

    print(f"IP7 raised {len(raised)}, taken {len(taken)}; masked slices {len(masked)}; VPro busy streaks {len(busy)}; RAD completions {len(unread)}, late {len(late)}")

    # The tick's latency: each raise to the next take.
    lat = []
    for r in raised:
        i = bisect.bisect_left(taken, r)
        if i < len(taken):
            lat.append((taken[i] - r, r))
    if lat:
        lat.sort(reverse=True)
        print(f"\nIP7 latency, raise to take: worst {ms(lat[0][0]):.2f} ms; over 10 ms: {sum(1 for l, _ in lat if l > 10e6)}, over 32 ms: {sum(1 for l, _ in lat if l > 32e6)}")
        for l, r in lat[:a.top]:
            print(f"  {ms(l):8.2f} ms at {rel(r):10.1f} ms")

    print(f"\nLongest IP7-masked stretches:")
    for b, e, _ in sorted(masked, key=lambda s: s[0] - s[1])[:a.top]:
        print(f"  {ms(e - b):8.2f} ms at {rel(b):10.1f} ms")

    print(f"\nLongest VPro busy streaks (offset, of which masked):")
    mstarts = [s[0] for s in masked]

    def masked_part(b, e):
        i = max(0, bisect.bisect_right(mstarts, b) - 1)
        tot = 0
        while i < len(masked) and masked[i][0] < e:
            tot += overlap(b, e, masked[i][0], masked[i][1])
            i += 1
        return tot

    for b, e, off in sorted(busy, key=lambda s: s[0] - s[1])[:a.top]:
        print(f"  {ms(e - b):8.2f} ms at {rel(b):10.1f} ms  offset {off:#x}  masked {ms(masked_part(b, e)):.2f} ms")

    mb = sorted(((masked_part(b, e), b, e, off) for b, e, off in busy), reverse=True)
    mb = [x for x in mb if x[0] > 0]
    print(f"\nVPro busy streaks while IP7 masked: {len(mb)}")
    by_off = collections.defaultdict(lambda: [0, 0])
    for m, _, _, off in mb:
        by_off[off][0] += 1
        by_off[off][1] = max(by_off[off][1], m)
    for off, (n, worst) in sorted(by_off.items(), key=lambda x: -x[1][1]):
        print(f"  offset {off:#x}: {n} streaks, worst {ms(worst):.2f} ms masked")

    if unread:
        peak = max(v for _, v in unread)
        print(f"\nRAD ring: most slots unread {peak}")
        hits = []
        for i, (ts, v) in enumerate(unread):
            if v >= peak and (i == 0 or unread[i - 1][1] < peak):
                hits.append(ts)
        w = a.window * 1e6
        for ts in hits[:a.top * 2]:
            b = ts - w
            ml = max((overlap(b, ts, s0, s1) for s0, s1, _ in masked), default=0)
            bl = [(overlap(b, ts, s0, s1), off) for s0, s1, off in busy if overlap(b, ts, s0, s1) > 0]
            bl.sort(reverse=True)
            lw = [x for x in lat if b <= x[1] <= ts]
            lt = [v for t, v in late if b <= t <= ts]
            print(f"  at {rel(ts):10.1f} ms: worst IP7 latency {ms(max((l for l, _ in lw), default=0)):.2f} ms, longest masked {ms(ml):.2f} ms, "
                  f"busy {ms(sum(x for x, _ in bl)):.2f} ms in {len(bl)} streaks" + (f" (longest {ms(bl[0][0]):.2f} ms at {bl[0][1]:#x})" if bl else "")
                  + (f", late completions {lt}" if lt else ""))


if __name__ == "__main__":
    main()
