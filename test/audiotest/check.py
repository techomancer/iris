#!/usr/bin/env python3
"""check.py [--o2] CAPTURE MARKS: judge the emulator's capture of an audio sweep.

CAPTURE is the card's capture, in either of two forms (told apart by the first
four bytes):

  WAV     a stereo 16-bit WAV (an [audio] file, 48 kHz). The rate comes from
          its header. MARKS holds byte offsets into its data chunk.
  dump    what the card played at its own rate (IRIS_AUDIO_DUMP; the layout
          is in src/dev/audio_dump.rs): records of rate:u32, pairs:u32, then
          pairs x (l:i16, r:i16), little-endian, with out:u8 ahead of each
          record for the O2 (found by which layout tiles the file; --o2
          and --plain force it, as a killed run's file tiles neither). MARKS holds byte offsets into the file.
          A case is the records that start inside its span; they are joined
          per (output, rate) and the group with the most sound (non-zero
          samples) is measured at its own rate. A program that sets the rate
          after it starts leaves records at the old rate: silence from the
          idle channel, which can outnumber an 8 kHz tone.

MARKS has one line per case, written by host.sh: "NAME START END", the offsets
before and after that case played. Each slice must hold one tone, so a case
passes when

    its length is within 5% of 2.0 s    (a wrong playback rate scales it), and
    its pitch is within 3% of 1000 Hz   (the same error scales the pitch), and
    it is not silent.

Length is the span of samples above 1/8 of the slice's peak; pitch is the
left channel's zero crossings over that span. Prints one line per case,
"PASS name" or "FAIL name: length, pitch", then the counts; the exit status is
the number of failures. Standard library only.
"""
import array
import struct
import sys
import wave

DEFAULT_RATE = 44100
EXPECT_S, LEN_TOL = 2.0, 0.05
EXPECT_HZ, HZ_TOL = 1000.0, 0.03


def measure(raw, rate=DEFAULT_RATE):
    """(seconds, hz, peak) of the tone in a 16-bit stereo slice at RATE."""
    raw = raw[: len(raw) // 4 * 4]
    s = array.array("h", raw)
    if sys.byteorder == "big":
        s.byteswap()
    left = s[0::2]
    peak = max((abs(x) for x in left), default=0)
    if peak == 0:
        return 0.0, 0.0, 0
    loud = [i for i, x in enumerate(left) if abs(x) > peak / 8]
    seg = left[loud[0] : loud[-1] + 1]
    crossings = sum(1 for i in range(1, len(seg)) if (seg[i - 1] < 0) != (seg[i] < 0))
    secs = len(seg) / rate
    return secs, crossings / 2 / secs, peak


def tiles(data, head):
    """Whether records with a HEAD-byte header (rate and count, and for the
    O2 the output byte) end exactly at the end of DATA."""
    pos = 0
    while pos + head <= len(data):
        pos += head + 4 * struct.unpack_from("<I", data, pos + head - 4)[0]
    return pos == len(data)


def dump_case(data, a, b, o2=False):
    """(raw samples, rate) for the dump records starting in [a, b)."""
    outs = {}  # (output, rate) -> [sample bytes]
    pos = 0
    head = 9 if o2 else 8
    while pos + head <= len(data):
        out, rate, pairs = struct.unpack_from("<BII", data, pos) if o2 else (0, *struct.unpack_from("<II", data, pos))
        body = data[pos + head : pos + head + pairs * 4]
        body = body[: len(body) // 4 * 4]  # a cut-off record ends mid-pair
        if a <= pos < b:
            outs.setdefault((out, rate), []).append(body)
        pos += head + pairs * 4
    if not outs:
        return b"", DEFAULT_RATE
    sound = lambda parts: sum(len(p) // 2 - array.array("h", p).count(0) for p in parts)
    (_, rate), parts = max(outs.items(), key=lambda kv: sound(kv[1]))
    return b"".join(parts), rate


def main():
    flags = {a for a in sys.argv[1:] if a in ("--o2", "--plain")}
    cap, marks = [a for a in sys.argv[1:] if a not in flags]
    o2 = "--o2" in flags
    with open(cap, "rb") as f:
        data = f.read()
    is_wav = data[:4] == b"RIFF"
    if is_wav:
        with wave.open(cap, "rb") as w:
            wav_rate = w.getframerate()
        at = 12  # past RIFF, size, WAVE; walk the chunks to "data"
        while data[at : at + 4] != b"data":
            at += 8 + struct.unpack_from("<I", data, at + 4)[0] + (data[at + 4] & 1)
        data = data[at + 8 :]
    elif not flags:
        if tiles(data, 8):
            pass
        elif tiles(data, 9):
            o2 = True  # only the O2's layout tiles the file
        else:
            sys.exit(
                f"{cap}: no dump layout tiles the file (a killed run?). Pass --o2 if the card is the O2 "
                "(records of out:u8, rate:u32, pairs:u32, then pairs), or --plain for the other cards "
                "(rate:u32, pairs:u32, then pairs)."
            )
    fails = cases = 0
    for line in open(marks):
        name, a, b = line.split()
        if is_wav:
            secs, hz, peak = measure(data[int(a) : int(b)], wav_rate)
        else:
            raw, rate = dump_case(data, int(a), int(b), o2)
            secs, hz, peak = measure(raw, rate)
        ok = (
            peak > 0
            and abs(secs - EXPECT_S) <= EXPECT_S * LEN_TOL
            and abs(hz - EXPECT_HZ) <= EXPECT_HZ * HZ_TOL
        )
        cases += 1
        if ok:
            print(f"PASS {name}")
        else:
            fails += 1
            print(f"FAIL {name}: {secs:.3f} s, {hz:.0f} Hz" + ("" if peak > 0 else ", silent"))
    print(f"{cases - fails} passed, {fails} failed")
    sys.exit(min(fails, 255))


main()
