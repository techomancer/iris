# The audio test suite (`test/audiotest`): rates x channels x widths

**Keywords:** audio,rad1,octane,o2,mace,sample rate,alSetParams,playaiff,sfplay,audiotest,sweep,capture,IRIS_AUDIO_DUMP

A guest program that plays a tone through the IRIX audio library, a sweep
that runs it and the stock players over every rate and format, and a host
checker that measures the emulator's capture of the output. It runs on the
Octane and the O2. It exists because
RAD1 once ignored the rate IRIX programmed
([#354](https://github.com/techomancer/iris/issues/354)); a game at 22050 Hz
played at the wrong speed, and so did every sound after it.

| File | Where | What |
|---|---|---|
| `audiotest.c` | guest | `audiotest -r RATE -c 1\|2 -b 8\|16 [-s SECS] [-f HZ] [-n]` plays a sine through `alSetParams(AL_RATE)` and an `ALport`. `-w FILE` writes the same tone as an AIFF instead, so no audio file is committed or made on the host. `-n` leaves the device rate alone |
| `irix.sh` | guest | builds it with MIPSpro `cc -mips3 -n32 -laudio -lm` |
| `sweep.sh` | guest | `cases`, `prep NAME`, `play NAME`, `all` (see its header) |
| `host.sh` | host | copies the test in, builds it, and plays each case, noting the capture's size before and after |
| `check.py` | host | slices the capture (Octane WAV or O2 dump) and prints PASS or FAIL per case |

## Cases

86, each a 1 kHz tone for 2.0 s at 8000, 11025, 16000, 22050, 32000, 44100
and 48000 Hz, mono and stereo, 8 and 16 bit:

- `gen_*`: `audiotest` through the audio library (28).
- `playaiff_*` and `sfplay_*`: the same tone as an AIFF through the stock
  players (28 each).
- `sticky_lib`, `sticky_aiff`: a 22050 Hz tone first (not measured), then
  44100 Hz. IRIX leaves the last rate set when a program exits, so this checks
  that the second program's rate wins, through the library and through
  `playaiff`.

## Tolerances

Fixed in `check.py`. A case passes when its slice has sound, its length is
within 5% of 2.0 s and its pitch is within 3% of 1000 Hz. Length is the span
above 1/8 of the slice's peak; pitch is the left channel's zero crossings over
that span. Playing at the wrong rate scales both by the same ratio, so one
failing number is enough. The fixed Octane runs 1.997 to 2.093 s and 995 to
1000 Hz; the nearest unfixed failure is 2.177 s and 919 Hz (48 kHz).

## Running it

Both machines need an IRIX 6.5 disk with the compiler and audio libraries
(`/usr/bin/cc`, `/usr/include/dmedia/audio.h`) and a config with the control
socket on (`ci = true`, `ci_socket`), plus a scratch volume for `iris-ci put`
(`[scsi.N]` with `scratch = true`). The emulator records the card's output
to a file with `IRIS_AUDIO_DUMP=FILE`, what the card played at the guest's own
rate, before resampling, as records of rate (u32) and pair count (u32), then
the pairs, little-endian. The O2 puts the output number (u8) in front of
every record. It needs no host sound device. `check.py` measures each case
at its records' rate. (A WAV from `[audio] file` works too, at the stream's
rate.)

```sh
IRIS_AUDIO_DUMP=/path/cap iris --config octane.toml &   # or o2.toml
# then: iris-ci start
sh test/audiotest/host.sh /path/ci.sock /path/cap /path/out [EGREP_PATTERN]
python3 test/audiotest/check.py /path/cap /path/out/marks.txt    # --o2 for the O2
```

The O2 needs `overlay = true` on a disk in a writable directory (the disk image must be
inside the run directory, e.g. hard-linked as `disk.raw`) and `ci = true`
to run headless. A full run takes about 4 minutes on the O2 and tens of
minutes on the Octane. The pattern picks cases (`'^gen_'`, `'sticky'`).
`check.py`'s exit status is the number of failed cases. When iris runs in a container, its
paths are not the shell's: either mount the run directory at the same path
inside the container, or set `PUT="cmd"` so that `host.sh` calls
`cmd HOSTFILE GUESTPATH` to copy the tarball in. For a guest with no
compiler, build `audiotest` elsewhere (it is MIPS III n32, so any 6.5 box
will do), put it in `/usr/tmp/at` and run with `SKIP_BUILD=1`.

## Results

Octane (RAD1), IRIX 6.5.30 (`si-octane` disk), 2026-10-29. Stereo 16-bit
`audiotest` shown; the other widths and channel counts, `playaiff` and `sfplay`
agree with it row for row.

| Rate | Upstream b188bb9 (length s, pitch Hz) | With the rate fix |
|---|---|---|
| 8000 | 11.174, 5430 FAIL | 2.000, 1000 ok |
| 11025 | 15.203, 3973 FAIL | 2.011, 995 ok |
| 16000 | 0.772, 2756 FAIL | 2.000, 1000 ok |
| 22050 | 1.000, 2000 FAIL | 2.000, 1000 ok |
| 32000 | 1.451, 1378 FAIL | 2.016, 1000 ok |
| 44100 | 2.000, 1000 ok | 2.023, 1000 ok |
| 48000 | 2.177, 919 FAIL | 2.000, 1000 ok |

| Octane | Pass | Fail |
|---|---|---|
| With the fix (4ec2939) | 86 of 86 | 0 |
| Upstream b188bb9 | 14 of 86 | 72 |

The upstream passes are all of 44.1 kHz (12) and both `sticky_*` cases. The
sticky cases cannot fail there: with no rate handling, every rate is 44.1 kHz.
They guard against the fix keeping a stale rate.

O2 (MACE audio), IRIX 6.5.30 (`o2-irix6530-20261028` disk, `uname -R`
`6.5 6.5.30m`), 2026-10-29. Ranges cover `gen`, `playaiff` and `sfplay`,
mono and stereo, 8 and 16 bit.

| Rate | Before the MACE fix (length s, pitch Hz) | With it |
|---|---|---|
| 8000 | 2.080 to 2.093, 1045 to 1166 FAIL | 2.000 to 2.003, 999 to 1000 ok |
| 11025 | 2.071 to 2.076, 1036 to 1092 FAIL | 2.000, 1000 ok |
| 16000 | 2.009 to 2.016, 1001 to 1008 ok | 2.000, 1000 ok |
| 22050 | 2.004 to 2.014, 1000 to 1004 ok | 1.998 to 2.001, 1000 ok |
| 32000 | 2.004 to 2.011, 1001 to 1005 ok | 1.998 to 2.000, 1000 ok |
| 44100 | 2.005 to 2.010, 1001 to 1006 ok | 2.000 to 2.004, 998 to 1000 ok |
| 48000 | 2.006 to 2.011, 1001 to 1006 ok | 1.999 to 2.001, 999 to 1000 ok |

| O2 | Pass | Fail |
|---|---|---|
| With the MACE fix | 86 of 86 | 0 |
| Before (4ec2939) | 62 of 86 | 24 |

Before the fix every 8000 and 11025 Hz case failed through all three paths.
The rate was right, but the stream had zeros inserted in the middle of the
sine (about one in 33 at 8 kHz), which added zero crossings. IRIX keeps an
output ring about 2 ms ahead of the read pointer it reads each millisecond,
and MACE only moved that pointer on the service thread's passes, about
2.1 ms apart on a quiet machine. A pass then wanted 16 or 17 frames at
8 kHz where 16 or fewer were queued, and played a zero for each missing
one. Register reads now bring the rings up to date first.

Indy and Indigo2 (HAL2): not covered by the sweep. `IRIS_AUDIO_DUMP` records
codec A at its own rate, in the same record layout minus the output number.


## Gotchas

- `cc` prints a MIPSpro license-password notice on this disk and builds
  anyway. `irix.sh` succeeds; check that `audiotest` exists.
- The guest's `/tmp` is cleared on boot. The sweep keeps its files in
  `/usr/tmp`; stage them again on a fresh boot (`host.sh` does).
- IRIX `grep` has no `\|`; use `egrep`.
- The root shell is csh. `$?` is an error there; use `$status`, or run a
  `sh` script.
- A command line longer than about 250 characters never reaches the shell
  and `iris-ci run` times out. Put long scripts in a file.
- iris creates the capture file on the first sound, so it can be missing
  before the first case; `host.sh` counts that as size 0. The O2's dump
  already holds the boot chime (22 MB at 34 and 48 kHz) by login; marks are
  offsets, so it does not matter.
- The O2 dump's cases are the records that start inside the case's span, in
  the group (output, rate) with the most non-zero samples. A program that
  sets the rate after it starts leaves records at the old rate: the idle
  channel's silence, 0.4 s of it at 48 kHz before the first case, which
  outnumbers an 8 kHz tone.
- `iris-ci put` fails with "scratch volume not configured" if the config
  has no scratch disk.
- `playaiff` and `sfplay` read the AIFF's rate and set it themselves; only
  `audiotest -n` plays at whatever the device was left on.
