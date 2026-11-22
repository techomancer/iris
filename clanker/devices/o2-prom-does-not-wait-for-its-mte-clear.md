# The O2 PROM does not wait for its MTE clear: a status read must

Status: fixed (2026-11-12).

**Symptom.** On a busy host the O2 now and then sat at `SL-9600-8E>` on
serial with nothing before it, instead of reaching the PROM menu. A later
form of the same race prints nothing and parks the CPU at `0xbfc003a0` on a
reserved instruction.

**Cause.** PROM 4.18 clears 32 KB at RAM base with the MTE (`0xbfc061f0`),
reads RE status once without acting on it (`0xbfc06400`), stores a cookie at
`0xa000_1008` and copies its POST code to `0xa000_4000`. It checks the cookie
after POST (`0xbfc044f4`) and drops to the serial loader if it changed. The MTE
runs on the CRIME-RE host thread, and a status read waited at most 100 us.
When the host kept that thread off a processor longer, the clear ran late and
zeroed the cookie, or later still the POST code.

**Rule.** A status read answers `BUS_BUSY` until every MTE transfer queued
before it is done (`transfers_queued` against `transfers_done` in `CrimeRe`).
Primitives keep the bounded wait. MTE copies longer than 100 us lose their
overlap with the CPU. Software that reuses memory a transfer wrote without
reading the status word still races the engine. The same race exists for
primitives, whose frame buffers are main memory: a read-back with no idle wait
could see pixels not yet drawn. It is left uncovered on purpose, as no known
guest does that and holding primitives would cost drawing overlap.

## Issue #301 is this race (checked 2026-11-20)

techomancer/iris#301 (filed 2026-10-24) reported about 1 boot in 70 stopping at
`SL-9600-8E>`, and once, under heavy host load, POST dying with
`EXC code:8, Read TLB Miss, Bad addr: 0xff800000`, and suspected the full cache
model the PROM then ran on. That model is gone (the O2 runs the shadow cache from
reset since 9350cf01), and the serial loader symptom is the cookie wipe above.

Measured with `tools/prom-boot-loop.py` on macOS arm64, release build without
jitv2 or lightning, R5000, the disk-less O2 config from the script's help. A good
start reaches `Option?` in about 5 s. "Stress" is six `while :; do :; done` shells
at nice 0 with iris at nice 10, on a host already running other emulators (load
13 to 72 on 16 cores):

| Build | Load | Starts | Not ok |
|---|---|---|---|
| upstream/main a7b9abca | other sessions only (load ~13) | 100 | 0 |
| upstream/main a7b9abca | stress | 300 | 0 |
| same, status guard disabled (`if false &&`) | stress | 150 | 1: no serial output at all, a POST hang |

The control hang is the late form of the race (the clear lands on the POST code),
so the stress does reach the window. 400 starts without a failure against the
reported 1 in 70 (about 6 expected) closes the serial loader half of #301.

The `0xff800000` TLB miss was seen once and not again; not reproduced here. A
wiped POST code or stack sends POST to arbitrary addresses, which fits, but that
is not shown. If it comes back on a current build, keep the serial log and look
at the CPU's PC and `ra` from the monitor before stopping iris.

## The race on demand: `IRIS_IP32_RE_LAG_US` (2026-11-22)

`IRIS_IP32_RE_LAG_US=N` makes the engine's thread sleep N us before every MTE
transfer, so the window opens without loading the host. Same O2 config, macOS
arm64, release build, R5000, a quiet host (load ~2), the status guard disabled
(`if false &&`) as the control:

| Lag (us) | Guard disabled | Upstream (guarded) |
|---|---|---|
| 0, 500 | 20/20 ok | 20/20 ok |
| 600 | 24 of 25 stop at `SL-9600-8E>` in about 1.5 s | 20/20 ok |
| 800 to 5000 | every start hangs, no serial output (50 of 50) | 40/40 ok |
| 20000 | 5 of 10 hang | 10/10 ok, 25 s a start |

The hang is the POST copy wiped: Cause RI, EPC `0xa0006008` (inside the copy at
`0xa0004000`), PC spinning at `0xbfc003a4` in the PROM's BEV handler. 600 us
here and 1400 us on the Linux host of e013d1e0 are the same window; where it sits
depends on how fast the CPU thread reaches the cookie store. A regression check
is one or two starts at 600 and at 5000: any start that is not ok means the
status read stopped waiting for the engine.
